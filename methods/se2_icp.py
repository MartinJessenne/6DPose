"""
SE(2)-constrained point-to-plane ICP (Gauss-Newton) with a GNC robust kernel.

Each Gauss-Newton increment xi = (omega, vx, vy) is composed onto the current
pose through the se(2) exponential map, so every iterate stays exactly on the
SE(2) manifold -- roll and pitch remain zero and z is never touched. Operates on
raw np.ndarray inputs; no open3d. Point-to-plane residuals use the SCENE normals,
matching Open3D's TransformationEstimationPointToPlane convention.

There is exactly one entry point, icp_se2, and it always solves the full 3 DoF
increment. There is no dual-hypothesis refinement: the flip is settled upstream
by FrontFaceGate on geometry, not downstream on a fitness tiebreak.

See docs/explanation/gnc_refinement.md for the graduated non-convexity
derivation, why the capture radius and the outlier threshold had to be split,
why the noise bound is per-correspondence, and the two deliberate departures
from the paper (re-association, and c_bar = 1 sigma).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from methods.depth_noise import DepthSensor
from methods.se2_lie_utils import se2_exp_about, se2_to_se3

# Terminal control parameter: at mu = 1 the surrogate IS the Geman-McClure cost.
# Copied into the schedule rather than divided down to, so `mu == _MU_FINAL` is
# an exact test.
_MU_FINAL = 1.0

# Unknowns per iteration: omega, vx, vy.
_DOF = 3

# Levenberg-Marquardt damping RELATIVE to the normal matrix's own scale, which
# caps the damped condition number at ~1/1e-6 = 1e6. Replaced an absolute
# 1e-12 * eye(3): on a matrix with O(1) entries that is below rounding and
# damped nothing, and np.linalg.solve raises only on EXACT singularity, so an
# ill-conditioned system returned a huge increment silently.
_LM_RELATIVE_DAMPING = 1e-6

# Condition number of the UNDAMPED normal matrix above which the visible
# geometry is not constraining all 3 DoF -- typically a single plane, where the
# along-face slide and the yaw are both unobservable. A diagnostic about the
# scene, not about numerics: the damped solve is still fine, which is why
# nothing would otherwise notice.
_DEGENERATE_CONDITION = 1e8


@dataclass(frozen=True)
class GncSchedule:
    """
    Annealing schedule for the Geman-McClure control parameter mu.

    The IRLS weight (Yang et al. RA-L 2020, eq. 12) for a correspondence with
    residual r and noise bound c is

        w = ( mu c^2 / (r^2 + mu c^2) )^2

    Large mu is nearly quadratic and so nearly convex; mu = 1 recovers the true
    Geman-McClure cost. Annealing downward re-introduces the non-convexity only
    once the iterate is inside the right basin.

    Attributes:
        shrink: Divisor per step, mu <- mu / shrink. 1.4 is Yang et al. Remark 5
            verbatim. Clamped up when the budget cannot fit the anneal; see
            iteration_mu.
    """

    shrink: float = 1.4  # Yang et al. RA-L 2020, Remark 5

    def __post_init__(self):
        if self.shrink <= 1.0:
            raise ValueError(f"shrink must be greater than 1, got {self.shrink}")

    def iteration_mu(self, mu_init: float, budget: int) -> list[float]:
        """
        The control parameter for each of `budget` iterations, annealing from
        mu_init down to 1 and holding there. Always exactly `budget` long, so
        the ICP loop needs no separate iteration counter.

        The anneal wants N = ceil(ln(mu_init) / ln(shrink)) + 1 entries. When
        that exceeds the budget the fix is to anneal FASTER, not to stop short:
        stopping short leaves the kernel wide, i.e. no outlier rejection at all,
        while still returning a pose. So shrink is clamped up to
        mu_init ** (1/(budget-1)), the smallest value that fits, and the clamp
        is logged because the right response is to fix the configuration.

        The `len(out) < budget - 1` guard is belt-and-braces against the
        exponentiation rounding down by an ulp.
        """
        if budget < 1:
            raise ValueError(f"budget must be at least 1 iteration, got {budget}")
        if mu_init <= _MU_FINAL:
            return [_MU_FINAL] * budget
        if budget == 1:
            logging.warning(
                f"GNC: a 1-iteration budget cannot anneal from mu={mu_init:.4g}; "
                f"running a single pass at mu={_MU_FINAL}, so the kernel does no "
                "graduated non-convexity at all."
            )
            return [_MU_FINAL]

        shrink = max(self.shrink, mu_init ** (1.0 / (budget - 1)))
        if shrink > self.shrink:
            logging.warning(
                f"GNC: annealing mu={mu_init:.4g} -> {_MU_FINAL} at "
                f"shrink={self.shrink:.4g} needs "
                f"{int(np.ceil(np.log(mu_init) / np.log(self.shrink))) + 1} iterations "
                f"but the budget is {budget}. Using effective shrink {shrink:.4g} "
                "instead, which is likelier to overshoot the capture basin -- raise "
                "max_iterations or lower max_correspondence_distance."
            )

        out: list[float] = []
        mu = float(mu_init)
        while mu > _MU_FINAL and len(out) < budget - 1:
            out.append(mu)
            mu /= shrink
        out.append(_MU_FINAL)
        return out + [_MU_FINAL] * (budget - len(out))


@dataclass(frozen=True)
class IcpResult:
    """
    What the GNC refinement computed. Deliberately not RegistrationResult, whose
    `fitness` is a hard inlier count -- correct for a global stage, meaningless
    here now that the radius is a capture basin rather than an outlier test.

    Attributes:
        transformation: 4x4, exactly planar.
        effective_inlier_fraction: sum(w) / n_model at mu = 1. An effective
            COUNT, not a residual: w = 0.5 means that correspondence counted
            half. Strictly lower than the old hard fitness (~0.68 -> ~0.54 at
            this rig's residuals) because points inside the radius now
            contribute w < 1.
        robust_rmse: sqrt(sum(w r^2) / sum(w)) over the POINT-TO-PLANE residual,
            the quantity actually minimised. The old inlier_rmse was Euclidean
            and so could not be compared against the kernel scale at all.
        median_kernel_scale: median c_bar in meters at the final pose. Without
            it the two numbers above are not interpretable, since both are
            relative to a per-correspondence scale that varies with range.
    """

    transformation: np.ndarray
    effective_inlier_fraction: float
    robust_rmse: float
    median_kernel_scale: float


def _gnc_weight(residuals: np.ndarray, mu: float, noise_bound: np.ndarray) -> np.ndarray:
    """
    Geman-McClure IRLS weight (Yang et al. eq. 12), elementwise. Broadcasts over
    an array of per-correspondence noise bounds exactly as it did over a scalar.
    """
    s2 = mu * noise_bound**2
    return (s2 / (s2 + residuals**2)) ** 2


def _make_evaluator(model_points, scene_tree, max_correspondence_distance):
    """
    Data association. The KD-tree radius is the capture basin and nothing else.
    Points with no neighbour come back as dists == inf and idx == len(scene),
    cKDTree's "not found" sentinel rather than a real index, so masking on
    np.isfinite before indexing is mandatory, not defensive.
    """

    def _evaluate(T):
        transformed = model_points @ T[:3, :3].T + T[:3, 3]
        dists, idx = scene_tree.query(
            transformed, k=1, distance_upper_bound=max_correspondence_distance
        )
        valid = np.isfinite(dists)
        return transformed, dists, idx, valid

    return _evaluate


def _associate(evaluate, T, scene_points, scene_normals, sensor):
    """
    One association pass. Returns None when fewer than _DOF correspondences
    survive the radius, otherwise (p, n, r, c_bar): transformed model points,
    matched scene normals, point-to-plane residuals, per-correspondence noise
    bounds.
    """
    transformed, _dists, idx, valid = evaluate(T)
    if valid.sum() < _DOF:
        return None
    p = transformed[valid]
    q = scene_points[idx[valid]]
    n = scene_normals[idx[valid]]
    r = np.einsum("ij,ij->i", n, p - q)
    return p, n, r, sensor.noise_bound(q)


def _initial_mu(residuals: np.ndarray, noise_bound: np.ndarray) -> float:
    """
    mu_0 = 2 * max(r / c_bar)^2 -- Yang et al. Remark 5's 2 r_max^2 / c_bar^2,
    generalised to a per-correspondence c_bar by normalising each residual by
    its own bound. Dimensionless, as mu must be.

    Deriving it from the data is what makes the anneal adaptive: an easy frame
    gets a short schedule. It cannot run away, since r <= radius by the KD-tree
    query and c_bar is bounded below by the near-range noise.
    """
    return 2.0 * float(np.max((residuals / noise_bound) ** 2))


def _jacobian_se2(p: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    (K, 3) Jacobian of r_i with respect to xi = (omega, vx, vy), rotation taken
    about `c`. A small rotation about c displaces p_i by
    omega * (-(p_y - c_y), p_x - c_x); dotting that into n_i gives column one.
    """
    return np.column_stack(
        [
            n[:, 1] * (p[:, 0] - c[0]) - n[:, 0] * (p[:, 1] - c[1]),  # d r / d omega
            n[:, 0],  # d r / d vx
            n[:, 1],  # d r / d vy
        ]
    )


def _compose_se2(T: np.ndarray, xi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Applies the increment on the LEFT: T <- exp_c(xi) . T. Left composition acts
    in world coordinates, which is the frame p, n and c are expressed in;
    composing on the right would apply it in the model frame and be silently
    wrong.
    """
    return se2_to_se3(se2_exp_about(xi, c)) @ T


def _score(T: np.ndarray, associated, n_model: int) -> IcpResult:
    """
    Builds the result at mu = _MU_FINAL regardless of where the loop stopped:
    mu is a homotopy parameter, not a property of the fit, so a run that
    exhausted its budget mid-anneal must still be scored against the true cost
    for its numbers to mean the same thing as a converged run's.
    """
    if associated is None:
        return IcpResult(T, 0.0, np.inf, np.nan)
    _p, _n, r, c_bar = associated
    w = _gnc_weight(r, _MU_FINAL, c_bar)
    w_sum = float(w.sum())
    return IcpResult(
        transformation=T,
        effective_inlier_fraction=w_sum / n_model if n_model else 0.0,
        robust_rmse=float(np.sqrt(np.sum(w * r**2) / w_sum)) if w_sum > 0 else np.inf,
        median_kernel_scale=float(np.median(c_bar)),
    )


def icp_se2(
    model_points,  # (N, 3) model points (already sampled from the CAD)
    scene_points,  # (M, 3) scene points in the Z-up robot base frame
    scene_normals,  # (M, 3) scene normals (oriented toward the sensor)
    T_init,  # (4, 4) SE(2)-embedded initial pose
    max_correspondence_distance,
    *,
    sensor: DepthSensor,
    gnc: GncSchedule,
    max_iterations: int = 100,
    tolerance: float = 1e-8,  # stop when the increment norm falls below this
) -> IcpResult:
    """
    Minimizes sum_i w_i [ n_i . (T p_i - q_i) ]^2 over T in SE(2), with w_i the
    Geman-McClure weight at the current control parameter. One iteration per
    schedule entry: associate, build the weighted normal equations, solve,
    compose.

    Args:
        sensor: Supplies the per-correspondence noise bound the kernel scale is
            measured in.
        gnc: Anneals mu from a data-derived mu_0 down to 1, after which the loop
            keeps iterating at mu = 1 until the increment falls below
            `tolerance`.

    Returns:
        IcpResult evaluated at the final pose.
    """
    model_points = np.asarray(model_points, dtype=float)
    scene_points = np.asarray(scene_points, dtype=float)
    scene_normals = np.asarray(scene_normals, dtype=float)
    T = np.array(T_init, dtype=float, copy=True)

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)
    evaluate = _make_evaluator(model_points, scene_tree, max_correspondence_distance)

    # One association ahead of the loop: the schedule must exist before the
    # first iteration, and its starting point depends on the residuals at T_init.
    associated = _associate(evaluate, T, scene_points, scene_normals, sensor)
    if associated is None:
        logging.warning(
            f"SE(2) ICP: fewer than {_DOF} correspondences within "
            f"{max_correspondence_distance} m at the initial pose; no refinement possible."
        )
        return _score(T, None, n_model)

    for mu in gnc.iteration_mu(_initial_mu(associated[2], associated[3]), max_iterations):
        if associated is None:
            logging.warning(
                f"SE(2) ICP: fewer than {_DOF} correspondences within "
                f"{max_correspondence_distance} m; stopping refinement."
            )
            break

        p, n, r, c_bar = associated
        c = p[:, :2].mean(axis=0)
        J = _jacobian_se2(p, n, c)
        w = _gnc_weight(r, mu, c_bar)

        # w[:, None] * J scales each ROW of J by that row's weight, which is what
        # down-weighting an observation means. Weighting a column would rescale a
        # parameter instead.
        JTWJ = J.T @ (w[:, None] * J)

        cond = np.linalg.cond(JTWJ)
        if cond > _DEGENERATE_CONDITION:
            logging.warning(
                f"SE(2) ICP: normal equations near-degenerate (cond={cond:.3g}) at "
                f"mu={mu:.4g} with {len(p)} correspondences -- the visible model "
                "geometry may be a single plane."
            )

        JTJ = JTWJ + (_LM_RELATIVE_DAMPING * np.trace(JTWJ) / _DOF) * np.eye(_DOF)
        try:
            xi = np.linalg.solve(JTJ, -(J.T @ (w * r)))
        except np.linalg.LinAlgError:
            logging.warning("SE(2) ICP: singular normal equations; stopping refinement.")
            break

        T = _compose_se2(T, xi, c)
        associated = _associate(evaluate, T, scene_points, scene_normals, sensor)
        # Converging mid-anneal is not convergence: the cost is still changing
        # under the iterate. Only a small step at mu = 1 ends the search.
        if np.linalg.norm(xi) < tolerance and mu == _MU_FINAL:
            break

    return _score(T, associated, n_model)
