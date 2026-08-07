"""
SE(2)-constrained point-to-plane ICP (Gauss-Newton) for ground-bounded objects.

Replaces the unconstrained SE(3) ICP refinement for the 3 DoF pipeline: each
Gauss-Newton increment xi = (omega, vx, vy) is composed onto the current pose
through the se(2) exponential map (methods/se2_lie_utils.se2_exp), so every
iterate stays exactly on the SE(2) manifold — roll and pitch remain zero and
the z translation is never touched. No final re-projection is needed.

Like the rest of the constrained pipeline, this module operates on raw
np.ndarray inputs and does not depend on open3d. Point-to-plane residuals use
the SCENE normals (target normals), matching Open3D's
TransformationEstimationPointToPlane convention.

There is deliberately no dual-hypothesis refinement here any more. Refining a
180-degree-flipped copy and picking a winner on fitness or RMSE was an attempt to
resolve the flip with scores that are symmetric under it -- the margins involved
were 2% and less on a near-symmetric object. The flip is now settled in the
global stage by FrontFaceGate (methods/ransac3dof.py), on geometry rather than
on fit, before a candidate is ever scored.

The single icp_max_correspondence_distance hyperparameter which was carrying all
the importance weight in the previous parameter sweeps was in fact doing two jobs at
the same time:
Job 1 -- the capture basin :
    it's the maximal distance at which a model point can be considered to have a valid
    correspondence in the scene, and therefore the maximal distance at which a
    model point can contribute to the residual sum-of-squares. This is the "capture basin" of the ICP refinement.
    Heuristically, the capture basin should be set at a value large enough to get a chance to converge.
Job 2 -- outlier rejection :
    it's how large a residual is still noise-plausible. It must be tight enough or ICP pairs model points
    with the wrong scene points and settles in a biased minimum. (This was the 2.2 cm fixed point of the previous iterations).

Gnc splits them and schedule progressively the job 2 with a robust Geman-McClure kernel toward a minimal value.

ONE SOLVER, ONE MOTION
----------------------
There is exactly one entry point, icp_se2, and it always solves the full 3 DoF
increment xi = (omega, vx, vy).

There used to be a second, yaw-frozen motion model, selected per call. It served
one caller: a post-hoc yaw guard in Ransac3DoFEstimator that re-ran a refinement
stage with rotation disabled whenever the first solve rotated too far. That guard
existed because front_crop_aspect defaulted to 1.0, which made the front slab
square BY CONSTRUCTION (depth = l_y, and the width is l_y) and put a genuine
90-degree minimum in the objective.

At the current default of 2.0 the slab is ~2:1 on all three carts and the twin is
gone. Measured, RNG-free, ICP initialised at a known yaw offset: at aspect 1.0
the guard fires at +/-70 degree starts and the frames still end 55-58 degrees
wrong -- it froze a bad yaw, it never recovered one. At aspect 2.0 nothing fires
out to +/-80 degrees and guarded and unguarded output is bit-identical on every
cart. The capture basin widened from about +/-45 to at least +/-80 degrees.

So the second motion model was not solving a problem, it was patching one the
crop had created. Both are deleted.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from methods.registration_result import RegistrationResult
from methods.se2_lie_utils import se2_exp_about, se2_to_se3


@dataclass(frozen=True)
class GncSchedule:
    """
    Graduated non-convexity schedule for the robust kernel scale.

    The kernel is Geman-McClure, whose IRLS weight at scale s is

        w(r; s) = ( s^2 / (s^2 + r^2) )^2

    with r the point-to-plane residual n . (T p - q). Read off the shape:
    w(0) = 1, w(s) = 1/4, w(3s) = 1/100. So s is the residual at which a
    correspondence is already down to quarter weight -- it is a noise scale, not
    a cut-off, and nothing is ever discarded discontinuously.

    Annealing starts at s = max_correspondence_distance (the KD-tree radius, so
    the first pass weights everything it can even see roughly equally) and
    divides s by `shrink` per step until it reaches `scale_min`. A large s makes
    the cost nearly quadratic and therefore nearly convex -- one broad basin,
    no local minima to fall into. Shrinking s progressively re-introduces the
    non-convexity, but by then the iterate is already inside the right basin.
    That ordering is the whole point of GNC: minimising the sharp cost directly
    from a mediocre initialisation is exactly what falls into a symmetry twin.

    Attributes:
        scale_min: Final kernel scale in meters -- the depth-sensor noise floor
            at the working range. Fixed by the sensor, NOT tuned: for a stereo
            pair, sigma_z = z^2 * sigma_d / (f * B), and this rig has
            f * B = 639.99768 px * 0.095 m = 60.80 px.m. At a typical cart range
            z = 3.0 m and a subpixel disparity accuracy sigma_d = 0.1 px,
            sigma_z = 9.0 * 0.1 / 60.80 = 0.0148 m. Annealing below the noise
            floor cannot help: it down-weights correspondences whose residual is
            real measurement noise, which throws away signal.
        shrink: Divisor applied to s per annealing step. sqrt(1.4) = 1.1832
            reproduces the mu / 1.4 schedule of Yang et al., "Graduated
            Non-Convexity for Robust Spatial Perception" (RA-L 2020), because
            their control parameter mu enters the weight as s^2 = mu * c^2 --
            dividing mu by 1.4 divides s by sqrt(1.4). Larger values anneal
            faster and risk overshooting the basin; smaller values cost
            iterations for nothing. Clamped UP when the iteration budget cannot
            fit the anneal -- see iteration_scales.
    """

    scale_min: float
    shrink: float = 1.1832  # sqrt(1.4), see Yang et al. RA-L 2020

    def __post_init__(self):
        if self.scale_min <= 0:
            raise ValueError(f"scale_min must be positive got {self.scale_min}")
        if self.shrink <= 1:
            raise ValueError(f"shrink must be greater than 1 got {self.shrink}")

    def iteration_scales(self, scale_init: float, budget: int) -> list[float]:
        """
        The kernel scale to use on each of `budget` iterations, annealing from
        scale_init down to scale_min and then holding there.

        The returned list is always exactly `budget` long, so the ICP loop can
        iterate over it directly and needs no separate iteration counter.

        HOW LONG THE ANNEAL WANTS TO BE
        -------------------------------
        Write s0 = scale_init, m = scale_min, k = shrink. The sequence is
        s_j = s0 / k^j, and it stops at the first j with s_j <= m, then appends
        m itself. That first index is

            J = ceil( ln(s0/m) / ln(k) )

        so the anneal wants N = J + 1 entries. Note what is NOT in that formula:
        the residuals. The length depends only on the ratio s0/m and on k, so it
        is known before the first correspondence is computed -- annealing cannot
        run away on a hard frame.

        WHEN THE BUDGET IS TOO SMALL
        ----------------------------
        If N > budget the anneal cannot finish. Ending it early would leave the
        kernel wide, which means no outlier rejection at all -- the silent
        version of this failure is worse than the loud one, because the pose
        still comes back and merely happens to be biased.

        The fix is to anneal FASTER, not to stop short, so `shrink` is clamped
        up to whatever the budget affords. Requiring N <= budget:

            J <= budget - 1
            ln(s0/m) / ln(k) <= budget - 1
            k >= (s0/m)^(1 / (budget - 1))

        so the effective shrink is

            k_eff = max( k, (s0/m)^(1 / (budget - 1)) )

        which is the smallest value that both fits the budget and never anneals
        slower than configured. The cost is real -- a coarser anneal is likelier
        to overshoot the basin -- so it is logged at WARNING with both values,
        because the correct response is to fix the configuration, not to rely on
        the clamp.

        The `len(out) < budget - 1` guard in the loop is belt-and-braces: the
        formula is exact over the reals, but `ratio ** (1/(budget-1))` can round
        down by an ulp and yield one entry more than intended. The guard makes
        the length bound hold arithmetically rather than by trust.

        Returns [scale_min] * budget when scale_init is already at or below the
        noise floor -- a capture radius tighter than the sensor noise leaves
        nothing to anneal, and iterating at a scale wider than the radius would
        be a lie about what the kernel is doing.

        The final element is `self.scale_min` itself, copied rather than arrived
        at by dividing. _at_final_scale depends on that: it compares with `==`,
        which is exact only because no arithmetic touches the value.
        """
        if budget < 1:
            raise ValueError(f"budget must be at least 1 iteration, got {budget}")
        if scale_init <= self.scale_min:
            return [self.scale_min] * budget
        if budget == 1:
            logging.warning(
                f"GNC: a 1-iteration budget cannot anneal from {scale_init:.4g} m; "
                f"running a single pass at scale_min={self.scale_min:.4g} m, so the "
                "kernel does no graduated non-convexity at all."
            )
            return [self.scale_min]

        ratio = scale_init / self.scale_min
        shrink = max(self.shrink, ratio ** (1.0 / (budget - 1)))
        if shrink > self.shrink:
            logging.warning(
                f"GNC: annealing {scale_init:.4g} m -> {self.scale_min:.4g} m at "
                f"shrink={self.shrink:.4g} needs "
                f"{int(np.ceil(np.log(ratio) / np.log(self.shrink))) + 1} iterations "
                f"but the budget is {budget}. Using effective shrink "
                f"{shrink:.4g} instead. This anneals faster than configured and is "
                "likelier to overshoot the capture basin -- raise max_iterations or "
                "lower max_correspondence_distance."
            )

        out: list[float] = []
        scale = scale_init
        while scale > self.scale_min and len(out) < budget - 1:
            out.append(scale)
            scale /= shrink
        out.append(self.scale_min)
        return out + [self.scale_min] * (budget - len(out))


def _geman_mcclure_weight(residuals: np.ndarray, scale: float) -> np.ndarray:
    """
    Geman-McClure IRLS weight for a given residual array and kernel scale element-wise.

    w(r; s) = ( s^2 / (s^2 + r^2) )^2
    """
    s2 = scale**2
    r2 = residuals**2
    return (s2 / (s2 + r2)) ** 2


def _make_evaluator(model_points, scene_tree, max_correspondence_distance):
    """
    Builds the closure that given a transformation T
    returns the results of the transform and of the kd-tree query
    """

    def _evaluate(T):
        transformed = model_points @ T[:3, :3].T + T[:3, 3]
        dists, idx = scene_tree.query(
            transformed, k=1, distance_upper_bound=max_correspondence_distance
        )
        valid = np.isfinite(dists)
        return transformed, dists, idx, valid

    return _evaluate


def _at_final_scale(scale: float, gnc: GncSchedule) -> bool:
    """
    True when `scale` is the last one the schedule will ever use.

    The `==` is exact, not approximate: GncSchedule.iteration_scales puts
    `gnc.scale_min` itself into the list, copied rather than arrived at by
    dividing. If the tail ever becomes a computed value this must become a
    tolerance comparison.
    """
    return scale == gnc.scale_min


# Unknowns per iteration: omega, vx, vy. Sizes both the minimum correspondence
# count (below it the normal equations are rank-deficient) and the ridge.
_DOF = 3


def _jacobian_se2(p: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    (K, 3) Jacobian of r_i with respect to xi = (omega, vx, vy), the rotation
    taken about `c`.

    A small rotation about c displaces p_i by omega * (-(p_y - c_y), p_x - c_x),
    perpendicular to the lever arm from the centre and proportional to its
    length; dotting that into n_i gives the first column. The other two are the
    translation columns above.
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
    Applies a full SE(2) increment on the LEFT: T <- exp_c(xi) . T.

    Left composition acts in world coordinates, which is the frame p, n and c
    are all expressed in. Composing on the right would apply the increment in
    the model's own frame and be silently wrong.
    """
    return se2_to_se3(se2_exp_about(xi, c)) @ T


def icp_se2(
    model_points,  # (N, 3) model points (already sampled from the CAD)
    scene_points,  # (M, 3) scene points in the Z-up robot base frame
    scene_normals,  # (M, 3) scene normals (oriented toward the sensor)
    T_init,  # (4, 4) SE(2)-embedded initial pose
    max_correspondence_distance,
    *,
    gnc: GncSchedule,
    max_iterations: int = 100,
    tolerance: float = 1e-8,  # stop when the increment norm falls below this
) -> RegistrationResult:
    """
    SE(2)-constrained point-to-plane ICP with a GNC-annealed robust kernel.

    Minimizes sum_i w_i [ n_i . (T p_i - q_i) ]^2 over T in SE(2) (embedded in
    SE(3) with fixed z), where q_i / n_i are the nearest scene point and its
    normal and w_i is the Geman-McClure weight at the current kernel scale. One
    iteration per entry of the annealing schedule: re-associate, build the
    weighted normal equations at that scale, solve, compose.

    The linearized residual for a left increment xi = (omega, vx, vy) rotating
    about the centroid c is
        dr_i = omega * (n_y (p_x - c_x) - n_x (p_y - c_y)) + n_x vx + n_y vy,
    giving a (K, 3) Jacobian solved through 3x3 normal equations.

    Inputs:
        gnc: Annealing schedule for the kernel scale, from
            max_correspondence_distance down to gnc.scale_min, one step per
            iteration, after which the loop keeps iterating at scale_min until
            the increment falls below `tolerance`. See GncSchedule for why the
            ordering (wide first) is the entire mechanism.

    Returns:
        RegistrationResult with `.transformation` (4x4, exactly planar),
        `.fitness` (inliers / N model points) and `.inlier_rmse`
        (Euclidean RMSE over matched pairs), all evaluated at the final pose.

        fitness and inlier_rmse keep their hard-threshold definition under the
        robust kernel, deliberately: they are cross-arm diagnostics, and
        redefining them to be weight-based would make a GNC run's numbers
        incomparable with every run already recorded.
    """
    model_points = np.asarray(model_points, dtype=float)
    scene_points = np.asarray(scene_points, dtype=float)
    scene_normals = np.asarray(scene_normals, dtype=float)
    T = np.array(T_init, dtype=float, copy=True)

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)
    _evaluate = _make_evaluator(model_points, scene_tree, max_correspondence_distance)

    for scale in gnc.iteration_scales(max_correspondence_distance, max_iterations):
        transformed, _dists, idx, valid = _evaluate(T)
        if valid.sum() < _DOF:
            logging.warning(
                f"SE(2) ICP: fewer than {_DOF} correspondences within "
                f"{max_correspondence_distance} m; stopping refinement."
            )
            break

        p = transformed[valid]
        q = scene_points[idx[valid]]
        n = scene_normals[idx[valid]]
        c = p[:, :2].mean(axis=0)

        r = np.einsum("ij,ij->i", n, p - q)
        J = _jacobian_se2(p, n, c)
        w = _geman_mcclure_weight(r, scale)

        # w[:, None] * J multiplies each ROW of J by that row's weight, which is
        # what "down-weight this observation" means. Weighting a COLUMN would
        # rescale a parameter instead -- a different, and wrong, operation.
        JTJ = J.T @ (w[:, None] * J) + 1e-12 * np.eye(_DOF)
        try:
            xi = np.linalg.solve(JTJ, -(J.T @ (w * r)))
        except np.linalg.LinAlgError:
            logging.warning("SE(2) ICP: singular normal equations; stopping refinement.")
            break

        T = _compose_se2(T, xi, c)
        # Converging mid-anneal is not convergence: the cost function is still
        # changing under the iterate, so a small step means the solver has
        # caught up with THIS scale, not that it has settled. Only a small step
        # at the final scale ends the search.
        if np.linalg.norm(xi) < tolerance and _at_final_scale(scale, gnc):
            break

    # Evaluate fitness/RMSE at the final pose (the loop metrics lag one update behind).
    _, dists, _, valid = _evaluate(T)
    n_inliers = valid.sum()
    fitness = n_inliers / n_model if n_model else 0.0
    inlier_rmse = np.sqrt(np.mean(dists[valid] ** 2)) if n_inliers > 0 else np.inf
    return RegistrationResult(T, fitness, inlier_rmse)
