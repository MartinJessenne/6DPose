import logging
from collections.abc import Generator
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, Annotated

import numpy as np
from scipy.spatial import cKDTree

from methods.base import SearchRange
from methods.constrained_ransac import RansacResult, match_correspondences_fpfh, se2_to_se3
from methods.ransac3dof import FrontFaceGate, Ransac3DoFEstimator, Ransac3DoFParams
from methods.se2_lie_utils import minimal_solver_se2

if TYPE_CHECKING:
    pass


@dataclass
class MatchedNpPointClouds:
    model_points: np.ndarray  # (N, 3) downsampled model points
    scene_points: np.ndarray  # (M, 3) downsampled scene points
    model_fpfh: np.ndarray  # (N, 33) FPFH descriptors of model
    scene_fpfh: np.ndarray  # (M, 33) FPFH descriptors of scene
    # Oriented normals, for the E1 normal-agreement term. Optional because every
    # caller predating that arm builds this without them, and because the term
    # is off by default -- vsac_se2 falls back to purely positional scoring when
    # either is None, which is exactly the control arm.
    model_normals: np.ndarray | None = None  # (N, 3) outward, CAD convention
    scene_normals: np.ndarray | None = None  # (M, 3) oriented towards the sensor


@dataclass
class VsacConfig:
    """Call-time settings for vsac_se2(). NOT a CLI params class -- these are
    computed per call by VSACSe2Estimator._global_registration (from
    VSACSe2Params plus per-frame state), never parsed from the command line.
    No SearchRange belongs here; VSACSe2Params owns every searchable range."""

    distance_threshold: float
    max_iterations: int = 10_000
    min_iterations: int = 1_000
    confidence: float = 0.999
    z_offset: float = 0.0
    z_gate_threshold: float | None = None
    edge_length_tolerance: float = 0.14
    min_sample_distance: float = 0.0
    scoring_subsample_size: int = 100
    rho: float = 0.3  # Spatial independence radius for flip-disambiguation
    normal_consistency: bool = False  # E1: require agreeing normals to count an inlier


def prosac_pairs(
    n_corr: int, max_iteration: int, rng: np.random.Generator
) -> Generator[tuple[int, int], None, None]:
    """
    This function yiels indices of matched pairs based on the PROSAC progressive sampling strategy

    Args:
        n_corr: the number of matched gated points after fpfh
        max_iteration: the number of RANSAC max iteration
        rng: the potential seeded numpy generator for this experiment

    Yields:
        (i1, i2): indices of matched pairs in the model and scence point clouds generated following the PROSAC schedule
    """
    n = 2  # Minimal number of points needed to estimate the transform so 2 in SE(2)

    if n_corr < n:
        return

    m = 2  # The prefix set starts with the minimal number points needed

    T_n = max_iteration
    T_m = (
        T_n * comb(m, n) / comb(n_corr, n)
    )  # We draw from the prefix set for a number of iterations that is proportional to T_n times the

    for t in range(1, max_iteration + 1):
        if t > T_m and m < n_corr:  # We've oversampled and we're still in the PROSAC Regime
            m += 1
            T_m = T_n * comb(m, n) / comb(n_corr, n)  # update of next saturation time

        if m < n_corr:
            i1 = int(rng.integers(0, m - 1))
            i2 = (
                m - 1
            )  # we automatically select m as one of the indices to prevent oversampling the same subset of progressive matches.

            yield (i1, i2)

        else:  # Revert to the classic RANSAC uniform sampling regime
            i1, i2 = rng.choice(n_corr, size=2, replace=False)
            yield (int(i1), int(i2))


def score_msac(
    transformed_model_points: np.ndarray,
    scene_kdtree_3d: cKDTree,
    tau: float,
    transformed_model_normals: np.ndarray | None = None,
    scene_normals: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Computes MSAC quality score, inlier ratio (fitness), inlier mask, and distance array
    in 3D Euclidean space (x, y, z).

    Positional scoring alone is provably blind to the cart's 180-degree symmetry:
    a flipped pose lands model points on the SAME surfaces, so it queries the same
    neighbours at the same distances and scores identically (see
    30.04.2 - E02 Failure Mode Characterisation). Position is invariant under the
    flip; oriented normals are not. The model's normals point outward from the CAD
    surface and the scene's point back towards the sensor, so on the visible side
    of a solid the two agree. Rotate the model by 180 degrees and the face now
    covering that surface is the far one, whose outward normal points AWAY from
    the sensor -- the dot product changes sign. Requiring
    n_model . n_scene > 0 therefore makes the score asymmetric, which no
    threshold on distance can be.

    This is arm E1 of the flip-disambiguation roadmap. It is a no-op unless both
    normal arrays are supplied, which is the control arm.

    Args:
        transformed_model_points: (N, 3) 3D transformed model point cloud.
        scene_kdtree_3d: 3D cKDTree of scene (x, y, z) points.
        tau: Distance threshold in meters.
        transformed_model_normals: (N, 3) model normals under the SAME candidate
            rotation as `transformed_model_points` (rotation only -- normals do
            not translate). None disables the term.
        scene_normals: (M, 3) scene normals, indexed like the KD-tree's points.
            None disables the term.

    Returns:
        quality: float MSAC quality score sum(tau^2 - r^2)
        fitness: float inlier ratio in [0.0, 1.0]
        inlier_mask: (N,) boolean mask where 3D distance < tau AND, when the
            normal term is active, the matched normals agree
        dists: (N,) 3D Euclidean distance array from cKDTree
    """
    # Compute the 3D distances from the transformed model points to the nearest scene points in (x, y, z)
    distances, indices = scene_kdtree_3d.query(
        transformed_model_points, k=1, distance_upper_bound=tau
    )

    inlier_mask = distances < np.inf

    if transformed_model_normals is not None and scene_normals is not None:
        # A miss returns index == len(scene_points), which would raise on gather.
        # Clip and let inlier_mask discard those rows: the clipped index reads a
        # real but irrelevant normal, and its agreement verdict is ANDed against
        # an already-False positional test.
        safe = np.minimum(indices, len(scene_normals) - 1)
        agree = np.einsum("ij,ij->i", transformed_model_normals, scene_normals[safe]) > 0.0
        inlier_mask &= agree

    fitness = float(np.mean(inlier_mask))

    # Compute the MSAC quality score
    if np.any(inlier_mask):
        quality = np.sum(tau**2 - distances[inlier_mask] ** 2)
    else:
        quality = 0.0

    return quality, fitness, inlier_mask, distances


def count_independent_inliers(
    inlier_mask: np.ndarray,
    model_points_2d: np.ndarray,
    rho: float,
) -> int:
    """
    Counts spatially independent inliers by greedily suppressing inliers within
    model-frame radius `rho`.

    Args:
        inlier_mask: (N,) boolean mask of model point inliers.
        model_points_2d: (N, 2) model points in 2D (x, y).
        rho: Spatial independence radius in meters (e.g. 0.3m).

    Returns:
        n_independent: Integer count of spatially independent inliers.
    """
    if not np.any(inlier_mask):
        return 0

    pts = model_points_2d[inlier_mask]  # Only use x, y for SE(2)
    if len(pts) == 0:
        return 0

    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    n_independent = 0

    for k in range(len(pts)):
        if not visited[k]:
            n_independent += 1
            # Mark all points within radius rho as visited
            indices = tree.query_ball_point(pts[k], r=rho)
            visited[indices] = True

    return n_independent


def vsac_se2(
    point_clouds: MatchedNpPointClouds,
    params: VsacConfig,
    rng: np.random.Generator | None = None,
    front_face_gate: "FrontFaceGate | None" = None,
) -> RansacResult:
    """SE(2) PROSAC + MSAC global registration.

    front_face_gate, when given, vetoes any candidate whose towing face points
    more than the allowed angle away from the camera. It is applied at CANDIDATE
    GENERATION rather than at promotion: the predicate costs about ten flops, so
    discarding the wrong half of the hypothesis space before it reaches a KD-tree
    query lets the same max_iterations budget buy roughly twice the search in the
    half that can still win. Gating at promotion would instead spend full MSAC
    scoring on poses about to be refused.
    """

    model_points = point_clouds.model_points
    scene_points = point_clouds.scene_points

    # E1 normal-agreement term. Requires the flag AND both normal arrays; a
    # missing array silently degrades to the control arm rather than raising,
    # because call sites without mesh context (unit tests, inspect_pose on a bare
    # cloud) legitimately have no normals to supply.
    use_normals = (
        params.normal_consistency
        and point_clouds.model_normals is not None
        and point_clouds.scene_normals is not None
    )
    model_normals = point_clouds.model_normals if use_normals else None
    scene_normals = point_clouds.scene_normals if use_normals else None

    rng = np.random.default_rng() if rng is None else rng

    DOF = 2  # SE(2) has 2 degrees of freedom, we need at least 2 points to estimate the transform

    # PROSAC Step :
    correspondences, distances = match_correspondences_fpfh(
        feat_model=point_clouds.model_fpfh, feat_scene=point_clouds.scene_fpfh, scored=True
    )

    if len(correspondences) < DOF:  # Not enough correspondence found to compute a transform
        return RansacResult(np.eye(4), 0.0, np.inf, reason="fpfh_insufficient")

    MODEL_COL = 0
    SCENE_COL = 1
    scene_correspondence_idx, model_correspondence_idx = (
        correspondences[:, SCENE_COL],
        correspondences[:, MODEL_COL],
    )

    Z_AXIS = 2

    z_gate = (
        params.z_gate_threshold
        if params.z_gate_threshold is not None
        else params.distance_threshold
    )
    scene_z = scene_points[scene_correspondence_idx][:, Z_AXIS]
    model_z = model_points[model_correspondence_idx][:, Z_AXIS] + params.z_offset
    dz = np.abs(model_z - scene_z)

    gated_correspondences = correspondences[dz < z_gate]
    gated_distances = distances[dz < z_gate]

    if len(gated_correspondences) < DOF:
        return RansacResult(np.eye(4), 0.0, np.inf, reason="z_gate_insufficient")

    # Sort the correspondences by distances
    sort_order = np.argsort(gated_distances)
    sorted_gated_correspondences = gated_correspondences[sort_order]

    # Run the PROSAC RANSAC
    sampler = prosac_pairs(
        n_corr=len(sorted_gated_correspondences), max_iteration=params.max_iterations, rng=rng
    )

    # Correspondence coordinates for the adaptive-termination inlier ratio w
    # (below), precomputed once -- mirrors constrained_ransac_se2's corr_p/corr_q
    # in 3D (x, y, z) for 3D MSAC scoring consistency.
    corr_p_3d = model_points[sorted_gated_correspondences[:, 0]]
    corr_q_3d = scene_points[sorted_gated_correspondences[:, 1]]

    # Prepare a random subset of the model points for sub quality MSAC scoring to unbias the process.
    n_model = len(model_points)
    subsample_size = min(params.scoring_subsample_size, n_model)
    subsample_indices = rng.choice(n_model, size=subsample_size, replace=False)

    # 3D KD-Tree for fast nearest neighbor search in Euclidean (x, y, z) space.
    # Evaluated in 3D to prevent vertical height ambiguities.
    # Example: If a scene point q sits at (1.0, 0.5, 2.5m) (e.g. on a wall or ceiling)
    # and a transformed model point p lands at (1.0, 0.5, 0.1m) (on the cart chassis),
    # their 2D distance is sqrt(dx^2 + dy^2) = 0.0m < tau, which would falsely mark
    # ceiling clutter as a cart inlier. 3D querying enforces r_3D = 2.4m > tau.
    scene_kdtree_3d = cKDTree(scene_points)  # Full 3D (x, y, z) point cloud
    best_quality = 0.0
    best_fitness = 0.0
    best_n_independent = 0
    best_transformation = np.eye(4)
    best_rmse = np.inf
    best_inlier_mask: np.ndarray | None = None
    # Instrumentation, not bookkeeping. Three previous gates in this pipeline
    # "improved" a headline rate by vetoing nearly everything and turning flips
    # into abstentions -- a rate with a success denominator always improves when
    # you empty it. The RATIO below is what distinguishes a working gate from a
    # silently inverted one, and it is cheap enough to log every frame.
    n_orientation_rejected = 0
    n_candidates_valid = 0

    # Adaptive early termination (ported from constrained_ransac_se2): without
    # this, the loop always drains the full max_iterations budget regardless
    # of how good the best hypothesis already is, making p95 latency
    # comparisons against constrained_ransac_se2 meaningless.
    required_iterations = params.max_iterations
    it = 0

    for i1, i2 in sampler:
        it += 1
        if it > min(params.max_iterations, required_iterations):
            break

        mi1, si1 = sorted_gated_correspondences[i1]
        mi2, si2 = sorted_gated_correspondences[i2]

        # Extract the 2D points (x, y) for SE(2) transformation estimation
        p1, p2 = model_points[mi1][:Z_AXIS], model_points[mi2][:Z_AXIS]

        q1, q2 = scene_points[si1][:Z_AXIS], scene_points[si2][:Z_AXIS]

        len_p = np.linalg.norm(p1 - p2)
        len_q = np.linalg.norm(q1 - q2)

        # Check if the sampled points are valid based on the edge length and minimum sample distance constraints
        if len_p < params.min_sample_distance or len_q < params.min_sample_distance:
            continue

        if abs(len_p - len_q) > params.edge_length_tolerance:
            continue

        # Compute the candidate transformation using the minimal solver for SE(2)
        sol = minimal_solver_se2(p1, p2, q1, q2)

        if sol is None:
            continue

        theta, t_xy = sol
        T_candidate = se2_to_se3(theta, t_xy, z=params.z_offset)  # 4x4 SE(3) matrix
        n_candidates_valid += 1

        # Orientation veto, before any scoring. See the docstring for why here
        # and not at promotion.
        if front_face_gate is not None and not front_face_gate.accepts(T_candidate):
            n_orientation_rejected += 1
            continue

        # Apply the candidate 4x4 transformation to the 3D model points
        transformed_model_points = model_points @ T_candidate[:3, :3].T + T_candidate[:3, 3]
        # Normals rotate but do not translate. The rotation is a yaw about z and
        # therefore orthogonal, so the rotated normals stay unit length and the
        # sign test below needs no renormalisation.
        transformed_model_normals = (
            model_normals @ T_candidate[:3, :3].T if model_normals is not None else None
        )

        # Prescoring on subsampled 3D points to speed up the process
        sub_quality, _, _, _ = score_msac(
            transformed_model_points=transformed_model_points[subsample_indices],
            scene_kdtree_3d=scene_kdtree_3d,
            tau=params.distance_threshold,
            transformed_model_normals=(
                transformed_model_normals[subsample_indices]
                if transformed_model_normals is not None
                else None
            ),
            scene_normals=scene_normals,
        )

        max_possible_quality = (
            sub_quality + (n_model - subsample_size) * params.distance_threshold**2
        )

        if max_possible_quality < best_quality:
            continue  # Early exit if the maximum possible quality is worse than the best found so far

        # Full 3D model scoring to compute the actual quality and fitness
        quality, fitness, inlier_mask, dists = score_msac(
            transformed_model_points=transformed_model_points,
            scene_kdtree_3d=scene_kdtree_3d,
            tau=params.distance_threshold,
            transformed_model_normals=transformed_model_normals,
            scene_normals=scene_normals,
        )

        n_independent = count_independent_inliers(
            inlier_mask=inlier_mask, model_points_2d=model_points[:, :Z_AXIS], rho=params.rho
        )

        # 1. Clear quality win (>2% MSAC improvement)
        is_quality_better = quality > best_quality * 1.02

        # 2. Quality tie (within 2% band) AND superior spatial support (higher independent inliers)
        is_tiebreak_win = (quality >= best_quality * 0.98) and (n_independent > best_n_independent)

        if (
            is_quality_better or is_tiebreak_win or best_inlier_mask is None
        ):  # Accept the first valid solution
            best_quality = quality
            best_fitness = fitness
            best_n_independent = n_independent
            best_transformation = T_candidate
            best_inlier_mask = inlier_mask
            best_rmse = np.sqrt(np.mean(dists[inlier_mask] ** 2)) if np.any(inlier_mask) else np.inf

            if best_fitness >= 1.0:
                break  # Perfect fit found

            # Update required RANSAC iterations based on correspondence inlier
            # ratio w: chance of picking 2 inliers in a row is w^2, so required
            # iterations = log(1-confidence)/log(1-w^2).
            residuals = np.linalg.norm(
                corr_p_3d @ T_candidate[:3, :3].T + T_candidate[:3, 3] - corr_q_3d,
                axis=1,
            )
            w = (residuals < params.distance_threshold).mean()
            if w >= 1.0:
                required_iterations = params.min_iterations
            elif w > 0.0:
                bound = np.log(1 - params.confidence) / np.log1p(-(w * w))
                required_iterations = int(
                    np.clip(bound, params.min_iterations, params.max_iterations)
                )
            else:
                required_iterations = params.max_iterations

    # Reject only the genuine no-support case: no candidate pose landed a single
    # inlier, so there is nothing to return.
    #
    # ABLATED (commit following 6256d59): a Poisson null-hypothesis gate used to
    # also reject any pose whose independent-inlier count fell below
    # I_delta = lam_hat + 6*sqrt(lam_hat*(1-delta_0)), with lam_hat calibrated from
    # low-fitness candidates seen during the search. At 6 sigma it was rejecting
    # poses wholesale rather than filtering random ones: VSAC's median pose
    # failures went 36 -> 116 out of 210 when it landed (W&B ffx1i5tn -> fgugxxrn)
    # while best AR fell 0.180 -> 0.122, and in the SmokeM2 run every single
    # abstention was attributed to this gate. It was converting flips into
    # abstentions, which the old success-denominator flip_rate scored as an
    # improvement. See git history for the implementation if it needs reviving.
    #
    # NOTE the independent-inlier machinery itself is NOT ablated:
    # count_independent_inliers still drives the is_tiebreak_win spatial-support
    # tiebreak above, which is the actual flip-disambiguation mechanism.
    if front_face_gate is not None and n_candidates_valid:
        ratio = n_orientation_rejected / n_candidates_valid
        # Read this every run. FPFH is flip-symmetric on this geometry, so roughly
        # half the candidates SHOULD point the wrong way:
        #   ~50%  healthy.
        #   ~100% the sign convention is inverted somewhere -- check the arrow is
        #         +x and that extrinsic is camera->robot, not its inverse. Do not
        #         trust a sweep run in this state.
        #   ~0%   the gate is inert: not wired, or the arm never got its override.
        # Both the 100% and the 0% cases otherwise produce a run that looks
        # entirely plausible.
        logging.info(
            f"Front-face gate rejected {n_orientation_rejected}/{n_candidates_valid} "
            f"candidates ({ratio:.1%})."
        )

    if best_fitness == 0.0:
        # "Nothing fit" and "everything that fit was vetoed" are different
        # failures, and the second is the gate's own doing, so it must not hide
        # inside no_inliers -- that is precisely how the earlier gates looked
        # healthy in the logs while emptying the eval set.
        reason = "orientation_rejected" if n_orientation_rejected > 0 else "no_inliers"
        return RansacResult(np.eye(4), 0.0, np.inf, reason=reason)

    return RansacResult(best_transformation, best_fitness, best_rmse)


@dataclass(frozen=True)
class VSACSe2Params(Ransac3DoFParams):
    """Ransac3DoFParams plus the knobs specific to the VSAC global-registration
    stage. Everything else (voxel_size, front_crop_aspect, front_face_max_angle_deg,
    z_gate_threshold, seed, ...) is shared with Ransac3DoFEstimator, since
    VSACSe2Estimator only swaps out _global_registration.

    rho: spatial-independence radius (meters) for the flip-disambiguation
        tiebreak in vsac_se2 -- inliers within rho of an already-accepted
        independent inlier (in the CAD model's own frame) count as
        dependent/clustered rather than adding new spatial support.
    normal_consistency: arm E1. When True, score_msac only counts a model point
        as an inlier if its normal agrees with the matched scene point's
        (n_model . n_scene > 0), making the MSAC score asymmetric under the
        180-degree flip that positional scoring cannot see. False (the default)
        is the control arm. Deliberately absent from suggest_params for the same
        reason as front_face_max_angle_deg: it is an A/B's independent variable,
        and letting Optuna tune it would turn an on/off contrast into a nuisance
        parameter. Requires the model normal-sign fix (methods/base.py
        reorient_normals_to_reference) to be meaningful -- without it the sign
        test reads rounding noise.
    """

    rho: Annotated[float, SearchRange(min=0.05, max=0.6)] = 0.3
    normal_consistency: bool = False


class VSACSe2Estimator(Ransac3DoFEstimator):
    """
    VSAC-style drop-in replacement for Ransac3DoFEstimator's global
    registration stage: PROSAC sampling + MSAC scoring + an independent-inlier
    tiebreak that prefers spatially spread support over the clustered support
    a 180-degree-flipped pose typically accumulates (see
    VSAC_Implementation_Plan.md). Overrides only _global_registration --
    prepare/crop/z-offset, single-pass SE(2) ICP refinement, and the SE(2)
    projection safety net are all inherited unchanged.

    This is also where the front-face orientation veto lives, because this is the
    stage that creates the 180-degree ambiguity in the first place: MSAC counts
    how many model points land near a scene point, which a pose and its flipped
    twin do equally well. Every earlier attempt at the flip acted downstream in
    ICP, on a choice the global stage had already made blind.
    """

    params_cls = VSACSe2Params
    params: VSACSe2Params

    def __init__(
        self,
        params: VSACSe2Params | None = None,
        extrinsic: np.ndarray | None = None,
    ):
        if params is None:
            params = VSACSe2Params()
        super().__init__(params=params, extrinsic=extrinsic)

    def _global_registration(self, model_down, pcd_down, model_fpfh, pcd_fpfh):
        """VSAC replacement for constrained_ransac_se2 (same call site contract
        as Ransac3DoFEstimator._global_registration, methods/ransac3dof.py)."""
        distance_threshold = self.params.voxel_size * 1.5
        point_clouds = MatchedNpPointClouds(
            model_points=np.asarray(model_down.points),
            scene_points=np.asarray(pcd_down.points),
            model_fpfh=np.asarray(model_fpfh.data).T,  # (33, N) -> (N, 33)
            scene_fpfh=np.asarray(pcd_fpfh.data).T,
            # Both clouds already carry oriented normals at this point -- the
            # model's outward (mesh convention, re-signed by
            # reorient_normals_to_reference) and the scene's towards-sensor. They
            # were simply never threaded through to the scorer. Passed
            # unconditionally; params.normal_consistency decides whether they are
            # used, so the two arms differ by one flag and not by a code path.
            model_normals=(np.asarray(model_down.normals) if model_down.has_normals() else None),
            scene_normals=(np.asarray(pcd_down.normals) if pcd_down.has_normals() else None),
        )
        vsac_params = VsacConfig(
            distance_threshold=distance_threshold,
            max_iterations=self.params.ransac_max_iterations,
            confidence=self.params.ransac_confidence,
            z_offset=self._active_z_offset,
            z_gate_threshold=self.params.z_gate_threshold,
            edge_length_tolerance=self.params.edge_length_tolerance,
            # Short sample baselines give yaw hypotheses dominated by voxel
            # noise; require the 2 sampled model points to span a few voxels
            # (matches Ransac3DoFEstimator._global_registration's own rule).
            min_sample_distance=3.0 * self.params.voxel_size,
            rho=self.params.rho,
            normal_consistency=self.params.normal_consistency,
        )
        return vsac_se2(
            point_clouds,
            vsac_params,
            rng=np.random.default_rng(self.params.seed),
            front_face_gate=self._build_front_face_gate(),
        )

    def _build_front_face_gate(self) -> FrontFaceGate | None:
        """
        The orientation veto for this frame, or None when it is switched off.

        Returns None when front_face_max_angle_deg is unset (the control arm) or
        when no cart geometry has been resolved. Returning None rather than
        raising keeps call sites with no mesh context -- unit tests, inspect_pose
        on a bare cloud -- working, instead of silently behaving like a third,
        undeclared arm.
        """
        if self.params.front_face_max_angle_deg is None:
            return None
        if self._front_face is None:
            return None
        return FrontFaceGate(
            front_face=self._front_face,
            camera_xy=np.asarray(self.extrinsic, dtype=float)[:3, 3][:2],
            max_angle_deg=float(self.params.front_face_max_angle_deg),
        )
