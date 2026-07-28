"""
SE(2)-constrained RANSAC replacing
o3d.pipelines.registration.registration_ransac_based_on_feature_matching
for ground-bounded objects (3 DoF: x, y, theta).

This module does NOT depend on open3d: it operates on raw np.ndarray inputs.
The open3d glue (extracting points/features) happens at the call site, see
methods/ransac3dof.py.

Key assumption: model_points and scene_points are expressed in frames that
share the same vertical (+Z up) axis — i.e. the scene cloud has been
transformed to the robot base frame and the CAD model is Z-up. Only the XY
projection is used to solve for theta and (x, y); z is fixed to z_offset in
the returned transformation. If the CAD origin is not on the ground plane,
z_offset must compensate for that gap.
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from methods.se2_lie_utils import minimal_solver_se2

# ---------------------------------------------------------------------------
# Result container, API-compatible with the open3d registration result
# (T_init = result.transformation ; result.fitness)
# ---------------------------------------------------------------------------


class RansacResult:
    """
    Result container returned by RANSAC registration.

    Attributes:
        transformation (np.ndarray): 4x4 homogeneous transformation matrix
            representing the estimated 6D/SE(2) pose of the CAD model in the
            scene frame (base_link).
        fitness (float): Fraction of downsampled model points that fall within
            `distance_threshold` of a scene point after applying `transformation`
            (in range [0.0, 1.0]).
        inlier_rmse (float): Root Mean Square Error (RMSE) of Euclidean distances
            for all inlier model points (points within `distance_threshold`).
        reason (str | None): On abstention (fitness == 0.0), WHICH check rejected
            the frame -- one of ABSTENTION_REASONS. Every abstention path used to
            collapse into the same anonymous `fitness == 0.0`, which made
            "the FPFH stage found no correspondences" indistinguishable from
            "every candidate pose was rejected" in the sweep logs. None on
            success.
    """

    def __init__(
        self,
        transformation: np.ndarray,
        fitness: float,
        inlier_rmse: float,
        reason: str | None = None,
    ):
        self.transformation = transformation
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.reason = reason


# Abstention causes reported by the registration stage (RansacResult.reason).
# Kept as a flat tuple of plain strings rather than an Enum so they survive the
# trip into a CSV column / W&B table unchanged.
ABSTENTION_REASONS = (
    "fpfh_insufficient",  # fewer than DOF mutual FPFH correspondences
    "z_gate_insufficient",  # z-gate left fewer than DOF correspondences
    "no_inliers",  # no candidate pose scored a single inlier
    # VSAC only, and deliberately separate from "no_inliers": the pose WAS
    # scored, the Poisson null gate then vetoed it. Disappears if that gate is
    # ablated -- which is exactly why it needs its own bucket while it exists.
    # Emitted only by the ablated Poisson null gate. Kept in the list because
    # sweeps run before the ablation (e.g. study VSAC_NullOn_M2) have frame CSVs
    # full of it, and a reader of those files needs the name to mean something.
    "null_rejected",
    # VSAC only: every candidate that fit was vetoed by FrontFaceGate for pointing
    # its towing face away from the camera. Separate from "no_inliers" because the
    # two demand opposite responses -- no_inliers means the scene gave nothing to
    # work with, this means the gate refused what it was given, and if it appears
    # in bulk the gate is misconfigured (see the rejection-ratio log in vsac_se2).
    "orientation_rejected",
    "empty_scene_cloud",  # segmentation/reprojection produced no scene points
    "registration",  # abstained without naming a cause (estimator not yet instrumented)
)


# ---------------------------------------------------------------------------
# Mutual FPFH matching -- numpy equivalent of mutual_filter=True
# ---------------------------------------------------------------------------


def match_correspondences_fpfh(
    feat_model: np.ndarray, feat_scene: np.ndarray, scored: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Computes mutual nearest-neighbor matches between model and scene descriptors.

    Args:
        feat_model: (N, 33) numpy array of 33-dimensional Fast Point Feature
            Histogram (FPFH) descriptors for N downsampled model points.
            Each row represents a local surface geometry signature.
        feat_scene: (M, 33) numpy array of 33-dimensional FPFH descriptors for
            M downsampled scene points.

    Returns:
        (K, 2) integer array of (i_model, j_scene) indices such that scene point
        j_scene is the nearest descriptor neighbor to model point i_model AND
        model point i_model is also the nearest descriptor neighbor to scene
        point j_scene (mutual nearest neighbors, equivalent to mutual_filter=True
        in Open3D). Returns shape (0, 2) array if inputs are empty or no matches exist.
    """
    # Guard against empty descriptor arrays resulting from aggressive downsampling/masking
    if len(feat_model) == 0 or len(feat_scene) == 0:
        empty = np.empty((0, 2), dtype=int)
        return (empty, np.empty((0,), dtype=int)) if scored else empty

    # 1. For each model point descriptor, find its nearest neighbor in scene descriptor space
    tree_scene = cKDTree(feat_scene)
    dist, nn_in_scene = tree_scene.query(
        feat_model, k=1
    )  # Shape (N,); nn_in_scene[i] = scene index for model i

    # 2. For each scene point descriptor, find its nearest neighbor in model descriptor space
    tree_model = cKDTree(feat_model)
    _, nn_in_model = tree_model.query(
        feat_scene, k=1
    )  # Shape (M,); nn_in_model[j] = model index for scene j

    # 3. Check mutual consistency:
    # `model_idx` is [0, 1, ..., N-1].
    # `nn_in_scene[i]` is the closest scene point `j` to model point `i`.
    # `nn_in_model[nn_in_scene[i]]` is the closest model point `i_prime` to scene point `j`.
    # If `i_prime == i`, then `i` and `j` are mutual nearest neighbors.
    model_idx = np.arange(len(feat_model))
    mutual = nn_in_model[nn_in_scene] == model_idx
    corr = np.column_stack([model_idx[mutual], nn_in_scene[mutual]])
    if scored:
        return corr, dist[mutual]
    else:
        return corr


# ---------------------------------------------------------------------------
# Lift (theta, t_xy) into a 4x4 homogeneous transform with fixed z
# ---------------------------------------------------------------------------


def se2_to_se3(theta: float, t_xy: np.ndarray, z: float = 0.0) -> np.ndarray:
    """
    Embeds 2D planar pose parameters (theta, x, y) into a 3D 4x4 SE(3) homogeneous
    transformation matrix with fixed elevation `z` and zero roll/pitch angles.

    The 6D pose pipeline operates using 4x4 matrices in 3D robot frame (base_link).
    This function converts the 2D solver results (yaw rotation `theta` and 2D translation `t_xy`)
    into the standard 4x4 matrix format expected by downstream components.
    """
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T[0, 3], T[1, 3], T[2, 3] = t_xy[0], t_xy[1], z
    return T


def project_to_se2(T: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """
    Projects an arbitrary SE(3) 4x4 matrix onto the SE(2) ground-plane manifold.

    Computes the closest planar yaw angle (Frobenius norm projection of the upper-left
    2x2 block onto SO(2)), sets roll = pitch = 0, and pins z to `z_offset`.

    Caller trace:
    - Called by `Ransac3DoFEstimator._project_pose` as a final safety-net assertion
      to guarantee that planar invariants are strictly preserved.
    """
    R = T[:3, :3]
    theta = np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    return se2_to_se3(theta, T[:2, 3], z_offset)


# ---------------------------------------------------------------------------
# Helper functions for RANSAC loop factoring
# ---------------------------------------------------------------------------


def _sample_and_validate_pair(
    correspondences: np.ndarray,
    model_points: np.ndarray,
    scene_points: np.ndarray,
    min_sample_distance: float,
    edge_length_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Draws 2 random correspondence pairs and validates their geometric consistency.

    Rejects candidate pairs whose model-point baseline is too small (< min_sample_distance)
    or whose model vs. scene segment lengths disagree significantly (edge length test).

    Returns:
        (p1, p2, q1, q2) 2D point pairs if valid, or None if rejected.
    """
    n_corr = len(correspondences)
    i1, i2 = rng.choice(n_corr, size=2, replace=False)
    mi1, si1 = correspondences[i1]
    mi2, si2 = correspondences[i2]

    p1, p2 = model_points[mi1, :2], model_points[mi2, :2]
    q1, q2 = scene_points[si1, :2], scene_points[si2, :2]

    # Measure 2D segment lengths in model and scene
    len_p = np.linalg.norm(p2 - p1)
    len_q = np.linalg.norm(q2 - q1)

    # 1. Enforce minimum sample distance baseline to prevent noise-dominated yaw estimates
    if len_p < max(1e-9, min_sample_distance):
        return None

    # 2. Edge-length ratio test (equivalent to Open3D CorrespondenceCheckerBasedOnEdgeLength):
    # Rigid bodies preserve distances between points. If length in model differs from scene, reject.
    if min(len_p, len_q) / max(len_p, len_q) < edge_length_threshold:
        return None

    return p1, p2, q1, q2


# ---------------------------------------------------------------------------
# The constrained RANSAC loop
# ---------------------------------------------------------------------------


def constrained_ransac_se2(
    model_points: np.ndarray,  # (N, 3) downsampled model points
    scene_points: np.ndarray,  # (M, 3) downsampled scene points
    model_fpfh: np.ndarray,  # (N, 33) FPFH descriptors of model
    scene_fpfh: np.ndarray,  # (M, 33) FPFH descriptors of scene
    distance_threshold: float,
    max_iterations: int = 100000,
    min_iterations: int = 1000,
    confidence: float = 0.999,
    z_offset: float = 0.0,  # height of model origin above ground plane
    z_gate_threshold: float | None = None,
    edge_length_threshold: float = 0.9,
    min_sample_distance: float = 0.0,
    scoring_subsample_size: int = 100,
    rng: np.random.Generator | None = None,
) -> RansacResult:
    """
    SE(2)-constrained (3 DoF: x, y, yaw) RANSAC global registration algorithm.

    ### Algorithm Explanation (RANSAC Step-by-Step):
    1. **Feature Matching**: Find initial candidate point pairs (correspondences) by matching
       33-dimensional FPFH feature descriptors between model and scene.
    2. **Z-Consistency Filtering**: Ground-bound objects have fixed vertical offset `z_offset`.
       Filter out candidate pairs whose vertical heights differ by more than `z_gate_threshold`.
    3. **RANSAC Hypotheses**: In 2D SE(2), 2 point correspondences (instead of 3 in 3D SE(3))
       suffice to uniquely solve for planar rotation angle theta and 2D translation (x, y).
    4. **Candidate Validation**:
       - Minimum sample distance: Ensure the 2 sampled model points are spaced apart.
       - Edge-length check: Verify that distance between the 2 model points matches distance
         between the 2 scene points (rigid transformation requirement).
    5. **Pre-scoring (Subsampling)**: To save computation time, score candidate pose `T` on a
       small random subset of model points first. Only evaluate the full point cloud if the
       subsample score improves upon the current best score.
    6. **Inlier Ratio & Early Exit**: Update the required RANSAC iterations based on the
       correspondence inlier ratio `w` of the best hypothesis found so far.

    Args:
        distance_threshold: Maximum Euclidean distance (in meters) for a scene point
            to be considered an inlier neighbor of a transformed model point.
        z_gate_threshold: Half-width of vertical height consistency filter (meters).
        min_sample_distance: Minimum distance required between the 2 sampled model points.
        scoring_subsample_size: Number of points used for fast candidate pre-scoring.
        rng: Random number generator for reproducible sampling.
    """
    rng = np.random.default_rng() if rng is None else rng

    correspondences = match_correspondences_fpfh(model_fpfh, scene_fpfh)
    n_matched = len(correspondences)
    if n_matched < 2:
        return RansacResult(np.eye(4), 0.0, np.inf, reason="fpfh_insufficient")

    # Z-consistency gate: Rotation about +Z preserves Z coordinates.
    # Any valid correspondence pair (model_i, scene_j) must satisfy:
    # scene_points[j, 2] ~= model_points[i, 2] + z_offset
    if z_gate_threshold is None:
        z_gate_threshold = distance_threshold

    scene_z = scene_points[correspondences[:, 1], 2]
    model_z = model_points[correspondences[:, 0], 2]
    dz = np.abs(scene_z - model_z - z_offset)

    correspondences = correspondences[dz < z_gate_threshold]
    n_corr = len(correspondences)
    logging.info(
        f"SE(2) RANSAC: {n_matched} mutual FPFH matches, {n_corr} after z-consistency gate."
    )
    if n_corr < 2:
        logging.warning(
            "SE(2) RANSAC: z-consistency gate removed (almost) all correspondences. "
            "Check that scene cloud is in Z-up frame and z_offset is correct."
        )
        return RansacResult(np.eye(4), 0.0, np.inf, reason="z_gate_insufficient")

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)

    # Correspondence pairs used to compute the correspondence inlier ratio (drives RANSAC early exit)
    corr_p = model_points[correspondences[:, 0]]
    corr_q = scene_points[correspondences[:, 1]]

    # Pre-scoring setup: sample a small subset of model points to score candidate transforms quickly.
    # Full model point cloud evaluation is only triggered when subsample fitness beats current best.
    n_sub = min(scoring_subsample_size, n_model)
    sub_points = model_points[rng.choice(n_model, size=n_sub, replace=False)]
    subsample_is_full = n_sub == n_model

    best_fitness = 0.0
    best_rmse = np.inf
    best_T = np.eye(4)

    it = 0
    required_iterations = max_iterations

    while it < min(max_iterations, required_iterations):
        it += 1

        # 1. Sample 2 correspondences and validate geometric constraints
        pair = _sample_and_validate_pair(
            correspondences,
            model_points,
            scene_points,
            min_sample_distance,
            edge_length_threshold,
            rng,
        )
        if pair is None:
            continue
        p1, p2, q1, q2 = pair

        # 2. Solve 2D minimal pose (theta, tx, ty) and embed into 4x4 SE(3) matrix with fixed z_offset
        theta, t_xy = minimal_solver_se2(p1, p2, q1, q2)
        T = se2_to_se3(theta, t_xy, z_offset)

        # 3. Fast pre-scoring on subsample
        sub_transformed = sub_points @ T[:3, :3].T + T[:3, 3]
        sub_dists, _ = scene_tree.query(sub_transformed, k=1)
        sub_fitness = (sub_dists < distance_threshold).mean()

        # Gate check: skip full evaluation if subsample fitness falls below current best
        if not subsample_is_full and sub_fitness <= best_fitness - 2.0 * np.sqrt(0.25 / n_sub):
            continue
        if subsample_is_full and sub_fitness <= best_fitness:
            continue

        # 4. Full evaluation on all model points
        if subsample_is_full:
            dists = sub_dists
        else:
            transformed = model_points @ T[:3, :3].T + T[:3, 3]
            dists, _ = scene_tree.query(transformed, k=1)

        inlier_mask = dists < distance_threshold
        fitness = inlier_mask.sum() / n_model

        # 5. Update best pose if full fitness improved
        if fitness > best_fitness:
            best_fitness = fitness
            best_rmse = np.sqrt(np.mean(dists[inlier_mask] ** 2)) if inlier_mask.any() else np.inf
            best_T = T

            if best_fitness >= 1.0:
                break  # Perfect fit found

            # 6. Update required RANSAC iterations based on correspondence inlier ratio w:
            # Chance of picking 2 inliers in a row is w^2, so required iterations = log(1-conf)/log(1-w^2).
            residuals = np.linalg.norm(corr_p @ T[:3, :3].T + T[:3, 3] - corr_q, axis=1)
            w = (residuals < distance_threshold).mean()
            if w >= 1.0:
                required_iterations = min_iterations
            elif w > 0.0:
                bound = np.log(1 - confidence) / np.log1p(-(w * w))
                required_iterations = int(np.clip(bound, min_iterations, max_iterations))
            else:
                required_iterations = max_iterations

    # best_fitness stays 0.0 when no candidate pose landed a single inlier --
    # same abstention the caller sees, but a different cause than the two
    # correspondence-starvation exits above, so name it.
    return RansacResult(
        best_T,
        best_fitness,
        best_rmse,
        reason="no_inliers" if best_fitness == 0.0 else None,
    )
