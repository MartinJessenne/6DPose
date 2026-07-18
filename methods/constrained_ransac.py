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
    def __init__(self, transformation, fitness, inlier_rmse):
        self.transformation = transformation
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse


# ---------------------------------------------------------------------------
# Mutual FPFH matching -- numpy equivalent of mutual_filter=True
# ---------------------------------------------------------------------------

def match_correspondences_fpfh(feat_model, feat_scene):
    """
    Args:
        feat_model: (N, 33) FPFH descriptors of the model
        feat_scene: (M, 33) FPFH descriptors of the scene

    Returns:
        (K, 2) integer array of (i_model, j_scene) pairs such that j_scene is
        the nearest scene neighbor of i_model AND vice versa (mutual nearest
        neighbors, like mutual_filter=True in Open3D). Shape (0, 2) if none.
    """
    tree_scene = cKDTree(feat_scene)
    _, nn_in_scene = tree_scene.query(feat_model, k=1)  # nearest scene index per model point

    tree_model = cKDTree(feat_model)
    _, nn_in_model = tree_model.query(feat_scene, k=1)  # nearest model index per scene point

    model_idx = np.arange(len(feat_model))
    mutual = nn_in_model[nn_in_scene] == model_idx
    return np.column_stack([model_idx[mutual], nn_in_scene[mutual]])


# ---------------------------------------------------------------------------
# Lift (theta, t_xy) into a 4x4 homogeneous transform with fixed z
# ---------------------------------------------------------------------------

def se2_to_se3(theta, t_xy, z=0.0):
    """Embeds (theta, x, y) into SE(3) with fixed z and roll = pitch = 0."""
    T = np.eye(4)
    T[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    T[0, 3], T[1, 3], T[2, 3] = t_xy[0], t_xy[1], z
    return T


def project_to_se2(T, z_offset=0.0):
    """
    Projects an arbitrary SE(3) transform back onto the SE(2) manifold:
    yaw is the closest planar rotation (Frobenius projection of the upper-left
    2x2 block onto SO(2)), roll = pitch = 0, and z is pinned to z_offset.

    Used after the unconstrained ICP refinement so the final pose respects
    the ground-bounded constraint.
    """
    R = T[:3, :3]
    theta = np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    return se2_to_se3(theta, T[:2, 3], z_offset)


# ---------------------------------------------------------------------------
# The constrained RANSAC loop
# ---------------------------------------------------------------------------

def constrained_ransac_se2(
    model_points,        # (N, 3) downsampled model points
    scene_points,        # (M, 3) downsampled scene points
    model_fpfh,          # (N, 33)
    scene_fpfh,          # (M, 33)
    distance_threshold,
    max_iterations=100000,
    min_iterations=1000,
    confidence=0.999,
    z_offset=0.0,        # height of the model origin above the scene's ground plane
    edge_length_threshold=0.9,
    min_sample_distance=0.0,
    scoring_subsample_size=100,
    rng=None,
):
    """
    SE(2)-constrained (3 DoF) equivalent of
    registration_ransac_based_on_feature_matching.

    Hypotheses are generated from 2 correspondences (instead of 3 for SE(3)),
    which lowers the iteration bound to log(1-confidence)/log(1-w^2), where w
    is the CORRESPONDENCE inlier ratio of the best hypothesis so far (the
    fraction of FPFH correspondences consistent with it) — not the
    model-coverage fitness, which measures a different quantity and would make
    the early exit far too optimistic. `min_iterations` additionally floors
    the budget so an early mediocre hypothesis cannot end the search.

    Because z is fixed by the constraint, correspondences whose z coordinates
    are inconsistent (|q_z - p_z - z_offset| >= distance_threshold) can never
    be inliers of a valid hypothesis and are filtered out upfront.

    Args:
        min_sample_distance: minimum distance between the 2 sampled model
            points; short baselines give yaw estimates dominated by voxel
            noise (callers typically pass a few voxel sizes).
        rng: np.random.Generator for reproducible runs (defaults to a fresh one).
    """
    rng = np.random.default_rng() if rng is None else rng

    correspondences = match_correspondences_fpfh(model_fpfh, scene_fpfh)
    n_matched = len(correspondences)
    if n_matched < 2:
        return RansacResult(np.eye(4), 0.0, np.inf)

    # z-consistency gate: a rotation about z preserves the model z coordinate,
    # so any valid correspondence must satisfy q_z ~= p_z + z_offset.
    dz = np.abs(
        scene_points[correspondences[:, 1], 2]
        - model_points[correspondences[:, 0], 2]
        - z_offset
    )
    correspondences = correspondences[dz < distance_threshold]
    n_corr = len(correspondences)
    logging.info(
        f"SE(2) RANSAC: {n_matched} mutual FPFH matches, {n_corr} after z-consistency gate."
    )
    if n_corr < 2:
        logging.warning(
            "SE(2) RANSAC: z-consistency gate removed (almost) all correspondences. "
            "Check that the scene cloud is in a Z-up frame and z_offset is correct."
        )
        return RansacResult(np.eye(4), 0.0, np.inf)

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)

    # Matched pairs used to measure the correspondence inlier ratio of a
    # hypothesis (drives the early-exit bound).
    corr_p = model_points[correspondences[:, 0]]
    corr_q = scene_points[correspondences[:, 1]]

    # Cheap hypothesis pre-scoring on a fixed random subsample; the full model
    # is only evaluated when the subsample beats the current best.
    n_sub = min(scoring_subsample_size, n_model)
    sub_points = model_points[rng.choice(n_model, size=n_sub, replace=False)]
    subsample_is_full = n_sub == n_model

    best_fitness = 0.0
    best_rmse = np.inf
    best_T = np.eye(4)

    # Early-exit criterion, mirroring RANSACConvergenceCriteria(max_iter, confidence)
    it = 0
    required_iterations = max_iterations

    while it < min(max_iterations, required_iterations):
        it += 1

        i1, i2 = rng.choice(n_corr, size=2, replace=False)
        mi1, si1 = correspondences[i1]
        mi2, si2 = correspondences[i2]

        p1, p2 = model_points[mi1, :2], model_points[mi2, :2]
        q1, q2 = scene_points[si1, :2], scene_points[si2, :2]

        # CorrespondenceCheckerBasedOnEdgeLength: reject sample pairs whose
        # p1-p2 / q1-q2 distances disagree (noise or mismatch)
        len_p = np.linalg.norm(p2 - p1)
        len_q = np.linalg.norm(q2 - q1)
        if len_p < max(1e-9, min_sample_distance):
            continue
        if min(len_p, len_q) / max(len_p, len_q) < edge_length_threshold:
            continue

        theta, t_xy = minimal_solver_se2(p1, p2, q1, q2)
        T = se2_to_se3(theta, t_xy, z_offset)

        sub_transformed = sub_points @ T[:3, :3].T + T[:3, 3]
        sub_dists, _ = scene_tree.query(sub_transformed, k=1)
        sub_fitness = (sub_dists < distance_threshold).mean()
        # The slack (~2 sigma of a binomial estimate at n_sub=100) prevents
        # subsample noise from rejecting a hypothesis that is actually better
        # than the current best. Random wrong hypotheses score near zero, so
        # the gate still prunes the vast majority of full evaluations.
        if not subsample_is_full and sub_fitness <= best_fitness - 2.0 * np.sqrt(0.25 / n_sub):
            continue
        if subsample_is_full and sub_fitness <= best_fitness:
            continue

        if subsample_is_full:
            dists = sub_dists
        else:
            transformed = model_points @ T[:3, :3].T + T[:3, 3]
            dists, _ = scene_tree.query(transformed, k=1)

        inlier_mask = dists < distance_threshold
        fitness = inlier_mask.sum() / n_model

        if fitness > best_fitness:
            best_fitness = fitness
            best_rmse = np.sqrt(np.mean(dists[inlier_mask] ** 2)) if inlier_mask.any() else np.inf
            best_T = T

            if best_fitness >= 1.0:
                break  # perfect fit: no hypothesis can improve on it

            # Update the iteration bound from the CORRESPONDENCE inlier ratio w
            # of the new best: the chance a random 2-sample is all-inlier is
            # ~w^2, so the classic bound is log(1-confidence)/log(1-w^2).
            # Model-coverage fitness must not be used here — a mediocre wrong
            # pose with moderate coverage would collapse the budget to a few
            # dozen iterations and the true pose would never be sampled.
            residuals = np.linalg.norm(
                corr_p @ T[:3, :3].T + T[:3, 3] - corr_q, axis=1
            )
            w = (residuals < distance_threshold).mean()
            if w < 1.0:
                required_iterations = max(min_iterations, int(
                    np.log(1 - confidence) / np.log(1 - w ** 2)
                ))
            else:
                required_iterations = min_iterations

    return RansacResult(best_T, best_fitness, best_rmse)
