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
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from methods.constrained_ransac import RansacResult
from methods.se2_lie_utils import se2_exp


def _lift_se2_increment(xi, center_xy):
    """
    Lifts the se(2) increment xi = (omega, vx, vy) to a 4x4 SE(3) matrix,
    applying the rotation about `center_xy` instead of the world origin.

    Centering the rotation on the (transformed) model centroid decouples
    omega from the translation columns of the Jacobian, which keeps the
    Gauss-Newton normal equations well conditioned when the object is far
    from the robot base origin.
    """
    T3 = se2_exp(xi)
    R2 = T3[:2, :2]
    t2 = T3[:2, 2]

    T = np.eye(4)
    T[:2, :2] = R2
    T[:2, 3] = center_xy - R2 @ center_xy + t2
    return T


def icp_translation_only(
    model_points,  # (N, 3) model points (already sampled from the CAD)
    scene_points,  # (M, 3) scene points in the Z-up robot base frame
    scene_normals,  # (M, 3) scene normals (oriented toward the sensor)
    T_init,  # (4, 4) SE(2)-embedded initial pose
    max_correspondence_distance,
    max_iterations=100,
    tolerance=1e-8,
):
    """
    Point-to-plane ICP over ground-plane translation only; yaw is frozen.

    The residual n_i . (R p_i + t - q_i) is already linear in t, so the Jacobian
    is just [n_x, n_y] and 2x2 normal equations solve it -- there is no rotation
    parameter to move, by construction rather than by penalty.

    This exists because the tightened refinement of
    Ransac3DoFEstimator._refine_pose is sharp enough to fall into a symmetry
    twin: the front slab of the smallest cart in the fleet is 0.735 x 0.704 m,
    near-square in plan, and a culled tight objective has a genuine 90-degree
    minimum there. When the yaw guard trips, this recovers the stage's
    translation gain while leaving the orientation the wide stage chose intact.

    Returns:
        RansacResult, same contract as icp_point_to_plane_se2.
    """
    model_points = np.asarray(model_points, dtype=float)
    scene_points = np.asarray(scene_points, dtype=float)
    scene_normals = np.asarray(scene_normals, dtype=float)
    T = np.array(T_init, dtype=float, copy=True)

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)

    def _evaluate(T_eval):
        transformed = model_points @ T_eval[:3, :3].T + T_eval[:3, 3]
        dists, idx = scene_tree.query(
            transformed, k=1, distance_upper_bound=max_correspondence_distance
        )
        valid = np.isfinite(dists)
        return transformed, dists, idx, valid

    for _ in range(max_iterations):
        transformed, _dists, idx, valid = _evaluate(T)
        if valid.sum() < 3:
            logging.warning(
                "SE(2) translation-only ICP: fewer than 3 correspondences within "
                f"{max_correspondence_distance} m; stopping refinement."
            )
            break

        p = transformed[valid]
        q = scene_points[idx[valid]]
        n = scene_normals[idx[valid]]

        r = np.einsum("ij,ij->i", n, p - q)
        J = n[:, :2]

        JTJ = J.T @ J + 1e-12 * np.eye(2)
        try:
            dt = np.linalg.solve(JTJ, -(J.T @ r))
        except np.linalg.LinAlgError:
            logging.warning("SE(2) translation-only ICP: singular normal equations; stopping.")
            break

        T[:2, 3] += dt
        if np.linalg.norm(dt) < tolerance:
            break

    _, dists, _, valid = _evaluate(T)
    n_inliers = valid.sum()
    fitness = n_inliers / n_model if n_model else 0.0
    inlier_rmse = np.sqrt(np.mean(dists[valid] ** 2)) if n_inliers > 0 else np.inf
    return RansacResult(T, fitness, inlier_rmse)


def icp_point_to_plane_se2(
    model_points,  # (N, 3) model points (already sampled from the CAD)
    scene_points,  # (M, 3) scene points in the Z-up robot base frame
    scene_normals,  # (M, 3) scene normals (oriented toward the sensor)
    T_init,  # (4, 4) SE(2)-embedded initial pose
    max_correspondence_distance,
    max_iterations=100,
    tolerance=1e-8,  # stop when the increment norm falls below this
):
    """
    SE(2)-constrained point-to-plane ICP.

    Minimizes sum_i [ n_i . (T p_i - q_i) ]^2 over T in SE(2) (embedded in
    SE(3) with fixed z), where q_i / n_i are the nearest scene point and its
    normal. The linearized residual for a left increment xi = (omega, vx, vy)
    rotating about the centroid c is
        dr_i = omega * (n_y (p_x - c_x) - n_x (p_y - c_y)) + n_x vx + n_y vy,
    giving a (K, 3) Jacobian solved through 3x3 normal equations per iteration.

    Returns:
        RansacResult with `.transformation` (4x4, exactly planar),
        `.fitness` (inliers / N model points) and `.inlier_rmse`
        (Euclidean RMSE over matched pairs), all evaluated at the final pose.
    """
    model_points = np.asarray(model_points, dtype=float)
    scene_points = np.asarray(scene_points, dtype=float)
    scene_normals = np.asarray(scene_normals, dtype=float)
    T = np.array(T_init, dtype=float, copy=True)

    scene_tree = cKDTree(scene_points)
    n_model = len(model_points)

    def _evaluate(T_eval):
        transformed = model_points @ T_eval[:3, :3].T + T_eval[:3, 3]
        dists, idx = scene_tree.query(
            transformed, k=1, distance_upper_bound=max_correspondence_distance
        )
        valid = np.isfinite(dists)
        return transformed, dists, idx, valid

    for _ in range(max_iterations):
        transformed, dists, idx, valid = _evaluate(T)
        if valid.sum() < 3:
            logging.warning(
                "SE(2) ICP: fewer than 3 correspondences within "
                f"{max_correspondence_distance} m; stopping refinement."
            )
            break

        p = transformed[valid]
        q = scene_points[idx[valid]]
        n = scene_normals[idx[valid]]
        c = p[:, :2].mean(axis=0)

        r = np.einsum("ij,ij->i", n, p - q)
        J = np.column_stack(
            [
                n[:, 1] * (p[:, 0] - c[0]) - n[:, 0] * (p[:, 1] - c[1]),  # d r / d omega
                n[:, 0],  # d r / d vx
                n[:, 1],  # d r / d vy
            ]
        )

        JTJ = J.T @ J + 1e-12 * np.eye(3)
        try:
            xi = np.linalg.solve(JTJ, -(J.T @ r))
        except np.linalg.LinAlgError:
            logging.warning("SE(2) ICP: singular normal equations; stopping refinement.")
            break

        T = _lift_se2_increment(xi, c) @ T
        if np.linalg.norm(xi) < tolerance:
            break

    # Evaluate fitness/RMSE at the final pose (the loop metrics lag one update behind).
    _, dists, _, valid = _evaluate(T)
    n_inliers = valid.sum()
    fitness = n_inliers / n_model
    inlier_rmse = np.sqrt(np.mean(dists[valid] ** 2)) if n_inliers > 0 else np.inf
    return RansacResult(T, fitness, inlier_rmse)
