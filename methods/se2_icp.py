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
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from methods.free_space import count_violations

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


def refine_pose_dual_hypothesis_se2(
    model_points,
    scene_points,
    scene_normals,
    T_init,
    max_correspondence_distance,
    max_iterations=100,
    visibility=None,
    free_space_points=None,
    free_space_margin=0.03,
    free_space_separation=0.02,
    free_space_min_observed=30,
):
    """
    SE(2)-constrained counterpart of methods.base.refine_pose_dual_hypothesis with
    flip disambiguation:

    1. Runs SE(2) ICP Pass 1 on the primary hypothesis T_init using model_points --
       the discriminative front slab when the caller crops for asymmetry, so ICP and
       RANSAC register against the same geometry.
    2. Runs SE(2) ICP Pass 2 on the 180-degree-flipped hypothesis, pivoted about the
       slab's own centre so the two hypotheses differ in FACING rather than location.
    3. Decides between them: free-space violations first, then ICP fitness, then RMSE.

    `model_points` and `free_space_points` are deliberately separate arguments and
    must not be conflated. Registration wants the slab (asymmetric, densely observed);
    the free-space test wants the FULL cart. Judging visibility on the slab alone is
    self-defeating: the flip pivots the slab about its own centre, so it maps roughly
    onto itself, both hypotheses occupy the same space, and both score ~0 violations.
    The body is the geometry that ends up between the camera and the measured front
    face when the cart faces the wrong way, and it is the entire signal.

    Free-space evidence is only trusted when at least `free_space_min_observed` model
    points landed on valid depth pixels for the hypothesis being judged. A view that
    puts the model where nothing was measured yields a 0/0 "clean" ratio carrying no
    information; without this guard that degenerate case is indistinguishable from a
    genuinely unambiguous pose, exactly on the hard views this check exists for.
    """

    # Pass 1: ICP refinement on primary hypothesis using model_points
    result_1 = icp_point_to_plane_se2(
        model_points,
        scene_points,
        scene_normals,
        T_init,
        max_correspondence_distance,
        max_iterations,
    )

    # Pass 2 always runs. There used to be an early exit here that returned Pass 1
    # whenever its violation ratio was low -- but the ratio was computed on the slab,
    # so it was low for every hypothesis including flipped ones, and the "gate"
    # rubber-stamped whatever global registration handed over. A cheap check that is
    # always satisfied is worse than no check: it looks like evidence in the logs.

    T_init_array = np.asarray(T_init)
    world_pts = model_points @ T_init_array[:3, :3].T + T_init_array[:3, 3]
    cx = 0.5 * (world_pts[:, 0].min() + world_pts[:, 0].max())
    cy = 0.5 * (world_pts[:, 1].min() + world_pts[:, 1].max())

    T_flip = np.eye(4)
    T_flip[:2, :2] = np.array([[-1, 0], [0, -1]])
    T_flip[0, 3] = 2.0 * cx
    T_flip[1, 3] = 2.0 * cy

    result_2 = icp_point_to_plane_se2(
        model_points,
        scene_points,
        scene_normals,
        T_flip @ T_init_array,
        max_correspondence_distance,
        max_iterations,
    )

    # Free-space evidence for both hypotheses, judged on the full cart.
    n_obs_1, viol_ratio_1 = None, None
    n_obs_2, viol_ratio_2 = None, None
    if visibility is not None and free_space_points is not None:
        _, n_obs_1, viol_ratio_1 = count_violations(
            visibility, free_space_points, result_1.transformation, free_space_margin
        )
        _, n_obs_2, viol_ratio_2 = count_violations(
            visibility, free_space_points, result_2.transformation, free_space_margin
        )

    # Decision logic:
    # A. If the violation ratios differ by more than free_space_separation, pick the
    # lower one. Only trusted when both hypotheses had enough observed points to make
    # the ratio meaningful.
    free_space_reliable = (
        viol_ratio_1 is not None
        and viol_ratio_2 is not None
        and n_obs_1 >= free_space_min_observed
        and n_obs_2 >= free_space_min_observed
    )
    if free_space_reliable and abs(viol_ratio_1 - viol_ratio_2) > free_space_separation:
        if viol_ratio_1 < viol_ratio_2:
            best, label = (
                result_1,
                f"Original (Free-space: {viol_ratio_1:.2%} vs {viol_ratio_2:.2%})",
            )
        else:
            best, label = (
                result_2,
                f"Flipped 180° (Free-space: {viol_ratio_2:.2%} vs {viol_ratio_1:.2%})",
            )
    # B. Else if ICP fitness differs significantly (>2%), pick higher fitness. This margin
    # avoids letting a coin-flip-sized fitness gap decide the hard, near-tied cases that flip
    # disambiguation exists for.
    elif abs(result_1.fitness - result_2.fitness) > 0.02:
        if result_1.fitness > result_2.fitness:
            best, label = (
                result_1,
                f"Original (Fitness: {result_1.fitness:.4f} vs {result_2.fitness:.4f})",
            )
        else:
            best, label = (
                result_2,
                f"Flipped 180° (Fitness: {result_2.fitness:.4f} vs {result_1.fitness:.4f})",
            )
    # C. Fallback: RMSE tiebreaker
    else:
        if result_1.inlier_rmse <= result_2.inlier_rmse:
            best, label = result_1, "Original [Tie breaker]"
        else:
            best, label = result_2, "Flipped 180° [Tie breaker]"

    logging.info(
        f"SE(2) ICP orientation selected: {label} "
        f"(Fitness: {best.fitness:.4f}, RMSE: {best.inlier_rmse:.4f})"
    )
    best.diagnostics = {
        "selected": "original" if best is result_1 else "flipped",
        "decision": label,
        "fitness_1": result_1.fitness,
        "fitness_2": result_2.fitness,
        "viol_ratio_1": viol_ratio_1,
        "viol_ratio_2": viol_ratio_2,
    }
    return best
