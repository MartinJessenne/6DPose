import copy
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    import optuna


class BasePoseEstimator(ABC):
    """Abstract base class representing a 6D Pose Estimation method."""

    _PREPARATION_CACHE = {}  # Global cache: (class_name, cart_type, prep_params_tuple) -> prepared_dict

    @abstractmethod
    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        cart_type: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Reconstructed scene point cloud in robot frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            cart_type (str, optional): Name of the cart type.
            **kwargs: Method-specific inputs (e.g., rgb_crop, depth_crop, camera intrinsics).

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix, or None if estimation fails.
        """
        pass

    def prepare(self, cad_mesh: o3d.geometry.TriangleMesh, cart_type: str) -> None:  # noqa: B027
        """
        Prepares and caches model-specific properties (e.g., FPFH features, PPF match database)
        for a given CAD model to speed up online evaluation.

        Args:
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            cart_type (str): Name of the cart type.
        """
        # Default implementation for estimators that do not require pre-computation.
        pass

    @classmethod
    def suggest_params(cls, trial: "optuna.Trial") -> dict[str, Any]:
        """
        Suggests hyperparameters for this matching method using an Optuna trial.

        Args:
            trial: The active Optuna trial.

        Returns:
            dict[str, Any]: Suggested parameter dictionary.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(
            f"Estimator class '{cls.__name__}' does not implement 'suggest_params' for parameter sweeps."
        )


def reorient_normals_to_reference(
    target: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
) -> None:
    """
    Re-imposes `reference`'s normal orientation convention on `target`, in place.

    Why this is needed at all: geometry determines a normal only up to sign (the
    orthogonal complement of the tangent plane is a line, and a line holds two
    unit vectors). PCA cannot pick between them -- its objective satisfies
    E(n) = E(-n) -- so `estimate_normals` resolves the sign by *inheriting* the
    normal already present at that point (Open3D >= 0.14). That inheritance is
    exactly the problem here: `voxel_down_sample` averages the normals falling
    into a voxel WITHOUT renormalising, so on this fleet's thin tubular frame the
    two walls of a tube land in one voxel and their outward normals cancel. The
    prior handed to `estimate_normals` is then a near-zero vector whose direction
    is rounding noise, and the estimated normal faithfully inherits that noise.
    Measured on colruyt.ply: ~12% of model normals affected at voxel_size 0.02,
    ~47.5% at 0.06.

    The repair uses the densely sampled cloud, whose normals are interpolated
    from the mesh's own consistently-wound triangles and are therefore exactly
    outward. Only the SIGN is taken from the reference: the estimated direction
    (and hence the support radius it was computed at) is left untouched, so this
    changes one variable and not two.

    Args:
        target: Point cloud whose normals are re-signed in place. Must have normals.
        reference: Densely sampled cloud carrying the trusted convention.
    """
    if target.is_empty() or reference.is_empty() or not reference.has_normals():
        return

    target_normals = np.asarray(target.normals)
    if len(target_normals) == 0:
        return

    reference_normals = np.asarray(reference.normals)
    kdtree = o3d.geometry.KDTreeFlann(reference)

    nearest = np.empty(len(target_normals), dtype=np.int64)
    for i, point in enumerate(np.asarray(target.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        nearest[i] = idx[0]

    reference_at_target = reference_normals[nearest]

    # A cancelled voxel average can leave an estimated normal of near-zero
    # length; renormalise so downstream dot products are comparable. Points whose
    # normal is degenerate beyond rescue fall back to the mesh normal outright.
    lengths = np.linalg.norm(target_normals, axis=1)
    degenerate = lengths < 1e-9
    target_normals[degenerate] = reference_at_target[degenerate]
    lengths[degenerate] = np.linalg.norm(target_normals[degenerate], axis=1)
    target_normals /= np.maximum(lengths, 1e-12)[:, None]

    flip = np.einsum("ij,ij->i", target_normals, reference_at_target) < 0.0
    target_normals[flip] *= -1.0

    target.normals = o3d.utility.Vector3dVector(target_normals)


def prepare_scene_point_cloud(
    pcd: o3d.geometry.PointCloud,
    T_robot_camera: np.ndarray,
    normal_radius: float = 0.05,
    normal_max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """
    Prepares the scene point cloud by estimating surface normals, orienting them
    towards the camera origin, and transforming the points to the robot's base frame.

    Note:
        This function does not mutate the input point cloud in-place. It operates on a
        deep copy to prevent coordinate transform contamination.

    Args:
        pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera coordinate frame.
        T_robot_camera (np.ndarray): 4x4 extrinsic transform from camera to robot base link.
        normal_radius (float): Radius parameter for hybrid KD-tree normal search.
        normal_max_nn (int): Max neighborhood size for KD-tree search.

    Returns:
        o3d.geometry.PointCloud: Transformed point cloud with oriented surface normals.
    """
    # Deepcopy to prevent mutating the caller's point cloud (double extrinsics bug)
    pcd = copy.deepcopy(pcd)
    if pcd.is_empty():
        return pcd
    # 1. Estimate surface normals using hybrid KD-tree search
    # This computes a local plane fit for each point's neighbors. If points lack normals,
    # ICP refinement (Point-to-Plane) and PPF matching will fail.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )

    # 2. Orient normals towards the camera center [0, 0, 0] in camera frame.
    # This guarantees consistent normal directions (pointing outward toward the sensor),
    # which is crucial for point-to-plane ICP to converge correctly.
    pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))

    # 3. Transform point cloud from camera frame to robot base link frame.
    # This aligns the coordinate system with the robot reference frame where CAD poses are evaluated.
    pcd.transform(T_robot_camera)

    return pcd


def refine_pose_dual_hypothesis(
    model_pc: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    icp_max_correspondence_distance: float,
    icp_max_iterations: int,
) -> np.ndarray:
    """
    Refines an initial pose estimate using point-to-plane ICP with a dual-hypothesis search.

    The dual-hypothesis search handles symmetric objects (like towing carts) where the global
    registration (PPF/RANSAC) might fit the object rotated 180 degrees. We run two parallel
    ICP optimizations:
      - Hypothesis 1: The original alignment guess.
      - Hypothesis 2: The alignment guess rotated 180 degrees around the object's local Z-axis.
    We then choose the hypothesis that yields the higher fitness score (overlap ratio). In the event
    of a tie, we break the tie using the lower Inlier Root Mean Squared Error (RMSE).

    Args:
        model_pc (o3d.geometry.PointCloud): Sampled point cloud from the reference CAD model.
        scene_pcd (o3d.geometry.PointCloud): Prepared scene point cloud in the robot reference frame.
        T_init (np.ndarray): 4x4 initial homogeneous registration guess.
        icp_max_correspondence_distance (float): Max correspondence search distance in meters.
        icp_max_iterations (int): Max ICP solver convergence iterations.

    Returns:
        np.ndarray: Refined 4x4 homogeneous transformation matrix.
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations)

    # Hypothesis 1: Run ICP on the original registration guess
    icp_result_1 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    # Hypothesis 2: Run ICP on the alignment rotated 180 degrees around the local Z-axis
    T_flip = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    T_init_flipped = T_init @ T_flip

    icp_result_2 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init_flipped,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    # Select the hypothesis that maximizes point cloud overlap (fitness score)
    if icp_result_1.fitness > icp_result_2.fitness:
        best_result = icp_result_1
        logging.info(
            f"Orientation selected: Original (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})"
        )
    elif icp_result_2.fitness > icp_result_1.fitness:
        best_result = icp_result_2
        logging.info(
            f"Orientation selected: Flipped 180° (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})"
        )
    else:
        # Tie breaker: pick the one with lower RMSE
        if icp_result_1.inlier_rmse <= icp_result_2.inlier_rmse:
            best_result = icp_result_1
            logging.info(
                f"Orientation selected: Original [Tie breaker] (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})"
            )
        else:
            best_result = icp_result_2
            logging.info(
                f"Orientation selected: Flipped 180° [Tie breaker] (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})"
            )

    return best_result.transformation
