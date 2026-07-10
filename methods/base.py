from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d

class BasePoseEstimator(ABC):
    """Abstract base class representing a 6D Pose Estimation method."""

    @abstractmethod
    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        **kwargs
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Reconstructed scene point cloud in robot frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            **kwargs: Method-specific inputs (e.g., rgb_crop, depth_crop, camera intrinsics).

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix, or None if estimation fails.
        """
        pass


def prepare_scene_point_cloud(
    pcd: o3d.geometry.PointCloud,
    T_robot_camera: np.ndarray,
    normal_radius: float = 0.05,
    normal_max_nn: int = 30
) -> o3d.geometry.PointCloud:
    """
    Prepares the scene point cloud by estimating surface normals, orienting them
    towards the camera origin, and transforming the points to the robot's base frame.

    Args:
        pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera coordinate frame.
        T_robot_camera (np.ndarray): 4x4 extrinsic transform from camera to robot base link.
        normal_radius (float): Radius parameter for hybrid KD-tree normal search.
        normal_max_nn (int): Max neighborhood size for KD-tree search.

    Returns:
        o3d.geometry.PointCloud: Transformed point cloud with oriented surface normals.
    """
    # 1. Estimate surface normals using hybrid KD-tree search
    # This computes a local plane fit for each point's neighbors. If points lack normals,
    # ICP refinement (Point-to-Plane) and PPF matching will fail.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
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
    icp_max_iterations: int
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
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=icp_max_iterations
    )
    
    # Hypothesis 1: Run ICP on the original registration guess
    icp_result_1 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    # Hypothesis 2: Run ICP on the alignment rotated 180 degrees around the local Z-axis
    T_flip = np.array([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    T_init_flipped = T_init @ T_flip
    
    icp_result_2 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init_flipped,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    # Select the hypothesis that maximizes point cloud overlap (fitness score)
    if icp_result_1.fitness > icp_result_2.fitness:
        best_result = icp_result_1
        print(f"Orientation selected: Original (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})")
    elif icp_result_2.fitness > icp_result_1.fitness:
        best_result = icp_result_2
        print(f"Orientation selected: Flipped 180° (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})")
    else:
        # Tie breaker: pick the one with lower RMSE
        if icp_result_1.inlier_rmse <= icp_result_2.inlier_rmse:
            best_result = icp_result_1
            print(f"Orientation selected: Original [Tie breaker] (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})")
        else:
            best_result = icp_result_2
            print(f"Orientation selected: Flipped 180° [Tie breaker] (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})")
            
    return best_result.transformation
