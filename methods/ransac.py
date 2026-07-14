from typing import Any
import numpy as np
import open3d as o3d
from methods.base import BasePoseEstimator, prepare_scene_point_cloud, refine_pose_dual_hypothesis


# =====================================================================
# 1. PARAMETER CLASS
# =====================================================================
class RansacParams:
    """Hyperparameters specific to the FPFH + RANSAC + ICP matching method."""
    def __init__(
        self,
        voxel_size: float = 0.06,
        ransac_max_iterations: int = 100000,
        icp_max_correspondence_distance: float = 0.15,
        icp_max_iterations: int = 100
    ):
        """
        Initializes hyperparameters for RANSAC global registration and ICP.
        
        Args:
            voxel_size (float): Voxel size (in meters) for downsampling point clouds.
            ransac_max_iterations (int): Maximum iterations for RANSAC validation checking.
            icp_max_correspondence_distance (float): Max correspondence distance threshold for ICP.
            icp_max_iterations (int): Max iteration count for the final ICP refinement.
        """
        self.voxel_size = voxel_size
        self.ransac_max_iterations = ransac_max_iterations
        self.icp_max_correspondence_distance = icp_max_correspondence_distance
        self.icp_max_iterations = icp_max_iterations


# =====================================================================
# 2. ESTIMATOR CLASS IMPLEMENTATION
# =====================================================================
class RansacEstimator(BasePoseEstimator):
    """6D Pose Estimator using FPFH features, RANSAC Global Registration, and Dual ICP refinement."""

    def __init__(self, params: RansacParams | dict = None, extrinsic: list | np.ndarray = None):
        """
        Initializes the RANSAC + ICP Estimator with matching parameters.
        
        Args:
            params (RansacParams, optional): Dedicated parameters. If None, uses defaults.
        """
        if params is not None:
            if not isinstance(params, RansacParams):
                self.params = RansacParams(**dict(params))
            else:
                self.params = params
        else:
            self.params = RansacParams()
            
        if extrinsic is not None:
            self.extrinsic = np.asarray(extrinsic, dtype=np.float64)
        else:
            self.extrinsic = np.array([
                [0.5, 0.0,  0.866, 0.439],
                [0.0, 1.0, -0.0,   0.0  ],
                [-0.866, 0.0, 0.5, 0.304],
                [0.0, 0.0,  0.0,   1.0  ]
            ])

    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        **kwargs: Any
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud using FPFH + RANSAC.
        
        Args:
            pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix in robot frame, 
                        or None if registration fails.
        """
        # Prepare scene point cloud using factored-out utility function
        pcd = prepare_scene_point_cloud(pcd, self.extrinsic)

        # Prepare CAD model by computing vertex normals and sampling points
        cad_mesh.compute_vertex_normals()
        model_pc = cad_mesh.sample_points_uniformly(number_of_points=2000)
        
        # Define search params for features
        voxel_size = self.params.voxel_size
        
        # 1. Downsample point clouds for fast descriptor calculation
        pcd_down = pcd.voxel_down_sample(voxel_size)
        model_down = model_pc.voxel_down_sample(voxel_size)
        
        # Normals are preserved during downsampling, but let's ensure they are updated
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
        model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
        
        # 2. Compute FPFH (Fast Point Feature Histograms) descriptors
        print("Computing FPFH descriptors...")
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        
        # 3. Perform RANSAC Global Registration based on feature matching
        distance_threshold = voxel_size * 1.5
        print("Running RANSAC global registration...")
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model_down, 
            pcd_down, 
            model_fpfh, 
            pcd_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.params.ransac_max_iterations, 0.999
            )
        )
        
        T_init = result_ransac.transformation
        
        # If RANSAC global registration returns an identity or invalid matrix, it failed
        if np.allclose(T_init, np.eye(4)):
            print("RANSAC global registration failed to find a valid transformation.")
            return None
            
        # 4. Refine pose using factored-out dual-hypothesis point-to-plane ICP
        T_refined = refine_pose_dual_hypothesis(
            model_pc=model_pc,
            scene_pcd=pcd,
            T_init=T_init,
            icp_max_correspondence_distance=self.params.icp_max_correspondence_distance,
            icp_max_iterations=self.params.icp_max_iterations
        )
        
        return T_refined
