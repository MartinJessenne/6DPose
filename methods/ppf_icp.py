import numpy as np
import cv2
import open3d as o3d
from methods.base import BasePoseEstimator, prepare_scene_point_cloud, refine_pose_dual_hypothesis




# =====================================================================
# 1. PARAMETER CLASS
# =====================================================================
class PPFICPParams:
    """Hyperparameters specific to the PPF + ICP matching method."""
    def __init__(
        self,
        ppf_sampling_step: float = 0.1,
        ppf_distance_step: float = 0.02,
        ppf_match_threshold: float = 0.06,
        ppf_match_tolerance: float = 0.03,
        icp_max_correspondence_distance: float = 0.17528702727791115,
        icp_max_iterations: int = 10
    ):
        """
        Initializes hyperparameters for PPF and ICP alignment.
        """
        self.ppf_sampling_step = ppf_sampling_step
        self.ppf_distance_step = ppf_distance_step
        self.ppf_match_threshold = ppf_match_threshold
        self.ppf_match_tolerance = ppf_match_tolerance
        self.icp_max_correspondence_distance = icp_max_correspondence_distance
        self.icp_max_iterations = icp_max_iterations


# =====================================================================
# 2. PPF HELPER FUNCTIONS
# =====================================================================
def o3d_to_ppf_format(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Extracts 3D coordinates and surface normals from an Open3D point cloud
    into a stacked array format required by OpenCV's PPF matching detector.

    OpenCV's PPF matching detector expects a float32 array of shape [N, 6]
    where each row contains [x, y, z, nx, ny, nz] representing the point coordinate
    and its associated surface normal vector.
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    
    pts = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack((pts, normals))


# =====================================================================
# 3. ESTIMATOR CLASS IMPLEMENTATION
# =====================================================================
class PPFICPEstimator(BasePoseEstimator):
    """6D Pose Estimator using Point Pair Features (PPF) and Dual ICP refinement."""

    def __init__(self, params: PPFICPParams | dict = None, extrinsic: list | np.ndarray = None):
        """
        Initializes the PPF + ICP Estimator with matching parameters.
        
        Args:
            params (PPFICPParams, optional): Dedicated parameters. If None, uses defaults.
        """
        if isinstance(params, dict):
            self.params = PPFICPParams(**params)
        else:
            self.params = params if params is not None else PPFICPParams()
            
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
        **kwargs
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud.
        
        Args:
            pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix in robot frame, 
                        or None if registration fails.
        """
        # Prepare scene point cloud using factored-out utility function
        pcd = prepare_scene_point_cloud(pcd, self.extrinsic)

        # Prepare CAD model
        cad_mesh.compute_vertex_normals()
        model_pc = cad_mesh.sample_points_uniformly(number_of_points=1000)
        
        # Convert geometries to PPF stacked formats [N, 6]
        ppf_model = o3d_to_ppf_format(model_pc)
        ppf_scene = o3d_to_ppf_format(pcd)
        
        # Initialize PPF 3D Detector (Notice the legacy underscore convention used by cv2)
        detector = cv2.ppf_match_3d_PPF3DDetector(
            relativeSamplingStep=self.params.ppf_sampling_step,
            relativeDistanceStep=self.params.ppf_distance_step
        )
        
        print("Training PPF Hash Table from CAD model...")
        detector.trainModel(ppf_model)
        
        print("Running PPF Match on scene...")
        match_results = detector.match(
            ppf_scene,
            self.params.ppf_match_threshold,
            self.params.ppf_match_tolerance
        )
        
        # Guard check if matching returned empty results
        if not match_results:
            print("PPF matching failed to return any candidate matches.")
            return None
            
        # Retrieve the best match result
        best_match = match_results[0]
        T_init = np.asarray(best_match.pose, dtype=np.float64).reshape(4, 4)
        print(f"PPF Alignment Complete. Best match votes: {best_match.numVotes}")
        
        # Refine pose using factored-out dual-hypothesis point-to-plane ICP
        T_refined = refine_pose_dual_hypothesis(
            model_pc=model_pc,
            scene_pcd=pcd,
            T_init=T_init,
            icp_max_correspondence_distance=self.params.icp_max_correspondence_distance,
            icp_max_iterations=self.params.icp_max_iterations
        )
        
        return T_refined
