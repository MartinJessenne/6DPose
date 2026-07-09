import numpy as np
import cv2
import open3d as o3d
from methods.base import BasePoseEstimator

# AGENT: Here or later in the documentation, add extensive documentation about the algorithm, the maths behind it and the methods used.
# Each theoretical step of the PPF variant implemented in this file should be extensively detailed. 
# We'll work that out together by designing a markdown 'textbook explanatory note' based on my prior knowledge and what I still need to be explained. 
# We could also inclue a diagram of the different components, steps and dataflow of the algorithm so to better understand how do they interact between each others
# What are the performance bottlenecks...


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
    """
    
    # AGENT: Maybe detail a bit what's happening below. What is the PPF model expecting as a data format. 
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    
    pts = np.array(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack((pts, normals))


# =====================================================================
# 3. ESTIMATOR CLASS IMPLEMENTATION
# =====================================================================
class PPFICPEstimator(BasePoseEstimator):
    """6D Pose Estimator using Point Pair Features (PPF) and Dual ICP refinement."""

    def __init__(self, params: PPFICPParams = None):
        """
        Initializes the PPF + ICP Estimator with matching parameters.
        
        Args:
            params (PPFICPParams, optional): Dedicated parameters. If None, uses defaults.
        """
        self.params = params if params is not None else PPFICPParams()

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
        # Prepare scene point cloud (estimate normals & transform to robot base frame)
        # AGENT: This should be more detailed, I don't really think this is trivial, and I think that even there some assumptions are made that should be clarified. 
        # Just a clear explanation of the o3d API in this context, why do we need to call them, what action are they performing, what are the expected results, and what are the failure modes of those operations. 
        from main import Config
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))
        pcd.transform(Config.T_ROBOT_CAMERA)

        
        # Prepare CAD model
        cad_mesh.compute_vertex_normals()
        model_pc = cad_mesh.sample_points_uniformly(number_of_points=1000)
        
        # Convert geometries to PPF stacked formats [N, 6]
        # Again quick and clear explanation of the expected format 
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
        
        # ICP Refinement (Point-to-Plane registration)
        # AGENT: This ICP refinement step deserves a whole separate and factored out implementation in itself, with a clear description of how the current algorithm is working, and what
        # tweaks we are additionnaly performing
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.params.icp_max_iterations
        )
        
        # Hypothesis 1: Original PPF alignment guess
        icp_result_1 = o3d.pipelines.registration.registration_icp(
            model_pc,
            pcd,
            self.params.icp_max_correspondence_distance,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria
        )
        
        # Hypothesis 2: 180-degree rotation along local Z-axis (handles vertical symmetry)
        # AGENT: Not really a code review comment, but rather a todo item: I think we should constrain the horizontal orientation of the cart during the algorithm. Indeed, in practice we're never
        # going to tow an upside-down cart. I think we should allow for a minimal pitch value (-\epsilon, +\epsilon) since the cart is mostly always going to be laid on a flat floor, and always upside up.  
        # So the deal is more to check for a possible a YZ plane symmetry inversion, and eventually a XZ plane inversion, but we can safely rule out a XZ plane inversion. 
        T_flip = np.array([
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0]
        ])
        T_init_flipped = T_init @ T_flip
        
        # AGENT: Again I think we need to be more explicit here about what is the o3d.pipelines.registration.registration_icp API doing. 
        # Just so that we can better understand the hidden execution costs, the hidden assumption made so that we can update them later on if we find additionnal assumptions. 
        icp_result_2 = o3d.pipelines.registration.registration_icp(
            model_pc,
            pcd,
            self.params.icp_max_correspondence_distance,
            T_init_flipped,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria
        )
        
        # Select the hypothesis that maximizes point cloud overlap (fitness)
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
