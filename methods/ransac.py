import copy
import logging
from typing import TYPE_CHECKING, Any
import numpy as np
import open3d as o3d
from methods.base import BasePoseEstimator, prepare_scene_point_cloud, refine_pose_dual_hypothesis

if TYPE_CHECKING:
    import optuna


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
            self.extrinsic = None

    @classmethod
    def suggest_params(cls, trial: "optuna.Trial") -> dict[str, Any]:
        """Suggests parameters for RANSAC + ICP registration."""
        return {
            "voxel_size": trial.suggest_float("voxel_size", 0.02, 0.10, step=0.01),
            "icp_max_correspondence_distance": trial.suggest_float("icp_max_correspondence_distance", 0.05, 0.25),
            "icp_max_iterations": trial.suggest_int("icp_max_iterations", 10, 100, step=10)
        }

    def _get_prep_params_key(self) -> tuple:
        """Returns only the parameters affecting offline preparation."""
        # Note: The trailing comma is required to instantiate a single-element tuple in Python.
        return (self.params.voxel_size,)

    def prepare(self, cad_mesh: o3d.geometry.TriangleMesh, cart_type: str) -> None:
        """Prepares CAD model by computing vertex normals, downsampling, and FPFH descriptor computation."""
        prep_params = self._get_prep_params_key()
        cache_key = (self.__class__.__name__, cart_type, prep_params)
        if cache_key in self._PREPARATION_CACHE:
            return

        # Evict old cached entries for the same cart type with different parameters to bound memory growth
        stale_keys = [
            k for k in self._PREPARATION_CACHE
            if k[0] == self.__class__.__name__ and k[1] == cart_type and k[2] != prep_params
        ]
        for k in stale_keys:
            del self._PREPARATION_CACHE[k]

        # Deepcopy to prevent mutating original mesh
        mesh_copy = copy.deepcopy(cad_mesh)
        mesh_copy.compute_vertex_normals()
        model_pc = mesh_copy.sample_points_uniformly(number_of_points=2000)
        
        voxel_size = self.params.voxel_size
        model_down = model_pc.voxel_down_sample(voxel_size)
        model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
        
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        
        self._PREPARATION_CACHE[cache_key] = {
            "model_pc": model_pc,
            "model_down": model_down,
            "model_fpfh": model_fpfh
        }

    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        cart_type: str | None = None,
        **kwargs: Any
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud using FPFH + RANSAC.
        
        Args:
            pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            cart_type (str, optional): Name of the cart type.
            
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix in robot frame, 
                        or None if registration fails.
        """
        # Prepare scene point cloud using factored-out utility function
        pcd = prepare_scene_point_cloud(pcd, self.extrinsic)
        if pcd.is_empty():
            logging.warning("Prepared scene point cloud is empty. Registration aborted.")
            return None

        voxel_size = self.params.voxel_size

        if cart_type is None:
            # Lazy local fallback (no caching, no side-effects)
            mesh_copy = copy.deepcopy(cad_mesh)
            mesh_copy.compute_vertex_normals()
            model_pc = mesh_copy.sample_points_uniformly(number_of_points=2000)
            model_down = model_pc.voxel_down_sample(voxel_size)
            model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
            model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                model_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
            )
        else:
            cache_key = (self.__class__.__name__, cart_type, self._get_prep_params_key())
            if cache_key not in self._PREPARATION_CACHE:
                logging.warning(f"Cache miss inside estimate_pose: preparing on-the-fly for key {cache_key}")
                self.prepare(cad_mesh, cart_type)
            
            # Retrieve prepared representations from the cache.
            # Rationale for deepcopy: Open3D C++ objects are mutable and passed by reference.
            # Downstream algorithms (such as ICP refinement) transform these point clouds in-place.
            # If we did not deepcopy, the cached objects would get corrupted across calls.
            # Performance: Deepcopying a small point cloud/feature descriptor array takes <0.1ms,
            # whereas recomputing voxelization, normals, and FPFH features from scratch takes ~10-30ms.
            prep_data = self._PREPARATION_CACHE[cache_key]
            model_pc = copy.deepcopy(prep_data["model_pc"])
            model_down = copy.deepcopy(prep_data["model_down"])
            model_fpfh = copy.deepcopy(prep_data["model_fpfh"])
        
        # 1. Downsample point clouds for fast descriptor calculation
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Normals are preserved during downsampling, but let's ensure they are updated
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
        
        # 2. Compute FPFH (Fast Point Feature Histograms) descriptors
        logging.info("Computing FPFH descriptors...")
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        
        # 3. Perform RANSAC Global Registration based on feature matching
        logging.info("Running RANSAC global registration...")
        result_ransac = self._global_registration(model_down, pcd_down, model_fpfh, pcd_fpfh)

        T_init = result_ransac.transformation
        
        # If RANSAC global registration has 0 fitness (no matches found), it failed
        if result_ransac.fitness == 0.0:
            logging.error("RANSAC global registration failed: zero correspondences found.")
            return None
            
        # 4. Refine pose using dual-hypothesis point-to-plane ICP
        T_refined = self._refine_pose(model_pc, pcd, T_init)

        return self._project_pose(T_refined)

    def _refine_pose(self, model_pc, scene_pcd, T_init: np.ndarray) -> np.ndarray:
        """
        Runs the refinement stage on the global-registration result. Subclasses
        may override to swap in a different refinement (e.g. SE(2)-constrained ICP).
        """
        return refine_pose_dual_hypothesis(
            model_pc=model_pc,
            scene_pcd=scene_pcd,
            T_init=T_init,
            icp_max_correspondence_distance=self.params.icp_max_correspondence_distance,
            icp_max_iterations=self.params.icp_max_iterations
        )

    def _global_registration(
        self, model_down, pcd_down, model_fpfh, pcd_fpfh
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Runs the global registration stage. Returns an object exposing
        `.transformation` (4x4) and `.fitness`. Subclasses may override to
        swap in a different registration strategy (e.g. SE(2)-constrained).
        """
        distance_threshold = self.params.voxel_size * 1.5
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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

    def _project_pose(self, T: np.ndarray) -> np.ndarray:
        """Hook for subclasses to project the refined pose onto a constrained manifold."""
        return T
