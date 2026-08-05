import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import open3d as o3d

from methods.base import (
    BaseParams,
    BasePoseEstimator,
    SearchRange,
    prepare_scene_point_cloud,
    refine_pose_dual_hypothesis,
    reorient_normals_to_reference,
)

if TYPE_CHECKING:
    pass


# =====================================================================
# 1. PARAMETER CLASS
# =====================================================================
@dataclass(frozen=True)
class RansacParams(BaseParams):
    """Hyperparameters specific to the FPFH + RANSAC + ICP matching method.

    Attributes:
        voxel_size: Voxel size (in meters) for downsampling point clouds.
        ransac_max_iterations: Maximum iterations for RANSAC validation checking.
        icp_max_correspondence_distance: Max correspondence distance threshold for ICP.
        icp_max_iterations: Max iteration count for the final ICP refinement.
    """

    voxel_size: float = 0.06
    ransac_max_iterations: int = 100000
    icp_max_correspondence_distance: Annotated[float, SearchRange(min=0.01, max=0.25)] = 0.15
    icp_max_iterations: int = 100  # Pinned (not swept)


# =====================================================================
# 2. ESTIMATOR CLASS IMPLEMENTATION
# =====================================================================
class RansacEstimator(BasePoseEstimator):
    """6D Pose Estimator using FPFH features, RANSAC Global Registration, and Dual ICP refinement."""

    params_cls = RansacParams  # Associate the parameter dataclass with this estimator

    def __init__(self, params: RansacParams | None = None, extrinsic: np.ndarray | None = None):
        """
        Initializes the RANSAC + ICP Estimator with matching parameters.

        Args:
            params (RansacParams, optional): Dedicated parameters. If None, uses defaults.
        """
        self.params = params or RansacParams()
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64) if extrinsic is not None else None

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
            k
            for k in self._PREPARATION_CACHE
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
        model_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        # voxel_down_sample averages normals without renormalising, so opposite
        # walls of a tube cancel; estimate_normals then inherits that near-zero
        # direction. Re-impose the mesh's outward convention before FPFH, which
        # is equivariant (not invariant) under per-point normal sign flips.
        reorient_normals_to_reference(model_down, model_pc)

        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )

        self._PREPARATION_CACHE[cache_key] = {
            "model_pc": model_pc,
            "model_down": model_down,
            "model_fpfh": model_fpfh,
        }

    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        cart_type: str | None = None,
        **kwargs: Any,
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
        # Why the pose was abandoned, for the caller's per-frame audit trail.
        # Reset per call: a stale reason from the previous frame would be worse
        # than none at all. None once a pose is successfully returned.
        self._last_failure_reason: str | None = None

        # Prepare scene point cloud using factored-out utility function
        pcd = prepare_scene_point_cloud(pcd, self.extrinsic)
        if pcd.is_empty():
            logging.warning("Prepared scene point cloud is empty. Registration aborted.")
            self._last_failure_reason = "empty_scene_cloud"
            return None

        voxel_size = self.params.voxel_size

        if cart_type is None:
            # Lazy local fallback (no caching, no side-effects)
            mesh_copy = copy.deepcopy(cad_mesh)
            mesh_copy.compute_vertex_normals()
            model_pc = mesh_copy.sample_points_uniformly(number_of_points=2000)
            model_down = model_pc.voxel_down_sample(voxel_size)
            model_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
            )
            reorient_normals_to_reference(model_down, model_pc)
            model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                model_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
            )
        else:
            cache_key = (self.__class__.__name__, cart_type, self._get_prep_params_key())
            if cache_key not in self._PREPARATION_CACHE:
                logging.warning(
                    f"Cache miss inside estimate_pose: preparing on-the-fly for key {cache_key}"
                )
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

        # estimate_normals inherits the sign already present at each point, and
        # voxel_down_sample's unrenormalised averaging can leave that prior
        # near-zero. Re-imposing the towards-sensor convention afterwards is what
        # keeps the scene cloud's signs comparable with the model's outward ones
        # (they agree on the visible side of a solid). In the robot frame the
        # camera centre is extrinsic[:3, 3], not the origin.
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        pcd_down.orient_normals_towards_camera_location(
            camera_location=np.asarray(self.extrinsic, dtype=np.float64)[:3, 3]
        )

        # 2. Compute FPFH (Fast Point Feature Histograms) descriptors
        logging.info("Computing FPFH descriptors...")
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )

        # 3. Perform RANSAC Global Registration based on feature matching
        logging.info("Running RANSAC global registration...")
        result_ransac = self._global_registration(model_down, pcd_down, model_fpfh, pcd_fpfh)

        T_init = result_ransac.transformation

        # If RANSAC global registration has 0 fitness, it abstained. Carry out
        # WHICH check rejected the frame (see RansacResult.reason) rather than a
        # bare None -- "not enough FPFH correspondences" and "the null gate
        # vetoed the best pose" are different bugs with different fixes.
        if result_ransac.fitness == 0.0:
            self._last_failure_reason = getattr(result_ransac, "reason", None) or "registration"
            logging.error(f"RANSAC global registration abstained: {self._last_failure_reason}.")
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
            icp_max_iterations=self.params.icp_max_iterations,
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
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.params.ransac_max_iterations, 0.999
            ),
        )

    def _project_pose(self, T: np.ndarray) -> np.ndarray:
        """Hook for subclasses to project the refined pose onto a constrained manifold."""
        return T
