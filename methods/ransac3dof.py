import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from methods.constrained_ransac import constrained_ransac_se2, project_to_se2
from methods.ransac import RansacEstimator, RansacParams
from methods.se2_icp import refine_pose_dual_hypothesis_se2

if TYPE_CHECKING:
    import optuna


# =====================================================================
# 1. PARAMETER CLASS
# =====================================================================
class Ransac3DoFParams(RansacParams):
    """Hyperparameters for the SE(2)-constrained FPFH + RANSAC + ICP method."""

    def __init__(
        self,
        voxel_size: float = 0.06,
        ransac_max_iterations: int = 100000,
        icp_max_correspondence_distance: float = 0.15,
        icp_max_iterations: int = 100,
        z_offset: float = 0.0,
        edge_length_threshold: float = 0.9,
        ransac_confidence: float = 0.999,
        seed: int | None = 0,
    ):
        """
        Args:
            z_offset (float): Height of the CAD model origin above the scene's
                ground plane, in the robot base frame (compensates for a CAD
                origin that is not on the ground).
            edge_length_threshold (float): Edge-length similarity checker ratio
                for RANSAC sample pairs (Open3D-equivalent, default 0.9).
            ransac_confidence (float): Early-exit confidence for the RANSAC
                iteration bound.
            seed (int, optional): Seed for the RANSAC random generator. Defaults
                to 0 so benchmark sweeps (which build params from suggest_params
                alone, bypassing the yaml) stay deterministic per trial; pass
                None explicitly for non-deterministic runs.
        """
        super().__init__(
            voxel_size=voxel_size,
            ransac_max_iterations=ransac_max_iterations,
            icp_max_correspondence_distance=icp_max_correspondence_distance,
            icp_max_iterations=icp_max_iterations,
        )
        self.z_offset = z_offset
        self.edge_length_threshold = edge_length_threshold
        self.ransac_confidence = ransac_confidence
        self.seed = seed


# =====================================================================
# 2. ESTIMATOR CLASS IMPLEMENTATION
# =====================================================================
class Ransac3DoFEstimator(RansacEstimator):
    """
    SE(2)-constrained pose estimator for ground-bounded carts (3 DoF: x, y, yaw).

    Reuses the FPFH preparation and dual-hypothesis ICP pipeline of
    RansacEstimator, but replaces the global registration with an
    SE(2)-constrained RANSAC and projects the final refined pose back onto
    the SE(2) manifold (roll = pitch = 0, z = z_offset).

    The SE(2) assumption is only valid in a Z-up frame, so the camera-to-robot
    extrinsic is mandatory: without it the scene cloud would stay in the
    camera frame (Z-forward) and the XY projection would be meaningless.
    """

    params: Ransac3DoFParams

    def __init__(
        self,
        params: Ransac3DoFParams | dict | None = None,
        extrinsic: list | np.ndarray | None = None,
    ):
        if params is None:
            params = Ransac3DoFParams()
        elif not isinstance(params, Ransac3DoFParams):
            params = Ransac3DoFParams(**dict(params))

        if extrinsic is None:
            raise ValueError(
                "Ransac3DoFEstimator requires the camera-to-robot extrinsic: the SE(2) "
                "constraint only holds once the scene cloud is in the Z-up robot base frame."
            )

        super().__init__(params=params, extrinsic=extrinsic)

    def _global_registration(self, model_down, pcd_down, model_fpfh, pcd_fpfh):
        """SE(2)-constrained replacement for Open3D's feature-matching RANSAC."""
        distance_threshold = self.params.voxel_size * 1.5
        return constrained_ransac_se2(
            model_points=np.asarray(model_down.points),
            scene_points=np.asarray(pcd_down.points),
            model_fpfh=np.asarray(model_fpfh.data).T,   # (33, N) -> (N, 33)
            scene_fpfh=np.asarray(pcd_fpfh.data).T,
            distance_threshold=distance_threshold,
            max_iterations=self.params.ransac_max_iterations,
            confidence=self.params.ransac_confidence,
            z_offset=self.params.z_offset,
            edge_length_threshold=self.params.edge_length_threshold,
            # Short sample baselines give yaw hypotheses dominated by voxel
            # noise; require the 2 sampled model points to span a few voxels.
            min_sample_distance=3.0 * self.params.voxel_size,
            rng=np.random.default_rng(self.params.seed),
        )

    def _refine_pose(self, model_pc, scene_pcd, T_init: np.ndarray) -> np.ndarray:
        """
        SE(2)-constrained Gauss-Newton point-to-plane ICP (dual hypothesis).
        Every increment is composed through the se(2) exponential map, so the
        refined pose never leaves the planar manifold.
        """
        scene_points = np.asarray(scene_pcd.points)
        scene_normals = np.asarray(scene_pcd.normals)
        if len(scene_normals) != len(scene_points):
            raise ValueError(
                "SE(2) ICP requires scene normals; the scene cloud should come from "
                "prepare_scene_point_cloud, which estimates and orients them."
            )

        result = refine_pose_dual_hypothesis_se2(
            model_points=np.asarray(model_pc.points),
            scene_points=scene_points,
            scene_normals=scene_normals,
            T_init=np.asarray(T_init),
            max_correspondence_distance=self.params.icp_max_correspondence_distance,
            max_iterations=self.params.icp_max_iterations,
        )
        return result.transformation

    def _project_pose(self, T: np.ndarray) -> np.ndarray:
        """
        Safety net: both the constrained RANSAC and the constrained ICP keep
        the pose on SE(2) by construction, so this projection is normally an
        exact no-op (up to float round-off). A visible correction indicates a
        bug upstream, so it is logged loudly rather than silently absorbed.
        """
        T_projected = project_to_se2(T, z_offset=self.params.z_offset)
        residual = np.abs(T_projected - T).max()
        if residual > 1e-6:
            logging.error(
                f"SE(2) invariant violated before projection (max-abs deviation {residual:.2e}); "
                "the constrained RANSAC/ICP should never leave the planar manifold."
            )
        return T_projected

    @classmethod
    def suggest_params(cls, trial: "optuna.Trial") -> dict[str, Any]:
        """Suggests parameters for the SE(2)-constrained RANSAC + ICP registration."""
        params = super().suggest_params(trial)
        params["edge_length_threshold"] = trial.suggest_float("edge_length_threshold", 0.8, 0.95)
        return params
