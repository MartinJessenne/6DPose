import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

from methods.constrained_ransac import constrained_ransac_se2, project_to_se2
from methods.ransac import RansacEstimator, RansacParams
from methods.se2_icp import refine_pose_dual_hypothesis_se2

if TYPE_CHECKING:
    import optuna


def crop_front_face(cad_mesh: "o3d.geometry.TriangleMesh", depth: float) -> "o3d.geometry.TriangleMesh":
    """
    Keeps only the slab within `depth` meters of the mesh's +x extreme.

    CAD convention for the cart fleet (colruyt/leanflow/picanol): origin on
    the floor at the towing face (x ~ 0), body extending toward -x, y the
    symmetry axis. Cropping to the front slab removes the near-180-degree
    symmetry that causes flipped registrations: the remaining face is
    asymmetric, so the dual-hypothesis selection can tell front from back.

    The cropped mesh is NOT recentered — it stays in the original CAD frame,
    so estimated poses remain directly comparable to full-model ground truth.
    """
    vertices = np.asarray(cad_mesh.vertices)
    x_max = float(vertices[:, 0].max())
    big = 1e6
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(x_max - depth, -big, -big),
        max_bound=(x_max + big, big, big),
    )
    cropped = cad_mesh.crop(aabb)
    if len(cropped.vertices) == 0:
        raise ValueError(
            f"front_crop_depth={depth} left an empty mesh (mesh x range "
            f"[{vertices[:, 0].min():.3f}, {x_max:.3f}])."
        )
    return cropped


def derive_z_offset(cad_mesh: "o3d.geometry.TriangleMesh") -> float:
    """
    Height of the CAD origin above the model's lowest vertex.

    For a cart resting on the floor in a frame whose z=0 is the ground plane
    (robot base_link), the pose's z translation must equal this value so the
    model's lowest point touches the ground.
    """
    return -float(np.asarray(cad_mesh.vertices)[:, 2].min())


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
        z_offset: float | None = None,
        z_gate_threshold: float = 0.09,
        edge_length_threshold: float = 0.9,
        ransac_confidence: float = 0.999,
        seed: int | None = 0,
        front_crop_depth: float | None = None,
    ):
        """
        Args:
            z_offset (float, optional): Height of the CAD model origin above
                the scene's ground plane, in the robot base frame. None (the
                default) derives it per cart from the CAD mesh as
                -min(vertex z), i.e. the model rests on the floor; set a float
                to override manually.
            z_gate_threshold (float): Half-width (meters) of the z-consistency
                gate on FPFH correspondences. A sensor-noise property, so it is
                tuned independently of voxel_size (default 0.09 matches the
                previous voxel_size * 1.5 coupling at voxel_size 0.06).
            edge_length_threshold (float): Edge-length similarity checker ratio
                for RANSAC sample pairs (Open3D-equivalent, default 0.9).
            ransac_confidence (float): Early-exit confidence for the RANSAC
                iteration bound.
            seed (int, optional): Seed for the RANSAC random generator. Defaults
                to 0 so benchmark sweeps (which build params from suggest_params
                alone, bypassing the yaml) stay deterministic per trial; pass
                None explicitly for non-deterministic runs.
            front_crop_depth (float, optional): When set, register against only
                the front slab of the CAD model (the `depth` meters nearest the
                +x face, i.e. the towing face) instead of the full cart. The
                slab is asymmetric, which disambiguates the 180-degree flip.
                None (default) uses the full mesh.
        """
        super().__init__(
            voxel_size=voxel_size,
            ransac_max_iterations=ransac_max_iterations,
            icp_max_correspondence_distance=icp_max_correspondence_distance,
            icp_max_iterations=icp_max_iterations,
        )
        self.z_offset = z_offset
        self.z_gate_threshold = z_gate_threshold
        self.edge_length_threshold = edge_length_threshold
        self.ransac_confidence = ransac_confidence
        self.seed = seed
        self.front_crop_depth = front_crop_depth


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
        self._active_z_offset = 0.0

    def _get_prep_params_key(self) -> tuple:
        # front_crop_depth changes the prepared model representation, so it
        # must be part of the cache key alongside voxel_size.
        return (self.params.voxel_size, self.params.front_crop_depth)

    def prepare(self, cad_mesh, cart_type: str) -> None:
        if self.params.front_crop_depth is not None:
            cad_mesh = crop_front_face(cad_mesh, self.params.front_crop_depth)
        super().prepare(cad_mesh, cart_type)

    def estimate_pose(self, pcd, cad_mesh, cart_type=None, **kwargs):
        # Resolve the z offset for THIS cart before the pipeline runs: the
        # hooks below (_global_registration, _project_pose) have no access to
        # the CAD mesh, so the resolved value is carried on the instance.
        # Derived from the FULL mesh — the front slab may not reach the floor.
        if self.params.z_offset is not None:
            self._active_z_offset = self.params.z_offset
        else:
            self._active_z_offset = derive_z_offset(cad_mesh)

        # Register against the front slab only (asymmetric -> no flips); the
        # crop stays in the CAD frame, so the output pose is unchanged in
        # meaning and remains comparable to full-model ground truth.
        if self.params.front_crop_depth is not None:
            cad_mesh = crop_front_face(cad_mesh, self.params.front_crop_depth)

        return super().estimate_pose(pcd, cad_mesh, cart_type=cart_type, **kwargs)

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
            z_offset=self._active_z_offset,
            z_gate_threshold=self.params.z_gate_threshold,
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
        T_projected = project_to_se2(T, z_offset=self._active_z_offset)
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
        # First z-gate sweep pressed against the old 0.20 ceiling (19/20 top
        # trials above 0.15): the optimum lies higher, so give it headroom.
        params["z_gate_threshold"] = trial.suggest_float("z_gate_threshold", 0.05, 0.35)
        # Let Optuna trade RANSAC budget against latency explicitly instead of
        # only implicitly through voxel_size.
        params["ransac_max_iterations"] = trial.suggest_int(
            "ransac_max_iterations", 2000, 100000, log=True
        )
        # Registering the asymmetric front slab instead of the full cart
        # halved the flip rate and doubled AR in A/B benchmarks; the slab
        # depth trades feature support against re-imported symmetry.
        params["front_crop_depth"] = trial.suggest_float("front_crop_depth", 0.2, 0.6)
        return params
