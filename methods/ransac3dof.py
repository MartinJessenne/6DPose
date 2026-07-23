import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d
import trimesh

from methods.constrained_ransac import constrained_ransac_se2, project_to_se2
from methods.ransac import RansacEstimator, RansacParams
from methods.se2_icp import refine_pose_dual_hypothesis_se2

if TYPE_CHECKING:
    import optuna


def crop_front_face(cad_mesh: "o3d.geometry.TriangleMesh",
                    depth: float,
                    min_height: float = 0.16
                    ) -> "o3d.geometry.TriangleMesh":
    """
    Keeps only the slab within `depth` meters of the mesh's +x extreme, and at
    least `min_height` meters above the floor.

    CAD convention for the cart fleet (colruyt/leanflow/picanol): origin on
    the floor at the towing face center bottom (x ~ 0), body extending toward -x,
    y the symmetry axis. Cropping to the front slab removes the near-180-degree
    symmetry that causes flipped registrations: the remaining face is
    asymmetric, so the dual-hypothesis selection can tell front from back.

    The height cut excludes the wheels/casters: their steering angle is fixed
    in the CAD (and in the synthetic training data) but arbitrary in reality,
    so they're a spurious registration cue rather than a useful one.

    Both cuts are true geometric plane slices (via trimesh), not a mesh.crop()
    bounding-box filter -- crop() only keeps triangles whose vertices already
    fall inside the box, so it snaps to existing mesh topology instead of
    cutting cleanly at the requested plane (observed drift up to ~9 cm on
    this fleet's meshes). Slicing re-triangulates at the exact cut plane.

    The cropped mesh is NOT recentered — it stays in the original CAD frame,
    so estimated poses remain directly comparable to full-model ground truth.
    """

    # vertices is an (N, 3) array where col 0 is X (longitudinal), col 1 is Y (transverse), col 2 is Z (height).
    vertices = np.asarray(cad_mesh.vertices)
    # x_max is the maximum X coordinate across all vertices, representing the front-most tip of the cart face.
    x_max = float(vertices[:, 0].max())
    z_floor = float(vertices[:, 2].min())

    # Two chained single-plane slices (X-depth, then Z-height). cap=False (the
    # default) deliberately leaves the cut open rather than sealing it with a
    # flat polygon: a real depth camera never sees an artificial cut face, so
    # capping would inject a fake planar surface into the FPFH/ICP point cloud.
    tm = trimesh.Trimesh(vertices=vertices, faces=np.asarray(cad_mesh.triangles))
    tm = trimesh.intersections.slice_mesh_plane(tm, plane_normal=[1, 0, 0], plane_origin=[x_max - depth, 0, 0])
    tm = trimesh.intersections.slice_mesh_plane(tm, plane_normal=[0, 0, 1], plane_origin=[0, 0, z_floor + min_height])

    if len(tm.vertices) == 0:
        raise ValueError(
            f"front_crop_depth={depth}, min_height={min_height} left an empty mesh "
            f"(mesh x range [{vertices[:, 0].min():.3f}, {x_max:.3f}], "
            f"z range [{z_floor:.3f}, {float(vertices[:, 2].max()):.3f}])."
        )

    cropped = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tm.vertices),
        o3d.utility.Vector3iVector(tm.faces),
    )
    # Slicing rebuilds the vertex set (including new vertices on the cut
    # plane), so normals from the source mesh don't carry over automatically
    # the way they did with crop()'s subset-preserving behavior.
    cropped.compute_vertex_normals()
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
@dataclass(frozen=True)
class Ransac3DoFParams(RansacParams):
    """Hyperparameters for the SE(2)-constrained FPFH + RANSAC + ICP method.

    Attributes:
        z_offset: Height of the CAD model origin above the scene's ground
            plane, in the robot base frame. None (the default) derives it per
            cart from the CAD mesh as -min(vertex z), i.e. the model rests on
            the floor; set a float to override manually.
        z_gate_threshold: Half-width (meters) of the z-consistency gate on
            FPFH correspondences. A sensor-noise property, so it is tuned
            independently of voxel_size (default 0.09 matches the previous
            voxel_size * 1.5 coupling at voxel_size 0.06).
        edge_length_threshold: Edge-length similarity checker ratio for
            RANSAC sample pairs (Open3D-equivalent, default 0.9).
        ransac_confidence: Early-exit confidence for the RANSAC iteration
            bound.
        seed: Seed for the RANSAC random generator. Defaults to 0 so
            benchmark sweeps (which build params from suggest_params alone,
            bypassing the yaml) stay deterministic per trial; pass None
            explicitly for non-deterministic runs.
        front_crop_depth: When set, register against only the front slab of
            the CAD model (the `depth` meters nearest the +x face, i.e. the
            towing face) instead of the full cart. The slab is asymmetric,
            which disambiguates the 180-degree flip. None (default) uses the
            full mesh.
        free_space_threshold: Maximum ratio of free-space depth projection
            violations allowed before triggering the 180-degree flip search.
        free_space_margin: Depth margin in meters for free-space violation checks.
        free_space_min_observed: Minimum number of model points that must
            land on valid (unmasked, in-bounds) depth pixels before a
            free-space violation ratio is trusted for flip disambiguation.
            Below this, the view doesn't show enough of the cart to judge
            front from back by visibility alone, and the decision falls
            through to front-slab fitness / ICP tiebreaker instead.
    """
    z_offset: float | None = None
    z_gate_threshold: float = 0.09
    edge_length_threshold: float = 0.9
    ransac_confidence: float = 0.999
    seed: int | None = 0
    front_crop_depth: float | None = None
    free_space_threshold: float = 0.02
    free_space_margin: float = 0.03
    free_space_min_observed: int = 30


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
        self._active_frame = None
        self._active_cart_type = None

    def _get_prep_params_key(self) -> tuple:
        # front_crop_depth changes the prepared model representation, so it
        # must be part of the cache key alongside voxel_size.
        return (self.params.voxel_size, self.params.front_crop_depth)

    def prepare(self, cad_mesh, cart_type: str) -> None:
        prep_params = self._get_prep_params_key()
        cache_key = (self.__class__.__name__, cart_type, prep_params)
        if cache_key in self._PREPARATION_CACHE:
            return

        stale_keys = [
            k for k in self._PREPARATION_CACHE
            if k[0] == self.__class__.__name__ and k[1] == cart_type and k[2] != prep_params
        ]
        for k in stale_keys:
            del self._PREPARATION_CACHE[k]

        mesh_copy = copy.deepcopy(cad_mesh)
        mesh_copy.compute_vertex_normals()

        # Single front-slab crop, shared by RANSAC (downsampled + FPFH) and ICP
        # (dense model_pc): the crop is what breaks the front/back symmetry, so
        # both stages should register against the same asymmetric geometry
        # instead of RANSAC seeing a crop while ICP falls back to the full,
        # near-symmetric cart. front_crop_depth=None (Ransac3DoFFullMeshEstimator's
        # ablation baseline) means no crop at all, not a fallback depth.
        if self.params.front_crop_depth is not None:
            try:
                slab_mesh = crop_front_face(mesh_copy, depth=self.params.front_crop_depth)
            except Exception:
                slab_mesh = mesh_copy
        else:
            slab_mesh = mesh_copy

        model_pc = slab_mesh.sample_points_uniformly(number_of_points=2000)

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
            "model_fpfh": model_fpfh,
        }

    def estimate_pose(self, pcd, cad_mesh, cart_type=None, **kwargs):
        # Resolve the z offset for THIS cart before the pipeline runs: the
        # hooks below (_global_registration, _project_pose) have no access to
        # the CAD mesh, so the resolved value is carried on the instance.
        # Derived from the FULL mesh — the front slab may not reach the floor.
        if self.params.z_offset is not None:
            self._active_z_offset = self.params.z_offset
        else:
            self._active_z_offset = derive_z_offset(cad_mesh)

        self._active_frame = kwargs.get("frame", None)
        self._active_cart_type = cart_type

        # Only crop cad_mesh for lazy local fallback (when cart_type is None).
        # When cart_type is specified, prepare() has already cached the cropped
        # model geometry in _PREPARATION_CACHE, avoiding per-frame cropping overhead.
        if cart_type is None and self.params.front_crop_depth is not None:
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

        frame = getattr(self, "_active_frame", None)

        result = refine_pose_dual_hypothesis_se2(
            model_points=np.asarray(model_pc.points),
            scene_points=scene_points,
            scene_normals=scene_normals,
            T_init=np.asarray(T_init),
            max_correspondence_distance=self.params.icp_max_correspondence_distance,
            max_iterations=self.params.icp_max_iterations,
            frame=frame,
            extrinsic=self.extrinsic,
            free_space_threshold=self.params.free_space_threshold,
            free_space_margin=self.params.free_space_margin,
            free_space_min_observed=self.params.free_space_min_observed,
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
        # depth trades feature support against re-imported symmetry. Upper
        # bound is the longest cart in the fleet (colruyt, ~2.57 m x-extent):
        # beyond that the crop no longer removes any mesh, silently
        # re-introducing the front/back symmetry this parameter exists to break.
        params["front_crop_depth"] = trial.suggest_float("front_crop_depth", 0.1, 2.5)
        params["free_space_threshold"] = trial.suggest_float("free_space_threshold", 0.005, 0.08)
        params["free_space_margin"] = trial.suggest_float("free_space_margin", 0.01, 0.08)
        return params


class Ransac3DoFFullMeshEstimator(Ransac3DoFEstimator):
    """Ransac3DoFEstimator variant that never crops the CAD mesh.

    front_crop_depth is simply never suggested, so it stays at its
    Ransac3DoFParams default of None (full mesh). This is the "before crop"
    ablation baseline for the SE(2) benchmark report -- superseded by
    Ransac3DoFEstimator once front-crop tuning landed, kept only to
    reproduce that historical data point.
    """

    @classmethod
    def suggest_params(cls, trial: "optuna.Trial") -> dict[str, Any]:
        params = RansacEstimator.suggest_params(trial)
        params["edge_length_threshold"] = trial.suggest_float("edge_length_threshold", 0.8, 0.95)
        params["z_gate_threshold"] = trial.suggest_float("z_gate_threshold", 0.05, 0.35)
        params["ransac_max_iterations"] = trial.suggest_int(
            "ransac_max_iterations", 2000, 100000, log=True
        )
        return params
