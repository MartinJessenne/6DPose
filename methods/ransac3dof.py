import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d
import trimesh

from methods.base import orient_normals_hoppe, reorient_normals_to_reference
from methods.constrained_ransac import constrained_ransac_se2, project_to_se2
from methods.ransac import RansacEstimator, RansacParams
from methods.se2_icp import icp_point_to_plane_se2, icp_translation_only

if TYPE_CHECKING:
    import optuna

# Below this many visible model points a refinement stage falls back to the full
# model cloud: the pose is far enough from the scene that the visible set is not
# a meaningful subset, and refining against a handful of points is worse than
# refining against a biased many.
_MIN_VISIBLE_MODEL_POINTS = 50


def _yaw_delta_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Absolute yaw difference between two SE(2) poses, wrapped to [0, 180]."""
    yaw_a = float(np.arctan2(T_a[1, 0], T_a[0, 0]))
    yaw_b = float(np.arctan2(T_b[1, 0], T_b[0, 0]))
    return abs(float(np.degrees(np.arctan2(np.sin(yaw_b - yaw_a), np.cos(yaw_b - yaw_a)))))


@dataclass(frozen=True)
class FrontFace:
    """
    Outward-facing arrow of a cart's towing face, in the CAD frame.

    No validation to perform for this class,
    every criterion would be an ad-hod criterion based on the current meshes,
    the only validation that can be done is a visual inspection of the mesh and the derived anchor and normal.

    So potentially a display method that would output a .glb file could be interesting.
    """

    anchor: np.ndarray  # (3,) center of the towing face
    normal: np.ndarray  # (3,) unit vector pointing outward from the towing face (+x convention)

    def __post_init__(self):
        for field_name in ("anchor", "normal"):
            arr = np.array(getattr(self, field_name), dtype=float)  # fresh contiguous copy
            arr.flags.writeable = False
            object.__setattr__(self, field_name, arr)


def front_face_from_mesh(mesh) -> FrontFace:
    """
    Derives the outward-facing arrow of a cart's towing face from its CAD mesh.

    Fleet CAD convention (measured on colruyt/leanflow/picanol): the origin sits
    at the bottom centre of the towing face, the face itself at x_max ~ 0, and
    the body extends toward -x. So the outward arrow is +x, anchored on the floor
    at the hitch point.

    Args:
        mesh: The CAD mesh of the cart, UNCROPPED. crop_front_face removes the
            body, which does not move x_max but leaves nothing to inspect if the
            convention is ever checked visually.

    Returns:
        A FrontFace with the anchor point and outward normal, both in the CAD frame.
    """
    vertices = np.asarray(mesh.vertices)
    x_max = float(vertices[:, 0].max())
    z_floor = float(vertices[:, 2].min())
    return FrontFace(
        anchor=np.array([x_max, 0.0, z_floor], dtype=float),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
    )


def front_face_angle_deg(T: np.ndarray, front_face: FrontFace, camera_xy: np.ndarray) -> float:
    """
    Signed angle in degrees between the cart's outward front-face arrow and the
    direction from the face to the camera. 0 means facing the camera head-on.

    Sign is retained deliberately: a cluster near +/-180 in the per-frame CSV is a
    flip, a spread around +/-30 is ordinary yaw error, and an unsigned magnitude
    would merge the two.

    XY-only, and exact rather than a projection. For an SE(2)-embedded pose,
    T[:3, :3] = Rz(theta) = [[c, -s, 0], [s, c, 0], [0, 0, 1]], so the rotated 3D
    normal is [c, s, 0] -- its z component is identically zero, not merely small --
    and the 3D anchor's xy equals the 2D result exactly. Including z would be the
    bug, not the fix: it would add the camera's elevation above the face, which is
    unrelated to yaw and worth up to 35 degrees on the tall picanol cart at 1 m.

    Args:
        T: (4, 4) candidate pose, CAD frame -> robot base_link frame.
        front_face: The cart's arrow in the CAD frame.
        camera_xy: (2,) camera centre in base_link, i.e. extrinsic[:3, 3][:2].
    """
    R2 = T[:2, :2]
    n2 = R2 @ front_face.normal[:2]  # (2,) rotated arrow, unit for any SE(2) pose
    a2 = R2 @ front_face.anchor[:2] + T[:2, 3]  # (2,) face centre in base_link
    v = camera_xy - a2
    # z component of the 2-D cross product. Written out rather than np.cross,
    # which is deprecated for 2-vectors in numpy 2.0.
    det = n2[0] * v[1] - n2[1] * v[0]
    return float(np.degrees(np.arctan2(det, float(n2 @ v))))


@dataclass(frozen=True)
class FrontFaceGate:
    """
    Rejects candidate poses whose towing face points away from the camera.

    This is not a score, and that is the whole point. MSAC, the z-gate and
    point-to-plane ICP are all symmetric under a 180-degree flip -- a cart and its
    twin occupy the same volume and tie, often exactly -- so no amount of tuning
    makes them prefer one. The arrow is a hard geometric fact about the model that
    the flip maps to its exact negation, evaluated in about ten flops.

    Measured on 560 ground-truth poses across all three dataset splits, using this
    exact anchor: |bearing| has p90 42.7, p99 46.1, max 48.79 degrees. A flipped
    pose therefore can come no closer to head-on than 180 - 48.79 = 131.2. Any
    threshold in the open interval (48.8, 131.2) separates the two perfectly.

    The arms use 60: 11 degrees of slack above the ground-truth maximum, 71 below
    the nearest flip. Note that 45 -- the angle the dataset's cart yaw is sampled
    within -- is NOT safe: measured against the towing face rather than the CAD
    origin, 18 of those 560 ground-truth poses exceed it, so a 45-degree gate
    would veto 3.2% of correct answers outright and could only turn them into
    abstentions.

    Attributes:
        front_face: The cart's arrow in the CAD frame.
        camera_xy: (2,) camera centre in base_link.
        max_angle_deg: Largest permitted absolute angle.
    """

    front_face: FrontFace
    camera_xy: np.ndarray
    max_angle_deg: float

    def accepts(self, T: np.ndarray) -> bool:
        """True when this pose's towing face points within the allowed angle."""
        return abs(front_face_angle_deg(T, self.front_face, self.camera_xy)) <= self.max_angle_deg


def export_front_face_glb(mesh, front_face: FrontFace, path: str, shaft_len: float = 0.6) -> None:
    """
    Writes the mesh plus its front-face arrow to a .glb for visual inspection.

    The convention this fleet relies on cannot be validated by assertion without
    inventing thresholds tuned to today's three meshes, so the honest check is to
    look at it. A mesh exported mirrored would invert the arrow and make the gate
    enforce the flip on every frame -- consistently, and with nothing in the log.

    Args:
        mesh: CAD mesh, uncropped.
        front_face: Arrow derived from that mesh.
        path: Output .glb path.
        shaft_len: Arrow length in meters.
    """
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    )

    # trimesh builds cylinders along +z, so rotate +z onto the arrow direction.
    arrow = trimesh.creation.cylinder(radius=0.02, height=shaft_len)
    arrow.apply_translation([0.0, 0.0, shaft_len / 2.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = np.asarray(front_face.normal, dtype=float)
    direction = direction / np.linalg.norm(direction)
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-9:
        angle = float(np.arccos(np.clip(z_axis @ direction, -1.0, 1.0)))
        arrow.apply_transform(trimesh.transformations.rotation_matrix(angle, axis / axis_norm))
    elif z_axis @ direction < 0:
        arrow.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]))
    arrow.apply_translation(np.asarray(front_face.anchor, dtype=float))
    arrow.visual.face_colors = [255, 40, 40, 255]

    scene.add_geometry(arrow)
    scene.export(path)


def crop_front_face(
    cad_mesh: "o3d.geometry.TriangleMesh", depth: float, min_height: float = 0.16
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
    # The cut plane comes from the same place as the orientation arrow, so the
    # crop and the gate can never disagree about where the towing face is.
    front_face = front_face_from_mesh(cad_mesh)
    x_max = float(front_face.anchor[0])
    z_floor = float(vertices[:, 2].min())

    # Two chained single-plane slices (X-depth, then Z-height). cap=False (the
    # default) deliberately leaves the cut open rather than sealing it with a
    # flat polygon: a real depth camera never sees an artificial cut face, so
    # capping would inject a fake planar surface into the FPFH/ICP point cloud.
    tm = trimesh.Trimesh(vertices=vertices, faces=np.asarray(cad_mesh.triangles))
    tm = trimesh.intersections.slice_mesh_plane(
        tm, plane_normal=[1, 0, 0], plane_origin=[x_max - depth, 0, 0]
    )
    tm = trimesh.intersections.slice_mesh_plane(
        tm, plane_normal=[0, 0, 1], plane_origin=[0, 0, z_floor + min_height]
    )

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
        front_face_max_angle_deg: Largest angle (degrees) permitted between the
            cart's outward towing-face arrow and the direction to the camera.
            None (the default) disables the constraint entirely, which is the
            control arm of the A/B; the treatment arm uses 60.0. See
            FrontFaceGate for why 60 rather than the 45 the data is generated
            with. Deliberately absent from suggest_params: it is the A/B's
            independent variable, and letting Optuna choose it would turn an
            on/off contrast into a tuned nuisance.
        hoppe_normal_orientation: Orient the sampled model normals by MST
            propagation (methods/base.py orient_normals_hoppe) instead of
            inheriting the mesh's winding. False (the default) keeps today's
            behaviour. Required by anything that reads a normal's DIRECTION
            rather than merely its axis -- all three cart meshes report
            is_orientable() == False, so their normal signs are arbitrary and
            VSACSe2Params.normal_consistency measures a coin flip without this.
            Changes the prepared model, hence part of _get_prep_params_key.
        icp_visibility_cull: Restrict the ICP model cloud to the points actually
            visible from the camera, recomputed at each refinement stage.
            prepare() samples uniformly over the whole slab, but only ~33% of
            those points can be seen: the meshes are zero-thickness shells, so
            every tube contributes a near sheet and a far sheet, and the far
            sheet is fiction as far as the depth camera is concerned. Left in,
            those points still demand a correspondence within
            icp_max_correspondence_distance and drag the model toward the
            sensor -- a ~2.2 cm bias along the cart's own +x, measured by
            initialising ICP AT the ground truth and watching it walk away.
            False (the default) keeps today's behaviour. See vault note
            "30.06 - T0 Translation Error and the Visibility Cull".
        icp_refine_ladder: Extra refinement stages run after the shipped wide
            stage, as decreasing correspondence distances in meters, e.g.
            (0.05, 0.02, 0.01). The wide stage keeps its large capture basin;
            these only tighten. None (the default) runs the single wide stage
            alone. Nearly all of the benefit needs icp_visibility_cull too --
            tightening against a cloud that is two thirds fictional just finds
            the same biased minimum more precisely.
        icp_yaw_guard_deg: Largest yaw change a ladder stage may make. A tight
            culled objective is sharp enough to fall into a symmetry twin: the
            leanflow slab is 0.735 x 0.704 m, near-square in plan, and the tight
            stage snapped every correct leanflow frame to ~90 degrees without
            this. When a stage exceeds the guard its rotation is discarded and
            the stage is re-run in translation only, which keeps the precision
            and forbids the jump. None disables the guard (not recommended with
            a ladder). Ignored when icp_refine_ladder is None.
    """

    z_offset: float | None = None
    z_gate_threshold: float = 0.09
    edge_length_threshold: float = 0.9
    ransac_confidence: float = 0.999
    seed: int | None = 0
    front_crop_depth: float | None = None
    front_face_max_angle_deg: float | None = None
    hoppe_normal_orientation: bool = False
    icp_visibility_cull: bool = False
    icp_refine_ladder: tuple[float, ...] | None = None
    icp_yaw_guard_deg: float | None = 5.0


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
        # This cart's towing-face arrow, resolved per frame in estimate_pose.
        # _global_registration receives only clouds and features, so an instance
        # attribute is the gate's only route in.
        self._front_face: FrontFace | None = None
        # Populated by _refine_pose: fitness, RMSE, the final pose's signed
        # front-face angle, and how far ICP rotated it. Read by benchmark.py's
        # evaluate_pipeline right after estimate_pose() returns, for per-frame
        # auditing.
        self._last_diagnostics: dict | None = None

    def _get_prep_params_key(self) -> tuple:
        # front_crop_depth changes the prepared model representation, so it
        # must be part of the cache key alongside voxel_size.
        return (
            self.params.voxel_size,
            self.params.front_crop_depth,
            self.params.hoppe_normal_orientation,
        )

    def prepare(self, cad_mesh, cart_type: str) -> None:
        prep_params = self._get_prep_params_key()
        cache_key = (self.__class__.__name__, cart_type, prep_params)
        if cache_key in self._PREPARATION_CACHE:
            return

        stale_keys = [
            k
            for k in self._PREPARATION_CACHE
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
        # near-symmetric cart.
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
        model_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        # See methods/base.py:reorient_normals_to_reference. This is the model
        # normal site the 3DoF and VSAC arms actually execute -- prepare() is
        # overridden here, so fixing RansacEstimator.prepare alone would leave
        # every arm on this branch running the unfixed path.
        #
        # Runs FIRST even under hoppe_normal_orientation, for its degenerate
        # rescue: a cancelled voxel average leaves a near-zero normal whose
        # AXIS is noise too, and MST propagation can only fix signs, not axes.
        reorient_normals_to_reference(model_down, model_pc)

        # Then impose a real convention, on model_down itself.
        #
        # Applying this to model_pc instead and letting the line above propagate
        # signs downward loses most of the benefit -- measured as local sign
        # coherence (fraction of near-parallel neighbours whose normals agree in
        # sign; 0.5 is a coin flip):
        #
        #                            colruyt  leanflow  picanol
        #   mesh normals only          0.517     0.514    0.484
        #   hoppe on model_pc          0.783     0.769    0.705
        #   hoppe on model_down        0.953     0.988    0.982
        #
        # The dense cloud's own orientation is fine; it is the propagation
        # through a downsampled cloud's re-estimated normals that degrades it.
        if self.params.hoppe_normal_orientation:
            model_down = orient_normals_hoppe(model_down)

        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )

        self._PREPARATION_CACHE[cache_key] = {
            "model_pc": model_pc,
            "model_down": model_down,
            "model_fpfh": model_fpfh,
            # From the UNCROPPED mesh: the crop leaves x_max untouched, so the
            # arrow is identical either way, but taking it from the full cart
            # keeps the stored geometry meaningful for visual inspection.
            "front_face": front_face_from_mesh(mesh_copy),
        }

    def estimate_pose(self, pcd, cad_mesh, cart_type=None, **kwargs):
        # Clear last frame's flip-disambiguation diagnostics: _refine_pose only
        # runs on frames that get past global registration, so an abstaining
        # frame would otherwise inherit the previous frame's decision record.
        self._last_diagnostics = None

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

        # Resolve the towing-face arrow for THIS cart. Taken from the prepared
        # cache when a cart_type is known (prepare() is a no-op on a hit), and
        # derived directly on the lazy path used by unit tests and inspect_pose
        # on a bare cloud -- if that path silently produced no arrow, those call
        # sites would exercise a different estimator than the benchmark does.
        if cart_type is not None:
            self.prepare(cad_mesh, cart_type)
            cache_key = (self.__class__.__name__, cart_type, self._get_prep_params_key())
            self._front_face = self._PREPARATION_CACHE[cache_key]["front_face"]
        else:
            self._front_face = front_face_from_mesh(cad_mesh)

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
            model_fpfh=np.asarray(model_fpfh.data).T,  # (33, N) -> (N, 33)
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

    def _visible_model_indices(self, model_points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Indices of the model points visible from the camera at pose `T`.

        Katz et al. hidden point removal, which is purely geometric: it needs no
        normals, which matters here because all three cart meshes report
        is_orientable() == False and their normal signs are therefore arbitrary
        (see methods/base.py reorient_normals_to_reference).

        The radius multiplier is deliberately large. Katz's parameter trades
        recall against precision on the visible set; at 100x the cloud diameter
        the spherical flip is nearly a plane, which keeps grazing-incidence
        surface -- real, if noisy, observations -- rather than discarding it.
        """
        world = model_points @ T[:3, :3].T + T[:3, 3]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
        diameter = float(np.linalg.norm(pc.get_max_bound() - pc.get_min_bound()))
        camera = np.asarray(self.extrinsic, dtype=float)[:3, 3]
        _mesh, kept = pc.hidden_point_removal(camera.tolist(), diameter * 100.0)
        return np.asarray(kept, dtype=int)

    def _refine_pose(self, model_pc, scene_pcd, T_init: np.ndarray) -> np.ndarray:
        """
        SE(2)-constrained Gauss-Newton point-to-plane ICP.
        Every increment is composed through the se(2) exponential map, so the
        refined pose never leaves the planar manifold.

        One wide stage always runs, at icp_max_correspondence_distance and over
        the full model cloud -- that is the stage with the capture basin, and it
        is bit-identical to what shipped before. icp_refine_ladder then adds
        optional tightening stages on top; params.icp_visibility_cull decides
        whether they see the whole model or only the part of it the camera can
        actually observe. Ordering matters: culling first and tightening after
        would hand the tight stage a smaller basin than the geometry warrants.

        There is deliberately no second, 180-degree-flipped hypothesis here. The
        flip is settled upstream by FrontFaceGate, geometrically, before a
        candidate is ever scored. Re-opening it downstream on a fitness or RMSE
        tiebreak -- margins of 2% and less, on a near-symmetric object -- is what
        the previous design did, and it could only undo a decision that had
        already been made on stronger evidence.
        """
        scene_points = np.asarray(scene_pcd.points)
        scene_normals = np.asarray(scene_pcd.normals)
        if len(scene_normals) != len(scene_points):
            raise ValueError(
                "SE(2) ICP requires scene normals; the scene cloud should come from "
                "prepare_scene_point_cloud, which estimates and orients them."
            )

        T_init = np.asarray(T_init)
        model_points = np.asarray(model_pc.points)
        result = icp_point_to_plane_se2(
            model_points=model_points,
            scene_points=scene_points,
            scene_normals=scene_normals,
            T_init=T_init,
            max_correspondence_distance=self.params.icp_max_correspondence_distance,
            max_iterations=self.params.icp_max_iterations,
        )

        n_guard_trips = 0
        for stage_distance in self.params.icp_refine_ladder or ():
            T_stage = result.transformation
            stage_points = model_points
            if self.params.icp_visibility_cull:
                visible = self._visible_model_indices(model_points, T_stage)
                # A near-empty visible set means the pose is nowhere near the
                # scene (a flipped or runaway candidate). Falling back to the
                # full cloud keeps such a frame on exactly the code path it
                # would have taken before the cull existed, rather than
                # refining against a handful of points.
                if len(visible) >= _MIN_VISIBLE_MODEL_POINTS:
                    stage_points = model_points[visible]

            stage = icp_point_to_plane_se2(
                model_points=stage_points,
                scene_points=scene_points,
                scene_normals=scene_normals,
                T_init=T_stage,
                max_correspondence_distance=stage_distance,
                max_iterations=self.params.icp_max_iterations,
            )

            guard = self.params.icp_yaw_guard_deg
            if guard is not None and _yaw_delta_deg(T_stage, stage.transformation) > guard:
                # The stage tried to jump to a symmetry twin. Discard its
                # rotation, keep its translation: the gain this ladder exists
                # for is positional, and yaw was already settled upstream.
                n_guard_trips += 1
                stage = icp_translation_only(
                    model_points=stage_points,
                    scene_points=scene_points,
                    scene_normals=scene_normals,
                    T_init=T_stage,
                    max_correspondence_distance=stage_distance,
                    max_iterations=self.params.icp_max_iterations,
                )
            result = stage

        diagnostics = {"fitness": result.fitness, "inlier_rmse": result.inlier_rmse}
        if self.params.icp_refine_ladder:
            # How often the tight stages tried to rotate away. Non-zero is not a
            # fault -- it is the guard doing its job -- but a frame where it
            # trips on every stage is a frame whose slab is symmetric enough
            # that front_crop_depth should be revisited for that cart.
            diagnostics["icp_yaw_guard_trips"] = n_guard_trips
        if self._front_face is not None:
            camera_xy = np.asarray(self.extrinsic, dtype=float)[:3, 3][:2]
            diagnostics["front_face_angle_deg"] = front_face_angle_deg(
                result.transformation, self._front_face, camera_xy
            )
            # How far ICP rotated the pose the gate approved. ICP is
            # unconstrained in yaw, so it can in principle walk a candidate
            # accepted at the boundary across it; this is the field that says
            # whether that actually happens, and it is what would size a bound
            # if it ever needs one.
            yaw_init = float(np.arctan2(T_init[1, 0], T_init[0, 0]))
            yaw_final = float(np.arctan2(result.transformation[1, 0], result.transformation[0, 0]))
            diagnostics["icp_yaw_delta_deg"] = float(
                np.degrees(np.arctan2(np.sin(yaw_final - yaw_init), np.cos(yaw_final - yaw_init)))
            )
        self._last_diagnostics = diagnostics
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
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Suggests parameters for the SE(2)-constrained RANSAC + ICP registration."""

        # Here for documentation purposes it would be nice to list all the parameters that are suggested through the base class.
        params = super().suggest_params(trial, fixed=fixed)

        if "edge_length_threshold" not in fixed:
            params["edge_length_threshold"] = trial.suggest_float(
                "edge_length_threshold", 0.8, 0.95
            )

        # I'm wondering if 0.35 is too high,this has also to be checked against noise resolution levels.
        if "z_gate_threshold" not in fixed:
            params["z_gate_threshold"] = trial.suggest_float("z_gate_threshold", 0.05, 0.35)

        # This will later be modified to be based on a fraction of the mesh to crop out
        if "front_crop_depth" not in fixed:
            params["front_crop_depth"] = trial.suggest_float("front_crop_depth", 0.1, 2.5)

        # front_face_max_angle_deg is NOT suggested: it is the arm's independent
        # variable, set per-arm via --param-overrides. A parameter Optuna sweeps
        # but nothing contrasts is a tuned nuisance, not a controlled comparison.
        #
        # icp_visibility_cull / icp_refine_ladder / icp_yaw_guard_deg are left
        # out for the same reason -- they are the arm's independent variables
        # and are selected by profile (tuned-vis) so control and treatment differ
        # by the profile token alone. icp_max_correspondence_distance IS swept
        # (inherited from RansacEstimator) and should be re-swept once the cull
        # lands: its current optimum was found against a biased objective.
        return params
