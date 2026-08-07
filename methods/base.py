import ast
import copy
import dataclasses
import inspect
import logging
import types
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import open3d as o3d

from methods.depth_noise import DepthSensor

if TYPE_CHECKING:
    import optuna

_BOOLS = {"true", "false", "none"}


# Defining custom error types
class ConfigError(Exception):
    """Base for every configuration-resolution failure.

    Having a base class means a caller can catch every failure from this module
    with one `except ConfigError`, without also swallowing unrelated ValueErrors.
    """


class UnknownOverrideError(ConfigError):
    def __init__(self, name: str, valid: set[str]) -> None:
        self.name = name
        self.valid = valid
        super().__init__(f"Unknown parameter override '{name!r}'. Available: {sorted(valid)}")


@dataclass(frozen=True)
class SearchRange:
    """
    Search range descriptor for optuna metadata.
    """

    min: float
    max: float
    step: float | None = None  # Optional step size for discrete search spaces
    log: bool = False  # Whether to keep logs of the parameter values

    def suggest(self, trial: "optuna.Trial", name: str, annotation) -> Any:
        """
        Suggests a value for this parameter using an Optuna trial.

        Args:
            trial: The active Optuna trial.
            name: The name of the parameter to suggest.
            annotation: The type annotation of the parameter, used to determine
                whether to suggest a float or int value.

        Returns:
            Any: The suggested value for the parameter.
        """
        target = _unwrap_optional(annotation)
        if target is int:
            step = int(self.step) if self.step is not None else 1
            return trial.suggest_int(name, int(self.min), int(self.max), step=step, log=self.log)
        return trial.suggest_float(
            name, float(self.min), float(self.max), step=self.step, log=self.log
        )


_NoneType = type(None)  # For type hinting of optional fields


def _unwrap_optional(annotation):
    """
    Annotated[T, ...] -> T and X | None -> X.

    The Annotated unwrap must happen first.
    """

    if hasattr(annotation, "__metadata__"):
        annotation = typing.get_args(annotation)[0]

    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        real = [a for a in typing.get_args(annotation) if a is not _NoneType]
        if len(real) == 1:
            return real[0]

    return annotation


def coerce_override(raw, annotation, name: str):
    """One CLI string -> the value the field is declared to hold.

    Parse first, narrow second. Parsing with ast.leteral_eval means `None`,
    `True`, `(0.05, 0.02)`, `1e-3` and `40` all become the right Python object
    without this function knowing which fields exist. Narrowing then rejects
    what the field cannot hold, at startup rather than mid-sweep.
    """

    if not isinstance(raw, str):
        return raw

    text = raw.strip()

    # tyro hands through the shell's spelling; literal_eval wants Python's.
    parsed_source = text.capitalize() if text.lower() in {"true", "false", "none"} else text
    try:
        value = ast.literal_eval(parsed_source)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{name}={raw!r} is not a Python literal") from exc

    target = _unwrap_optional(annotation)

    if value is None:
        return None

    if target is bool and not isinstance(value, bool):
        raise ValueError(f"{name}={raw!r} is not a boolean")

    if target is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if target is int and isinstance(value, bool):
        raise ValueError(f"{name}={raw!r} is not an integer")
    return value


@dataclass(frozen=True)
class BaseParams:
    seed: int | None = 0

    def with_overrides(self, **overrides: Any) -> Self:
        """
        Returns a new instance of the dataclass with specified fields overridden.
        Validates parameter names and coerces string CLI values to field target types.
        """
        if not overrides:
            return self

        hints = typing.get_type_hints(type(self))
        valid = {f.name for f in dataclasses.fields(self)}
        coerced = {}
        for name, raw in overrides.items():
            if name not in valid:
                raise UnknownOverrideError(name, valid)
            coerced[name] = coerce_override(raw, hints[name], name)
        return dataclasses.replace(self, **coerced)

    @classmethod
    def search_space(cls) -> dict[str, SearchRange]:
        """Every field this params class declares as searchable, by name.
        Returns:
            dict[str, SearchRange]: Mapping of field names to their SearchRange metadata.
        """
        hints = typing.get_type_hints(cls, include_extras=True)
        out = {}

        for f in dataclasses.fields(cls):
            meta = getattr(
                hints[f.name], "__metadata__", ()
            )  # __metadata__ is a tuple of all Annotated metadata
            rng = next((m for m in meta if isinstance(m, SearchRange)), None)
            if rng:
                out[f.name] = rng

        return out

    @classmethod
    def sample_optuna(cls, trial: "optuna.Trial", base: Self | None = None, fixed=()) -> Self:
        """
        `base` (default: class defaults) with every non-fixed searchable field resampled.
        """
        base = base if base else cls()
        hints = typing.get_type_hints(cls, include_extras=True)
        sampled = {
            name: rng.suggest(trial, name, hints[name])
            for name, rng in cls.search_space().items()
            if name not in fixed
        }
        return dataclasses.replace(base, **sampled)

    def __post_init__(self):
        hints = typing.get_type_hints(type(self), include_extras=True)

        for name, rng in self.search_space().items():
            v = getattr(self, name)
            if v is None:
                continue

            declared = _unwrap_optional(hints[name])
            if not isinstance(v, declared) or isinstance(v, bool) is not (declared is bool):
                raise TypeError(f"{type(self).__name__}.{name} = {v!r} is not {declared.__name__}")

            if not (rng.min <= v <= rng.max):
                raise ValueError(
                    f"{type(self).__name__}.{name} = {v} outside [{rng.min}, {rng.max}]"
                )


class BasePoseEstimator(ABC):
    """Abstract base class representing a 6D Pose Estimation method."""

    _PREPARATION_CACHE = {}  # Global cache: (class_name, cart_type, prep_params_tuple) -> prepared_dict

    # Each subclass must define its own dataclass for parameters, which will be used for validation and type coercion.
    params_cls: type[BaseParams]

    @classmethod
    def build(
        cls,
        *,
        profile_params: BaseParams | None = None,
        overrides: dict[str, Any] | None = None,
        sensor: DepthSensor,
    ) -> Self:
        """1-line construction: loads baseline profile, applies overrides, and instantiates the estimator."""
        base = cls.params_cls() if profile_params is None else profile_params
        if overrides:
            base = base.with_overrides(**overrides)
        return cls(params=base, sensor=sensor)

    @abstractmethod
    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        cart_type: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Reconstructed scene point cloud in robot frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            cart_type (str, optional): Name of the cart type.
            **kwargs: Method-specific inputs (e.g., rgb_crop, depth_crop, camera intrinsics).

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix, or None if estimation fails.
        """
        pass

    def prepare(self, cad_mesh: o3d.geometry.TriangleMesh, cart_type: str) -> None:  # noqa: B027
        """
        Prepares and caches model-specific properties (e.g., FPFH features, PPF match database)
        for a given CAD model to speed up online evaluation.

        Args:
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model.
            cart_type (str): Name of the cart type.
        """
        # Default implementation for estimators that do not require pre-computation.
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and not isinstance(getattr(cls, "params_cls", None), type):
            raise TypeError(
                f"Concrete subclass {cls.__name__} must define a 'params_cls' attribute "
                f"that is a dataclass type inheriting from BaseParams."
            )


def reorient_normals_to_reference(
    target: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
) -> None:
    """
    Re-imposes `reference`'s normal orientation convention on `target`, in place.

    Why this is needed at all: geometry determines a normal only up to sign (the
    orthogonal complement of the tangent plane is a line, and a line holds two
    unit vectors). PCA cannot pick between them -- its objective satisfies
    E(n) = E(-n) -- so `estimate_normals` resolves the sign by *inheriting* the
    normal already present at that point (Open3D >= 0.14). That inheritance is
    exactly the problem here: `voxel_down_sample` averages the normals falling
    into a voxel WITHOUT renormalising, so on this fleet's thin tubular frame the
    two walls of a tube land in one voxel and their outward normals cancel. The
    prior handed to `estimate_normals` is then a near-zero vector whose direction
    is rounding noise, and the estimated normal faithfully inherits that noise.
    Measured on colruyt.ply: ~12% of model normals affected at voxel_size 0.02,
    ~47.5% at 0.06.

    The repair uses the densely sampled cloud, whose normals are interpolated
    from the mesh's own consistently-wound triangles and are therefore exactly
    outward. Only the SIGN is taken from the reference: the estimated direction
    (and hence the support radius it was computed at) is left untouched, so this
    changes one variable and not two.

    Args:
        target: Point cloud whose normals are re-signed in place. Must have normals.
        reference: Densely sampled cloud carrying the trusted convention.
    """
    if target.is_empty() or reference.is_empty() or not reference.has_normals():
        return

    target_normals = np.asarray(target.normals)
    if len(target_normals) == 0:
        return

    reference_normals = np.asarray(reference.normals)
    kdtree = o3d.geometry.KDTreeFlann(reference)

    nearest = np.empty(len(target_normals), dtype=np.int64)
    for i, point in enumerate(np.asarray(target.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        nearest[i] = idx[0]

    reference_at_target = reference_normals[nearest]

    # A cancelled voxel average can leave an estimated normal of near-zero
    # length; renormalise so downstream dot products are comparable. Points whose
    # normal is degenerate beyond rescue fall back to the mesh normal outright.
    lengths = np.linalg.norm(target_normals, axis=1)
    degenerate = lengths < 1e-9
    target_normals[degenerate] = reference_at_target[degenerate]
    lengths[degenerate] = np.linalg.norm(target_normals[degenerate], axis=1)
    target_normals /= np.maximum(lengths, 1e-12)[:, None]

    flip = np.einsum("ij,ij->i", target_normals, reference_at_target) < 0.0
    target_normals[flip] *= -1.0

    target.normals = o3d.utility.Vector3dVector(target_normals)


def orient_normals_hoppe(pcd: o3d.geometry.PointCloud, k: int = 30) -> o3d.geometry.PointCloud:
    """
    Gives a point cloud a globally consistent, outward-pointing normal
    convention without using mesh topology. Returns a new cloud.

    Why this is needed instead of the mesh's own normals: all three cart meshes
    report `is_orientable() == False` (and are neither edge- nor vertex-manifold,
    nor watertight -- they are open surface shells, as CAD tube geometry modelled
    without wall thickness usually is). On a non-orientable surface there is NO
    consistent assignment of triangle winding, so `compute_vertex_normals()`
    returns signs fixed by whatever winding each triangle happens to carry, and
    "outward" is not a property the mesh can express. Mesh cleanup does not help:
    `remove_duplicated_*` / `remove_degenerate_triangles` leave
    `is_orientable()` False and `orient_triangles()` returns False on all three.

    That makes `reorient_normals_to_reference` insufficient on its own. It
    propagates the reference's convention faithfully, which is the right
    behaviour -- but the reference's convention is arbitrary, so consistency is
    achieved with respect to noise.

    This uses the standard alternative (Hoppe et al., "Surface Reconstruction
    from Unorganized Points", SIGGRAPH 1992): build a Riemannian graph over the
    points, and propagate orientation along its minimum spanning tree, choosing
    at each edge the sign that maximises agreement between neighbouring tangent
    planes. That fixes all signs RELATIVE to one another; the one remaining
    global degree of freedom is resolved by majority vote against the outward
    radial direction from the centroid.

    Measured effect on the E1 normal-agreement signal (18 fixtures, voxel 0.02,
    fraction of positional inliers surviving the normal test at ground truth
    minus the same at the 180-degree twin -- see
    scripts/probe_normal_agreement.py):

        mesh normals   keep_gt 0.510  keep_flip 0.502  separation +0.008
        this function  keep_gt 0.628  keep_flip 0.384  separation +0.247

    +0.008 is a coin flip at both poses, i.e. no signal at all. It is not a
    complete fix -- 0.628 is well short of the ~1.0 a correctly oriented model
    would give, and the residual is the MST wandering between the frame's
    disconnected tube components -- but it is the difference between a mechanism
    that can carry information and one that provably cannot.

    Args:
        pcd: Cloud with normals already estimated. Not mutated.
        k: Neighbourhood size for the Riemannian graph.

    Returns:
        A new point cloud with re-signed normals.
    """
    oriented = o3d.geometry.PointCloud(pcd)
    if oriented.is_empty() or not oriented.has_normals():
        return oriented

    oriented.orient_normals_consistent_tangent_plane(k)

    points = np.asarray(oriented.points)
    normals = np.asarray(oriented.normals)
    radial = points - points.mean(axis=0)
    if np.mean(np.einsum("ij,ij->i", normals, radial) > 0.0) < 0.5:
        oriented.normals = o3d.utility.Vector3dVector(-normals)
    return oriented


def prepare_scene_point_cloud(
    pcd: o3d.geometry.PointCloud,
    T_robot_camera: np.ndarray,
    normal_radius: float = 0.05,
    normal_max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """
    Prepares the scene point cloud by estimating surface normals, orienting them
    towards the camera origin, and transforming the points to the robot's base frame.

    Note:
        This function does not mutate the input point cloud in-place. It operates on a
        deep copy to prevent coordinate transform contamination.

    Args:
        pcd (o3d.geometry.PointCloud): Segmented scene point cloud in camera coordinate frame.
        T_robot_camera (np.ndarray): 4x4 extrinsic transform from camera to robot base link.
        normal_radius (float): Radius parameter for hybrid KD-tree normal search.
        normal_max_nn (int): Max neighborhood size for KD-tree search.

    Returns:
        o3d.geometry.PointCloud: Transformed point cloud with oriented surface normals.
    """
    # Deepcopy to prevent mutating the caller's point cloud (double extrinsics bug)
    pcd = copy.deepcopy(pcd)
    if pcd.is_empty():
        return pcd
    # 1. Estimate surface normals using hybrid KD-tree search
    # This computes a local plane fit for each point's neighbors. If points lack normals,
    # ICP refinement (Point-to-Plane) and PPF matching will fail.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
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
    icp_max_iterations: int,
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
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations)

    # Hypothesis 1: Run ICP on the original registration guess
    icp_result_1 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    # Hypothesis 2: Run ICP on the alignment rotated 180 degrees around the local Z-axis
    T_flip = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    T_init_flipped = T_init @ T_flip

    icp_result_2 = o3d.pipelines.registration.registration_icp(
        model_pc,
        scene_pcd,
        icp_max_correspondence_distance,
        T_init_flipped,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    # Select the hypothesis that maximizes point cloud overlap (fitness score)
    if icp_result_1.fitness > icp_result_2.fitness:
        best_result = icp_result_1
        logging.info(
            f"Orientation selected: Original (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})"
        )
    elif icp_result_2.fitness > icp_result_1.fitness:
        best_result = icp_result_2
        logging.info(
            f"Orientation selected: Flipped 180° (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})"
        )
    else:
        # Tie breaker: pick the one with lower RMSE
        if icp_result_1.inlier_rmse <= icp_result_2.inlier_rmse:
            best_result = icp_result_1
            logging.info(
                f"Orientation selected: Original [Tie breaker] (Fitness: {icp_result_1.fitness:.4f}, RMSE: {icp_result_1.inlier_rmse:.4f})"
            )
        else:
            best_result = icp_result_2
            logging.info(
                f"Orientation selected: Flipped 180° [Tie breaker] (Fitness: {icp_result_2.fitness:.4f}, RMSE: {icp_result_2.inlier_rmse:.4f})"
            )

    return best_result.transformation
