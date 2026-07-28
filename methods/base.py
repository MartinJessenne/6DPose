import copy
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    import optuna


class BasePoseEstimator(ABC):
    """Abstract base class representing a 6D Pose Estimation method."""

    _PREPARATION_CACHE = {}  # Global cache: (class_name, cart_type, prep_params_tuple) -> prepared_dict

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

    @classmethod
    def suggest_params(cls, trial: "optuna.Trial") -> dict[str, Any]:
        """
        Suggests hyperparameters for this matching method using an Optuna trial.

        Args:
            trial: The active Optuna trial.

        Returns:
            dict[str, Any]: Suggested parameter dictionary.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(
            f"Estimator class '{cls.__name__}' does not implement 'suggest_params' for parameter sweeps."
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


def orient_normals_hoppe(
    pcd: o3d.geometry.PointCloud, k: int = 30
) -> o3d.geometry.PointCloud:
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
