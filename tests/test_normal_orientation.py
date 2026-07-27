"""
Tests for the normal sign convention re-imposed after voxel downsampling.

Background, because the failure is not visible from the call site: a surface
determines a normal only up to sign, and PCA cannot choose between the two --
its objective satisfies E(n) = E(-n). Open3D's estimate_normals therefore
resolves the sign by INHERITING whatever normal is already at that point. That
inheritance is what makes voxel_down_sample's behaviour matter: it averages the
normals falling in a voxel without renormalising, so on a thin tube the two walls
cancel, and estimate_normals faithfully inherits a near-zero vector whose
direction is rounding noise. FPFH is equivariant (not invariant) under per-point
sign flips, so those points feed corrupted descriptors into matching.
"""

import unittest

import numpy as np
import open3d as o3d

from methods.base import reorient_normals_to_reference
from methods.ransac3dof import Ransac3DoFEstimator, Ransac3DoFParams


def two_walled_slab(thickness: float = 0.01, n_per_wall: int = 400) -> o3d.geometry.PointCloud:
    """
    Two parallel sheets a hair apart with opposing outward normals -- the tube
    cross-section that defeats unrenormalised averaging.
    """
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.25, 0.25, size=(n_per_wall, 2))

    front = np.column_stack([xy, np.full(n_per_wall, +thickness / 2)])
    back = np.column_stack([xy, np.full(n_per_wall, -thickness / 2)])

    normals = np.zeros((2 * n_per_wall, 3))
    normals[:n_per_wall, 2] = 1.0
    normals[n_per_wall:, 2] = -1.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([front, back]))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


class TestVoxelAveragingDefect(unittest.TestCase):
    """Characterises the defect itself, so the fix cannot be removed silently."""

    def test_downsampling_cancels_opposing_normals(self):
        dense = two_walled_slab()
        # A voxel far larger than the wall separation merges both walls.
        down = dense.voxel_down_sample(0.1)

        lengths = np.linalg.norm(np.asarray(down.normals), axis=1)

        # Not merely short -- essentially annihilated. Direction here is noise.
        self.assertLess(lengths.max(), 0.1)

    def test_estimate_normals_inherits_rather_than_replaces_the_sign(self):
        # The premise correction: estimate_normals does NOT hand back an
        # arbitrary sign, it propagates the one already present. So it cannot
        # repair a cancelled average, and dropping it would not help either.
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).sample_points_uniformly(2000)
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 10.0]))
        planted = np.asarray(pcd.normals).copy()

        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        after = np.asarray(pcd.normals)

        agreement = np.mean(np.einsum("ij,ij->i", planted, after) > 0)
        self.assertGreater(agreement, 0.99)


class TestReorientToReference(unittest.TestCase):
    def test_flips_normals_that_disagree_with_the_reference(self):
        reference = two_walled_slab()
        target = o3d.geometry.PointCloud(reference)

        corrupted = np.asarray(reference.normals).copy()
        corrupted[::2] *= -1.0
        target.normals = o3d.utility.Vector3dVector(corrupted)

        reorient_normals_to_reference(target, reference)

        dots = np.einsum("ij,ij->i", np.asarray(target.normals), np.asarray(reference.normals))
        self.assertTrue(np.all(dots > 0))

    def test_rescues_cancelled_normals_to_unit_length(self):
        dense = two_walled_slab()
        down = dense.voxel_down_sample(0.1)
        self.assertLess(np.linalg.norm(np.asarray(down.normals), axis=1).max(), 0.1)

        reorient_normals_to_reference(down, dense)

        lengths = np.linalg.norm(np.asarray(down.normals), axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-9)

    def test_leaves_already_consistent_normals_untouched(self):
        reference = two_walled_slab()
        target = o3d.geometry.PointCloud(reference)
        before = np.asarray(target.normals).copy()

        reorient_normals_to_reference(target, reference)

        np.testing.assert_allclose(np.asarray(target.normals), before, atol=1e-12)


class TestPreparedModelNormals(unittest.TestCase):
    """End-to-end invariants on the geometry the 3DoF/VSAC arms actually use.

    prepare() here overrides RansacEstimator.prepare, so this is a distinct code
    path from the inherited one -- fixing only methods/ransac.py would leave it
    broken while every unit test on the parent passed.
    """

    @classmethod
    def setUpClass(cls):
        cls.mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=0.02)
        cls.mesh.compute_vertex_normals()

    def prepared(self, voxel_size: float) -> dict:
        estimator = Ransac3DoFEstimator(
            params=Ransac3DoFParams(voxel_size=voxel_size, front_crop_depth=None),
            extrinsic=np.eye(4),
        )
        cart_type = f"unit_test_{voxel_size}"
        estimator.prepare(self.mesh, cart_type)
        return estimator._PREPARATION_CACHE[
            (type(estimator).__name__, cart_type, estimator._get_prep_params_key())
        ]

    def test_all_model_normals_are_unit_length(self):
        for voxel_size in (0.02, 0.06):
            with self.subTest(voxel_size=voxel_size):
                normals = np.asarray(self.prepared(voxel_size)["model_down"].normals)
                np.testing.assert_allclose(
                    np.linalg.norm(normals, axis=1), 1.0, atol=1e-9
                )

    def test_all_model_normals_agree_with_the_mesh(self):
        for voxel_size in (0.02, 0.06):
            with self.subTest(voxel_size=voxel_size):
                prep = self.prepared(voxel_size)
                down, dense = prep["model_down"], prep["model_pc"]

                tree = o3d.geometry.KDTreeFlann(dense)
                dense_normals = np.asarray(dense.normals)
                nearest = np.array(
                    [tree.search_knn_vector_3d(p, 1)[1][0] for p in np.asarray(down.points)]
                )
                dots = np.einsum(
                    "ij,ij->i", np.asarray(down.normals), dense_normals[nearest]
                )

                self.assertTrue(np.all(dots >= 0.0))


class TestFreeSpaceGeometryIsCached(unittest.TestCase):
    def test_prepare_caches_full_cart_points_alongside_the_slab(self):
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=0.5)
        mesh.compute_vertex_normals()
        estimator = Ransac3DoFEstimator(
            params=Ransac3DoFParams(voxel_size=0.05, front_crop_depth=0.3), extrinsic=np.eye(4)
        )
        estimator.prepare(mesh, "cache_probe")
        prep = estimator._PREPARATION_CACHE[
            (type(estimator).__name__, "cache_probe", estimator._get_prep_params_key())
        ]

        slab = np.asarray(prep["model_pc"].points)
        full = prep["full_points"]

        # The whole point of the fix: the visibility geometry must extend beyond
        # the registration slab. If these x-extents matched, the free-space test
        # would again be judging a shape that is near-invariant under the flip.
        self.assertGreater(
            full[:, 0].max() - full[:, 0].min(),
            slab[:, 0].max() - slab[:, 0].min(),
        )
        self.assertGreater(len(prep["gate_points"]), 0)
        self.assertLessEqual(len(prep["gate_points"]), len(full))

    def test_gate_subsample_size_is_part_of_the_cache_key(self):
        # _PREPARATION_CACHE is class-level: a parameter that changes cached
        # content but is missing from the key makes two configurations collide
        # inside one process, so the second silently reuses the first's geometry.
        a = Ransac3DoFEstimator(
            params=Ransac3DoFParams(free_space_gate_points=100), extrinsic=np.eye(4)
        )
        b = Ransac3DoFEstimator(
            params=Ransac3DoFParams(free_space_gate_points=500), extrinsic=np.eye(4)
        )
        self.assertNotEqual(a._get_prep_params_key(), b._get_prep_params_key())


if __name__ == "__main__":
    unittest.main()
