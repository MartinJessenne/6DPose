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

Recovered from the reverted commit 54ba373, which bundled this normals fix
(part A) with an unrelated free-space rework (part B) whose arms came back
catastrophic; the whole commit was reverted in 594bef1 and the normals fix was
never evaluated on its own. Only part A is re-landed here, so 54ba373's
TestFreeSpaceGeometryIsCached class is deliberately absent -- it asserted on
prepare() caching free-space geometry that does not exist on this branch.

IMPORTANT, measured after this file was written: consistency with the mesh is
NOT the same as correctness. All three cart meshes report is_orientable() ==
False, so the mesh's own winding -- the reference these tests check against --
is itself arbitrary, and no amount of faithful propagation makes a normal point
outward. See methods/base.py orient_normals_hoppe. These tests remain valid as
what they are: a check that the propagation is faithful.

That is also why the end-to-end class below no longer checks model_down against
the dense model_pc. prepare() deliberately applies Hoppe to model_down ONLY
(hoppe on model_down scores 0.953/0.988/0.982 against 0.783/0.769/0.705 for
hoppe on model_pc), so the two clouds now carry different sign conventions and
cross-cloud agreement is false BY DESIGN. Harmless downstream: model_pc feeds
point-to-plane ICP, whose residual ((p-q).n)^2 is invariant under n -> -n,
while model_down feeds FPFH and the normal-consistency test, which do read
direction. The invariant that survives is internal to model_down.
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


def same_sheet_dissenter_fraction(
    pcd: o3d.geometry.PointCloud,
    tangency: float = 0.3,
    k: int = 12,
    min_support: int = 3,
) -> float:
    """
    Fraction of points whose normal disagrees in sign with its own neighbourhood.

    A globally consistent orientation cannot be checked directly without a
    trusted reference, and on a non-orientable mesh no such reference exists.
    What CAN be checked is the property Hoppe's MST propagation actually
    delivers: neighbouring points on the same surface sheet end up on the same
    side of it.

    "Same sheet" is the load-bearing qualifier. On a thin shell the opposite
    wall is a near neighbour in space while carrying a legitimately opposite
    normal, so a naive k-nearest vote would score correct geometry as broken.
    Neighbour j is admitted only when the separation is nearly tangential to
    i's normal:

        |(p_j - p_i) . n_i| < tangency * ||p_j - p_i||

    The ratio is |cos| of the angle between the separation and n_i, so
    tangency=0.3 admits neighbours lying within arccos(0.3) = 72.5 deg of the
    normal, i.e. within 17.5 deg of i's tangent plane, and rejects the opposite
    wall (separation parallel to n_i, ratio ~1.0). Measured insensitive over
    tangency in [0.2, 0.5]: the result moves by <0.002.

    k=12 spans roughly a two-voxel disc at voxel-size point spacing -- enough
    votes for a majority to mean something without reaching across the shell.
    Points with fewer than min_support admitted neighbours are excluded rather
    than counted as passing: a two-vote majority is noise.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    tree = o3d.geometry.KDTreeFlann(pcd)
    neighbours = np.array(
        [
            [j for j in tree.search_knn_vector_3d(p, k + 1)[1] if j != i][:k]
            for i, p in enumerate(points)
        ]
    )

    delta = points[neighbours] - points[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    along_normal = np.abs(np.einsum("ik,ijk->ij", normals, delta))
    same_sheet = along_normal < tangency * np.maximum(distance, 1e-12)

    agrees = np.einsum("ik,ijk->ij", normals, normals[neighbours]) > 0
    support = same_sheet.sum(axis=1)
    in_favour = (same_sheet & agrees).sum(axis=1)

    testable = support >= min_support
    dissents = testable & (in_favour * 2 < support)
    return float(dissents.sum() / max(int(testable.sum()), 1))


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
        # Open3D keeps its OWN global RNG, separate from numpy's, and prepare()
        # draws from it via sample_points_uniformly. Unseeded, the dissenter
        # fraction below wanders over 0.018-0.036 across repeats, which is not a
        # band a threshold can sit in. Seeding pins prepare() bit-for-bit.
        o3d.utility.random.seed(0)
        estimator = Ransac3DoFEstimator(
            params=Ransac3DoFParams(voxel_size=voxel_size, front_crop_aspect=None),
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
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)

    def test_model_normals_are_locally_sign_coherent(self):
        """
        The surviving direction invariant, once cross-cloud agreement is gone.

        The bound is 5%, and it is not arbitrary at either end.

        Floor -- the fixture is a 2.0 x 1.0 x 0.02 box, so at voxel_size 0.02
        the shell thickness EQUALS the voxel. The local covariance is then one
        voxel thick and a few voxels wide, its smallest eigenvalue is not
        cleanly separated, and a few percent of normals come out with an
        ill-defined AXIS -- 53 of 64 dissenters at that setting have x/y-facing
        normals on a box whose large faces are z-facing. Hoppe re-signs, it
        cannot re-aim (see ransac3dof.py, the comment above the
        reorient_normals_to_reference call), so those points stay. Measured
        with the seed above: 0.037 at voxel 0.02, exactly 0.000 at 0.06, where
        the voxel is 3x the thickness and the fit is well conditioned.

        Ceiling -- corrupting the same seeded cloud reads 0.310 (voxel 0.02) /
        0.375 (0.06) with hoppe_normal_orientation off, 0.172 for 15% of signs
        flipped, and 0.43 for pure noise directions. The defect class this test
        exists for therefore clears the bound by 6-8x.

        Known blind spot: a 5% random sign flip reads 0.086 at voxel 0.02
        (caught) but 0.040 at 0.06 (MISSED) -- that cloud is only 582 points,
        so the bound is ~29 of them and the detection floor there is around 6%
        of points. One bound is set from the worst case rather than a per-voxel
        pair, which leaves 0.06 loose against corruptions no mechanism in this
        codebase actually produces.

        This is the test that fails if prepare() stops orienting model_down, or
        orients the wrong cloud. It does NOT check unit length -- that is
        test_all_model_normals_are_unit_length, and it is the magnitude half of
        the same voxel-averaging defect.
        """
        for voxel_size in (0.02, 0.06):
            with self.subTest(voxel_size=voxel_size):
                down = self.prepared(voxel_size)["model_down"]
                self.assertLess(same_sheet_dissenter_fraction(down), 0.05)
