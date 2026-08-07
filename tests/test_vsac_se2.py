import unittest

import numpy as np
from scipy.spatial import cKDTree

from methods.vsac_se2 import (
    MatchedNpPointClouds,
    VsacConfig,
    count_independent_inliers,
    prosac_pairs,
    score_msac,
    vsac_se2,
)
from tests.helpers import se2_pose


class TestProsacPairs(unittest.TestCase):
    def test_prosac_pairs_schedule_properties(self):
        n_corr = 100
        max_iter = 10000
        rng = np.random.default_rng(42)

        pairs = list(prosac_pairs(n_corr=n_corr, max_iteration=max_iter, rng=rng))
        self.assertEqual(len(pairs), max_iter)

        # 1. First iteration must sample from initial pool m=2 -> index 1 and index 0
        i1, i2 = pairs[0]
        self.assertEqual(i2, 1)  # newest element in pool of size 2 is index 1
        self.assertEqual(i1, 0)  # drawn uniformly from {0}

        # 2. All generated indices must stay within valid bounds [0, n_corr)
        for i1, i2 in pairs:
            self.assertTrue(0 <= i1 < n_corr, f"i1={i1} out of bounds")
            self.assertTrue(0 <= i2 < n_corr, f"i2={i2} out of bounds")
            self.assertNotEqual(i1, i2, "Sampled identical point pair")

    def test_prosac_pairs_insufficient_points(self):
        rng = np.random.default_rng(0)
        pairs = list(prosac_pairs(n_corr=1, max_iteration=100, rng=rng))
        self.assertEqual(len(pairs), 0)


class TestIndependentInliers(unittest.TestCase):
    def test_clustered_vs_spread_points(self):
        # 1. Dense cluster within 5cm (0.05m radius)
        rng = np.random.default_rng(42)
        clustered_pts = rng.uniform(-0.02, 0.02, size=(100, 2))
        mask_all = np.ones(100, dtype=bool)

        n_indep_clustered = count_independent_inliers(
            inlier_mask=mask_all, model_points_2d=clustered_pts, rho=0.3
        )
        self.assertEqual(
            n_indep_clustered, 1, "Tight 5cm cluster should collapse to 1 independent point"
        )

        # 2. Grid of 100 points spaced 0.5m apart
        x = np.linspace(0, 4.5, 10)
        y = np.linspace(0, 4.5, 10)
        xx, yy = np.meshgrid(x, y)
        spread_pts = np.column_stack([xx.ravel(), yy.ravel()])

        n_indep_spread = count_independent_inliers(
            inlier_mask=mask_all, model_points_2d=spread_pts, rho=0.3
        )
        self.assertEqual(
            n_indep_spread, 100, "Points spaced 0.5m apart (rho=0.3m) should all be independent"
        )

    def test_empty_mask_returns_zero(self):
        pts = np.zeros((50, 2))
        mask_none = np.zeros(50, dtype=bool)
        n_indep = count_independent_inliers(inlier_mask=mask_none, model_points_2d=pts, rho=0.3)
        self.assertEqual(n_indep, 0)


class TestVsacSe2(unittest.TestCase):
    def _make_scene(self, rng, theta_true, t_true, z_offset, n=200, noise=0.002):
        model_points = rng.uniform(-1, 1, size=(n, 3))
        model_points[:, 2] = rng.uniform(0.0, 0.5, size=n)

        T_true = se2_pose(theta_true, t_true, z_offset)
        scene_points = model_points @ T_true[:3, :3].T + T_true[:3, 3]
        scene_points += rng.normal(scale=noise, size=scene_points.shape)

        model_fpfh = rng.normal(size=(n, 33))
        scene_fpfh = model_fpfh.copy()
        return model_points, scene_points, model_fpfh, scene_fpfh

    def test_recovery_with_outliers_and_z_offset(self):
        rng = np.random.default_rng(42)
        theta_true = np.deg2rad(52.0)
        t_true = np.array([0.6, -0.3])
        z_offset = 0.13

        model_points, scene_points, model_fpfh, scene_fpfh = self._make_scene(
            rng, theta_true, t_true, z_offset
        )

        # Corrupt 30% of scene descriptors so they act as outliers
        n_bad = int(0.3 * len(scene_fpfh))
        bad_idx = rng.choice(len(scene_fpfh), size=n_bad, replace=False)
        scene_fpfh[bad_idx] = rng.normal(size=(n_bad, 33))

        pt_clouds = MatchedNpPointClouds(
            model_points=model_points,
            scene_points=scene_points,
            model_fpfh=model_fpfh,
            scene_fpfh=scene_fpfh,
        )
        params = VsacConfig(
            distance_threshold=0.01,
            max_iterations=2000,
            z_offset=z_offset,
            min_sample_distance=0.3,
        )

        result = vsac_se2(pt_clouds, params, rng=rng)

        self.assertGreater(result.fitness, 0.8)
        T = result.transformation
        theta_est = np.arctan2(T[1, 0], T[0, 0])
        self.assertAlmostEqual(theta_est, theta_true, delta=np.deg2rad(1.0))
        np.testing.assert_allclose(T[:2, 3], t_true, atol=0.01)
        self.assertAlmostEqual(T[2, 3], z_offset, places=12)
        np.testing.assert_allclose(T[2, :3], [0.0, 0.0, 1.0], atol=1e-12)

    def test_flip_ambiguous_tiebreak_prefers_spread_support(self):
        """
        Branch payoff test (VSAC_Implementation_Plan.md Step 12): two candidate
        poses each explain exactly half the model (tied MSAC quality, tied
        fitness) but the supporting points are spread across the object under
        one pose and packed into one small region under the other -- exactly
        what a 180-degree flip on a near-symmetric object looks like (plenty of
        raw correspondences, but clustered instead of spread across the
        asymmetric part). vsac_se2 must prefer the spread-support pose via the
        independent-inlier tiebreak, not whichever pose PROSAC happens to
        sample first.
        """
        rng = np.random.default_rng(7)

        # "Spread" group: 4 well-separated corners (1.0 m apart, >> rho=0.3),
        # a few points per corner so its point count matches the clustered
        # group. Small per-corner jitter keeps each corner one independent
        # cluster (well under rho) without making same-corner pairs unusable.
        corners = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
        spread_xy = np.repeat(corners, 4, axis=0) + rng.normal(scale=0.005, size=(16, 2))
        spread_model = np.column_stack([spread_xy, np.zeros(16)])

        # "Clustered" group: same point count (16), all within one small
        # patch (jitter std 0.02, so pairwise spread stays << rho=0.3 -- the
        # whole group collapses to a single independent inlier) far from the
        # spread group so the two never cross-contaminate each other's fit.
        clustered_xy = np.array([2.0, 2.0]) + rng.normal(scale=0.02, size=(16, 2))
        clustered_model = np.column_stack([clustered_xy, np.zeros(16)])

        model_points = np.vstack([spread_model, clustered_model])
        n_model = len(model_points)

        theta_spread, t_spread = np.deg2rad(20.0), np.array([1.0, -0.5])
        theta_clustered, t_clustered = np.deg2rad(-40.0), np.array([-3.0, 2.0])
        T_spread = se2_pose(theta_spread, t_spread)
        T_clustered = se2_pose(theta_clustered, t_clustered)

        scene_points = np.vstack(
            [
                spread_model @ T_spread[:3, :3].T + T_spread[:3, 3],
                clustered_model @ T_clustered[:3, :3].T + T_clustered[:3, 3],
            ]
        )

        # Perfect, exclusive descriptor correspondence (model point i <-> scene
        # point i), matching the construction the other vsac_se2 tests already
        # use, so mutual-NN matching is exact and both candidate poses are
        # reliably sampled as 2-point hypotheses.
        model_fpfh = rng.normal(size=(n_model, 33))
        scene_fpfh = model_fpfh.copy()

        pt_clouds = MatchedNpPointClouds(
            model_points=model_points,
            scene_points=scene_points,
            model_fpfh=model_fpfh,
            scene_fpfh=scene_fpfh,
        )
        params = VsacConfig(
            distance_threshold=0.03,
            max_iterations=3000,
            min_sample_distance=0.003,
            rho=0.3,
        )

        result = vsac_se2(pt_clouds, params, rng=np.random.default_rng(7))

        # Sanity check the tie is real (each hypothesis explains exactly its
        # own half of the model), not a construction bug where one pose
        # accidentally explains both groups.
        self.assertAlmostEqual(result.fitness, 0.5, delta=0.05)

        theta_est = np.arctan2(result.transformation[1, 0], result.transformation[0, 0])
        self.assertAlmostEqual(theta_est, theta_spread, delta=np.deg2rad(2.0))
        np.testing.assert_allclose(result.transformation[:2, 3], t_spread, atol=0.02)

    def test_seeded_runs_reproducible(self):
        theta_true, t_true, z_offset = 0.9, np.array([-0.4, 0.7]), 0.05

        results = []
        for _ in range(2):
            rng = np.random.default_rng(123)
            model_points, scene_points, model_fpfh, scene_fpfh = self._make_scene(
                rng, theta_true, t_true, z_offset
            )
            pt_clouds = MatchedNpPointClouds(
                model_points=model_points,
                scene_points=scene_points,
                model_fpfh=model_fpfh,
                scene_fpfh=scene_fpfh,
            )
            params = VsacConfig(
                distance_threshold=0.01,
                max_iterations=1000,
                z_offset=z_offset,
            )
            result = vsac_se2(pt_clouds, params, rng=rng)
            results.append(result)

        np.testing.assert_array_equal(results[0].transformation, results[1].transformation)
        self.assertEqual(results[0].fitness, results[1].fitness)

    def test_insufficient_points_returns_failure(self):
        pt_clouds = MatchedNpPointClouds(
            model_points=np.zeros((1, 3)),
            scene_points=np.zeros((1, 3)),
            model_fpfh=np.ones((1, 33)),
            scene_fpfh=np.ones((1, 33)),
        )
        params = VsacConfig(distance_threshold=0.01, max_iterations=10)
        result = vsac_se2(pt_clouds, params)
        self.assertEqual(result.fitness, 0.0)
        np.testing.assert_array_equal(result.transformation, np.eye(4))


class TestRandomModelRejection(unittest.TestCase):
    """The Poisson null gate that used to live here was ablated -- see the
    comment at vsac_se2's final rejection. What must survive the ablation is
    that a pure-noise scene still yields no pose, via the plain no-inliers
    check rather than a statistical test."""

    def test_pure_noise_scene_yields_no_pose(self):
        rng = np.random.default_rng(999)
        # Random model points
        model_points = rng.uniform(-1, 1, size=(200, 3))
        # Scene points in same volume with random clutter geometry
        scene_points = rng.uniform(-1, 1, size=(200, 3))

        model_fpfh = rng.normal(size=(200, 33))
        scene_fpfh = rng.normal(size=(200, 33))

        pt_clouds = MatchedNpPointClouds(
            model_points=model_points,
            scene_points=scene_points,
            model_fpfh=model_fpfh,
            scene_fpfh=scene_fpfh,
        )
        params = VsacConfig(
            distance_threshold=0.01,
            max_iterations=100,
            z_gate_threshold=100.0,  # wide z-gate so correspondences pass gate
        )
        result = vsac_se2(pt_clouds, params, rng=rng)
        self.assertEqual(result.fitness, 0.0)
        self.assertEqual(result.reason, "no_inliers")
        np.testing.assert_array_equal(result.transformation, np.eye(4))


class TestNormalConsistencyIsAsymmetric(unittest.TestCase):
    """
    Arm E1. The whole claim is that adding the normal term makes score_msac
    distinguish a pose from its 180-degree twin, which positional scoring
    provably cannot. Both halves are asserted here: WITHOUT normals the two
    scores must be identical (that is the defect), and WITH them the correct
    pose must win. A test that only checked the second half would pass just as
    happily if the scorer had never been blind in the first place.
    """

    def _one_sided_surface(self):
        """A planar patch seen from +z: outward normals up, sensor above."""
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 12), np.linspace(-0.2, 0.2, 6))
        pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
        normals = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
        return pts, normals

    def _rotate_z(self, v, deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return v @ R.T

    def test_positional_scoring_cannot_see_a_normal_flip(self):
        pts, normals = self._one_sided_surface()
        tree = cKDTree(pts)

        # Same points, opposite normals: geometrically indistinguishable.
        q_correct, _, _, _ = score_msac(pts, tree, tau=0.02)
        q_flipped, _, _, _ = score_msac(pts, tree, tau=0.02)
        self.assertAlmostEqual(q_correct, q_flipped, places=12)

    def test_normal_term_separates_the_twin(self):
        pts, normals = self._one_sided_surface()
        tree = cKDTree(pts)

        q_agree, fit_agree, mask_agree, _ = score_msac(
            pts,
            tree,
            tau=0.02,
            transformed_model_normals=normals,
            scene_normals=normals,
        )
        # A 180-degree yaw does not move a z-normal, so use the physically
        # meaningful case instead: the model face now covering this surface is
        # the opposite one, whose outward normal points away from the sensor.
        q_oppose, fit_oppose, mask_oppose, _ = score_msac(
            pts,
            tree,
            tau=0.02,
            transformed_model_normals=-normals,
            scene_normals=normals,
        )

        self.assertTrue(np.all(mask_agree))
        self.assertFalse(np.any(mask_oppose))
        self.assertAlmostEqual(fit_agree, 1.0)
        self.assertAlmostEqual(fit_oppose, 0.0)
        self.assertGreater(q_agree, q_oppose)

    def test_yaw_flip_of_a_vertical_face_changes_the_verdict(self):
        """The operational case: a vertical face, flipped 180 degrees in yaw."""
        ys, zs = np.meshgrid(np.linspace(-0.3, 0.3, 10), np.linspace(0.0, 0.6, 8))
        pts = np.stack([np.zeros(ys.size), ys.ravel(), zs.ravel()], axis=1)
        scene_normals = np.tile([-1.0, 0.0, 0.0], (len(pts), 1))  # towards sensor at -x
        model_normals = np.tile([-1.0, 0.0, 0.0], (len(pts), 1))  # outward, agrees
        tree = cKDTree(pts)

        _, fit_correct, _, _ = score_msac(
            pts,
            tree,
            tau=0.02,
            transformed_model_normals=model_normals,
            scene_normals=scene_normals,
        )
        _, fit_flipped, _, _ = score_msac(
            pts,
            tree,
            tau=0.02,
            transformed_model_normals=self._rotate_z(model_normals, 180.0),
            scene_normals=scene_normals,
        )
        self.assertAlmostEqual(fit_correct, 1.0)
        self.assertAlmostEqual(fit_flipped, 0.0)

    def test_flag_off_or_missing_normals_is_the_control_arm(self):
        """vsac_se2 must fall back to positional scoring, not raise."""
        rng = np.random.default_rng(0)
        pts = rng.random((60, 3))
        clouds = MatchedNpPointClouds(
            model_points=pts,
            scene_points=pts,
            model_fpfh=rng.random((60, 33)),
            scene_fpfh=rng.random((60, 33)),
            model_normals=None,
            scene_normals=None,
        )
        params = VsacConfig(distance_threshold=0.05, max_iterations=50, normal_consistency=True)
        vsac_se2(clouds, params, rng=np.random.default_rng(0))  # must not raise


if __name__ == "__main__":
    unittest.main()
