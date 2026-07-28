import unittest

import numpy as np

from methods.constrained_ransac import (
    constrained_ransac_se2,
    match_correspondences_fpfh,
    project_to_se2,
    se2_to_se3,
)
from methods.se2_icp import icp_point_to_plane_se2
from methods.se2_lie_utils import (
    minimal_solver_se2,
    se2_exp,
    se2_hat,
    se2_log,
    se2_matrix,
    se2_vee,
    so2_exp,
)


class TestSe2LieUtils(unittest.TestCase):
    def test_so2_exp_is_rotation(self):
        for theta in [0.0, 0.3, -2.5, np.pi]:
            R = so2_exp(theta)
            np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_minimal_solver_exact_recovery(self):
        theta_true = np.deg2rad(37.0)
        t_true = np.array([2.0, -1.5])
        R_true = so2_exp(theta_true)

        p1, p2 = np.array([1.0, 0.0]), np.array([0.0, 2.0])
        q1, q2 = R_true @ p1 + t_true, R_true @ p2 + t_true

        theta_est, t_est = minimal_solver_se2(p1, p2, q1, q2)
        angle_err = np.arctan2(np.sin(theta_est - theta_true), np.cos(theta_est - theta_true))
        self.assertAlmostEqual(angle_err, 0.0, places=10)
        np.testing.assert_allclose(t_est, t_true, atol=1e-10)

    def test_hat_vee_roundtrip(self):
        xi = np.array([0.4, 1.2, -0.7])
        np.testing.assert_allclose(se2_vee(se2_hat(xi)), xi, atol=1e-12)

    def test_exp_log_roundtrip(self):
        for xi in [
            np.array([0.4, 1.2, -0.7]),
            np.array([0.0, 0.5, 0.3]),  # pure translation
            np.array([1e-10, 0.5, 0.3]),  # near-zero rotation (Taylor branch)
            np.array([3.0, -0.2, 0.9]),  # large rotation, near pi
            np.array([-2.0, 0.0, 0.0]),  # pure rotation
        ]:
            T = se2_exp(xi)
            np.testing.assert_allclose(se2_log(T), xi, atol=1e-8, err_msg=f"xi={xi}")

    def test_composition_half_turn(self):
        # Advance 1m, then half-turn plus 1m: net translation must return to origin.
        T1 = se2_matrix(0.0, np.array([1.0, 0.0]))
        T2 = se2_matrix(np.pi, np.array([1.0, 0.0]))
        T_comp = T2 @ T1
        np.testing.assert_allclose(T_comp[:2, 2], [0.0, 0.0], atol=1e-12)


class TestSe2Se3Embedding(unittest.TestCase):
    def test_se2_to_se3_structure(self):
        T = se2_to_se3(0.7, np.array([1.0, 2.0]), z=0.3)
        np.testing.assert_allclose(T[2, :3], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 0.3], atol=1e-12)
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_project_to_se2_recovers_yaw_under_roll_perturbation(self):
        theta, z_offset = 0.8, 0.3
        T = se2_to_se3(theta, np.array([1.5, -0.4]), z=z_offset)

        # Perturb with a roll rotation and a z drift, as an unconstrained ICP would.
        roll = 0.1
        R_x = np.eye(4)
        R_x[1:3, 1:3] = [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]]
        T_drifted = T @ R_x
        T_drifted[2, 3] += 0.05

        T_proj = project_to_se2(T_drifted, z_offset=z_offset)
        yaw = np.arctan2(T_proj[1, 0], T_proj[0, 0])
        self.assertAlmostEqual(yaw, theta, places=10)
        self.assertAlmostEqual(T_proj[2, 3], z_offset, places=12)
        np.testing.assert_allclose(T_proj[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(T_proj[:2, 3], T_drifted[:2, 3], atol=1e-12)


class TestFpfhMatching(unittest.TestCase):
    def test_mutual_matching_identity(self):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(50, 33))
        matches = match_correspondences_fpfh(feats, feats.copy())
        self.assertEqual(len(matches), 50)
        np.testing.assert_array_equal(matches[:, 0], matches[:, 1])

    def test_mutual_matching_rejects_non_mutual(self):
        # Scene point 0 is nearest to model 0, but model 0's nearest scene
        # point is 1 (closer copy) -> the (0, 0) pair must not be mutual.
        feat_model = np.zeros((2, 33))
        feat_model[1] += 100.0
        feat_scene = np.zeros((3, 33))
        feat_scene[0] += 0.5  # near model 0
        feat_scene[1] += 0.1  # nearer to model 0
        feat_scene[2] += 100.0  # matches model 1
        matches = match_correspondences_fpfh(feat_model, feat_scene)
        self.assertEqual(matches.shape[1], 2)
        match_set = {tuple(m) for m in matches}
        self.assertIn((0, 1), match_set)
        self.assertIn((1, 2), match_set)
        self.assertNotIn((0, 0), match_set)


class TestConstrainedRansac(unittest.TestCase):
    def _make_scene(self, rng, theta_true, t_true, z_offset, n=200, noise=0.002):
        # Non-flat model: z spans a range so the z-consistency gate is exercised.
        model_points = rng.uniform(-1, 1, size=(n, 3))
        model_points[:, 2] = rng.uniform(0.0, 0.5, size=n)

        T_true = se2_to_se3(theta_true, t_true, z_offset)
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
        # Corrupt 30% of the scene descriptors: their mutual matches become
        # outlier correspondences (or drop out of the mutual set).
        n_bad = int(0.3 * len(scene_fpfh))
        bad_idx = rng.choice(len(scene_fpfh), size=n_bad, replace=False)
        scene_fpfh[bad_idx] = rng.normal(size=(n_bad, 33))

        result = constrained_ransac_se2(
            model_points,
            scene_points,
            model_fpfh,
            scene_fpfh,
            distance_threshold=0.01,
            max_iterations=2000,
            z_offset=z_offset,
            min_sample_distance=0.3,
            rng=rng,
        )

        self.assertGreater(result.fitness, 0.8)
        T = result.transformation
        theta_est = np.arctan2(T[1, 0], T[0, 0])
        self.assertAlmostEqual(theta_est, theta_true, delta=np.deg2rad(1.0))
        np.testing.assert_allclose(T[:2, 3], t_true, atol=0.01)
        self.assertAlmostEqual(T[2, 3], z_offset, places=12)
        # The transform must be exactly planar by construction.
        np.testing.assert_allclose(T[2, :3], [0.0, 0.0, 1.0], atol=1e-12)

    def test_z_gate_rejects_inconsistent_frame(self):
        # Shift every scene point 1m up without adjusting z_offset: the
        # z-consistency gate must remove all correspondences and fail cleanly.
        rng = np.random.default_rng(7)
        model_points, scene_points, model_fpfh, scene_fpfh = self._make_scene(
            rng, 0.3, np.array([0.2, 0.1]), z_offset=0.0
        )
        scene_points[:, 2] += 1.0

        result = constrained_ransac_se2(
            model_points,
            scene_points,
            model_fpfh,
            scene_fpfh,
            distance_threshold=0.01,
            max_iterations=500,
            z_offset=0.0,
            rng=rng,
        )
        self.assertEqual(result.fitness, 0.0)
        np.testing.assert_array_equal(result.transformation, np.eye(4))

    def test_z_gate_threshold_decoupled_from_distance_threshold(self):
        # Scene shifted 5cm up relative to the modeled z_offset (e.g. depth
        # bias): a 1cm gate must reject everything, a 20cm gate must let the
        # solver recover the planar pose since inliers still fit within the
        # 10cm registration distance_threshold.
        rng = np.random.default_rng(17)
        theta_true, t_true = 0.5, np.array([0.3, -0.1])
        model_points, scene_points, model_fpfh, scene_fpfh = self._make_scene(
            rng, theta_true, t_true, z_offset=0.0
        )
        scene_points[:, 2] += 0.05

        tight = constrained_ransac_se2(
            model_points,
            scene_points,
            model_fpfh,
            scene_fpfh,
            distance_threshold=0.1,
            max_iterations=500,
            z_offset=0.0,
            z_gate_threshold=0.01,
            rng=rng,
        )
        self.assertEqual(tight.fitness, 0.0)

        wide = constrained_ransac_se2(
            model_points,
            scene_points,
            model_fpfh,
            scene_fpfh,
            distance_threshold=0.1,
            max_iterations=2000,
            z_offset=0.0,
            z_gate_threshold=0.20,
            rng=rng,
        )
        self.assertGreater(wide.fitness, 0.8)
        T = wide.transformation
        theta_est = np.arctan2(T[1, 0], T[0, 0])
        self.assertAlmostEqual(theta_est, theta_true, delta=np.deg2rad(2.0))
        np.testing.assert_allclose(T[:2, 3], t_true, atol=0.02)

    def test_seeded_runs_are_reproducible(self):
        theta_true, t_true, z_offset = 0.9, np.array([-0.4, 0.7]), 0.05
        results = []
        for _ in range(2):
            rng = np.random.default_rng(123)
            model_points, scene_points, model_fpfh, scene_fpfh = self._make_scene(
                rng, theta_true, t_true, z_offset
            )
            result = constrained_ransac_se2(
                model_points,
                scene_points,
                model_fpfh,
                scene_fpfh,
                distance_threshold=0.01,
                max_iterations=1000,
                z_offset=z_offset,
                rng=rng,
            )
            results.append(result)
        np.testing.assert_array_equal(results[0].transformation, results[1].transformation)
        self.assertEqual(results[0].fitness, results[1].fitness)

    def test_too_few_correspondences_returns_failure(self):
        model_points = np.zeros((1, 3))
        scene_points = np.zeros((1, 3))
        feats = np.ones((1, 33))
        result = constrained_ransac_se2(
            model_points,
            scene_points,
            feats,
            feats.copy(),
            distance_threshold=0.01,
            max_iterations=10,
        )
        self.assertEqual(result.fitness, 0.0)


class TestSe2Icp(unittest.TestCase):
    """
    Test geometry: a floor patch plus two perpendicular walls. The wall
    normals (+X, +Y) make x, y and yaw fully observable for point-to-plane
    residuals; the floor only constrains the (already fixed) z direction.
    """

    @staticmethod
    def _make_corner_scene(rng, n_per=150):
        floor = np.column_stack(
            [rng.uniform(-1, 1, n_per), rng.uniform(-1, 1, n_per), np.zeros(n_per)]
        )
        wall_x = np.column_stack(
            [np.full(n_per, 1.0), rng.uniform(-1, 1, n_per), rng.uniform(0, 0.5, n_per)]
        )
        wall_y = np.column_stack(
            [rng.uniform(-1, 1, n_per), np.full(n_per, 1.0), rng.uniform(0, 0.5, n_per)]
        )
        points = np.vstack([floor, wall_x, wall_y])
        normals = np.vstack(
            [
                np.tile([0.0, 0.0, 1.0], (n_per, 1)),
                np.tile([1.0, 0.0, 0.0], (n_per, 1)),
                np.tile([0.0, 1.0, 0.0], (n_per, 1)),
            ]
        )
        return points, normals

    def _make_problem(self, rng, theta_true, t_true, z_offset, noise=0.001):
        model_points, model_normals = self._make_corner_scene(rng)
        T_true = se2_to_se3(theta_true, t_true, z_offset)
        scene_points = model_points @ T_true[:3, :3].T + T_true[:3, 3]
        scene_points += rng.normal(scale=noise, size=scene_points.shape)
        scene_normals = model_normals @ T_true[:3, :3].T
        return model_points, scene_points, scene_normals, T_true

    def test_converges_from_perturbed_init(self):
        rng = np.random.default_rng(3)
        theta_true, t_true, z_offset = 0.4, np.array([0.3, -0.2]), 0.1
        model_points, scene_points, scene_normals, _ = self._make_problem(
            rng, theta_true, t_true, z_offset
        )

        T_init = se2_to_se3(theta_true - 0.1, t_true + [-0.05, 0.1], z_offset)
        result = icp_point_to_plane_se2(
            model_points,
            scene_points,
            scene_normals,
            T_init,
            max_correspondence_distance=0.3,
            max_iterations=50,
        )

        T = result.transformation
        theta_est = np.arctan2(T[1, 0], T[0, 0])
        self.assertAlmostEqual(theta_est, theta_true, delta=5e-3)
        np.testing.assert_allclose(T[:2, 3], t_true, atol=5e-3)
        self.assertGreater(result.fitness, 0.95)

    def test_iterates_stay_exactly_planar(self):
        rng = np.random.default_rng(5)
        z_offset = 0.25
        model_points, scene_points, scene_normals, _ = self._make_problem(
            rng, -0.7, np.array([-0.2, 0.4]), z_offset
        )
        T_init = se2_to_se3(-0.55, np.array([-0.1, 0.3]), z_offset)
        result = icp_point_to_plane_se2(
            model_points,
            scene_points,
            scene_normals,
            T_init,
            max_correspondence_distance=0.3,
            max_iterations=50,
        )

        T = result.transformation
        # The se2_exp composition preserves the z row and z translation exactly.
        np.testing.assert_array_equal(T[2], [0.0, 0.0, 1.0, z_offset])
        np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(T[:2, 2], [0.0, 0.0], atol=1e-12)
        R2 = T[:2, :2]
        np.testing.assert_allclose(R2.T @ R2, np.eye(2), atol=1e-10)

    def test_no_correspondences_returns_init(self):
        rng = np.random.default_rng(11)
        model_points, scene_points, scene_normals, _ = self._make_problem(
            rng, 0.0, np.array([0.0, 0.0]), 0.0
        )
        T_init = se2_to_se3(0.0, np.array([100.0, 100.0]), 0.0)  # far off the scene
        result = icp_point_to_plane_se2(
            model_points,
            scene_points,
            scene_normals,
            T_init,
            max_correspondence_distance=0.05,
            max_iterations=20,
        )
        self.assertEqual(result.fitness, 0.0)
        np.testing.assert_array_equal(result.transformation, T_init)


if __name__ == "__main__":
    unittest.main()
