import copy
import unittest

import numpy as np
import open3d as o3d
from pydantic import ValidationError

from methods.base import prepare_scene_point_cloud, refine_pose_dual_hypothesis
from metrics import (
    GROSS_YAW_DEG,
    PoseErrorMetrics,
    compute_average_recall,
    compute_trial_metrics,
    extract_pose_errors,
    finite_or_none,
)


def _err(trans_xy: float, yaw: float) -> PoseErrorMetrics:
    """A PoseErrorMetrics with only the two fields the AR grid and the
    gross-yaw rate actually look at; pitch/roll/z are structurally ~0 for the
    SE(2) estimators under test."""
    return PoseErrorMetrics(
        trans_xy=trans_xy, trans_z=0.0, yaw=yaw, pitch=0.0, roll=0.0, geodesic_rot=abs(yaw)
    )


class TestBenchmarkMetrics(unittest.TestCase):
    def test_decomposed_metrics_mathematical_correctness(self):
        # 1. Test translation decomposition
        t_gt = np.array([1.0, 2.0, 3.0])
        t_est = np.array([1.03, 2.04, 3.05])

        # rotation matrices = identity
        R_gt = np.eye(3)
        R_est = np.eye(3)

        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt

        T_est = np.eye(4)
        T_est[:3, :3] = R_est
        T_est[:3, 3] = t_est

        metrics = extract_pose_errors(T_est, T_gt)
        self.assertAlmostEqual(metrics.trans_xy, 0.05, places=5)
        self.assertAlmostEqual(metrics.trans_z, 0.05, places=5)
        self.assertAlmostEqual(metrics.yaw, 0.0, places=5)
        self.assertAlmostEqual(metrics.pitch, 0.0, places=5)
        self.assertAlmostEqual(metrics.roll, 0.0, places=5)

        # 2. Test rotation decomposition (Yaw=30 deg, Pitch=10 deg, Roll=5 deg)
        yaw_rad = np.radians(30.0)
        pitch_rad = np.radians(10.0)
        roll_rad = np.radians(5.0)

        R_z = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R_y = np.array(
            [
                [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
            ]
        )
        R_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0.0, np.sin(roll_rad), np.cos(roll_rad)],
            ]
        )

        # R_err = R_z @ R_y @ R_x
        R_err = R_z @ R_y @ R_x

        T_gt = np.eye(4)  # Identity rotation for GT
        T_est = np.eye(4)
        T_est[:3, :3] = R_err  # Since R_gt is identity, R_err = R_est

        metrics = extract_pose_errors(T_est, T_gt)
        self.assertAlmostEqual(metrics.yaw, 30.0, places=5)
        self.assertAlmostEqual(metrics.pitch, 10.0, places=5)
        self.assertAlmostEqual(metrics.roll, 5.0, places=5)

    def test_pydantic_validation(self):
        # Valid metrics should validate without errors
        metrics = PoseErrorMetrics(
            trans_xy=0.02, trans_z=0.01, yaw=120.0, pitch=45.0, roll=-90.0, geodesic_rot=150.0
        )
        self.assertEqual(metrics.yaw, 120.0)

        # Invalid values should raise ValidationError
        with self.assertRaises(ValidationError):
            # yaw > 180
            PoseErrorMetrics(
                trans_xy=0.0, trans_z=0.0, yaw=185.0, pitch=0.0, roll=0.0, geodesic_rot=0.0
            )

        with self.assertRaises(ValidationError):
            # pitch < -90
            PoseErrorMetrics(
                trans_xy=0.0, trans_z=0.0, yaw=0.0, pitch=-95.0, roll=0.0, geodesic_rot=0.0
            )

        with self.assertRaises(ValidationError):
            # geodesic_rot < 0
            PoseErrorMetrics(
                trans_xy=0.0, trans_z=0.0, yaw=0.0, pitch=0.0, roll=0.0, geodesic_rot=-10.0
            )

    def test_average_recall_grid_calculation(self):
        # We manually calculate average recall for a few cases
        # Thresholds:
        # translation: [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10] -> 7 thresholds
        # rotation: [0.5, 1.0, 2.0, 5.0, 10.0, 15.0] -> 6 thresholds
        # Total threshold pairs = 42

        # Case A: 1 sample that succeeds at all thresholds (trans_xy = 0.001, yaw = 0.1)
        errs = [
            PoseErrorMetrics(
                trans_xy=0.001, trans_z=0.0, yaw=0.1, pitch=0.0, roll=0.0, geodesic_rot=0.0
            )
        ]
        ar = compute_average_recall(errs, 1)
        self.assertEqual(ar, 1.0)

        # Case B: 1 sample that fails everything (trans_xy = 1.0, yaw = 90.0)
        errs = [
            PoseErrorMetrics(
                trans_xy=1.0, trans_z=0.0, yaw=90.0, pitch=0.0, roll=0.0, geodesic_rot=0.0
            )
        ]
        ar = compute_average_recall(errs, 1)
        self.assertEqual(ar, 0.0)

        # Case C: Failures / misses check
        # Evaluating 2 samples, 1 succeeds at all, 1 is a miss (not in the list)
        errs = [
            PoseErrorMetrics(
                trans_xy=0.001, trans_z=0.0, yaw=0.1, pitch=0.0, roll=0.0, geodesic_rot=0.0
            )
        ]
        ar = compute_average_recall(errs, 2)
        self.assertEqual(ar, 0.5)

    def test_rates_partition_attempted(self):
        """good + gross_yaw + abstention must cover every attempted frame.

        This is the invariant whose absence let VSAC trade flips for
        abstentions unnoticed: flip_rate used to divide by the SUCCESS count
        while average_recall divided by the evaluated count.
        """
        errs = [
            _err(trans_xy=0.01, yaw=1.0),  # good
            _err(trans_xy=0.02, yaw=3.0),  # good
            _err(trans_xy=0.50, yaw=180.0),  # gross yaw (a true flip)
            _err(trans_xy=0.30, yaw=40.0),  # gross yaw (merely imprecise)
        ]
        m = compute_trial_metrics(errs, [0.1] * 4, detection_failures=2, pose_failures=4)

        self.assertEqual(m.n_attempted, 8)  # 4 matched + 4 abstained, detections excluded
        self.assertEqual(m.n_eval, 10)
        self.assertAlmostEqual(m.good_rate + m.gross_yaw_rate + m.abstention_rate, 1.0)
        self.assertAlmostEqual(m.good_rate, 0.25)
        self.assertAlmostEqual(m.gross_yaw_rate, 0.25)
        self.assertAlmostEqual(m.abstention_rate, 0.5)
        # Detection failures stay out of the estimator denominator entirely.
        self.assertAlmostEqual(m.detection_failure_rate, 0.2)

    def test_pose_ar_never_exceeds_good_rate(self):
        """Every AR grid cell requires |yaw| < 15, so AR is bounded by good_rate.

        A violation means the two metrics' denominators have diverged -- exactly
        the class of bug this suite exists to catch.
        """
        cases = [
            [_err(0.001, 0.1)],  # perfect
            [_err(0.001, 0.1), _err(0.5, 170.0)],  # mixed
            [_err(0.02, 14.9)],  # good but only just
            [_err(0.02, 15.1)],  # gross by a hair
        ]
        for errs in cases:
            for pose_failures in (0, 3):
                with self.subTest(errs=len(errs), pose_failures=pose_failures):
                    m = compute_trial_metrics(
                        errs, [0.1] * len(errs), detection_failures=0, pose_failures=pose_failures
                    )
                    self.assertLessEqual(m.pose_ar, m.good_rate + 1e-9)

    def test_total_abstention_scores_worst_not_flip_free(self):
        """A trial that returns no pose at all must be the WORST trial.

        Regression guard for W&B run fgugxxrn step 11, which logged
        pose_failures=207, average_recall=0 and flip_rate=0.0 -- i.e. a
        completely failed trial presented as having a perfect flip rate.
        """
        m = compute_trial_metrics([], [], detection_failures=0, pose_failures=10)

        self.assertEqual(m.abstention_rate, 1.0)
        self.assertEqual(m.gross_yaw_rate, 0.0)
        self.assertEqual(m.good_rate, 0.0)
        # The objective, which is what the sweep actually optimizes, is floored.
        self.assertEqual(m.pose_ar, 0.0)
        # No successful estimation => no latency to report (logged as None, not
        # float('inf'), which W&B stores as the string "Infinity").
        self.assertFalse(np.isfinite(m.p95_latency_s))
        self.assertIsNone(finite_or_none(m.p95_latency_s))
        self.assertIsNone(m.trans_xy_p50)

    def test_gross_yaw_threshold_matches_ar_grid(self):
        """GROSS_YAW_DEG must equal the AR grid's largest rotation threshold.

        If these drift apart, gross_yaw_rate and pose_ar stop answering the same
        question and the bound in test_pose_ar_never_exceeds_good_rate breaks.
        """
        self.assertEqual(GROSS_YAW_DEG, 15.0)

    def test_dual_hypothesis_flip_correction(self):
        # Load colruyt model ply
        mesh = o3d.io.read_triangle_mesh("meshes/colruyt.ply")
        mesh.compute_vertex_normals()
        model_pc = mesh.sample_points_uniformly(number_of_points=1000)

        # Prepare ground truth and scene
        T_gt = np.eye(4)

        # Create a mock scene with normals
        scene_pcd = copy.deepcopy(model_pc)

        # Perturb the initial alignment guess T_init by rotating 180 degrees around Z
        T_flip = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T_init = T_gt @ T_flip

        # Run dual-hypothesis refinement
        T_refined = refine_pose_dual_hypothesis(
            model_pc=model_pc,
            scene_pcd=scene_pcd,
            T_init=T_init,
            icp_max_correspondence_distance=0.1,
            icp_max_iterations=50,
        )

        # Assert that the refinement correctly flipped it back to identity T_gt
        # (meaning the flipped hypothesis was preferred because it matched the scene perfectly)
        np.testing.assert_array_almost_equal(T_refined, T_gt, decimal=4)

    def test_estimator_state_mutability(self):
        # Create a simple point cloud
        pts = np.random.rand(10, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Store original coordinates
        orig_pts = np.asarray(pcd.points).copy()

        # Call prepare_scene_point_cloud
        T_extrinsic = np.eye(4)
        _ = prepare_scene_point_cloud(pcd, T_extrinsic)

        # Check that the input pcd was NOT mutated
        current_pts = np.asarray(pcd.points)
        np.testing.assert_array_equal(current_pts, orig_pts)

    def test_sweep_seeding_and_reproducibility(self):
        # Verify that generating indices from a fixed seed is perfectly reproducible
        seed = 42
        total_samples = 1000
        size = 20

        rng1 = np.random.default_rng(seed)
        indices1 = rng1.choice(total_samples, size, replace=False).tolist()

        rng2 = np.random.default_rng(seed)
        indices2 = rng2.choice(total_samples, size, replace=False).tolist()

        self.assertEqual(indices1, indices2)

    def test_icp_tie_breaker_logic(self):
        from unittest.mock import MagicMock, patch

        # Create dummy point clouds
        model_pc = o3d.geometry.PointCloud()
        scene_pcd = o3d.geometry.PointCloud()
        T_init = np.eye(4)

        # We mock o3d.pipelines.registration.registration_icp
        with patch("open3d.pipelines.registration.registration_icp") as mock_icp:
            # We want to simulate equal fitness but different inlier_rmse
            # Result 1: fitness = 0.8, rmse = 0.02, transformation = T_init
            # Result 2: fitness = 0.8, rmse = 0.01, transformation = T_flipped
            res1 = MagicMock()
            res1.fitness = 0.8
            res1.inlier_rmse = 0.02
            res1.transformation = np.eye(4)

            res2 = MagicMock()
            res2.fitness = 0.8
            res2.inlier_rmse = 0.01
            res2.transformation = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.1],
                    [0.0, -1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            # Side effect: first call returns res1, second returns res2
            mock_icp.side_effect = [res1, res2]

            T_refined = refine_pose_dual_hypothesis(
                model_pc=model_pc,
                scene_pcd=scene_pcd,
                T_init=T_init,
                icp_max_correspondence_distance=0.1,
                icp_max_iterations=10,
            )

            # The tie-breaker should pick res2 because its inlier_rmse (0.01) is lower than res1 (0.02)
            np.testing.assert_array_equal(T_refined, res2.transformation)


if __name__ == "__main__":
    unittest.main()
