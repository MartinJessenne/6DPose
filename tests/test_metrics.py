import unittest
import numpy as np
import open3d as o3d
import copy
from pydantic import ValidationError

from benchmark import PoseErrorMetrics, extract_pose_errors, compute_average_recall
from methods.base import refine_pose_dual_hypothesis, prepare_scene_point_cloud


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
        
        R_z = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0.0],
            [0.0,              0.0,             1.0]
        ])
        R_y = np.array([
            [np.cos(pitch_rad),  0.0, np.sin(pitch_rad)],
            [0.0,                1.0, 0.0],
            [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)]
        ])
        R_x = np.array([
            [1.0, 0.0,              0.0],
            [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0.0, np.sin(roll_rad),  np.cos(roll_rad)]
        ])
        
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
            trans_xy=0.02,
            trans_z=0.01,
            yaw=120.0,
            pitch=45.0,
            roll=-90.0,
            geodesic_rot=150.0
        )
        self.assertEqual(metrics.yaw, 120.0)
        
        # Invalid values should raise ValidationError
        with self.assertRaises(ValidationError):
            # yaw > 180
            PoseErrorMetrics(trans_xy=0.0, trans_z=0.0, yaw=185.0, pitch=0.0, roll=0.0, geodesic_rot=0.0)
            
        with self.assertRaises(ValidationError):
            # pitch < -90
            PoseErrorMetrics(trans_xy=0.0, trans_z=0.0, yaw=0.0, pitch=-95.0, roll=0.0, geodesic_rot=0.0)

        with self.assertRaises(ValidationError):
            # geodesic_rot < 0
            PoseErrorMetrics(trans_xy=0.0, trans_z=0.0, yaw=0.0, pitch=0.0, roll=0.0, geodesic_rot=-10.0)

    def test_average_recall_grid_calculation(self):
        # We manually calculate average recall for a few cases
        # Thresholds:
        # translation: [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10] -> 7 thresholds
        # rotation: [0.5, 1.0, 2.0, 5.0, 10.0, 15.0] -> 6 thresholds
        # Total threshold pairs = 42
        
        # Case A: 1 sample that succeeds at all thresholds (trans_xy = 0.001, yaw = 0.1)
        errs = [PoseErrorMetrics(trans_xy=0.001, trans_z=0.0, yaw=0.1, pitch=0.0, roll=0.0, geodesic_rot=0.0)]
        ar = compute_average_recall(errs, 1)
        self.assertEqual(ar, 1.0)
        
        # Case B: 1 sample that fails everything (trans_xy = 1.0, yaw = 90.0)
        errs = [PoseErrorMetrics(trans_xy=1.0, trans_z=0.0, yaw=90.0, pitch=0.0, roll=0.0, geodesic_rot=0.0)]
        ar = compute_average_recall(errs, 1)
        self.assertEqual(ar, 0.0)

        # Case C: Failures / misses check
        # Evaluating 2 samples, 1 succeeds at all, 1 is a miss (not in the list)
        errs = [PoseErrorMetrics(trans_xy=0.001, trans_z=0.0, yaw=0.1, pitch=0.0, roll=0.0, geodesic_rot=0.0)]
        ar = compute_average_recall(errs, 2)
        self.assertEqual(ar, 0.5)

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
        T_flip = np.array([
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0]
        ])
        T_init = T_gt @ T_flip
        
        # Run dual-hypothesis refinement
        T_refined = refine_pose_dual_hypothesis(
            model_pc=model_pc,
            scene_pcd=scene_pcd,
            T_init=T_init,
            icp_max_correspondence_distance=0.1,
            icp_max_iterations=50
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


if __name__ == '__main__':
    unittest.main()
