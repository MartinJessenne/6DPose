import unittest
import numpy as np
import torch
import open3d as o3d

from pipeline import Camera, MaskedImageFrame
from methods.free_space import compute_free_space_violations
from methods.se2_icp import refine_pose_dual_hypothesis_se2
from methods.ransac3dof import Ransac3DoFEstimator, Ransac3DoFParams


class TestFlipDisambiguation(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        self.T_robot_camera = np.eye(4)

    def test_compute_free_space_violations_no_violations(self):
        # 100 model points at z = 2.0 m in front of camera
        model_pts = np.zeros((100, 3))
        model_pts[:, 2] = 2.0

        # Depth map where measured surface is at z = 2.0 m
        depth = torch.full((100, 100), 2.0, dtype=torch.float32)
        rgb = torch.zeros((100, 100, 3), dtype=torch.uint8)

        frame = MaskedImageFrame(
            rgb=rgb, depth=depth, camera=self.camera, xmin=270, ymin=190
        )

        T_pose = np.eye(4)
        n_viol, n_obs, ratio = compute_free_space_violations(
            model_pts, T_pose, self.T_robot_camera, frame, margin=0.03
        )

        self.assertGreater(n_obs, 0)
        self.assertEqual(n_viol, 0)
        self.assertEqual(ratio, 0.0)

    def test_compute_free_space_violations_with_violations(self):
        # Model points placed at z = 1.0 m (claimed solid structure)
        model_pts = np.zeros((100, 3))
        model_pts[:, 2] = 1.0

        # Measured depth is further back at z = 2.0 m (open space where model claims cart sits)
        depth = torch.full((100, 100), 2.0, dtype=torch.float32)
        rgb = torch.zeros((100, 100, 3), dtype=torch.uint8)

        frame = MaskedImageFrame(
            rgb=rgb, depth=depth, camera=self.camera, xmin=270, ymin=190
        )

        T_pose = np.eye(4)
        n_viol, n_obs, ratio = compute_free_space_violations(
            model_pts, T_pose, self.T_robot_camera, frame, margin=0.03
        )

        self.assertGreater(n_obs, 0)
        self.assertEqual(n_viol, n_obs)
        self.assertEqual(ratio, 1.0)

    def test_early_exit_gate_triggers_on_clean_pose(self):
        # Create synthetic scene and model point clouds
        model_pc = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.5, depth=0.5).sample_points_uniformly(200)
        scene_points = np.asarray(model_pc.points)
        scene_normals = np.zeros_like(scene_points)
        scene_normals[:, 2] = 1.0

        depth = torch.full((100, 100), 2.0, dtype=torch.float32)
        rgb = torch.zeros((100, 100, 3), dtype=torch.uint8)
        frame = MaskedImageFrame(rgb=rgb, depth=depth, camera=self.camera, xmin=270, ymin=190)

        # Pose that produces zero free space violations
        T_init = np.eye(4)

        # refine_pose_dual_hypothesis_se2 should execute fast early exit (returns clean result_1)
        res = refine_pose_dual_hypothesis_se2(
            model_points=np.asarray(model_pc.points),
            scene_points=scene_points,
            scene_normals=scene_normals,
            T_init=T_init,
            max_correspondence_distance=0.1,
            max_iterations=10,
            frame=frame,
            extrinsic=np.eye(4),
            free_space_threshold=0.05,
        )

        self.assertIsNotNone(res)
        self.assertTrue(np.allclose(res.transformation, T_init, atol=1e-3))

    def test_ransac3dof_params_has_free_space_threshold(self):
        params = Ransac3DoFParams()
        self.assertEqual(params.free_space_threshold, 0.02)
        self.assertEqual(params.free_space_margin, 0.03)


if __name__ == "__main__":
    unittest.main()
