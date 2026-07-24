import unittest

import numpy as np
import open3d as o3d

from cli_config import CameraConfig
from methods.base import BasePoseEstimator
from methods.ppf import PPFEstimator, PPFParams
from methods.ransac import RansacEstimator, RansacParams
from pipeline import compute_ground_truth_pose


class TestEstimatorPreparation(unittest.TestCase):
    def setUp(self):
        # Clear global cache before each test
        BasePoseEstimator._PREPARATION_CACHE.clear()

        # Load extrinsic from cli_config's CameraConfig as the source of truth
        # (replaces config/camera/default.yaml, removed in the tyro migration).
        self.extrinsic = np.array(CameraConfig().extrinsic, dtype=np.float64)

        # Create a simple box mesh to act as a CAD model
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(50, 3))

    def test_ransac_cache_hit_and_parameter_isolation(self):
        # 1. Initialize RansacEstimator with first parameters
        params1 = RansacParams(voxel_size=0.06, icp_max_iterations=100)
        est1 = RansacEstimator(params=params1)

        # Prepare mesh
        est1.prepare(self.mesh, "test_cart")

        # Cache key should exist in class-level dictionary
        cache_key1 = (RansacEstimator.__name__, "test_cart", est1._get_prep_params_key())
        self.assertIn(cache_key1, BasePoseEstimator._PREPARATION_CACHE)

        # Get references to cached objects
        cached_data1 = BasePoseEstimator._PREPARATION_CACHE[cache_key1]
        model_pc1 = cached_data1["model_pc"]
        model_down1 = cached_data1["model_down"]

        # Call prepare again with identical prep parameters but different estimator
        est2 = RansacEstimator(params=RansacParams(voxel_size=0.06, icp_max_iterations=50))
        est2.prepare(self.mesh, "test_cart")

        # Verify it was a cache hit (pointing to the exact same underlying objects)
        cache_key2 = (RansacEstimator.__name__, "test_cart", est2._get_prep_params_key())
        self.assertEqual(cache_key1, cache_key2)
        cached_data2 = BasePoseEstimator._PREPARATION_CACHE[cache_key2]
        self.assertIs(cached_data1, cached_data2)
        self.assertIs(model_pc1, cached_data2["model_pc"])
        self.assertIs(model_down1, cached_data2["model_down"])

        # 2. Parameter isolation: Change prep-relevant parameter (voxel_size)
        est3 = RansacEstimator(params=RansacParams(voxel_size=0.04))
        est3.prepare(self.mesh, "test_cart")

        cache_key3 = (RansacEstimator.__name__, "test_cart", est3._get_prep_params_key())
        self.assertNotEqual(cache_key1, cache_key3)
        self.assertIn(cache_key3, BasePoseEstimator._PREPARATION_CACHE)

    def test_ppf_cache_hit_and_parameter_isolation(self):
        # 1. Initialize PPFEstimator
        params1 = PPFParams(ppf_sampling_step=0.05, ppf_distance_step=0.02, icp_max_iterations=50)
        est1 = PPFEstimator(params=params1)

        est1.prepare(self.mesh, "test_cart")

        cache_key1 = (PPFEstimator.__name__, "test_cart", est1._get_prep_params_key())
        self.assertIn(cache_key1, BasePoseEstimator._PREPARATION_CACHE)

        cached_data1 = BasePoseEstimator._PREPARATION_CACHE[cache_key1]
        detector1 = cached_data1["detector"]

        # Verify cache hit with identical prep params but different online params
        est2 = PPFEstimator(
            params=PPFParams(ppf_sampling_step=0.05, ppf_distance_step=0.02, icp_max_iterations=100)
        )
        est2.prepare(self.mesh, "test_cart")

        cached_data2 = BasePoseEstimator._PREPARATION_CACHE[cache_key1]
        self.assertIs(detector1, cached_data2["detector"])

        # 2. Parameter isolation: Change prep-relevant parameters
        est3 = PPFEstimator(params=PPFParams(ppf_sampling_step=0.06))
        est3.prepare(self.mesh, "test_cart")
        cache_key3 = (PPFEstimator.__name__, "test_cart", est3._get_prep_params_key())
        self.assertNotEqual(cache_key1, cache_key3)

    def test_copy_on_retrieval_and_mutation_safety(self):
        # Ransac copy safety test
        params = RansacParams(voxel_size=0.05)
        est = RansacEstimator(params=params, extrinsic=self.extrinsic)
        est.prepare(self.mesh, "test_cart")

        # Retrieve representation and perform an in-place transformation (which mimics downstream mutate)
        T = np.eye(4)
        T[0, 3] = 100.0  # translate by 100 meters

        # Call estimate_pose twice to check if retrieved geometries are modified in the cache
        # Inside estimate_pose, model_pc/model_down are retrieved and passed to refine_pose/ICP
        # We will directly fetch from estimate_pose and assert they remain unmodified.
        cache_key = (RansacEstimator.__name__, "test_cart", est._get_prep_params_key())
        cached_pc_initial = np.asarray(
            BasePoseEstimator._PREPARATION_CACHE[cache_key]["model_pc"].points
        ).copy()

        # Call estimate_pose which retrieves them and could expose them to mutation
        est.estimate_pose(self.pcd, self.mesh, cart_type="test_cart")

        # Assert cached point cloud remains completely unchanged
        cached_pc_after = np.asarray(
            BasePoseEstimator._PREPARATION_CACHE[cache_key]["model_pc"].points
        )
        np.testing.assert_array_equal(cached_pc_initial, cached_pc_after)

    def test_local_fallback_no_side_effects(self):
        # Running estimate_pose with cart_type=None (lazy local fallback)
        params = RansacParams(voxel_size=0.06)
        est = RansacEstimator(params=params, extrinsic=self.extrinsic)

        # Verify cache is empty initially
        self.assertEqual(len(BasePoseEstimator._PREPARATION_CACHE), 0)

        # Run estimate_pose with cart_type=None
        # It should complete without inserting anything into the global cache
        est.estimate_pose(self.pcd, self.mesh, cart_type=None)

        self.assertEqual(len(BasePoseEstimator._PREPARATION_CACHE), 0)

    def test_cache_memory_eviction(self):
        params = RansacParams(voxel_size=0.06)
        est1 = RansacEstimator(params=params)

        # Prepare for 'test_cart' with voxel_size=0.06
        est1.prepare(self.mesh, "test_cart")
        self.assertEqual(len(BasePoseEstimator._PREPARATION_CACHE), 1)

        # Prepare for 'test_cart' with voxel_size=0.04 (should evict the 0.06 entry for 'test_cart')
        est2 = RansacEstimator(params=RansacParams(voxel_size=0.04))
        est2.prepare(self.mesh, "test_cart")

        self.assertEqual(len(BasePoseEstimator._PREPARATION_CACHE), 1)
        stale_key = (RansacEstimator.__name__, "test_cart", (0.06,))
        active_key = (RansacEstimator.__name__, "test_cart", (0.04,))
        self.assertNotIn(stale_key, BasePoseEstimator._PREPARATION_CACHE)
        self.assertIn(active_key, BasePoseEstimator._PREPARATION_CACHE)

    def test_transform_chain_regression(self):
        # Expected robot frame ground truth translation is [3.2056, 0.0, 0.0100] when using orthonormal matrix
        # Load extrinsic from cli_config's CameraConfig, the single source of truth.
        T_robot_camera = np.array(CameraConfig().extrinsic, dtype=np.float64)

        T_world_camera = np.eye(4)

        T_world_cart = np.array(
            [
                [1.0, 0.0, 0.0, 1.6379],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -2.2489],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Run calculation
        T_gt = compute_ground_truth_pose(T_world_camera, T_world_cart, T_robot_camera)
        t_gt = T_gt[:3, 3]

        # Verify Z is close to 0.0100 (rests on floor)
        self.assertAlmostEqual(t_gt[2], 0.0100, places=4)
        # Verify Y is close to 0.0000 (centered on robot axis)
        self.assertAlmostEqual(t_gt[1], 0.0, places=4)
        # Verify X is close to 3.2056 (orthonormal translation)
        self.assertAlmostEqual(t_gt[0], 3.2056, places=4)

    def test_empty_point_cloud_returns_none(self):
        """D1: Test that empty scene point clouds abort registration early and return None."""
        empty_pcd = o3d.geometry.PointCloud()

        # Test RANSAC
        ransac_est = RansacEstimator(params=RansacParams(voxel_size=0.05), extrinsic=self.extrinsic)
        res_ransac = ransac_est.estimate_pose(empty_pcd, self.mesh, cart_type="test_cart")
        self.assertIsNone(res_ransac)

        # Test PPF
        ppf_est = PPFEstimator(
            params=PPFParams(ppf_sampling_step=0.05, ppf_distance_step=0.05),
            extrinsic=self.extrinsic,
        )
        res_ppf = ppf_est.estimate_pose(empty_pcd, self.mesh, cart_type="test_cart")
        self.assertIsNone(res_ppf)

    def test_estimate_pose_cache_hit_on_the_fly(self):
        """D2: Verify that estimate_pose correctly triggers on-the-fly preparation and hits cache on subsequent calls."""
        ppf_est = PPFEstimator(
            params=PPFParams(ppf_sampling_step=0.05, ppf_distance_step=0.05),
            extrinsic=self.extrinsic,
        )
        cache_key = (PPFEstimator.__name__, "test_cart", ppf_est._get_prep_params_key())

        # Cache must be empty for this key initially
        self.assertNotIn(cache_key, BasePoseEstimator._PREPARATION_CACHE)

        # Running estimate_pose should trigger preparation on-the-fly (cache miss fallback)
        ppf_est.estimate_pose(self.pcd, self.mesh, cart_type="test_cart")
        self.assertIn(cache_key, BasePoseEstimator._PREPARATION_CACHE)

        # Retrieve the cached detector object
        cached_detector = BasePoseEstimator._PREPARATION_CACHE[cache_key]["detector"]

        # Run estimate_pose again, it should use the exact same cached detector
        ppf_est.estimate_pose(self.pcd, self.mesh, cart_type="test_cart")
        self.assertIs(BasePoseEstimator._PREPARATION_CACHE[cache_key]["detector"], cached_detector)


class TestMaskedImageFrameIntrinsics(unittest.TestCase):
    """D3: Tests for MaskedImageFrame.get_o3d_intrinsics principal-point shift."""

    def test_principal_point_shift(self):
        """Verify that cropping offsets are correctly subtracted from the principal point."""
        import torch

        from pipeline import Camera, MaskedImageFrame

        # Original camera with known principal point
        camera = Camera(fx=640.0, fy=640.0, cx=640.0, cy=400.0)

        # Simulate a crop at (xmin=100, ymin=50) producing a 200x150 patch
        crop_w, crop_h = 200, 150
        rgb = torch.zeros(crop_h, crop_w, 3, dtype=torch.uint8)
        depth = torch.zeros(crop_h, crop_w, dtype=torch.float32)

        frame = MaskedImageFrame(rgb=rgb, depth=depth, camera=camera, xmin=100, ymin=50)

        intrinsics = frame.get_o3d_intrinsics()

        # Principal point should be shifted by the crop offset
        intrinsic_matrix = intrinsics.intrinsic_matrix
        result_cx = intrinsic_matrix[0, 2]
        result_cy = intrinsic_matrix[1, 2]

        self.assertAlmostEqual(result_cx, 640.0 - 100, places=5)  # cx - xmin
        self.assertAlmostEqual(result_cy, 400.0 - 50, places=5)  # cy - ymin

        # Focal lengths should be unchanged
        self.assertAlmostEqual(intrinsic_matrix[0, 0], 640.0, places=5)
        self.assertAlmostEqual(intrinsic_matrix[1, 1], 640.0, places=5)

        # Width and height should match the crop
        self.assertEqual(intrinsics.width, crop_w)
        self.assertEqual(intrinsics.height, crop_h)

    def test_zero_offset_preserves_original(self):
        """When crop starts at origin, intrinsics should match the original camera."""
        import torch

        from pipeline import Camera, MaskedImageFrame

        camera = Camera(fx=639.99768, fy=639.99768, cx=640.0, cy=400.0)

        rgb = torch.zeros(800, 1280, 3, dtype=torch.uint8)
        depth = torch.zeros(800, 1280, dtype=torch.float32)

        frame = MaskedImageFrame(rgb=rgb, depth=depth, camera=camera, xmin=0, ymin=0)

        intrinsics = frame.get_o3d_intrinsics()
        intrinsic_matrix = intrinsics.intrinsic_matrix

        self.assertAlmostEqual(intrinsic_matrix[0, 2], 640.0, places=5)
        self.assertAlmostEqual(intrinsic_matrix[1, 2], 400.0, places=5)


if __name__ == "__main__":
    unittest.main()
