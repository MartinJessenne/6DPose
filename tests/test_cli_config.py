"""Regression guard for cli_config.py's dataclass defaults.

These values are transcribed from config/camera/default.yaml,
config/dataset/default.yaml, and the yolo: block of config/config.yaml (the
Hydra config tree being replaced by tyro). This test pins them so a copy/paste
mistake during the migration doesn't silently regress a tuned value -- and it
stays meaningful even after config/ is deleted, since the expected values
below are hardcoded, not re-read from the YAML.
"""

import unittest

import tyro

from cli_config import (
    CameraConfig,
    DatasetConfig,
    ModelPreset,
    PPFPreset,
    RansacPreset,
    Ransac3DoFPreset,
    YoloConfig,
)
from methods.ppf import PPFEstimator
from methods.ransac import RansacEstimator
from methods.ransac3dof import Ransac3DoFEstimator


class TestYoloConfig(unittest.TestCase):
    def test_defaults_match_config_yaml(self):
        cfg = YoloConfig()
        self.assertEqual(cfg.repo, "UItraviolet/yolo_multicart")
        self.assertEqual(cfg.file, "runs/segment/train-2/weights/best.pt")
        self.assertEqual(cfg.local_path, "best.pt")


class TestCameraConfig(unittest.TestCase):
    def test_defaults_match_camera_default_yaml(self):
        cfg = CameraConfig()
        self.assertEqual(cfg.fx, 639.99768)
        self.assertEqual(cfg.fy, 639.99768)
        self.assertEqual(cfg.cx, 640.0)
        self.assertEqual(cfg.cy, 400.0)
        self.assertEqual(
            cfg.extrinsic,
            (
                (0.5, 0.0, 0.8660254037844386, 0.439),
                (0.0, 1.0, -0.0, 0.0),
                (-0.8660254037844386, 0.0, 0.5, 0.304),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )


class TestDatasetConfig(unittest.TestCase):
    def test_defaults_match_dataset_default_yaml(self):
        cfg = DatasetConfig()
        self.assertEqual(cfg.path, "parquet")
        self.assertEqual(cfg.train_glob, "dataset/data/train-*-of-00127.parquet")
        self.assertEqual(cfg.val_glob, "dataset/data/validation-*-of-00016.parquet")
        self.assertEqual(cfg.test_glob, "dataset/data/test-*-of-00016.parquet")


def _select(*tokens: str):
    """Parse a ModelPreset CLI selection, e.g. _select('ransac3dof', 'profile:acc-opt')."""
    return tyro.cli(ModelPreset, args=list(tokens))


class TestModelPresetEstimatorClasses(unittest.TestCase):
    """One value per YAML file's `_target_`, confirming the algorithm-level dispatch."""

    def test_ppf_targets_ppf_estimator(self):
        self.assertIs(PPFPreset.ESTIMATOR_CLS, PPFEstimator)

    def test_ransac_targets_ransac_estimator(self):
        self.assertIs(RansacPreset.ESTIMATOR_CLS, RansacEstimator)

    def test_ransac3dof_targets_ransac3dof_estimator(self):
        self.assertIs(Ransac3DoFPreset.ESTIMATOR_CLS, Ransac3DoFEstimator)


class TestPPFProfiles(unittest.TestCase):
    """One block per config/model/ppf*.yaml file."""

    def test_default_matches_ppf_yaml(self):
        r = _select("ppf", "profile:default")
        self.assertEqual(r.profile.depth_trunc, 3.0)
        p = r.profile.params
        self.assertEqual(p.ppf_sampling_step, 0.1)
        self.assertEqual(p.ppf_distance_step, 0.02)
        self.assertEqual(p.ppf_match_threshold, 0.06)
        self.assertEqual(p.ppf_match_tolerance, 0.03)
        self.assertEqual(p.icp_max_correspondence_distance, 0.17528702727791115)
        self.assertEqual(p.icp_max_iterations, 10)

    def test_rt_opt_matches_ppf_rt_opt_yaml(self):
        r = _select("ppf", "profile:rt-opt")
        self.assertEqual(r.profile.depth_trunc, 3.8)
        p = r.profile.params
        self.assertEqual(p.ppf_sampling_step, 0.03)
        self.assertEqual(p.ppf_distance_step, 0.06)
        self.assertEqual(p.ppf_match_threshold, 0.04)
        self.assertEqual(p.ppf_match_tolerance, 0.08)
        self.assertEqual(p.icp_max_correspondence_distance, 0.1724415146443418)
        self.assertEqual(p.icp_max_iterations, 60)


class TestRansacProfiles(unittest.TestCase):
    """One block per config/model/ransac*.yaml file (excluding ransac3dof*)."""

    def test_default_matches_ransac_yaml(self):
        r = _select("ransac", "profile:default")
        self.assertEqual(r.profile.depth_trunc, 3.0)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.06)
        self.assertEqual(p.ransac_max_iterations, 100000)
        self.assertEqual(p.icp_max_correspondence_distance, 0.15)
        self.assertEqual(p.icp_max_iterations, 100)

    def test_pareto1_matches_ransac_pareto1_yaml(self):
        r = _select("ransac", "profile:pareto1")
        self.assertEqual(r.profile.depth_trunc, 6.2)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.08)
        self.assertEqual(p.icp_max_correspondence_distance, 0.1200659534)
        self.assertEqual(p.icp_max_iterations, 90)
        # Not overridden by this preset -- falls back to RansacParams' own default.
        self.assertEqual(p.ransac_max_iterations, 100000)

    def test_realtime_matches_ransac_realtime_yaml(self):
        r = _select("ransac", "profile:realtime")
        self.assertEqual(r.profile.depth_trunc, 3.6)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.09)
        self.assertEqual(p.icp_max_correspondence_distance, 0.1292972835)
        self.assertEqual(p.icp_max_iterations, 20)
        self.assertEqual(p.ransac_max_iterations, 100000)

    def test_rt_opt_matches_ransac_rt_opt_yaml(self):
        r = _select("ransac", "profile:rt-opt")
        self.assertEqual(r.profile.depth_trunc, 2.1)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.09)
        self.assertEqual(p.icp_max_correspondence_distance, 0.09794024637923043)
        self.assertEqual(p.icp_max_iterations, 20)
        self.assertEqual(p.ransac_max_iterations, 100000)


class TestRansac3DoFProfiles(unittest.TestCase):
    """One block per config/model/ransac3dof*.yaml file."""

    def test_default_matches_ransac3dof_yaml(self):
        r = _select("ransac3dof", "profile:default")
        self.assertEqual(r.profile.depth_trunc, 3.0)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.06)
        self.assertEqual(p.ransac_max_iterations, 100000)
        self.assertEqual(p.icp_max_correspondence_distance, 0.15)
        self.assertEqual(p.icp_max_iterations, 100)
        self.assertEqual(p.z_offset, 0.01)
        self.assertEqual(p.z_gate_threshold, 0.09)
        self.assertEqual(p.edge_length_threshold, 0.9)
        self.assertEqual(p.front_crop_depth, 0.35)
        self.assertEqual(p.ransac_confidence, 0.999)
        self.assertEqual(p.seed, 0)

    def test_acc_opt_matches_ransac3dof_acc_opt_yaml(self):
        r = _select("ransac3dof", "profile:acc-opt")
        self.assertEqual(r.profile.depth_trunc, 3.8)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.02)
        self.assertEqual(p.ransac_max_iterations, 14029)
        self.assertEqual(p.icp_max_correspondence_distance, 0.07280260785158235)
        self.assertEqual(p.icp_max_iterations, 80)
        self.assertEqual(p.z_offset, 0.01)
        self.assertEqual(p.z_gate_threshold, 0.24838996628120097)
        self.assertEqual(p.edge_length_threshold, 0.8315299309347685)
        self.assertEqual(p.front_crop_depth, 0.32839186866723463)
        self.assertEqual(p.ransac_confidence, 0.999)
        self.assertEqual(p.seed, 0)

    def test_rt_opt_matches_ransac3dof_rt_opt_yaml(self):
        r = _select("ransac3dof", "profile:rt-opt")
        self.assertEqual(r.profile.depth_trunc, 2.2)
        p = r.profile.params
        self.assertEqual(p.voxel_size, 0.04)
        self.assertEqual(p.ransac_max_iterations, 2772)
        self.assertEqual(p.icp_max_correspondence_distance, 0.06404478356487074)
        self.assertEqual(p.icp_max_iterations, 50)
        self.assertEqual(p.z_offset, 0.01)
        self.assertEqual(p.z_gate_threshold, 0.12938125580720045)
        self.assertEqual(p.edge_length_threshold, 0.831518113921687)
        self.assertEqual(p.front_crop_depth, 0.47964349266741707)
        self.assertEqual(p.ransac_confidence, 0.999)
        self.assertEqual(p.seed, 0)


if __name__ == "__main__":
    unittest.main()
