"""
Test suit to verify that the parameter overrides are correctly applied to the estimators.
In all the different scenarios : sweep, benchmark, CLI, presets...
"""

import unittest

import numpy as np

from .base.py import resolve_param_overrides


class RecordingTrial:
    """A mock Optuna trial that records the parameters suggested to it."""

    def __init__(self):
        self.asked: list[str] = []

    def suggest_float(self, name, low, high, step=None, log=False):
        self.asked.append(name)
        return low

    def suggest_int(self, name, low, high, step=1, log=False):
        self.asked.append(name)
        return low


class ResolveParamOverridesTest(unittest.TestCase):
    def setUp(self):
        self.extrinsic = np.array(CameraConfig().extrinsic, dtype=np.float64)

    def resolve(self, overrides):
        return resolve_param_overrides(VSACSe2Estimator, self.extrinsic, overrides)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve({"icp_visibility_culll": "true"})  # deliberate typo to test unknown name
        self.assertIn("icp_visibility_culll", str(ctx.exception))

    def test_bools_become_real_bools(self):
        out = self.resolve({"icp_visibility_cull": "true", "hoppe_normal_estimation": "false"})
        self.assertIsInstance(out["icp_visibility_culling"], bool)
        self.assertTrue(out["icp_visibility_culling"])
