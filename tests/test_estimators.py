import unittest
import optuna
from methods.ransac import RansacEstimator
from methods.ransac3dof import Ransac3DoFEstimator, Ransac3DoFParams
from methods.ppf import PPFEstimator

class TestEstimatorSweeps(unittest.TestCase):
    def test_ransac_suggest_params(self):
        # Create a mock trial with pre-fixed parameters
        fixed_values = {
            "voxel_size": 0.05,
            "icp_max_correspondence_distance": 0.15,
            "icp_max_iterations": 50
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        params = RansacEstimator.suggest_params(trial)
        
        self.assertEqual(params["voxel_size"], 0.05)
        self.assertEqual(params["icp_max_correspondence_distance"], 0.15)
        self.assertEqual(params["icp_max_iterations"], 50)

    def test_ppf_suggest_params(self):
        fixed_values = {
            "ppf_sampling_step": 0.05,
            "ppf_distance_step": 0.05,
            "ppf_match_threshold": 0.05,
            "ppf_match_tolerance": 0.05,
            "icp_max_correspondence_distance": 0.10,
            "icp_max_iterations": 30
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        params = PPFEstimator.suggest_params(trial)

        self.assertEqual(params["ppf_sampling_step"], 0.05)
        self.assertEqual(params["icp_max_iterations"], 30)

    def test_ransac3dof_suggest_params_construct_params(self):
        # Every suggested key must land as an attribute on the params object:
        # the sweep builds trial estimators from this dict alone, so a key
        # accepted by __init__ but not stored silently breaks every trial.
        fixed_values = {
            "voxel_size": 0.05,
            "icp_max_correspondence_distance": 0.15,
            "icp_max_iterations": 50,
            "edge_length_threshold": 0.85,
            "z_gate_threshold": 0.12,
            "ransac_max_iterations": 20000,
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        suggested = Ransac3DoFEstimator.suggest_params(trial)

        params = Ransac3DoFParams(**suggested)
        for key, value in fixed_values.items():
            self.assertEqual(getattr(params, key), value)

    def test_ransac3dof_requires_extrinsic(self):
        with self.assertRaises(ValueError):
            Ransac3DoFEstimator(extrinsic=None)
