import unittest
import optuna
from methods.ransac import RansacEstimator
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
