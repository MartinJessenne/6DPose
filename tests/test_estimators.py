import unittest

import open3d as o3d
import optuna

from methods.ppf import PPFEstimator
from methods.ransac import RansacEstimator
from methods.ransac3dof import (
    Ransac3DoFEstimator,
    Ransac3DoFParams,
    crop_front_face,
    derive_z_offset,
)


class TestEstimatorSweeps(unittest.TestCase):
    def test_ransac_suggest_params(self):
        # Create a mock trial with pre-fixed parameters
        fixed_values = {
            "voxel_size": 0.05,
            "icp_max_correspondence_distance": 0.15,
            "icp_max_iterations": 50,
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
            "icp_max_iterations": 30,
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
            "front_crop_depth": 0.35,
            "free_space_margin": 0.03,
            "free_space_separation": 0.02,
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        suggested = Ransac3DoFEstimator.suggest_params(trial)

        params = Ransac3DoFParams(**suggested)
        for key, value in fixed_values.items():
            self.assertEqual(getattr(params, key), value)

    def test_ransac3dof_requires_extrinsic(self):
        with self.assertRaises(ValueError):
            Ransac3DoFEstimator(extrinsic=None)

    def test_ransac3dof_derive_z_offset_from_mesh(self):
        # A unit box translated so its lowest vertex sits at z = -0.07: the
        # derived offset must lift the model so that point touches the floor.
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.translate((0.0, 0.0, -0.07))
        self.assertAlmostEqual(derive_z_offset(mesh), 0.07, places=6)

        # A mesh whose origin is already on the ground derives 0.
        mesh_grounded = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.assertAlmostEqual(derive_z_offset(mesh_grounded), 0.0, places=6)

    def test_ransac3dof_crop_front_face(self):
        import numpy as np

        # Box spanning x in [0, 2]: a 0.5 crop keeps only the +x slab, in the
        # ORIGINAL frame (no recentering).
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=1.0)
        cropped = crop_front_face(mesh, depth=0.5)
        v = np.asarray(cropped.vertices)
        self.assertGreater(len(v), 0)
        self.assertGreaterEqual(v[:, 0].min(), 1.5 - 1e-6)
        self.assertAlmostEqual(v[:, 0].max(), 2.0, places=6)

        # A crop that removes everything must raise rather than silently
        # register against an empty model.
        with self.assertRaises(ValueError):
            crop_front_face(mesh, depth=-1.0)


class TestSweepParamOverrides(unittest.TestCase):
    """A sweep trial's params come from suggest_params alone, so a flag named on
    the command line used to be silently dropped -- the treatment arm ran the
    control a second time while its logs claimed otherwise. These guard the
    mechanism that closes that hole."""

    def setUp(self):
        import numpy as np

        from benchmark import resolve_param_overrides
        from methods.vsac_se2 import VSACSe2Estimator

        self.resolve = resolve_param_overrides
        self.cls = VSACSe2Estimator
        self.extrinsic = np.eye(4)

    def test_parses_literals_by_type(self):
        resolved = self.resolve(
            self.cls,
            self.extrinsic,
            {"free_space_gate": "true", "free_space_gate_max_ratio": "0.05", "z_offset": "none"},
        )
        self.assertIs(resolved["free_space_gate"], True)
        self.assertEqual(resolved["free_space_gate_max_ratio"], 0.05)
        self.assertIsNone(resolved["z_offset"])

    def test_unknown_parameter_is_an_error_not_a_warning(self):
        with self.assertRaises(ValueError):
            self.resolve(self.cls, self.extrinsic, {"free_space_gaet": "true"})

    def test_override_wins_over_a_swept_value(self):
        # Overrides are merged last precisely so a parameter that is also swept
        # cannot drift away from the declared arm.
        suggested = {"voxel_size": 0.05, "free_space_gate": False}
        resolved = self.resolve(self.cls, self.extrinsic, {"free_space_gate": "true"})
        merged = {**suggested, **resolved}
        self.assertIs(merged["free_space_gate"], True)

    def test_empty_overrides_are_a_noop(self):
        self.assertEqual(self.resolve(self.cls, self.extrinsic, None), {})
        self.assertEqual(self.resolve(self.cls, self.extrinsic, {}), {})
