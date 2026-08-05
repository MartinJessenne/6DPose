import unittest

import open3d as o3d
import optuna

from methods.ppf import PPFEstimator
from methods.ransac import RansacEstimator
from methods.ransac3dof import (
    Ransac3DoFEstimator,
    crop_front_face,
    derive_z_offset,
)


class TestEstimatorSweeps(unittest.TestCase):
    def test_ransac_sample_optuna(self):
        # Create a mock trial with pre-fixed parameters
        fixed_values = {
            "icp_max_correspondence_distance": 0.05,
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        params = RansacEstimator.params_cls.sample_optuna(trial)

        self.assertEqual(params.icp_max_correspondence_distance, 0.05)

    def test_ppf_sample_optuna(self):
        fixed_values = {
            "ppf_sampling_step": 0.05,
            "ppf_distance_step": 0.05,
            "ppf_match_threshold": 0.05,
            "icp_max_correspondence_distance": 0.10,
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        params = PPFEstimator.params_cls.sample_optuna(trial)

        self.assertEqual(params.ppf_sampling_step, 0.05)
        self.assertEqual(params.ppf_distance_step, 0.05)

    def test_ransac3dof_sample_optuna(self):
        fixed_values = {
            "icp_max_correspondence_distance": 0.05,
            "z_gate_threshold": 0.12,
        }
        trial = optuna.trial.FixedTrial(fixed_values)
        params = Ransac3DoFEstimator.params_cls.sample_optuna(trial)

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
