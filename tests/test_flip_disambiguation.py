import unittest

import numpy as np
import open3d as o3d
import torch

from methods.free_space import VisibilityContext, count_violations
from methods.ransac3dof import Ransac3DoFParams
from methods.se2_icp import refine_pose_dual_hypothesis_se2
from methods.vsac_se2 import FreeSpaceGate
from pipeline import Camera, MaskedImageFrame


def make_frame(depth: torch.Tensor, camera: Camera) -> MaskedImageFrame:
    """
    Builds a frame whose masked crop is deliberately EMPTY while the full-frame
    depth carries the measurements.

    This mirrors the real pipeline, where `depth` is blacked out outside the YOLO
    silhouette and `depth_full` is not, and it makes these tests fail if the
    visibility check ever goes back to reading the masked crop.
    """
    h, w = depth.shape
    return MaskedImageFrame(
        rgb=torch.zeros((h, w, 3), dtype=torch.uint8),
        depth=torch.zeros_like(depth),
        camera=camera,
        xmin=0,
        ymin=0,
        depth_full=depth,
    )


class TestFreeSpaceEvaluator(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        self.extrinsic = np.eye(4)
        # Measured surface at 2 m everywhere.
        self.depth = torch.full((480, 640), 2.0, dtype=torch.float32)
        self.ctx = VisibilityContext.from_frame(make_frame(self.depth, self.camera), self.extrinsic)

    def test_on_surface_pose_has_no_violations(self):
        points = np.zeros((100, 3))
        points[:, 2] = 2.0

        n_viol, n_obs, ratio = count_violations(self.ctx, points, np.eye(4), margin=0.03)

        self.assertGreater(n_obs, 0)
        self.assertEqual(n_viol, 0)
        self.assertEqual(ratio, 0.0)

    def test_geometry_in_front_of_measured_surface_violates(self):
        # Solid structure claimed at 1 m along rays measured as free out to 2 m.
        points = np.zeros((100, 3))
        points[:, 2] = 1.0

        n_viol, n_obs, ratio = count_violations(self.ctx, points, np.eye(4), margin=0.03)

        self.assertGreater(n_obs, 0)
        self.assertEqual(n_viol, n_obs)
        self.assertEqual(ratio, 1.0)

    def test_geometry_behind_measured_surface_is_not_a_violation(self):
        # Occlusion safety: a point hidden BEHIND the measured surface is
        # unobserved, not contradicted. Counting it would reject every pose whose
        # far side is occluded, which is every pose.
        points = np.zeros((100, 3))
        points[:, 2] = 3.0

        n_viol, n_obs, _ = count_violations(self.ctx, points, np.eye(4), margin=0.03)

        self.assertGreater(n_obs, 0)
        self.assertEqual(n_viol, 0)

    def test_unmeasured_pixels_are_not_counted_as_evidence(self):
        # A pose thrown into space the sensor never measured must report 0/0, not
        # "clean". n_observed is what lets a caller tell those apart.
        ctx = VisibilityContext.from_frame(
            make_frame(torch.zeros((480, 640), dtype=torch.float32), self.camera), self.extrinsic
        )
        points = np.zeros((100, 3))
        points[:, 2] = 1.0

        n_viol, n_obs, ratio = count_violations(ctx, points, np.eye(4), margin=0.03)

        self.assertEqual(n_obs, 0)
        self.assertEqual(n_viol, 0)
        self.assertEqual(ratio, 0.0)

    def test_reads_full_frame_depth_not_the_masked_crop(self):
        # make_frame zeroes the masked crop, so a context that read `depth`
        # instead of `depth_full` would see no measurements at all.
        points = np.zeros((100, 3))
        points[:, 2] = 1.0

        _, n_obs, ratio = count_violations(self.ctx, points, np.eye(4), margin=0.03)

        self.assertGreater(n_obs, 0)
        self.assertEqual(ratio, 1.0)

    def test_context_is_none_without_a_frame(self):
        self.assertIsNone(VisibilityContext.from_frame(None, self.extrinsic))


class TestFreeSpaceGate(unittest.TestCase):
    """The stage-1 veto: it must reject a pose that occupies observed free space
    and leave a plausible one alone."""

    def setUp(self):
        self.camera = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        depth = torch.full((480, 640), 2.0, dtype=torch.float32)
        self.ctx = VisibilityContext.from_frame(make_frame(depth, self.camera), np.eye(4))
        self.points = np.zeros((200, 3))
        self.points[:, 2] = 2.0
        self.gate = FreeSpaceGate(
            context=self.ctx, points=self.points, margin=0.07, max_ratio=0.10, min_observed=30
        )

    def test_accepts_pose_consistent_with_the_measured_surface(self):
        self.assertFalse(self.gate.rejects(np.eye(4)))

    def test_rejects_pose_pulled_into_observed_free_space(self):
        T = np.eye(4)
        T[2, 3] = -1.0  # a metre nearer the camera than anything measured
        self.assertTrue(self.gate.rejects(T))

    def test_abstains_when_too_few_points_are_observed(self):
        gate = FreeSpaceGate(
            context=self.ctx,
            points=self.points,
            margin=0.07,
            max_ratio=0.10,
            min_observed=10_000,
        )
        T = np.eye(4)
        T[2, 3] = -1.0
        # Grossly violating, but the sample is below the trust threshold, so the
        # gate must abstain rather than veto.
        self.assertFalse(gate.rejects(T))


class TestDualHypothesisSelection(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    def test_falls_through_to_fitness_when_free_space_is_uninformative(self):
        # Scene is the flipped model, and there is no visibility evidence at all,
        # so the ICP fitness comparison must decide. Regression guard for the
        # deleted early-exit gate, which used to return Pass 1 unconditionally.
        model_points = np.asarray(
            o3d.geometry.TriangleMesh.create_box(1.0, 0.5, 0.5).sample_points_uniformly(200).points
        )
        cx = 0.5 * (model_points[:, 0].min() + model_points[:, 0].max())
        cy = 0.5 * (model_points[:, 1].min() + model_points[:, 1].max())
        T_flip = np.eye(4)
        T_flip[:2, :2] = np.array([[-1, 0], [0, -1]])
        T_flip[0, 3] = 2.0 * cx
        T_flip[1, 3] = 2.0 * cy

        scene_points = model_points @ T_flip[:3, :3].T + T_flip[:3, 3]
        scene_normals = np.zeros_like(scene_points)
        scene_normals[:, 2] = 1.0

        res = refine_pose_dual_hypothesis_se2(
            model_points=model_points,
            scene_points=scene_points,
            scene_normals=scene_normals,
            T_init=np.eye(4),
            max_correspondence_distance=0.05,
            max_iterations=20,
            visibility=None,
            free_space_points=None,
        )

        self.assertTrue(np.allclose(res.transformation, T_flip, atol=1e-2))

    def test_free_space_points_are_independent_of_model_points(self):
        # The signature must keep registration geometry (slab) and visibility
        # geometry (full cart) separate; passing only model_points is what made
        # the old check inert.
        import inspect

        sig = inspect.signature(refine_pose_dual_hypothesis_se2)
        self.assertIn("free_space_points", sig.parameters)
        self.assertIn("model_points", sig.parameters)


class TestParamsDefaults(unittest.TestCase):
    def test_free_space_defaults(self):
        params = Ransac3DoFParams()
        self.assertEqual(params.free_space_margin, 0.03)
        self.assertEqual(params.free_space_separation, 0.02)
        self.assertEqual(params.free_space_min_observed, 30)

    def test_stage1_gate_is_off_by_default(self):
        # It is an experimental arm, not the default pipeline.
        self.assertFalse(Ransac3DoFParams().free_space_gate)

    def test_stage1_margin_is_looser_than_the_post_icp_one(self):
        # The stage-1 hypothesis is pre-ICP and coarse; a tight margin there
        # rejects good poses over centimetre-scale error.
        params = Ransac3DoFParams()
        self.assertGreater(params.free_space_gate_margin, params.free_space_margin)


if __name__ == "__main__":
    unittest.main()
