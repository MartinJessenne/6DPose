"""
Tests for the T0 refinement stage: visibility culling, the tightening ladder and
the yaw guard (methods/ransac3dof.py, methods/se2_icp.py).

The mechanism under test is a BIAS in the ICP objective, so the tests are built
the way the finding was: place the model at a known pose, refine from there, and
assert on where the objective's minimum actually sits. A scene rendered from the
real cart meshes is enough to reproduce it -- the bias comes from the model cloud
carrying surfaces the sensor cannot see, which is a property of the mesh, not of
the dataset.

See the vault note "30.06 - T0 Translation Error and the Visibility Cull".
"""

import unittest

import numpy as np
import open3d as o3d

from cli_config import CameraConfig
from methods.ransac3dof import Ransac3DoFEstimator, Ransac3DoFParams, crop_front_face
from methods.se2_icp import icp_point_to_plane_se2, icp_translation_only
from pipeline import load_cad_meshes

EXTRINSIC = np.array(CameraConfig().extrinsic, dtype=np.float64)
CAMERA_XY = EXTRINSIC[:3, 3]


def se2(x: float, y: float, yaw_deg: float, z: float = 0.01) -> np.ndarray:
    """4x4 pose with rotation about z only."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def render_visible_cloud(mesh, T, n_points=20000):
    """
    A stand-in for a depth observation: sample the posed mesh, keep only what is
    visible from the camera, and attach normals oriented toward the sensor --
    which is exactly the contract prepare_scene_point_cloud provides.
    """
    posed = o3d.geometry.TriangleMesh(mesh)
    posed.transform(T)
    pc = posed.sample_points_uniformly(number_of_points=n_points)
    diameter = float(np.linalg.norm(pc.get_max_bound() - pc.get_min_bound()))
    _m, kept = pc.hidden_point_removal(CAMERA_XY.tolist(), diameter * 100.0)
    visible = pc.select_by_index(kept)
    visible.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    visible.orient_normals_towards_camera_location(CAMERA_XY)
    return visible


class TestTranslationOnlyIcp(unittest.TestCase):
    """The 2-DoF solver: it must move translation and must not move yaw."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # A single plane facing -x: its normals span one direction, so only the
        # along-normal component of a translation error is observable.
        pts = np.column_stack([np.zeros(400), rng.uniform(-1, 1, 400), rng.uniform(0, 1, 400)])
        self.scene_points = pts
        self.scene_normals = np.tile([-1.0, 0.0, 0.0], (400, 1))

    def test_recovers_translation_along_normals(self):
        model = self.scene_points + np.array([0.03, 0.0, 0.0])
        result = icp_translation_only(
            model_points=model,
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=np.eye(4),
            max_correspondence_distance=0.2,
            max_iterations=50,
        )
        self.assertAlmostEqual(result.transformation[0, 3], -0.03, places=6)

    def test_yaw_is_frozen(self):
        model = self.scene_points + np.array([0.03, 0.0, 0.0])
        T_init = se2(0.0, 0.0, 12.0, z=0.0)
        result = icp_translation_only(
            model_points=model,
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=T_init,
            max_correspondence_distance=0.5,
            max_iterations=50,
        )
        np.testing.assert_allclose(result.transformation[:3, :3], T_init[:3, :3], atol=1e-12)

    def test_stays_on_se2(self):
        result = icp_translation_only(
            model_points=self.scene_points + np.array([0.02, 0.01, 0.0]),
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=se2(0.0, 0.0, 0.0, z=0.37),
            max_correspondence_distance=0.5,
            max_iterations=20,
        )
        self.assertAlmostEqual(result.transformation[2, 3], 0.37, places=12)


class TestVisibilityCull(unittest.TestCase):
    """The cull itself, on the real meshes."""

    @classmethod
    def setUpClass(cls):
        cls.meshes = load_cad_meshes()
        cls.estimator = Ransac3DoFEstimator(
            params=Ransac3DoFParams(front_crop_depth=0.7352383501440559, z_offset=0.01),
            extrinsic=EXTRINSIC,
        )

    def test_culls_most_of_the_slab(self):
        """
        The premise of the whole arm: prepare()'s uniformly sampled model cloud
        is mostly surface the camera cannot see. If this ratio ever climbs back
        toward 1.0, the meshes gained wall thickness and the cull is moot.
        """
        for cart, mesh in self.meshes.items():
            with self.subTest(cart=cart):
                slab = crop_front_face(mesh, depth=0.7352383501440559)
                pts = np.asarray(slab.sample_points_uniformly(number_of_points=2000).points)
                T = se2(2.5, 0.0, 180.0)
                visible = self.estimator._visible_model_indices(pts, T)
                fraction = len(visible) / len(pts)
                self.assertLess(fraction, 0.75, f"{cart}: cull kept {fraction:.2f} of the slab")
                self.assertGreater(fraction, 0.02, f"{cart}: cull kept only {fraction:.2f}")

    def test_returns_indices_into_the_model_cloud(self):
        pts = np.asarray(
            crop_front_face(self.meshes["colruyt"], depth=0.7)
            .sample_points_uniformly(number_of_points=1000)
            .points
        )
        visible = self.estimator._visible_model_indices(pts, se2(2.5, 0.0, 180.0))
        self.assertEqual(visible.dtype.kind, "i")
        self.assertEqual(len(np.unique(visible)), len(visible))
        self.assertTrue((visible >= 0).all() and (visible < len(pts)).all())


class TestRefinementBias(unittest.TestCase):
    """
    The finding, as a regression test: refine FROM the ground truth and assert on
    how far the objective's minimum sits from it. Search plays no part here, so a
    failure is unambiguously a refinement-stage regression.
    """

    @classmethod
    def setUpClass(cls):
        cls.mesh = load_cad_meshes()["colruyt"]
        cls.T_gt = se2(2.5, 0.0, 180.0)
        cls.scene = render_visible_cloud(cls.mesh, cls.T_gt)
        slab = crop_front_face(cls.mesh, depth=0.7352383501440559)
        cls.model_pc = slab.sample_points_uniformly(number_of_points=2000)

    def _refine(self, **param_kwargs):
        params = Ransac3DoFParams(
            front_crop_depth=0.7352383501440559,
            z_offset=0.01,
            icp_max_correspondence_distance=0.13768813892484938,
            icp_max_iterations=100,
            **param_kwargs,
        )
        estimator = Ransac3DoFEstimator(params=params, extrinsic=EXTRINSIC)
        estimator._front_face = None
        T = estimator._refine_pose(self.model_pc, self.scene, self.T_gt)
        return float(np.linalg.norm((T[:3, 3] - self.T_gt[:3, 3])[:2])), estimator

    def test_unculled_refinement_drifts_off_the_truth(self):
        """Baseline behaviour, kept as the control the treatment is measured against."""
        drift, _ = self._refine()
        self.assertGreater(drift, 0.005, "the unculled bias this arm exists to remove has gone")

    def test_culled_ladder_lands_closer_to_the_truth(self):
        drift_base, _ = self._refine()
        drift_culled, estimator = self._refine(
            icp_visibility_cull=True,
            icp_refine_ladder=(0.05, 0.02, 0.01),
            icp_yaw_guard_deg=5.0,
        )
        self.assertLess(drift_culled, drift_base)
        self.assertIn("icp_yaw_guard_trips", estimator._last_diagnostics)

    def test_ladder_without_cull_is_not_the_mechanism(self):
        """
        Tightening alone is a different (weaker) lever than culling. Recorded so
        that a future simplification collapsing the two flags is caught.
        """
        drift_ladder, _ = self._refine(icp_refine_ladder=(0.05, 0.02, 0.01))
        drift_culled, _ = self._refine(
            icp_visibility_cull=True, icp_refine_ladder=(0.05, 0.02, 0.01)
        )
        self.assertLessEqual(drift_culled, drift_ladder + 1e-9)

    def test_defaults_are_byte_identical_to_the_single_wide_stage(self):
        """
        The new parameters must be inert at their defaults: `tuned` has to keep
        meaning exactly what it meant before this landed, or every number already
        recorded against it is invalidated.
        """
        params = Ransac3DoFParams(
            front_crop_depth=0.7352383501440559,
            z_offset=0.01,
            icp_max_correspondence_distance=0.13768813892484938,
            icp_max_iterations=100,
        )
        estimator = Ransac3DoFEstimator(params=params, extrinsic=EXTRINSIC)
        estimator._front_face = None
        T_new = estimator._refine_pose(self.model_pc, self.scene, self.T_gt)

        expected = icp_point_to_plane_se2(
            model_points=np.asarray(self.model_pc.points),
            scene_points=np.asarray(self.scene.points),
            scene_normals=np.asarray(self.scene.normals),
            T_init=self.T_gt,
            max_correspondence_distance=params.icp_max_correspondence_distance,
            max_iterations=params.icp_max_iterations,
        ).transformation
        np.testing.assert_array_equal(T_new, expected)
        self.assertNotIn("icp_yaw_guard_trips", estimator._last_diagnostics)


class TestYawGuard(unittest.TestCase):
    """The guard that keeps a tight stage from jumping to a symmetry twin."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = load_cad_meshes()["leanflow"]  # the near-square slab
        cls.T_gt = se2(2.2, 0.0, 180.0)
        cls.scene = render_visible_cloud(cls.mesh, cls.T_gt)
        slab = crop_front_face(cls.mesh, depth=0.7352383501440559)
        cls.model_pc = slab.sample_points_uniformly(number_of_points=2000)

    def _yaw_after_refine(self, guard):
        params = Ransac3DoFParams(
            front_crop_depth=0.7352383501440559,
            z_offset=0.01,
            icp_max_correspondence_distance=0.13768813892484938,
            icp_max_iterations=100,
            icp_visibility_cull=True,
            icp_refine_ladder=(0.05, 0.02, 0.01),
            icp_yaw_guard_deg=guard,
        )
        estimator = Ransac3DoFEstimator(params=params, extrinsic=EXTRINSIC)
        estimator._front_face = None
        T = estimator._refine_pose(self.model_pc, self.scene, self.T_gt)
        yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        return float(
            np.degrees(np.arctan2(np.sin(np.radians(yaw - 180.0)), np.cos(np.radians(yaw - 180.0))))
        )

    def test_guard_bounds_the_total_yaw_excursion(self):
        """
        With a 5 degree guard no ladder stage may rotate more than 5 degrees, so
        three stages can contribute at most 15 -- nothing near the 90 degree twin.
        """
        self.assertLess(abs(self._yaw_after_refine(5.0)), 15.0)


if __name__ == "__main__":
    unittest.main()
