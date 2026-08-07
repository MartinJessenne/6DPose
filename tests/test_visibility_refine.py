"""
Tests for the T0 refinement stage: visibility culling (methods/ransac3dof.py,
methods/se2_icp.py).

The mechanism under test is a BIAS in the ICP objective, so the tests are built
the way the finding was: place the model at a known pose, refine from there, and
assert on where the objective's minimum actually sits. A scene rendered from the
real cart meshes is enough to reproduce it -- the bias comes from the model cloud
carrying surfaces the sensor cannot see, which is a property of the mesh, not of
the dataset.

See the vault note "30.06 - T0 Translation Error and the Visibility Cull".
"""

import unittest
import unittest.mock

import numpy as np
import open3d as o3d

from cli_config import CameraConfig
from methods import ransac3dof
from methods.ransac3dof import (
    Ransac3DoFEstimator,
    Ransac3DoFParams,
    crop_front_face,
)
from methods.se2_icp import GncSchedule, icp_se2
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


class TestVisibilityCull(unittest.TestCase):
    """The cull itself, on the real meshes."""

    @classmethod
    def setUpClass(cls):
        cls.meshes = load_cad_meshes()
        cls.estimator = Ransac3DoFEstimator(
            params=Ransac3DoFParams(front_crop_aspect=2.0, z_offset=0.01),
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
                verts = np.asarray(mesh.vertices)
                l_y = float(verts[:, 1].max() - verts[:, 1].min())
                slab = crop_front_face(mesh, depth=l_y / 2.0)
                pts = np.asarray(slab.sample_points_uniformly(number_of_points=2000).points)
                T = se2(2.5, 0.0, 180.0)
                visible = self.estimator._visible_model_indices(pts, T)
                fraction = len(visible) / len(pts)
                self.assertLess(fraction, 0.75, f"{cart}: cull kept {fraction:.2f} of the slab")
                self.assertGreater(fraction, 0.02, f"{cart}: cull kept only {fraction:.2f}")

    def test_returns_indices_into_the_model_cloud(self):
        verts = np.asarray(self.meshes["colruyt"].vertices)
        l_y = float(verts[:, 1].max() - verts[:, 1].min())
        pts = np.asarray(
            crop_front_face(self.meshes["colruyt"], depth=l_y / 2.0)
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
        o3d.utility.random.seed(0)
        cls.mesh = load_cad_meshes()["colruyt"]
        cls.T_gt = se2(2.5, 0.0, 180.0)
        cls.scene = render_visible_cloud(cls.mesh, cls.T_gt)
        verts = np.asarray(cls.mesh.vertices)
        l_y = float(verts[:, 1].max() - verts[:, 1].min())
        slab = crop_front_face(cls.mesh, depth=l_y / 2.0)
        cls.model_pc = slab.sample_points_uniformly(number_of_points=2000)

    def _refine(self, **param_kwargs):
        params = Ransac3DoFParams(
            front_crop_aspect=2.0,
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
        """
        The control. With every model point kept, two thirds of them are
        far-sheet fiction that still demands a correspondence, and the minimum
        the solver converges to is measurably off the truth it started from.
        """
        drift, _ = self._refine(icp_visibility_cull=False)
        self.assertGreater(drift, 0.005, "the unculled bias this arm exists to remove has gone")

    def test_cull_lands_closer_to_the_truth(self):
        """The treatment, against that control. This is the whole arm, in one assertion."""
        drift_base, _ = self._refine(icp_visibility_cull=False)
        drift_culled, _ = self._refine(icp_visibility_cull=True)
        self.assertLess(drift_culled, drift_base)

    def test_cull_off_is_exactly_one_stage(self):
        """
        With the cull off there is nothing for a second stage to do differently,
        so _refine_pose must reduce to a single icp_se2 call -- bit-identical,
        not merely close. A drift here means a stage is running that should not.
        """
        params = Ransac3DoFParams(
            front_crop_aspect=2.0,
            z_offset=0.01,
            icp_max_correspondence_distance=0.13768813892484938,
            icp_max_iterations=100,
            icp_visibility_cull=False,
        )
        estimator = Ransac3DoFEstimator(params=params, extrinsic=EXTRINSIC)
        estimator._front_face = None
        T_new = estimator._refine_pose(self.model_pc, self.scene, self.T_gt)

        expected = icp_se2(
            model_points=np.asarray(self.model_pc.points),
            scene_points=np.asarray(self.scene.points),
            scene_normals=np.asarray(self.scene.normals),
            T_init=self.T_gt,
            gnc=GncSchedule(
                scale_min=params.icp_gnc_scale_min, shrink=params.icp_gnc_shrink
            ),
            max_correspondence_distance=params.icp_max_correspondence_distance,
            max_iterations=params.icp_max_iterations,
        ).transformation
        np.testing.assert_array_equal(T_new, expected)

    def test_both_stages_use_the_same_capture_radius(self):
        """
        The radius is the capture basin and nothing else now. If a future change
        reintroduces per-stage tightening, this catches it: every icp_se2 call
        must be handed icp_max_correspondence_distance verbatim.
        """
        seen = []
        real = ransac3dof.icp_se2

        def spy(**kwargs):
            seen.append(kwargs["max_correspondence_distance"])
            return real(**kwargs)

        params = Ransac3DoFParams(
            front_crop_aspect=2.0,
            z_offset=0.01,
            icp_max_correspondence_distance=0.13768813892484938,
            icp_max_iterations=100,
            icp_visibility_cull=True,
        )
        estimator = Ransac3DoFEstimator(params=params, extrinsic=EXTRINSIC)
        estimator._front_face = None
        with unittest.mock.patch.object(ransac3dof, "icp_se2", spy):
            estimator._refine_pose(self.model_pc, self.scene, self.T_gt)

        self.assertEqual(len(seen), 2, "expected a wide stage and a culled stage")
        self.assertEqual(set(seen), {params.icp_max_correspondence_distance})


if __name__ == "__main__":
    unittest.main()
