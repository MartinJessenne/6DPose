"""
Tests for the front-face orientation constraint, on REAL data.

Every previous flip mechanism in this pipeline passed its synthetic tests and
then failed on the benchmark, so these are built on the committed fixtures
(tests/fixtures, see scripts/fetch_test_samples.py) and on real ground-truth
poses rather than on hand-made point clouds.

The central construction is `rotate_about_anchor`. Rotating a pose about the
cart's own front-face anchor leaves the anchor -- and therefore the direction to
the camera -- unchanged, while rotating the arrow by exactly theta. So the
resulting angle is exactly `gt_angle + theta`, and the gate's verdict can be
asserted against a closed-form expectation for every cart type and every frame,
with no threshold invented for the test.
"""

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from cli_config import CameraConfig
from methods.ransac3dof import (
    FrontFace,
    FrontFaceGate,
    front_face_angle_deg,
    front_face_from_mesh,
)
from pipeline import compute_ground_truth_pose, load_cad_meshes, load_parquet_dataset

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MAX_ANGLE = 60.0


def wrap180(deg: float) -> float:
    """Wrap degrees into (-180, 180]."""
    return float(np.degrees(np.arctan2(np.sin(np.radians(deg)), np.cos(np.radians(deg)))))


def rotate_about_anchor(T: np.ndarray, front_face: FrontFace, theta_deg: float) -> np.ndarray:
    """Rotate pose T by theta about its own front-face anchor, in the ground plane.

    Leaves the anchor's world position fixed, so the direction to the camera is
    untouched and the arrow rotates by exactly theta.

    Note the reported angle then changes by MINUS theta: front_face_angle_deg
    measures from the arrow to the camera, so rotating the arrow forward reduces
    it.
    """
    a_world = T[:3, :3] @ np.asarray(front_face.anchor) + T[:3, 3]
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    rot = np.eye(4)
    rot[:2, :2] = np.array([[c, -s], [s, c]])
    # Conjugate the rotation so it acts about a_world rather than the origin.
    rot[:2, 3] = a_world[:2] - rot[:2, :2] @ a_world[:2]
    return rot @ T


def load_fixture_frames():
    """(cart_type, T_gt) for every committed fixture frame, across all splits."""
    extrinsic = np.array(CameraConfig().extrinsic, dtype=float)
    frames = []
    for split in ("test", "validation", "train"):
        ds = load_parquet_dataset(dataset_path=str(FIXTURES), test_glob=f"data/{split}-*.parquet")
        for row in ds:
            t_world_camera = np.asarray(row["camera_view_transform"], dtype=float).reshape(4, 4).T
            t_world_cart = np.asarray(row["bbox_3d_transform"][0], dtype=float).reshape(4, 4).T
            frames.append(
                (
                    row["bbox_3d_class_name"][0],
                    compute_ground_truth_pose(t_world_camera, t_world_cart, extrinsic),
                )
            )
    return frames


class TestFrontFaceGateOnFixtures(unittest.TestCase):
    """The gate, exercised against real ground-truth poses for all three carts."""

    @classmethod
    def setUpClass(cls):
        if not (FIXTURES / "data").exists():
            raise unittest.SkipTest("fixtures missing; run: uv run scripts/fetch_test_samples.py")
        cls.meshes = load_cad_meshes()
        cls.front_faces = {k: front_face_from_mesh(m) for k, m in cls.meshes.items()}
        cls.camera_xy = np.array(CameraConfig().extrinsic, dtype=float)[:3, 3][:2]
        cls.frames = load_fixture_frames()

    def test_fixtures_cover_all_carts_and_span_the_bearing_range(self):
        """Guards the fixture set itself, not the gate.

        A gate wrongly written against a fixed base_link axis instead of the
        direction to the camera passes on head-on carts and only fails on angled
        ones. If the fixtures ever lose their high-bearing frames, the tests
        below would keep passing while that bug shipped.
        """
        by_cart = {}
        for cart, T_gt in self.frames:
            angle = abs(front_face_angle_deg(T_gt, self.front_faces[cart], self.camera_xy))
            by_cart.setdefault(cart, []).append(angle)

        self.assertEqual(set(by_cart), {"colruyt", "leanflow", "picanol"})
        for cart, angles in by_cart.items():
            self.assertLess(min(angles), 15.0, f"{cart}: no near-head-on frame")
            self.assertGreater(max(angles), 30.0, f"{cart}: no high-bearing frame")

    def test_ground_truth_poses_are_all_accepted(self):
        """The constraint must never veto the truth.

        This is the check that would have caught the free-space gate before three
        hours of sweep: a gate that rejects correct answers can only convert them
        into abstentions, which every success-denominator rate reads as progress.
        """
        for cart, T_gt in self.frames:
            gate = FrontFaceGate(self.front_faces[cart], self.camera_xy, MAX_ANGLE)
            angle = front_face_angle_deg(T_gt, self.front_faces[cart], self.camera_xy)
            self.assertTrue(
                gate.accepts(T_gt),
                f"{cart}: ground-truth pose rejected at {angle:.2f} deg",
            )

    def test_flipped_ground_truth_poses_are_all_rejected(self):
        """The 180-degree flip -- the entire point of the mechanism."""
        for cart, T_gt in self.frames:
            ff = self.front_faces[cart]
            gate = FrontFaceGate(ff, self.camera_xy, MAX_ANGLE)
            T_flipped = rotate_about_anchor(T_gt, ff, 180.0)
            angle = front_face_angle_deg(T_flipped, ff, self.camera_xy)
            self.assertFalse(
                gate.accepts(T_flipped),
                f"{cart}: flipped pose accepted at {angle:.2f} deg",
            )
            # Confirms the separation the threshold relies on. Ground-truth
            # bearings reach 48.8 deg over 560 frames, so a flipped pose can come
            # no closer to head-on than 180 - 48.8 = 131.2.
            self.assertGreater(abs(angle), 130.0)

    def test_verdict_matches_closed_form_across_the_full_rotation_range(self):
        """Sweep theta all the way round, for every frame and every cart.

        The expected verdict is computed in closed form from the frame's own
        ground-truth angle, so this asserts the boundary at 5-degree resolution
        without a single hand-tuned constant.
        """
        for cart, T_gt in self.frames:
            ff = self.front_faces[cart]
            gate = FrontFaceGate(ff, self.camera_xy, MAX_ANGLE)
            gt_angle = front_face_angle_deg(T_gt, ff, self.camera_xy)

            for theta in range(-180, 180, 5):
                T = rotate_about_anchor(T_gt, ff, theta)
                measured = front_face_angle_deg(T, ff, self.camera_xy)
                # Minus, not plus: the angle is measured FROM the arrow TO the
                # camera, so rotating the arrow by +theta lowers it by theta.
                expected = wrap180(gt_angle - theta)
                # The rotation identity itself: exact up to float round-off.
                self.assertAlmostEqual(
                    wrap180(measured - expected),
                    0.0,
                    places=6,
                    msg=f"{cart} theta={theta}: angle identity broken",
                )
                self.assertEqual(
                    gate.accepts(T),
                    abs(expected) <= MAX_ANGLE,
                    f"{cart} theta={theta}: verdict disagrees at {measured:.2f} deg",
                )

    def test_angle_is_measured_against_the_camera_not_a_fixed_axis(self):
        """Move the camera, keep the pose: the verdict must follow the camera.

        A gate comparing the arrow to a fixed base_link axis would return the same
        answer here. That bug is invisible on head-on carts, which is why it gets
        its own test rather than relying on the sweep above.
        """
        cart, T_gt = self.frames[0]
        ff = self.front_faces[cart]
        anchor_xy = (T_gt[:3, :3] @ np.asarray(ff.anchor) + T_gt[:3, 3])[:2]
        arrow_xy = T_gt[:2, :2] @ np.asarray(ff.normal)[:2]

        # A camera placed straight along the arrow reads 0; one placed behind
        # reads 180 -- for the identical pose.
        cam_ahead = anchor_xy + 3.0 * arrow_xy
        cam_behind = anchor_xy - 3.0 * arrow_xy
        self.assertAlmostEqual(front_face_angle_deg(T_gt, ff, cam_ahead), 0.0, places=6)
        self.assertAlmostEqual(abs(front_face_angle_deg(T_gt, ff, cam_behind)), 180.0, places=6)


class TestFrontFaceGeometry(unittest.TestCase):
    """Properties of the arrow itself, on the real meshes."""

    @classmethod
    def setUpClass(cls):
        cls.meshes = load_cad_meshes()

    # NOTE: there is deliberately no geometric validation of the +x convention.
    # A check was attempted -- "at least 90% of vertices lie behind anchor[0]" --
    # and the mirrored-mesh test proved it vacuous: anchor[0] IS x_max, so every
    # vertex of every mesh satisfies it, mirrored or not. Nothing intrinsic to the
    # geometry distinguishes the towing face from the far end; which end tows is
    # semantic information the mesh does not carry. Any surviving check would be a
    # threshold fitted to today's three meshes.
    #
    # The real validation is visual: export_front_face_glb (methods/ransac3dof.py)
    # writes the mesh with its arrow attached, and the fixture-based tests above
    # catch an inverted convention immediately -- a mirrored mesh would put every
    # ground-truth pose near 180 deg and fail test_ground_truth_poses_are_all_accepted.

    def test_fields_are_immutable(self):
        """FrontFace instances are cached and shared, including across A/B arms.

        _PREPARATION_CACHE is class-level, so a mutable arrow could be modified
        through one estimator and read by another -- silently, as a comparison
        that quietly ran the same arm twice.
        """
        ff = front_face_from_mesh(self.meshes["colruyt"])
        # writeable=False blocks in-place writes...
        with self.assertRaises(ValueError):
            ff.anchor[0] = 99.0
        # ...and frozen=True blocks rebinding the field.
        with self.assertRaises(FrozenInstanceError):
            ff.normal = np.zeros(3)


if __name__ == "__main__":
    unittest.main()
