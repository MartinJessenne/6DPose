"""
Graduated non-convexity in the SE(2) ICP refinement.

The claim under test is the one that motivated the change: a single
max_correspondence_distance was doing two incompatible jobs, and splitting them
strictly dominates either setting of the single parameter.

  Job B, capture basin -- must be WIDE or a pose left 3 cm out by the global
    stage retrieves no neighbours and cannot converge at all.
  Job A, outlier rejection -- must be TIGHT or model points with no true scene
    counterpart count at full weight and drag the fit.

TestJobSeparation builds a scene where those two demands are in direct conflict
and shows all three outcomes: wide-hard converges to the wrong pose, tight-hard
starves for correspondences and barely moves, GNC does both jobs at once.

TestOptimalFixedRadiusIsNotKnowable is the argument for the change rather than
merely a check on it: the best single radius is a function of how far off the
global stage left the pose, so it cannot be chosen when the config is written.
"""

import unittest

import numpy as np

from methods.se2_icp import GncSchedule, _geman_mcclure_weight, icp_se2
from tests.helpers import se2_pose

# The sensor noise floor: sigma_z = z^2 sigma_d / (f B) with f B = 60.80 px.m,
# z = 3.0 m, sigma_d = 0.1 px. See Ransac3DoFParams.icp_gnc_scale_min.
SCALE_MIN = 0.0148
# The shipped capture radius (Optuna study VSAC_NullOff_M2 trial #33).
CAPTURE = 0.13768813892484938

# The control arm: hard truncation, i.e. what ran before GNC. There is no
# `gnc=None` to pass -- the schedule is mandatory -- so it is expressed instead
# as a kernel whose scale sits far above any residual in these fixtures. At
# s = 1e6 m the Geman-McClure weight (s^2/(s^2+r^2))^2 is 1 to within 1e-13 for
# every r under a metre, so every correspondence inside the KD-tree radius
# counts equally and the solve is plain least squares. That IS hard truncation,
# and expressing it this way keeps the control on the same code path as the
# treatment rather than on a separate branch that could rot.
NO_GNC = GncSchedule(scale_min=1e6)


class TestGncSchedule(unittest.TestCase):
    """
    The anneal must always fit the iteration budget, because overrunning it
    silently means the kernel never reaches the noise floor and the solver does
    no outlier rejection at all -- while still returning a pose.
    """

    def test_anneals_from_the_capture_radius_down_to_the_noise_floor(self):
        s = GncSchedule(scale_min=SCALE_MIN, shrink=1.1832).iteration_scales(CAPTURE, 100)
        self.assertAlmostEqual(s[0], CAPTURE)
        self.assertAlmostEqual(s[-1], SCALE_MIN)
        # Non-increasing: a schedule that ever widens would undo annealing.
        self.assertTrue(all(a >= b for a, b in zip(s, s[1:], strict=False)))

    def test_step_count_matches_the_closed_form(self):
        """
        The anneal is ceil(log(s0/s_min) / log(shrink)) + 1 entries long -- one
        per division plus the final clamp to scale_min -- after which the
        schedule holds at scale_min for the rest of the budget. Asserted so a
        future change to `shrink` cannot silently alter the anneal length.
        """
        sched = GncSchedule(scale_min=SCALE_MIN, shrink=1.1832)
        expected = int(np.ceil(np.log(CAPTURE / SCALE_MIN) / np.log(1.1832))) + 1
        s = sched.iteration_scales(CAPTURE, 100)
        self.assertEqual(len([x for x in s if x > SCALE_MIN]) + 1, expected)
        self.assertEqual(s[expected - 1], SCALE_MIN)

    def test_length_is_always_the_budget(self):
        gnc = GncSchedule(scale_min=SCALE_MIN)
        for scale_init, budget in [(0.3, 100), (0.5, 20), (0.5, 2), (0.01, 30), (0.2, 1)]:
            with self.subTest(scale_init=scale_init, budget=budget):
                self.assertEqual(len(gnc.iteration_scales(scale_init, budget)), budget)

    def test_radius_already_at_the_noise_floor_leaves_nothing_to_anneal(self):
        """Nothing to anneal, and widening the kernel past the KD-tree radius would lie."""
        sched = GncSchedule(scale_min=SCALE_MIN)
        self.assertEqual(sched.iteration_scales(SCALE_MIN / 2, 4), [SCALE_MIN] * 4)

    def test_rejects_a_schedule_that_cannot_anneal(self):
        with self.assertRaises(ValueError):
            GncSchedule(scale_min=SCALE_MIN, shrink=1.0)
        with self.assertRaises(ValueError):
            GncSchedule(scale_min=0.0)

    def test_tight_budget_clamps_instead_of_raising(self):
        """
        0.5 m -> 0.0148 m at 1.1832 wants 22 entries. A 10-iteration budget used
        to raise ValueError and abort the frame. Raising is wrong on a robot: a
        misconfiguration should degrade the anneal, loudly, not kill the
        pipeline. It must now anneal faster and still reach the floor.
        """
        gnc = GncSchedule(scale_min=SCALE_MIN)
        with self.assertLogs(level="WARNING") as captured:
            s = gnc.iteration_scales(0.5, 10)
        self.assertIn("effective shrink", "".join(captured.output))
        self.assertEqual(len(s), 10)
        self.assertEqual(s[0], 0.5)
        self.assertEqual(s[-1], gnc.scale_min)
        self.assertGreater(s[0] / s[1], gnc.shrink)

    def test_schedule_ends_exactly_on_scale_min(self):
        """
        _at_final_scale compares with ==, so the tail must be scale_min copied,
        never a value arrived at by dividing.
        """
        gnc = GncSchedule(scale_min=SCALE_MIN)
        for scale_init, budget in [(0.3, 100), (0.5, 10), (0.5, 3)]:
            with self.subTest(scale_init=scale_init, budget=budget):
                self.assertIs(gnc.iteration_scales(scale_init, budget)[-1], gnc.scale_min)

    def test_zero_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            GncSchedule(scale_min=SCALE_MIN).iteration_scales(0.3, 0)


class TestGemanMcClureWeights(unittest.TestCase):
    def test_weight_landmarks(self):
        """w(0) = 1, w(s) = 1/4, w(3s) = 1/100 -- the shape quoted in GncSchedule."""
        s = 0.02
        w = _geman_mcclure_weight(np.array([0.0, s, 3 * s]), s)
        np.testing.assert_allclose(w, [1.0, 0.25, 0.01], atol=1e-12)

    def test_weights_are_even_in_the_residual_sign(self):
        """A point-to-plane residual is signed; the kernel must not prefer one side."""
        w = _geman_mcclure_weight(np.array([-0.03, 0.03]), 0.02)
        self.assertAlmostEqual(w[0], w[1])

    def test_the_control_arm_really_is_unweighted(self):
        """NO_GNC only works as a hard-truncation control if its weights are all 1."""
        w = _geman_mcclure_weight(np.array([0.0, 0.05, 0.5]), NO_GNC.scale_min)
        np.testing.assert_allclose(w, 1.0, atol=1e-12)


class TestJobSeparation(unittest.TestCase):
    """
    An L-shaped surface of which the model carries a SECOND, fictional copy
    offset 0.10 m along each face normal -- the zero-thickness-shell far sheet
    of vault note 30.06, in synthetic form. The scene contains only the near
    sheet, so every far-sheet model point is an outlier with no true
    counterpart, and at the optimum its nearest scene point sits 0.10 m away.

    Two perpendicular faces rather than one: with a single plane the yaw and
    the along-face translation are unobservable, so the 3x3 normal equations
    would be rank 1 and the test would be measuring the regulariser.
    """

    FAR_SHEET = 0.10

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 500

        # Face A: the plane x = 0, outward normal -x. Face B: y = -1, normal -y.
        a_near = np.column_stack([np.zeros(n), rng.uniform(-1, 1, n), rng.uniform(0, 1, n)])
        b_near = np.column_stack([rng.uniform(0, 1, n), -np.ones(n), rng.uniform(0, 1, n)])
        n_a = np.tile([-1.0, 0.0, 0.0], (n, 1))
        n_b = np.tile([0.0, -1.0, 0.0], (n, 1))

        cls.scene_points = np.vstack([a_near, b_near])
        cls.scene_normals = np.vstack([n_a, n_b])

        # The model adds the far sheet: the same faces pushed 0.10 m INWARD
        # (along +x and +y), which is where the back of a zero-thickness tube
        # would be sampled.
        a_far = a_near + np.array([cls.FAR_SHEET, 0.0, 0.0])
        b_far = b_near + np.array([0.0, cls.FAR_SHEET, 0.0])
        cls.model_points = np.vstack([a_near, b_near, a_far, b_far])

        cls.T_true = np.eye(4)
        # Off by 3 cm and 2 cm in plane and 2 degrees in yaw: a plausible
        # residual error for the global RANSAC stage to leave behind.
        cls.T_init = se2_pose(np.radians(2.0), np.array([0.03, 0.02]), 0.0)

    def _run(self, max_correspondence_distance, gnc):
        result = icp_se2(
            model_points=self.model_points,
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=self.T_init,
            max_correspondence_distance=max_correspondence_distance,
            gnc=gnc,
            max_iterations=100,
        )
        T = result.transformation
        trans_err = float(np.linalg.norm(T[:2, 3] - self.T_true[:2, 3]))
        yaw_err = float(abs(np.degrees(np.arctan2(T[1, 0], T[0, 0]))))
        return trans_err, yaw_err, result

    def test_wide_hard_truncation_converges_to_a_biased_pose(self):
        """
        The far sheet is inside a 0.1377 m radius, so it counts at full weight.
        Half the model wants x = 0 and half wants x = -0.10, and least squares
        splits the difference -- roughly 0.05 m off, an order of magnitude
        larger than the sensor noise.
        """
        trans_err, _, result = self._run(CAPTURE, gnc=NO_GNC)
        self.assertGreater(trans_err, 0.02, "the far-sheet bias this change removes has gone")
        self.assertGreater(result.fitness, 0.5, "it did converge -- it converged to the wrong pose")

    def test_tight_hard_truncation_starves_the_association(self):
        """
        Shrinking the single parameter to the noise floor is the obvious fix and
        it does not work. From 3 cm out only the handful of model points that
        happen to start within 0.0148 m retrieve a neighbour at all -- fitness
        collapses to a few percent -- and the solver corrects only the
        directions those survivors happen to constrain. This is Job B failing,
        and it is why the sweeps could not simply keep shrinking.

        It is also, precisely, what icp_refine_ladder's last rung was doing.
        """
        trans_err, _, result = self._run(SCALE_MIN, gnc=NO_GNC)
        self.assertLess(result.fitness, 0.10, "it should retrieve almost no correspondences")
        self.assertGreater(trans_err, SCALE_MIN)

    def test_gnc_gets_both(self):
        """
        Wide radius for the association, annealed kernel for the weighting: the
        pose lands two orders of magnitude inside the noise floor and well under
        a degree in yaw, with full fitness. This failing while the two above
        pass means the annealing is not down-weighting the far sheet.
        """
        sched = GncSchedule(scale_min=SCALE_MIN, shrink=1.1832)
        trans_err, yaw_err, result = self._run(CAPTURE, gnc=sched)
        self.assertLess(trans_err, SCALE_MIN / 10.0)
        self.assertLess(yaw_err, 1.0)
        self.assertGreater(result.fitness, 0.9)

    def test_gnc_strictly_beats_the_wide_hard_stage_it_replaces(self):
        """The comparison the change has to win, stated as one assertion."""
        hard_err, _, _ = self._run(CAPTURE, gnc=NO_GNC)
        gnc_err, _, _ = self._run(CAPTURE, gnc=GncSchedule(scale_min=SCALE_MIN))
        self.assertLess(gnc_err, hard_err / 10.0)


class TestOptimalFixedRadiusIsNotKnowable(unittest.TestCase):
    """
    The argument for GNC, as a test.

    A fixed radius is not merely hard to tune -- its optimum is a function of
    how far off the GLOBAL stage left the pose, which varies frame to frame and
    is unknown when the config is written. Scanning 25 radii at each of six
    initialisation errors, the best radius climbs in step with the error, and a
    radius that is optimal at 3 cm is off by more than 10 cm at 12 cm. GNC's
    error does not move at all across the same range.

    This is also the direct case against icp_refine_ladder, whose rungs were
    exactly such hand-picked fixed radii.

    If this test ever fails because GNC has become init-dependent, the annealing
    is starting too tight -- check that the schedule begins at the full capture
    radius and not at something smaller.
    """

    INIT_OFFSETS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    RADIUS_GRID = tuple(np.geomspace(0.01, 0.25, 25))

    def setUp(self):
        self.fixture = TestJobSeparation
        self.fixture.setUpClass()

    def _err(self, T_init, radius, gnc):
        result = icp_se2(
            model_points=self.fixture.model_points,
            scene_points=self.fixture.scene_points,
            scene_normals=self.fixture.scene_normals,
            T_init=T_init,
            max_correspondence_distance=radius,
            gnc=gnc,
            max_iterations=100,
        )
        return float(np.linalg.norm(result.transformation[:2, 3]))

    def _inits(self):
        for off in self.INIT_OFFSETS:
            yield off, se2_pose(np.radians(2.0), np.array([off, off * 0.67]), 0.0)

    def test_the_best_fixed_radius_tracks_the_initialisation_error(self):
        best = []
        for _off, T_init in self._inits():
            errs = [self._err(T_init, r, NO_GNC) for r in self.RADIUS_GRID]
            best.append(self.RADIUS_GRID[int(np.argmin(errs))])
        self.assertLess(best[0], best[-1] / 5.0, f"best radii barely moved: {best}")

    def test_gnc_is_flat_across_the_same_range(self):
        errs = [
            self._err(T_init, CAPTURE, GncSchedule(scale_min=SCALE_MIN))
            for _, T_init in self._inits()
        ]
        self.assertLess(max(errs), SCALE_MIN / 10.0, f"GNC lost accuracy somewhere: {errs}")
        self.assertLess(max(errs) - min(errs), 1e-3, f"GNC became init-dependent: {errs}")

    def test_a_radius_tuned_at_one_offset_fails_at_another(self):
        """The concrete cost of shipping a single number, in meters."""
        inits = dict(self._inits())
        tuned_here = 0.0196  # optimal at a 3 cm initialisation error
        self.assertLess(self._err(inits[0.03], tuned_here, NO_GNC), 1e-3)
        self.assertGreater(self._err(inits[0.12], tuned_here, NO_GNC), 0.10)


class TestGncStaysOnTheManifold(unittest.TestCase):
    """The weighting must not disturb the SE(2) constraint the module exists for."""

    def test_z_row_is_untouched_by_the_weighted_solve(self):
        rng = np.random.default_rng(7)
        n = 400
        z_offset = 0.31
        scene = np.column_stack([np.zeros(n), rng.uniform(-1, 1, n), rng.uniform(0, 1, n)])
        scene = np.vstack(
            [scene, np.column_stack([rng.uniform(0, 1, n), -np.ones(n), rng.uniform(0, 1, n)])]
        )
        normals = np.vstack([np.tile([-1.0, 0.0, 0.0], (n, 1)), np.tile([0.0, -1.0, 0.0], (n, 1))])
        model = scene - np.array([0.0, 0.0, z_offset])

        result = icp_se2(
            model_points=model,
            scene_points=scene,
            scene_normals=normals,
            T_init=se2_pose(0.05, np.array([0.02, -0.01]), z_offset),
            max_correspondence_distance=CAPTURE,
            gnc=GncSchedule(scale_min=SCALE_MIN),
            max_iterations=100,
        )
        T = result.transformation
        np.testing.assert_array_equal(T[2], [0.0, 0.0, 1.0, z_offset])
        np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
        R2 = T[:2, :2]
        np.testing.assert_allclose(R2.T @ R2, np.eye(2), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
