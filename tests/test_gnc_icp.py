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
starves for correspondences, GNC does both jobs at once.

TestOptimalFixedRadiusIsNotKnowable is the argument for the change rather than
merely a check on it: the best single radius is a function of how far off the
global stage left the pose, so it cannot be chosen when the config is written.

TestDegenerateGeometry covers the failure the LM damping exists for, on the
single-plane fixture the other suites deliberately avoid.
"""

import unittest

import numpy as np

from methods.se2_icp import GncSchedule, _gnc_weight, icp_se2
from tests.helpers import depth_sensor, se2_pose, unweighted_sensor

# The shipped capture radius (Optuna study VSAC_NullOff_M2 trial #33).
CAPTURE = 0.13768813892484938

# Fixtures sit around the origin; the sensor views them from 3 m along -x, so
# c_bar ~ 0.0148 m there -- the constant this suite used before the noise model
# became per-point.
SENSOR = depth_sensor()
NOISE_FLOOR = 0.0148

# The control arm: hard truncation, i.e. what ran before GNC. See
# tests.helpers.unweighted_sensor for why a preposterous sigma_disparity is the
# honest way to express it.
NO_GNC = unweighted_sensor()


class TestGncSchedule(unittest.TestCase):
    """
    The anneal must always fit the iteration budget, because overrunning it
    silently means the kernel never reaches mu = 1 and the solver does no
    outlier rejection at all -- while still returning a pose.
    """

    def test_anneals_down_to_one(self):
        s = GncSchedule(shrink=1.4).iteration_mu(500.0, 100)
        self.assertAlmostEqual(s[0], 500.0)
        self.assertAlmostEqual(s[-1], 1.0)
        # Non-increasing: a schedule that ever widens would undo annealing.
        self.assertTrue(all(a >= b for a, b in zip(s, s[1:], strict=False)))

    def test_step_count_matches_the_closed_form(self):
        """
        The anneal is ceil(log(mu_0) / log(shrink)) + 1 entries long -- one per
        division plus the final clamp to 1 -- after which the schedule holds at 1
        for the rest of the budget. Asserted so a future change to `shrink`
        cannot silently alter the anneal length.
        """
        sched = GncSchedule(shrink=1.4)
        mu_init = 500.0
        expected = int(np.ceil(np.log(mu_init) / np.log(1.4))) + 1
        s = sched.iteration_mu(mu_init, 100)
        self.assertEqual(len([x for x in s if x > 1.0]) + 1, expected)
        self.assertEqual(s[expected - 1], 1.0)

    def test_length_is_always_the_budget(self):
        gnc = GncSchedule()
        for mu_init, budget in [(500.0, 100), (900.0, 20), (900.0, 2), (0.5, 30), (50.0, 1)]:
            with self.subTest(mu_init=mu_init, budget=budget):
                self.assertEqual(len(gnc.iteration_mu(mu_init, budget)), budget)

    def test_residuals_inside_the_noise_bound_leave_nothing_to_anneal(self):
        """mu_0 <= 1 means every residual is already noise-plausible."""
        self.assertEqual(GncSchedule().iteration_mu(0.5, 4), [1.0] * 4)

    def test_rejects_a_schedule_that_cannot_anneal(self):
        with self.assertRaises(ValueError):
            GncSchedule(shrink=1.0)

    def test_tight_budget_clamps_instead_of_raising(self):
        """
        mu = 900 -> 1 at shrink 1.4 wants 21 entries. A 10-iteration budget must
        anneal faster and still reach 1, loudly. Raising would be wrong on a
        robot: a misconfiguration should degrade the anneal, not kill the frame.
        """
        gnc = GncSchedule()
        with self.assertLogs(level="WARNING") as captured:
            s = gnc.iteration_mu(900.0, 10)
        self.assertIn("effective shrink", "".join(captured.output))
        self.assertEqual(len(s), 10)
        self.assertEqual(s[0], 900.0)
        self.assertEqual(s[-1], 1.0)
        self.assertGreater(s[0] / s[1], gnc.shrink)

    def test_schedule_ends_exactly_on_one(self):
        """The loop's `mu == _MU_FINAL` test is exact, so the tail must be copied."""
        gnc = GncSchedule()
        for mu_init, budget in [(500.0, 100), (900.0, 10), (900.0, 3)]:
            with self.subTest(mu_init=mu_init, budget=budget):
                self.assertEqual(gnc.iteration_mu(mu_init, budget)[-1], 1.0)

    def test_zero_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            GncSchedule().iteration_mu(500.0, 0)


class TestGemanMcClureWeights(unittest.TestCase):
    def test_weight_landmarks(self):
        """At mu = 1 the effective scale is c_bar: w(0)=1, w(c)=1/4, w(3c)=1/100."""
        c = np.full(3, 0.02)
        w = _gnc_weight(np.array([0.0, 0.02, 0.06]), 1.0, c)
        np.testing.assert_allclose(w, [1.0, 0.25, 0.01], atol=1e-12)

    def test_mu_scales_the_kernel_as_sqrt(self):
        """s^2 = mu c^2, so mu = 4 must behave exactly like a doubled scale."""
        c = np.full(3, 0.02)
        r = np.array([0.0, 0.04, 0.12])
        np.testing.assert_allclose(_gnc_weight(r, 4.0, c), _gnc_weight(r, 1.0, 2 * c), atol=1e-15)

    def test_weights_are_even_in_the_residual_sign(self):
        """A point-to-plane residual is signed; the kernel must not prefer one side."""
        w = _gnc_weight(np.array([-0.03, 0.03]), 1.0, np.full(2, 0.02))
        self.assertAlmostEqual(w[0], w[1])

    def test_per_point_bounds_are_applied_per_point(self):
        """
        The whole reason the bound became an array: the same residual must be
        judged differently at different ranges. A point twice as far has 4x the
        noise bound and so keeps far more weight for an identical residual.
        """
        r = np.full(2, 0.03)
        w = _gnc_weight(r, 1.0, np.array([0.0148, 4 * 0.0148]))
        self.assertLess(w[0], 0.1)
        self.assertGreater(w[1], 0.6)


class TestJobSeparation(unittest.TestCase):
    """
    An L-shaped surface of which the model carries a SECOND, fictional copy
    offset 0.10 m along each face normal -- the zero-thickness-shell far sheet
    of vault note 30.06, in synthetic form. The scene contains only the near
    sheet, so every far-sheet model point is an outlier with no true
    counterpart, and at the optimum its nearest scene point sits 0.10 m away.

    Two perpendicular faces rather than one: with a single plane the yaw and
    the along-face translation are unobservable, so the 3x3 normal equations
    would be rank 1 and the test would be measuring the regulariser. That case
    gets its own suite, TestDegenerateGeometry.
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

    def _run(self, max_correspondence_distance, sensor):
        result = icp_se2(
            model_points=self.model_points,
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=self.T_init,
            max_correspondence_distance=max_correspondence_distance,
            sensor=sensor,
            gnc=GncSchedule(),
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
        trans_err, _, result = self._run(CAPTURE, NO_GNC)
        self.assertGreater(trans_err, 0.02, "the far-sheet bias this change removes has gone")
        self.assertGreater(
            result.effective_inlier_fraction, 0.5, "it did converge -- to the wrong pose"
        )

    def test_tight_hard_truncation_starves_the_association(self):
        """
        Shrinking the single parameter to the noise floor is the obvious fix and
        it does not work. From 3 cm out only the handful of model points that
        happen to start within 0.0148 m retrieve a neighbour at all, and the
        solver corrects only the directions those survivors constrain. This is
        Job B failing, and it is why the sweeps could not simply keep shrinking.
        """
        trans_err, _, result = self._run(NOISE_FLOOR, NO_GNC)
        self.assertLess(result.effective_inlier_fraction, 0.10)
        self.assertGreater(trans_err, NOISE_FLOOR)

    def test_gnc_gets_both(self):
        """
        Wide radius for the association, annealed kernel for the weighting: the
        pose lands well inside the noise floor and under a degree in yaw. This
        failing while the two above pass means the annealing is not
        down-weighting the far sheet.
        """
        trans_err, yaw_err, _ = self._run(CAPTURE, SENSOR)
        self.assertLess(trans_err, NOISE_FLOOR / 10.0)
        self.assertLess(yaw_err, 1.0)

    def test_gnc_strictly_beats_the_wide_hard_stage_it_replaces(self):
        """The comparison the change has to win, stated as one assertion."""
        hard_err, _, _ = self._run(CAPTURE, NO_GNC)
        gnc_err, _, _ = self._run(CAPTURE, SENSOR)
        self.assertLess(gnc_err, hard_err / 10.0)

    def test_effective_fraction_is_below_the_hard_count(self):
        """
        The soft fitness must read LOWER than the hard one on identical
        geometry, because points inside the radius now contribute w < 1 rather
        than 1. If it ever reads equal, the weights are not reaching the score.
        """
        _, _, hard = self._run(CAPTURE, NO_GNC)
        _, _, soft = self._run(CAPTURE, SENSOR)
        self.assertLess(soft.effective_inlier_fraction, hard.effective_inlier_fraction)

    def test_robust_rmse_is_a_point_to_plane_residual(self):
        """
        Reported in the metric the cost function minimises, so it is comparable
        against the kernel scale -- which the old Euclidean inlier_rmse was not.
        A converged fit on a noiseless fixture must sit far below c_bar.
        """
        _, _, result = self._run(CAPTURE, SENSOR)
        self.assertLess(result.robust_rmse, result.median_kernel_scale)
        self.assertAlmostEqual(result.median_kernel_scale, NOISE_FLOOR, delta=0.02)


class TestOptimalFixedRadiusIsNotKnowable(unittest.TestCase):
    """
    The argument for GNC, as a test.

    A fixed radius is not merely hard to tune -- its optimum is a function of
    how far off the GLOBAL stage left the pose, which varies frame to frame and
    is unknown when the config is written. Scanning 25 radii at each of six
    initialisation errors, the best radius climbs in step with the error. GNC's
    error does not move at all across the same range.

    If test_gnc_is_flat_across_the_same_range ever fails because GNC has become
    init-dependent, the annealing is starting too tight -- check _initial_mu.
    """

    INIT_OFFSETS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    RADIUS_GRID = tuple(np.geomspace(0.01, 0.25, 25))

    def setUp(self):
        self.fixture = TestJobSeparation
        self.fixture.setUpClass()

    def _err(self, T_init, radius, sensor):
        result = icp_se2(
            model_points=self.fixture.model_points,
            scene_points=self.fixture.scene_points,
            scene_normals=self.fixture.scene_normals,
            T_init=T_init,
            max_correspondence_distance=radius,
            sensor=sensor,
            gnc=GncSchedule(),
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
        errs = [self._err(T_init, CAPTURE, SENSOR) for _, T_init in self._inits()]
        self.assertLess(max(errs), NOISE_FLOOR / 10.0, f"GNC lost accuracy somewhere: {errs}")
        self.assertLess(max(errs) - min(errs), 1e-3, f"GNC became init-dependent: {errs}")

    def test_a_radius_tuned_at_one_offset_fails_at_another(self):
        """The concrete cost of shipping a single number, in meters."""
        inits = dict(self._inits())
        tuned_here = 0.0196  # optimal at a 3 cm initialisation error
        self.assertLess(self._err(inits[0.03], tuned_here, NO_GNC), 1e-3)
        self.assertGreater(self._err(inits[0.12], tuned_here, NO_GNC), 0.10)

    def test_gnc_error_is_flat_across_the_radius(self):
        """
        The theory that lets the capture radius be DERIVED rather than swept:
        under GNC the error as a function of the radius should be a step, not a
        peak -- bad below some knee, flat above, because everything above the
        knee is the kernel's job.

        If this is flat, a sweep maximising over that plateau is returning a draw
        from evaluation noise, and the radius belongs in a formula instead. If it
        is NOT flat, that argument collapses and the radius must stay tuned.
        """
        _off, T_init = next(iter(self._inits()))
        T_init = se2_pose(np.radians(2.0), np.array([0.03, 0.02]), 0.0)
        errs = {r: self._err(T_init, r, SENSOR) for r in self.RADIUS_GRID}
        # The knee sits where the radius stops covering the 3 cm init error.
        above = [e for r, e in errs.items() if r >= 0.06]
        self.assertGreater(len(above), 5, "grid too coarse above the knee to say anything")
        self.assertLess(
            max(above) - min(above),
            1e-3,
            f"no plateau -- the radius still matters under GNC: {errs}",
        )


class TestDegenerateGeometry(unittest.TestCase):
    """
    A single plane, where the along-face slide and the yaw are both unobservable
    and the 3x3 normal equations are rank 1.

    This is not hypothetical here: after the visibility cull a cart seen head-on
    can leave a model cloud dominated by its front face. np.linalg.solve raises
    only on EXACT singularity, so before the relative LM damping such a frame
    returned a large increment in an unconstrained direction, silently, and the
    composition applied it.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        n = 800
        # The plane x = 0, outward normal -x. Only x is observable.
        cls.scene_points = np.column_stack(
            [np.zeros(n), rng.uniform(-1, 1, n), rng.uniform(0, 1, n)]
        )
        cls.scene_normals = np.tile([-1.0, 0.0, 0.0], (n, 1))
        cls.model_points = cls.scene_points.copy()

    def _run(self):
        return icp_se2(
            model_points=self.model_points,
            scene_points=self.scene_points,
            scene_normals=self.scene_normals,
            T_init=se2_pose(0.0, np.array([0.03, 0.0]), 0.0),
            max_correspondence_distance=CAPTURE,
            sensor=SENSOR,
            gnc=GncSchedule(),
            max_iterations=50,
        )

    def test_corrects_the_observable_direction(self):
        T = self._run().transformation
        self.assertLess(abs(T[0, 3]), NOISE_FLOOR)

    def test_does_not_run_away_along_the_unobservable_ones(self):
        """
        The assertion the damping exists for. With the previous absolute
        1e-12 ridge -- below rounding on a matrix with O(1) entries -- the
        unconstrained slide and yaw were free to take an arbitrary value.
        """
        T = self._run().transformation
        self.assertLess(abs(T[1, 3]), 1e-3, "the unobservable slide moved")
        yaw = abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        self.assertLess(yaw, 0.1, "the unobservable yaw moved")

    def test_the_degeneracy_is_reported(self):
        """A silently-degenerate frame is the failure mode; it must be loud."""
        with self.assertLogs(level="WARNING") as captured:
            self._run()
        self.assertIn("near-degenerate", "".join(captured.output))


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
            sensor=SENSOR,
            gnc=GncSchedule(),
            max_iterations=100,
        )
        T = result.transformation
        np.testing.assert_array_equal(T[2], [0.0, 0.0, 1.0, z_offset])
        np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
        R2 = T[:2, :2]
        np.testing.assert_allclose(R2.T @ R2, np.eye(2), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
