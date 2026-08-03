# =====================================================================
# 0. POSE ERROR METRICS & DECOMPOSITION MATH
# =====================================================================
import csv
import dataclasses
import os

import numpy as np
from pydantic import BaseModel, Field


class PoseErrorMetrics(BaseModel):
    trans_xy: float = Field(ge=0.0)
    trans_z: float
    yaw: float = Field(ge=-180.0, le=180.0)
    pitch: float = Field(ge=-90.0, le=90.0)
    roll: float = Field(ge=-180.0, le=180.0)
    geodesic_rot: float = Field(ge=0.0, le=180.0)


def extract_pose_errors(T_est: np.ndarray, T_gt: np.ndarray) -> PoseErrorMetrics:
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    trans_xy = float(np.linalg.norm(t_est[:2] - t_gt[:2]))
    trans_z = float(t_est[2] - t_gt[2])

    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]

    # Error rotation matrix from ground truth to estimate
    R_err = R_gt.T @ R_est

    r11, _r12, _r13 = R_err[0, 0], R_err[0, 1], R_err[0, 2]
    r21, _r22, _r23 = R_err[1, 0], R_err[1, 1], R_err[1, 2]

    r31, r32, r33 = R_err[2, 0], R_err[2, 1], R_err[2, 2]

    # Extract pitch (asin(-r31)), roll (atan2(r32, r33)), yaw (atan2(r21, r11))
    pitch_rad = np.arcsin(np.clip(-r31, -1.0, 1.0))
    roll_rad = np.arctan2(r32, r33)
    yaw_rad = np.arctan2(r21, r11)

    pitch = float(np.degrees(pitch_rad))
    roll = float(np.degrees(roll_rad))
    yaw = float(np.degrees(yaw_rad))

    # Standard geodesic rotation error
    trace_val = np.trace(R_err)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    geodesic_rot = float(np.degrees(np.arccos(cos_theta)))

    return PoseErrorMetrics(
        trans_xy=trans_xy,
        trans_z=trans_z,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        geodesic_rot=geodesic_rot,
    )


def compute_average_recall(errors: list[PoseErrorMetrics], total_samples: int) -> float:
    """
    Computes a BOP-style Average Recall (AR) metric over a grid of error thresholds.

    Why a BOP-style loss?
    --------------------
    1. Prevents Selection Bias:
       Evaluating only the mean error of successful matches introduces selection bias, where
       an estimator that fails on all but one easy sample can achieve a misleadingly low mean error.
       This metric penalizes failures implicitly by treating them as misses (0 recall contribution).
    2. Avoids Arbitrary Magic Multipliers:
       Traditional objectives like `mean_trans + alpha * mean_rot + beta * failures` mix units (meters,
       degrees, counts) and rely on arbitrary scaling weights (e.g., 5.0 for failure). Average Recall
       normalizes all parameters to a bounded [0, 1] range, ensuring equal scaling.
    3. Smooth Optimization Landscape for Optuna TPE:
       Using a single threshold (e.g., <1cm and <2°) results in a step-like discontinuous objective function
       (e.g., 21 discrete values for 20 samples). This is highly hostile to Bayesian optimization.
       Averaging recall over a grid of 42 threshold combinations (7 translation x 6 rotation) creates
       a much smoother landscape, assisting the Tree-structured Parzen Estimator (TPE) search.
    4. Focus on Ground-Plane Constraints (XY + Yaw):
       Since the towing cart is physically floor-bound, the pitch, roll, and Z-axis translation are
       highly constrained by gravity and the ground surface. Therefore, we focus the success criteria
       strictly on `trans_xy` (ground-plane drift) and `yaw` (heading orientation).

    Args:
        errors (list[PoseErrorMetrics]): Computed pose errors for successfully registered samples.
        total_samples (int): Denominator -- MUST be n_attempted (samples that
            actually reached the estimator), not len(errors). Passing the
            success count instead is the selection bias this metric exists to
            avoid: an estimator that abstains on every hard frame would score
            perfectly. See compute_trial_metrics.

    Returns:
        float: The average recall score in the range [0.0, 1.0].
    """
    if total_samples <= 0:
        return 0.0

    translation_thresholds = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    rotation_thresholds = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    recalls = []
    for t_thresh in translation_thresholds:
        for r_thresh in rotation_thresholds:
            successes = sum(1 for e in errors if e.trans_xy < t_thresh and abs(e.yaw) < r_thresh)
            recalls.append(successes / total_samples)

    return float(np.mean(recalls))


# Yaw error above which a returned pose is counted as grossly mis-oriented.
# Deliberately equal to the LARGEST rotation threshold in compute_average_recall's
# grid: that makes gross_yaw_rate and pose_ar answer the same question, and it is
# what makes the `pose_ar <= good_rate` invariant below hold. The old 90 degree
# threshold answered a different question than the objective did, so the two
# could drift apart without anyone noticing.
GROSS_YAW_DEG = 15.0


@dataclasses.dataclass(frozen=True)
class TrialMetrics:
    """The five headline metrics for one trial, plus a small diagnostics tail.

    The three estimator rates share ONE denominator, n_attempted, so they
    partition every frame the estimator was actually handed:

        good_rate + gross_yaw_rate + abstention_rate == 1.0

    That invariant is the entire point. The previous scheme divided flip_rate by
    the SUCCESS count while dividing average_recall by the evaluated count, so
    an estimator could drive its flip rate to zero purely by refusing to return
    poses -- and did: VSAC run fgugxxrn logged flip_rate=0.0 on trials with 207
    pose failures out of 210 frames.

    YOLO detection failures are excluded from n_attempted and reported
    separately: they happen upstream of the estimator and are invariant to the
    swept parameters, so folding them in would just add a constant that dilutes
    every estimator rate.
    """

    # --- the five ---
    pose_ar: float  # objective 1, maximize
    p95_latency_s: float  # objective 2, minimize (inf when nothing succeeded)
    gross_yaw_rate: float
    abstention_rate: float
    detection_failure_rate: float

    # --- diagnostics (never objectives) ---
    n_eval: int
    n_attempted: int
    good_rate: float
    # Conditional on GOOD samples only -- readable error magnitudes for humans,
    # useless as objectives precisely because of that conditioning.
    trans_xy_p50: float | None
    yaw_p50: float | None


def compute_trial_metrics(
    errors: list[PoseErrorMetrics],
    times: list[float],
    detection_failures: int,
    pose_failures: int,
) -> TrialMetrics:
    """
    Reduces one trial's raw per-frame outcomes to the five headline metrics.

    Args:
        errors: Pose errors for frames that produced a pose AND a GT comparison.
        times: Wall-clock estimation seconds, one per entry in `errors`.
        detection_failures: Frames where YOLO found no cart (upstream).
        pose_failures: Frames that reached the estimator but produced no pose.

    Returns:
        TrialMetrics: see that class for the invariants it guarantees.
    """
    n_matched = len(errors)
    n_attempted = n_matched + pose_failures
    n_eval = n_attempted + detection_failures

    if n_attempted == 0:
        # Nothing ever reached the estimator -- every estimator-scoped rate is
        # undefined rather than zero. Report the worst possible objective so a
        # degenerate trial can never win the sweep.
        return TrialMetrics(
            pose_ar=0.0,
            p95_latency_s=float("inf"),
            gross_yaw_rate=0.0,
            abstention_rate=0.0,
            detection_failure_rate=(detection_failures / n_eval) if n_eval else 0.0,
            n_eval=n_eval,
            n_attempted=0,
            good_rate=0.0,
            trans_xy_p50=None,
            yaw_p50=None,
        )

    good = [e for e in errors if abs(e.yaw) <= GROSS_YAW_DEG]
    n_gross = n_matched - len(good)

    pose_ar = compute_average_recall(errors, n_attempted)
    good_rate = len(good) / n_attempted
    gross_yaw_rate = n_gross / n_attempted
    abstention_rate = pose_failures / n_attempted

    # The two invariants, asserted rather than documented-and-hoped-for.
    assert abs((good_rate + gross_yaw_rate + abstention_rate) - 1.0) < 1e-9, (
        "estimator rates must partition n_attempted: "
        f"{good_rate} + {gross_yaw_rate} + {abstention_rate} != 1"
    )
    assert pose_ar <= good_rate + 1e-9, (
        f"pose_ar ({pose_ar}) exceeded good_rate ({good_rate}); every AR grid cell "
        f"requires |yaw| < {GROSS_YAW_DEG}, so this means the denominators diverged"
    )

    return TrialMetrics(
        pose_ar=pose_ar,
        p95_latency_s=float(np.percentile(times, 95)) if times else float("inf"),
        gross_yaw_rate=gross_yaw_rate,
        abstention_rate=abstention_rate,
        detection_failure_rate=(detection_failures / n_eval) if n_eval else 0.0,
        n_eval=n_eval,
        n_attempted=n_attempted,
        good_rate=good_rate,
        trans_xy_p50=float(np.median([e.trans_xy for e in good])) if good else None,
        yaw_p50=float(np.median([abs(e.yaw) for e in good])) if good else None,
    )


def finite_or_none(value: float) -> float | None:
    """W&B stores float('inf') as the string "Infinity", which breaks any chart
    the metric appears in. Log None instead -- an absent point, which is what a
    trial that never completed an estimation actually has."""
    return None if not np.isfinite(value) else float(value)


@dataclasses.dataclass
class FrameRecord:
    """
    One evaluated frame's outcome, for offline per-frame auditing -- e.g.
    "is gross_yaw_rate driven by the same handful of frames every time?" (the
    question that motivated this: a sweep's minimum flip rate recurring
    identically across trials with no way to tell whether it was the same
    frames or not).

    fitness_1/2, viol_ratio_1/2, selected, and decision come from whatever
    flip-disambiguation diagnostics the estimator exposes via a
    `_last_diagnostics` attribute (see Ransac3DoFEstimator._refine_pose,
    methods/ransac3dof.py) -- estimators that don't set one (PPF, the SE(3)
    RansacEstimator) just leave those fields None.

    FAILED frames get a record too, with outcome != "good" and the pose fields
    left None. They used to be dropped on the floor (evaluate_pipeline simply
    `continue`d), which meant the CSV could not answer the one question worth
    asking of an abstaining estimator: which check rejected these frames?
    """

    sample_idx: int
    cart_type: str | None
    # "good" | "gross_yaw" | "abstained" | "not_detected" -- mirrors the rate
    # partition in TrialMetrics, so grouping the CSV by this column reproduces
    # good_rate / gross_yaw_rate / abstention_rate exactly.
    outcome: str
    # Named cause, abstentions only (see methods/constrained_ransac.py's
    # ABSTENTION_REASONS). None for frames that produced a pose.
    failure_reason: str | None = None
    flipped: bool | None = None
    trans_xy: float | None = None
    yaw_err: float | None = None
    fitness: float | None = None
    inlier_rmse: float | None = None
    # Signed angle between the final pose's towing-face arrow and the direction
    # to the camera. Signed on purpose: a cluster near +/-180 is a flip that got
    # through, a spread around +/-30 is ordinary yaw error, and the unsigned
    # magnitude would merge two failures that call for different fixes.
    front_face_angle_deg: float | None = None
    # How far ICP rotated the pose after the gate approved it. ICP is
    # unconstrained in yaw, so this is what says whether a candidate accepted at
    # the boundary can be walked across it -- and what would size a bound.
    icp_yaw_delta_deg: float | None = None


def write_frame_records_csv(path: str, records: list[FrameRecord]) -> None:
    """Dumps per-frame records to CSV for offline analysis (pandas/notebook)."""
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [f.name for f in dataclasses.fields(FrameRecord)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(dataclasses.asdict(r))
