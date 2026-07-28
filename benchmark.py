"""
6D Pose Estimation Benchmark & Hyperparameter Tuning Suite.

This script evaluates and optimizes 6D Pose Estimation algorithms (such as PPF and RANSAC)
against ground truth labels from Isaac Sim. It runs in two distinct modes:

Usage Modes:
------------
1. Default Benchmark Evaluation (--sweep not passed, or --no-sweep)
   Evaluates a chosen method on a random subset of the test split using its default/optimized
   hyperparameters, printing a detailed performance report (Mean/Median Translation & Rotation
   errors, Match Success Rate, and Mean Execution time).

   Example (PPF):
     uv run benchmark.py --eval-size 30 model:ppf model.profile:default

   Example (RANSAC):
     uv run benchmark.py --eval-size 30 model:ransac model.profile:default

2. Hyperparameter Sweep Optimization (--sweep)
   Launches a Multi-Objective Bayesian Optimization sweep using Optuna to find the Pareto Front
   of optimal accuracy vs. speed trade-offs. The search space is dynamically configured based
   on the selected method, and results are persisted in a local SQLite database for visualization.

   Example (PPF Sweep):
     uv run benchmark.py --sweep --name PPF_Sweep --trials 50 --eval-size 30 model:ppf model.profile:default

   Example (RANSAC Sweep):
     uv run benchmark.py --sweep --name RANSAC_Sweep --trials 50 --eval-size 30 model:ransac model.profile:default

CLI Configuration Overrides:
----------------------------
  model:ppf|ransac|ransac3dof   The 6D pose estimation method to run (required, no default --
                                see docs/explanation/tyro_cli_config.md for why).
  model.profile:<name>          Tuning profile for the chosen method (required); run
                                `uv run benchmark.py model:<algo> --help` to list them.
  --sweep / --no-sweep          Flag to execute the Optuna hyperparameter sweep (default: no-sweep).
  --name NAME                   Name for this run -- the Optuna study (-> 'optuna_<NAME>.db')
                                in sweep mode, or this benchmark's W&B run name otherwise.
  --trials NUM                  Number of optimization trials to execute (default: 30).
  --eval-size NUM                Number of validation samples to evaluate per trial/benchmark (default: 20).
  --seed NUM                    Optional fixed seed to ensure reproducibility.
  --model.profile.params.<field> <value>   Override a single hyperparameter of the chosen profile.
  --model.profile.depth-trunc <value>      Override the chosen profile's depth truncation.
"""

import ast
import csv
import dataclasses
import glob
import logging
import os
import time

import numpy as np
import open3d as o3d
import optuna
import plotly.graph_objects as go
import tyro
from datasets import Dataset
from pydantic import BaseModel, Field

import wandb
from cli_config import BenchmarkArgs
from methods.base import BasePoseEstimator
from pipeline import (
    Camera,
    compute_ground_truth_pose,
    instance_detected,
    load_cad_meshes,
    load_hf_model,
    load_parquet_dataset,
    process_and_reconstruct,
)

# tyro hands override values through as raw strings; ast.literal_eval wants the
# Python spellings, so these three are title-cased before parsing.
_BOOLS = {"true", "false", "none"}


# =====================================================================
# 0. POSE ERROR METRICS & DECOMPOSITION MATH
# =====================================================================
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


# Set logging level for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def draw_eval_indices(total_samples: int, size: int, seed: int) -> list[int]:
    """
    Draws a deterministic evaluation subset from the dataset.

    Uses the modern Generator API exclusively (np.random.default_rng) so a
    given seed selects the same frames regardless of call site. Sweep and
    single-run evaluation used to draw from two different PRNG streams
    (default_rng's PCG64 vs. the legacy np.random.seed/np.random.choice
    global MT19937 state) -- same seed value, silently different frames,
    which made cross-run comparisons (e.g. an A/B between two estimators)
    untrustworthy unless both happened to use the same code path.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(total_samples, min(size, total_samples), replace=False).tolist()


def derive_internal_seeds(base_seed: int, n: int, salt: int = 0) -> list[int]:
    """
    Derives n distinct, reproducible estimator-internal RANSAC seeds from
    base_seed (+ an optional salt, e.g. an Optuna trial number, so different
    trials/calls don't accidentally share the same repeat seeds).

    Deliberately NOT exposed as an Optuna trial.suggest_int(...) search
    dimension: a pure-noise dimension has no smooth relationship to the
    objective and would mislead TPE/NSGA-II's surrogate model. Instead, the
    caller evaluates across these seeds and pools the results, making the
    search (or a final report) robust to seed luck instead of resting on a
    single, possibly-lucky draw.
    """
    rng = np.random.default_rng([base_seed, salt])
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


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
    selected: str | None = None
    decision: str | None = None
    fitness_1: float | None = None
    fitness_2: float | None = None
    viol_ratio_1: float | None = None
    viol_ratio_2: float | None = None


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


# =====================================================================
# 2. CORE EVALUATION PIPELINE
# =====================================================================
def evaluate_pipeline(
    dataset: Dataset,
    model,
    camera: Camera,
    estimator: BasePoseEstimator,
    sample_indices: list[int],
    meshes: dict[str, o3d.geometry.TriangleMesh],
    depth_trunc: float = 3.0,
) -> tuple[list[PoseErrorMetrics], list[float], int, int, list[FrameRecord]]:
    """
    Evaluates the pose estimation pipeline on a set of sample indices.

    Returns:
        tuple containing:
            - error_metrics (list[PoseErrorMetrics]): Decomposed errors for successfully matched samples.
            - times (list[float]): Pose estimation duration in seconds for each successful sample.
            - detection_failures (int): Count of samples where YOLO failed to detect a cart.
            - pose_failures (int): Count of samples where pose estimation failed.
            - frame_records (list[FrameRecord]): Per-frame outcome + failure cause +
              flip-disambiguation diagnostics for EVERY sample, for offline auditing.
    """
    error_metrics = []
    times = []
    frame_records = []
    detection_failures = 0
    pose_failures = 0

    def record_failure(sample_idx: int, cart_type: str | None, outcome: str, reason: str) -> None:
        frame_records.append(
            FrameRecord(
                sample_idx=int(sample_idx),
                cart_type=cart_type,
                outcome=outcome,
                failure_reason=reason,
            )
        )

    for sample_idx in sample_indices:
        row = dataset[int(sample_idx)]
        img = row["rgb"]
        depth_bytes = row["depth"]

        # 1. Run YOLO detection
        # We execute the 2D instance segmentation model. If no target cart instance
        # is detected, we cannot proceed with point cloud reconstruction or pose
        # registration, so we log it as a detection failure and continue to the next sample.
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            detection_failures += 1
            record_failure(sample_idx, None, "not_detected", "yolo_no_instance")
            continue

        # 2. Segment and Reconstruct Point Cloud
        try:
            cart_type, pcd, frame = process_and_reconstruct(
                img, depth_bytes, result, camera, depth_trunc=depth_trunc, return_frame=True
            )
        except Exception:
            logging.exception(f"PointCloud processing failed for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, None, "abstained", "pointcloud_error")
            continue

        # Retrieve preloaded mesh
        cad_mesh = meshes.get(cart_type)
        if cad_mesh is None:
            logging.error(f"CAD mesh not found for cart type '{cart_type}' (sample {sample_idx})")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "mesh_missing")
            continue

        # 3. Perform 6D Pose Estimation with timing
        start_time = time.time()
        try:
            T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=cart_type, frame=frame)
            if T_final is None:
                # The estimator names WHICH check rejected the frame (see
                # RansacResult.reason -> RansacEstimator._last_failure_reason).
                # Estimators that aren't instrumented fall back to "estimator_none".
                reason = getattr(estimator, "_last_failure_reason", None) or "estimator_none"
                logging.error(f"Pose estimator abstained on index {sample_idx}: {reason}")
                pose_failures += 1
                record_failure(sample_idx, cart_type, "abstained", reason)
                continue
        except Exception:
            logging.exception(f"Pose estimator raised an exception for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "estimator_exception")
            continue
        elapsed_time = time.time() - start_time

        # 4. Calculate Ground Truth pose and compare
        extrinsic = getattr(estimator, "extrinsic", None)
        if extrinsic is None:
            raise ValueError(
                "Estimator must have an extrinsic camera-to-robot transform configured."
            )

        try:
            T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
            T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
            T_ground_truth = compute_ground_truth_pose(
                T_world_camera, T_world_cart, T_robot_camera=extrinsic
            )
            metrics = extract_pose_errors(T_final, T_ground_truth)
            error_metrics.append(metrics)
            times.append(elapsed_time)

            diagnostics = getattr(estimator, "_last_diagnostics", None) or {}
            frame_records.append(
                FrameRecord(
                    sample_idx=int(sample_idx),
                    cart_type=cart_type,
                    outcome="good" if abs(metrics.yaw) <= GROSS_YAW_DEG else "gross_yaw",
                    # Kept alongside `outcome` on purpose: `flipped` is the
                    # near-180 degree failure mode specifically, while a
                    # "gross_yaw" outcome also catches merely-imprecise poses
                    # in the 15-90 degree band. Separating them is how we tell
                    # "still flipping" from "converging badly".
                    flipped=abs(metrics.yaw) > 90.0,
                    trans_xy=metrics.trans_xy,
                    yaw_err=metrics.yaw,
                    selected=diagnostics.get("selected"),
                    decision=diagnostics.get("decision"),
                    fitness_1=diagnostics.get("fitness_1"),
                    fitness_2=diagnostics.get("fitness_2"),
                    viol_ratio_1=diagnostics.get("viol_ratio_1"),
                    viol_ratio_2=diagnostics.get("viol_ratio_2"),
                )
            )
        except Exception:
            logging.exception(f"Error metric extraction failed for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "metric_error")
            continue

    return error_metrics, times, detection_failures, pose_failures, frame_records


# =====================================================================
# 3. OPTUNA SWEEP STUDY (MULTI-OBJECTIVE OPTIMIZATION)
# =====================================================================
def log_input_artifacts(run, yolo_cfg, dataset_cfg):
    """Register the YOLO weights and dataset as W&B *reference* artifacts.

    checksum=True records a content hash of each referenced file -- no bytes are
    uploaded, so an unchanged model/dataset produces no new version while any
    modification mints one. That version bump is the "was this input touched?"
    signal for the cross-commit report.
    """
    model_art = wandb.Artifact(
        "yolo-detector", type="model", metadata={"hf_repo": yolo_cfg.repo, "hf_file": yolo_cfg.file}
    )
    model_art.add_reference(f"file://{os.path.abspath(yolo_cfg.local_path)}", checksum=True)
    run.log_artifact(model_art)

    dataset_art = wandb.Artifact(
        "dataset",
        type="dataset",
        metadata={
            "path": dataset_cfg.path,
            "train_glob": dataset_cfg.train_glob,
            "val_glob": dataset_cfg.val_glob,
            "test_glob": dataset_cfg.test_glob,
        },
    )
    for glob_pattern in (dataset_cfg.train_glob, dataset_cfg.val_glob, dataset_cfg.test_glob):
        for shard in sorted(glob.glob(glob_pattern)):
            dataset_art.add_reference(f"file://{os.path.abspath(shard)}", checksum=True)
    run.log_artifact(dataset_art)


def build_pareto_figure(pareto, dominated, param_names, study_name):
    """Build the interactive Pareto-front scatter logged to W&B at sweep end.

    x = p95 latency (objective 2, minimized), y = pose_ar (objective 1, MAXIMIZED),
    so the bottom-right corner is best. Dominated trials form a recessive
    gray field, Pareto-optimal trials (study.best_trials) are highlighted in blue and
    connected by the frontier line. Every point's hover carries its trial number and
    full hyperparameter set, so any point on the frontier is traceable straight back
    to the iteration and config that produced it -- which a bare wandb.plot.scatter
    (fixed tooltip, x/y only) cannot do.
    """
    BLUE, GRAY, GRID, AXIS = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7"
    INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

    # customdata columns, in order:
    # trial_number, *params, gross_yaw_rate, abstention_rate.
    # Both failure rates ride along in the tooltip so a frontier point that looks
    # accurate can be checked for how much it simply declined to answer.
    def customdata(trials):
        return [
            [t.number]
            + [t.params.get(name) for name in param_names]
            + [t.user_attrs.get("gross_yaw_rate"), t.user_attrs.get("abstention_rate")]
            for t in trials
        ]

    hover_lines = [
        "<b>Trial %{customdata[0]}</b>",
        "pose_ar: %{y:.4f}",
        "p95 latency: %{x:.3f}s",
    ]
    for i, name in enumerate(param_names, start=1):
        hover_lines.append(f"{name}: %{{customdata[{i}]}}")
    hover_lines.append(f"gross_yaw_rate: %{{customdata[{len(param_names) + 1}]:.3f}}")
    hover_lines.append(f"abstention_rate: %{{customdata[{len(param_names) + 2}]:.3f}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure()

    # Dominated trials -- recessive gray field, drawn first (underneath).
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in dominated],
            y=[t.values[0] for t in dominated],
            mode="markers",
            name="Dominated trials",
            marker=dict(color=GRAY, size=8, opacity=0.55),
            customdata=customdata(dominated),
            hovertemplate=hovertemplate,
        )
    )

    # Pareto frontier -- straight line through the optimal trials sorted by latency.
    pareto_sorted = sorted(pareto, key=lambda t: t.values[1])
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in pareto_sorted],
            y=[t.values[0] for t in pareto_sorted],
            mode="lines",
            name="Pareto frontier",
            line=dict(color=BLUE, width=2),
            hoverinfo="skip",
        )
    )

    # Pareto-optimal trials -- highlighted, drawn on top.
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in pareto],
            y=[t.values[0] for t in pareto],
            mode="markers",
            name="Pareto-optimal",
            marker=dict(color=BLUE, size=12, line=dict(color=SURFACE, width=1.5)),
            customdata=customdata(pareto),
            hovertemplate=hovertemplate,
        )
    )

    fig.update_layout(
        title=dict(text=f"Pareto Front — {study_name}", font=dict(size=18, color=INK)),
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=13),
        xaxis=dict(
            title="p95 latency (s)  →  slower",
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS,
            ticks="outside",
            tickcolor=AXIS,
        ),
        yaxis=dict(
            title="pose_ar  →  better",
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS,
            ticks="outside",
            tickcolor=AXIS,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        margin=dict(l=70, r=30, t=70, b=90),
        annotations=[
            dict(
                text="← better (fast & accurate)",
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=GRAY, size=12),
            )
        ],
    )
    return fig


def resolve_param_overrides(
    estimator_cls: type[BasePoseEstimator],
    extrinsic: np.ndarray,
    overrides: dict | None,
) -> dict:
    """
    Validates and type-coerces CLI parameter overrides against the estimator's
    own params dataclass.

    An unknown field name raises rather than warning. The whole reason this
    mechanism exists is that a sweep silently ignored parameters set on the
    command line; replacing one silent no-op with another (a warning nobody reads
    in a 200-trial log) would leave the same failure available -- an arm that
    reports it is testing something while running the control.

    Values arrive as strings from tyro and are parsed as Python literals, so
    `free_space_gate=true`, `voxel_size=0.04` and `z_offset=None` all land as the
    right type; anything unparseable is kept as the raw string.
    """
    if not overrides:
        return {}

    probe_params = estimator_cls(extrinsic=extrinsic).params
    valid_fields = {f.name for f in dataclasses.fields(probe_params)}

    resolved = {}
    for name, raw in overrides.items():
        if name not in valid_fields:
            raise ValueError(
                f"Unknown parameter override '{name}' for {estimator_cls.__name__}. "
                f"Available: {sorted(valid_fields)}"
            )
        if isinstance(raw, str):
            try:
                resolved[name] = ast.literal_eval(raw.capitalize() if raw in _BOOLS else raw)
            except (ValueError, SyntaxError):
                resolved[name] = raw
        else:
            resolved[name] = raw
    return resolved


def run_parameter_sweep(
    dataset,
    model,
    camera,
    study_name,
    estimator_cls: type[BasePoseEstimator],
    sweep_size: int,
    n_trials: int,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    yolo_cfg,
    dataset_cfg,
    extrinsic: np.ndarray = None,
    seed: int = None,
    n_seeds: int = 1,
    supports_seed: bool = True,
    dump_frames: bool = True,
    param_overrides: dict | None = None,
):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.

    n_seeds: number of estimator-internal RANSAC seeds each trial is pooled
    over (see derive_internal_seeds). supports_seed: whether the chosen
    estimator's params dataclass actually has a `seed` field to vary --
    when False, n_seeds is ignored (nothing to vary) rather than silently
    re-running identical repeats. dump_frames: write each trial's per-frame
    CSV (see FrameRecord) under sweeps/<study_name>_frames/.
    param_overrides: estimator params pinned for every trial, declaring the arm
    (see BenchmarkArgs.param_overrides). Validated here, before any compute, so a
    typo cannot turn a treatment arm into a silent second copy of the control.
    """
    resolved_overrides = resolve_param_overrides(estimator_cls, extrinsic, param_overrides)
    if resolved_overrides:
        print(f"Parameter overrides pinned for every trial: {resolved_overrides}")

    # Stable, run-independent storage location: restarting the same command
    # after a crash (or on a new instance with the file restored) resumes the
    # study instead of starting a fresh DB in a new Hydra timestamped dir.
    project_root = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(project_root, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    db_name = os.path.join(sweep_dir, f"optuna_{study_name}.db")
    db_url = f"sqlite:///{db_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        # Maximize pose_ar, minimize p95 latency. NOTE this flipped from
        # ["minimize", "minimize"] when the objective stopped being the
        # redundant `accuracy_score = 1 - AR` and became pose_ar itself.
        # Optuna refuses to load a study whose directions changed, so studies
        # created before that switch cannot be resumed -- use a fresh --name.
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    # Retrieve run attributes from the Optuna study's metadata to check if we are resuming
    existing_seed = study.user_attrs.get("seed")
    existing_eval_size = study.user_attrs.get("eval_size")
    existing_indices = study.user_attrs.get("sweep_indices")

    total_samples = len(dataset)

    # CASE 1: Resuming an existing study that has proper validation metadata.
    if existing_seed is not None:
        # Integrity Guard: Ensure the requested sweep size matches what was already evaluated.
        if existing_eval_size != sweep_size:
            raise ValueError(
                f"Validation size mismatch: DB has eval_size={existing_eval_size}, "
                f"but sweep requested eval_size={sweep_size}."
            )
        # Integrity Guard: If a seed was explicitly passed, verify it matches the stored seed.
        if seed is not None and seed != existing_seed:
            raise ValueError(
                f"Seed mismatch: DB has seed={existing_seed}, but sweep requested seed={seed}."
            )
        # Load the existing seed and sample indices to ensure evaluation is on the exact same validation subset.
        seed = existing_seed
        sweep_indices = existing_indices
        print(f"Resuming existing study. Loaded seed={seed}, eval_size={sweep_size}")

    # CASE 2: Fresh study initialization.
    else:
        # Generate a seed if not explicitly provided.
        # We retrieve a high-entropy random integer from the OS (via SeedSequence().entropy)
        # and cast it to fit within a standard 31-bit integer range (modulo 2**31 - 1)
        # to make sure it is a valid seed for all downstream library generators.
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        # Draw indices deterministically using a generator seeded with our selected seed
        sweep_indices = draw_eval_indices(total_samples, sweep_size, seed)

        # Persist attributes in the study so future runs can resume with the same setup
        study.set_user_attr("seed", seed)
        study.set_user_attr("eval_size", sweep_size)
        study.set_user_attr("sweep_indices", sweep_indices)
        print(f"Created new study. Seed={seed}, eval_size={sweep_size}")

    print(f"Sweep validation indices: {sweep_indices}\n")

    # Seed global random number generators
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    if n_seeds > 1 and not supports_seed:
        print(
            f"Note: {estimator_cls.__name__}'s params have no 'seed' field; "
            "--n-seeds > 1 has no effect for this estimator, running each trial once."
        )
    effective_n_seeds = n_seeds if supports_seed else 1

    # ONE W&B run for this whole sweep (1 CLI execution <-> 1 run), not one per
    # trial -- a 200-trial sweep would otherwise flood the workspace with 200
    # separate run pages. Each trial logs into the SAME run at step=trial.number,
    # so every metric/param gets a real per-trial history (a genuine trend line
    # in the W&B UI, not the single-point "history" a one-shot log produces).
    with wandb.init(
        project="6dpose",
        name=study_name,
        group=estimator_cls.__name__,
        job_type="sweep",
        tags=[study_name],
        config={"eval_size": sweep_size, "n_trials": n_trials, "seed": seed},
    ) as run:
        log_input_artifacts(run, yolo_cfg, dataset_cfg)

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            # 1. Suggest global parameters
            depth_trunc = trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)

            # 2. Dynamically suggest model-specific parameters, then force the
            # arm's fixed parameters on top. The overrides go LAST so a value
            # that is also swept cannot drift away from the declared arm.
            suggested_params = {**estimator_cls.suggest_params(trial), **resolved_overrides}

            # 3. Evaluate across effective_n_seeds estimator-internal RANSAC
            # seeds and pool the resulting frames together (rather than
            # sampling `seed` as its own Optuna dimension -- see
            # derive_internal_seeds), so the search is robust to seed luck
            # instead of resting on a single, possibly-lucky draw.
            trial_seeds = (
                derive_internal_seeds(seed, effective_n_seeds, salt=trial.number)
                if supports_seed
                else [None]
            )

            error_metrics: list[PoseErrorMetrics] = []
            times: list[float] = []
            frame_records: list[FrameRecord] = []
            det_failed = 0
            pose_failed = 0
            gross_yaw_rate_per_seed = []

            for seed_i in trial_seeds:
                params_i = (
                    {**suggested_params, "seed": seed_i} if seed_i is not None else suggested_params
                )
                trial_estimator = estimator_cls(params=params_i, extrinsic=extrinsic)

                # Offline CAD mesh preparation (voxelization, normals, and FPFH/PPF
                # database generation). Cached per (class, cart_type, voxel_size,
                # front_crop_depth) -- seed isn't part of that key, so repeats hit
                # the cache instead of recomputing, and offline prep costs stay off
                # the timed online pose estimation latency metric either way.
                for cart_type, mesh in meshes.items():
                    trial_estimator.prepare(mesh, cart_type)

                em, t, df, pf, fr = evaluate_pipeline(
                    dataset,
                    model,
                    camera,
                    trial_estimator,
                    sweep_indices,
                    meshes,
                    depth_trunc=depth_trunc,
                )
                error_metrics.extend(em)
                times.extend(t)
                det_failed += df
                pose_failed += pf
                frame_records.extend(fr)
                # Per-seed rate uses the same n_attempted denominator as the
                # pooled metric, so the spread is comparable to the mean.
                gross_yaw_rate_per_seed.append(
                    compute_trial_metrics(em, t, df, pf).gross_yaw_rate
                )

            if dump_frames:
                write_frame_records_csv(
                    os.path.join(sweep_dir, f"{study_name}_frames", f"trial_{trial.number}.csv"),
                    frame_records,
                )

            # The five headline metrics over the pooled (all-seeds) samples.
            # p95 latency covers successful estimations only: abstentions are
            # already counted against objective 1 via the n_attempted
            # denominator, and charging them a fabricated latency too would
            # corrupt the latency objective.
            m = compute_trial_metrics(error_metrics, times, det_failed, pose_failed)

            # Optuna's own record, independent of W&B.
            trial.set_user_attr("pose_ar", m.pose_ar)
            trial.set_user_attr("p95_latency_s", m.p95_latency_s)
            trial.set_user_attr("gross_yaw_rate", m.gross_yaw_rate)
            trial.set_user_attr("abstention_rate", m.abstention_rate)
            trial.set_user_attr("detection_failure_rate", m.detection_failure_rate)
            trial.set_user_attr("n_attempted", m.n_attempted)
            # Per-seed breakdown (not just the pooled mean) so a later analysis
            # can check how seed-sensitive a config is -- see analyze_sweep.py.
            trial.set_user_attr("n_seeds", len(trial_seeds))
            trial.set_user_attr("trial_seeds", trial_seeds)
            trial.set_user_attr("gross_yaw_rate_per_seed", gross_yaw_rate_per_seed)

            # Log params + metrics together, indexed by trial number -- this is
            # what makes each key's W&B history a real per-trial trend line
            # instead of a single point.
            run.log(
                {
                    **suggested_params,
                    "depth_trunc": depth_trunc,
                    # --- the five ---
                    "pose_ar": m.pose_ar,
                    "p95_latency_s": finite_or_none(m.p95_latency_s),
                    "gross_yaw_rate": m.gross_yaw_rate,
                    "abstention_rate": m.abstention_rate,
                    "detection_failure_rate": m.detection_failure_rate,
                    # --- diagnostics, namespaced so they can never be mistaken
                    # for the headline five ---
                    "diag/good_rate": m.good_rate,
                    "diag/n_attempted": m.n_attempted,
                    "diag/trans_xy_p50": m.trans_xy_p50,
                    "diag/yaw_p50": m.yaw_p50,
                    "diag/gross_yaw_rate_std": float(np.std(gross_yaw_rate_per_seed))
                    if len(trial_seeds) > 1
                    else 0.0,
                },
                step=trial.number,
            )

            return m.pose_ar, m.p95_latency_s

        print(f"Sweep results are being saved to SQLite database: '{db_name}'")

        # n_trials is a TOTAL target for the study, not an increment: a crashed
        # sweep restarted with the same command only runs the remaining trials.
        # Trials left in RUNNING state by a crash are not counted as finished.
        finished = sum(1 for t in study.trials if t.state.is_finished())
        remaining = max(0, n_trials - finished)
        if finished:
            print(f"Resuming: {finished} finished trials in study, running {remaining} more.")
        try:
            if remaining > 0:
                study.optimize(objective, n_trials=remaining)
        finally:
            # Build the Pareto-front scatter from every COMPLETE trial in the
            # study (not just ones run in this process) -- reads straight from
            # Optuna's persistent SQLite storage, so a resumed sweep's chart is
            # always the complete picture, and an interrupted sweep (Ctrl+C
            # mid-study.optimize) still gets a chart for whatever finished
            # before the interrupt, since `finally` runs either way.
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                param_names = list(completed[0].params.keys())

                # Sortable/filterable raw table -- keeps the per-trial data
                # queryable in W&B alongside the chart (and is the searchable
                # companion to the frontier plot's per-point hover).
                columns = [
                    "trial_number",
                    *param_names,
                    "pose_ar",
                    "p95_latency_s",
                    "gross_yaw_rate",
                    "abstention_rate",
                    "diag_n_attempted",
                ]
                rows = [
                    [
                        t.number,
                        *[t.params.get(name) for name in param_names],
                        t.values[0],
                        t.values[1],
                        t.user_attrs.get("gross_yaw_rate"),
                        t.user_attrs.get("abstention_rate"),
                        t.user_attrs.get("n_attempted"),
                    ]
                    for t in completed
                ]

                # Split trials into Pareto-optimal (study.best_trials, the
                # non-dominated set) vs. everything else (dominated).
                best_numbers = {t.number for t in study.best_trials}
                pareto = [t for t in completed if t.number in best_numbers]
                dominated = [t for t in completed if t.number not in best_numbers]

                run.log(
                    {
                        "pareto_front": build_pareto_figure(
                            pareto, dominated, param_names, study_name
                        ),
                        "pareto_table": wandb.Table(columns=columns, data=rows),
                    }
                )

    print("\n" + "=" * 50)
    print("SWEEP COMPLETE (PARETO FRONT FINDINGS)")
    print("=" * 50)
    print(f"Found {len(study.best_trials)} optimal trade-off trials on the Pareto Front:")
    for _, trial in enumerate(study.best_trials):
        print(f"\n[Trial {trial.number}]")
        print(f"  - Accuracy Loss (1-AR):  {trial.values[0]:.4f}")
        print(f"  - p95 Execution Time:    {trial.values[1]:.4f}s")
        print("  - Hyperparameters:")
        for name, val in trial.params.items():
            print(f"    * {name}: {val}")
    print("=" * 50)


# =====================================================================
# 4. CLI ENTRY POINT
# =====================================================================
# tyro.cli(BenchmarkArgs) builds the parser directly from the BenchmarkArgs
# dataclass (cli_config.py) -- no external config file, no dynamic string
# resolution. `args.model` is already a concrete, type-checked *Preset
# instance by the time we get here: `.ESTIMATOR_CLS` is the real class object
# and `.profile.params`/`.profile.depth_trunc` are the chosen profile's
# values. See docs/explanation/tyro_cli_config.md for the full picture.
def main():
    args = tyro.cli(BenchmarkArgs)

    # Load model, camera, and dataset
    print("Loading pipeline assets...")
    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    dataset = load_parquet_dataset(dataset_path=args.dataset.path, test_glob=args.dataset.test_glob)

    estimator_cls = args.model.ESTIMATOR_CLS
    # Whether this estimator's params dataclass has a 'seed' field to vary at
    # all (only Ransac3DoFParams and its subclasses do today) -- --n-seeds is
    # a no-op, not a crash, for estimators with no such field.
    supports_seed = "seed" in {f.name for f in dataclasses.fields(args.model.profile.params)}

    # Hoist CAD mesh loading
    meshes = load_cad_meshes()

    # Camera extrinsic is always present now (CameraConfig has a real default,
    # not an optional YAML key), so no None-fallback is needed here anymore.
    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)

    if args.sweep:
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            study_name=args.name,
            estimator_cls=estimator_cls,
            sweep_size=args.eval_size,
            n_trials=args.trials,
            meshes=meshes,
            yolo_cfg=args.yolo,
            dataset_cfg=args.dataset,
            extrinsic=extrinsic,
            seed=args.seed,
            n_seeds=args.n_seeds,
            supports_seed=supports_seed,
            dump_frames=args.dump_frames,
            param_overrides=args.param_overrides,
        )

    else:
        # Default Evaluation mode
        seed = args.seed
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

        np.random.seed(seed)
        o3d.utility.random.seed(seed)

        total_samples = len(dataset)
        eval_indices = draw_eval_indices(total_samples, args.eval_size, seed)

        if args.n_seeds > 1 and not supports_seed:
            print(
                f"Note: {type(args.model.profile.params).__name__} has no 'seed' field; "
                "--n-seeds > 1 has no effect for this estimator, running once."
            )
        effective_n_seeds = args.n_seeds if supports_seed else 1
        internal_seeds = derive_internal_seeds(seed, effective_n_seeds) if supports_seed else [None]

        print(
            f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples..."
        )
        print(f"Seed: {seed}")
        if effective_n_seeds > 1:
            print(f"Pooled over {effective_n_seeds} internal RANSAC seeds: {internal_seeds}")
        print(f"Indices: {eval_indices}\n")

        wandb_config = {
            **dataclasses.asdict(args.model.profile.params),
            "depth_trunc": args.model.profile.depth_trunc,
            "eval_size": args.eval_size,
            "seed": seed,
            "n_seeds": effective_n_seeds,
        }
        with wandb.init(
            project="6dpose",
            name=args.name,
            group=estimator_cls.__name__,
            job_type="benchmark",
            config=wandb_config,
        ) as run:
            log_input_artifacts(run, args.yolo, args.dataset)

            # Pool results across internal seeds (a no-op loop of length 1 when
            # effective_n_seeds == 1) instead of trusting a single draw -- every
            # downstream statistic below then runs once over the pooled data.
            error_metrics: list[PoseErrorMetrics] = []
            times: list[float] = []
            frame_records: list[FrameRecord] = []
            det_failed = 0
            pose_failed = 0
            for seed_i in internal_seeds:
                params_i = (
                    dataclasses.replace(args.model.profile.params, seed=seed_i)
                    if seed_i is not None
                    else args.model.profile.params
                )
                # Directly construct the chosen preset's estimator -- no
                # _target_ string resolution needed, args.model.ESTIMATOR_CLS
                # is already the concrete class.
                estimator = estimator_cls(params=params_i, extrinsic=extrinsic)
                for cart_type, mesh in meshes.items():
                    estimator.prepare(mesh, cart_type)

                em, t, df, pf, fr = evaluate_pipeline(
                    dataset,
                    model,
                    camera,
                    estimator,
                    eval_indices,
                    meshes,
                    depth_trunc=args.model.profile.depth_trunc,
                )
                error_metrics.extend(em)
                times.extend(t)
                det_failed += df
                pose_failed += pf
                frame_records.extend(fr)

            if args.dump_frames:
                frames_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "benchmark_runs"
                )
                write_frame_records_csv(
                    os.path.join(frames_dir, f"{args.name}_frames.csv"), frame_records
                )

            m = compute_trial_metrics(error_metrics, times, det_failed, pose_failed)

            print("\n" + "=" * 58)
            print("BENCHMARK REPORT")
            print("=" * 58)
            print(f"Evaluated {m.n_eval} samples; {m.n_attempted} reached the estimator.")
            print("")
            print("THE FIVE:")
            print(f"  pose_ar                 {m.pose_ar:.4f}   (accuracy, higher is better)")
            print(
                f"  p95_latency_s           {m.p95_latency_s:.4f}   (speed, lower is better)"
                if np.isfinite(m.p95_latency_s)
                else "  p95_latency_s           n/a      (no successful estimation)"
            )
            print(f"  gross_yaw_rate          {m.gross_yaw_rate:.4f}   (|yaw| > {GROSS_YAW_DEG:g}°)")
            print(f"  abstention_rate         {m.abstention_rate:.4f}   (no pose returned)")
            print(f"  detection_failure_rate  {m.detection_failure_rate:.4f}   (YOLO, upstream)")
            print("")
            # good + gross + abstention == 1.0 by construction; printing the sum
            # makes that visible rather than something you have to trust.
            print(
                f"  partition check: good {m.good_rate:.4f} + gross {m.gross_yaw_rate:.4f} "
                f"+ abstained {m.abstention_rate:.4f} = "
                f"{m.good_rate + m.gross_yaw_rate + m.abstention_rate:.4f}"
            )

            log_payload = {
                "pose_ar": m.pose_ar,
                "p95_latency_s": finite_or_none(m.p95_latency_s),
                "gross_yaw_rate": m.gross_yaw_rate,
                "abstention_rate": m.abstention_rate,
                "detection_failure_rate": m.detection_failure_rate,
                "diag/good_rate": m.good_rate,
                "diag/n_attempted": m.n_attempted,
                "diag/n_eval": m.n_eval,
                "diag/trans_xy_p50": m.trans_xy_p50,
                "diag/yaw_p50": m.yaw_p50,
            }

            if m.trans_xy_p50 is not None:
                print("")
                print(f"CONDITIONAL ON GOOD SAMPLES ONLY (|yaw| <= {GROSS_YAW_DEG:g}°):")
                print(f"  median XY translation error  {m.trans_xy_p50:.4f} m")
                print(f"  median yaw error             {m.yaw_p50:.2f}°")
                print("  (readable magnitudes -- NOT comparable across configs with")
                print("   different abstention rates, since the conditioning set differs)")

            # Abstention causes -- the point of the per-frame records. Answers
            # "did the estimator starve at the FPFH stage or reject its own
            # candidates?", which a single pose_failures count cannot.
            failures = [r for r in frame_records if r.outcome != "good"]
            if failures:
                by_reason: dict[str, int] = {}
                for r in failures:
                    # gross_yaw frames produced a pose, so they carry no
                    # abstention cause -- don't render a "/None" suffix for them.
                    key = f"{r.outcome}/{r.failure_reason}" if r.failure_reason else r.outcome
                    by_reason[key] = by_reason.get(key, 0) + 1
                print("")
                print("FAILURE BREAKDOWN:")
                for key, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
                    print(f"  {key:<34} {count:>4}  ({count / m.n_eval * 100:.1f}% of evaluated)")
                    log_payload[f"diag/failures/{key.replace('/', '_')}"] = count

            # True ~180 degree flips, separated from merely-imprecise poses in
            # the 15-90 degree band: same headline bucket, different root cause.
            flipped = [r for r in frame_records if r.flipped]
            if m.n_attempted:
                flip_share = len(flipped) / m.n_attempted
                print("")
                print(f"  of which true flips (|yaw| > 90°): {len(flipped)} ({flip_share:.4f})")
                log_payload["diag/flip_share"] = flip_share

            run.log(log_payload)
            print("=" * 58)


if __name__ == "__main__":
    main()
