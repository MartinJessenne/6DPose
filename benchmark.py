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

import dataclasses
import glob
import os
import time

import numpy as np

import open3d as o3d
import optuna
import plotly.graph_objects as go
import tyro
import wandb
from datasets import Dataset

import logging
from pydantic import BaseModel, Field

from cli_config import BenchmarkArgs
from pipeline import (
    Camera, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, compute_ground_truth_pose,
    instance_detected, load_cad_meshes
)
from methods.base import BasePoseEstimator

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
    
    r11, r12, r13 = R_err[0, 0], R_err[0, 1], R_err[0, 2]
    r21, r22, r23 = R_err[1, 0], R_err[1, 1], R_err[1, 2]
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
        geodesic_rot=geodesic_rot
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
        total_samples (int): Total number of samples evaluated (including failures).
        
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
            successes = sum(
                1 for e in errors
                if e.trans_xy < t_thresh and abs(e.yaw) < r_thresh
            )
            recalls.append(successes / total_samples)
            
    return float(np.mean(recalls))

# Set logging level for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    depth_trunc: float = 3.0
) -> tuple[list[PoseErrorMetrics], list[float], int, int]:
    """
    Evaluates the pose estimation pipeline on a set of sample indices.
    
    Returns:
        tuple containing:
            - error_metrics (list[PoseErrorMetrics]): Decomposed errors for successfully matched samples.
            - times (list[float]): Pose estimation duration in seconds for each successful sample.
            - detection_failures (int): Count of samples where YOLO failed to detect a cart.
            - pose_failures (int): Count of samples where pose estimation failed.
    """
    error_metrics = []
    times = []
    detection_failures = 0
    pose_failures = 0
    
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
            continue
            
        # 2. Segment and Reconstruct Point Cloud
        try:
            cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera, depth_trunc=depth_trunc)
        except Exception:
            logging.exception(f"PointCloud processing failed for index {sample_idx}")
            pose_failures += 1
            continue
            
        # Retrieve preloaded mesh
        cad_mesh = meshes.get(cart_type)
        if cad_mesh is None:
            logging.error(f"CAD mesh not found for cart type '{cart_type}' (sample {sample_idx})")
            pose_failures += 1
            continue
        
        # 3. Perform 6D Pose Estimation with timing
        start_time = time.time()
        try:
            T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=cart_type)
            if T_final is None:
                logging.error(f"Pose estimator returned None for index {sample_idx}")
                pose_failures += 1
                continue
        except Exception:
            logging.exception(f"Pose estimator raised an exception for index {sample_idx}")
            pose_failures += 1
            continue
        elapsed_time = time.time() - start_time
            
        # 4. Calculate Ground Truth pose and compare
        extrinsic = getattr(estimator, "extrinsic", None)
        if extrinsic is None:
            raise ValueError("Estimator must have an extrinsic camera-to-robot transform configured.")
        
        try:
            T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
            T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
            T_ground_truth = compute_ground_truth_pose(T_world_camera, T_world_cart, T_robot_camera=extrinsic)
            metrics = extract_pose_errors(T_final, T_ground_truth)
            error_metrics.append(metrics)
            times.append(elapsed_time)
        except Exception:
            logging.exception(f"Error metric extraction failed for index {sample_idx}")
            pose_failures += 1
            continue
        
    return error_metrics, times, detection_failures, pose_failures


# =====================================================================
# 3. OPTUNA SWEEP STUDY (MULTI-OBJECTIVE OPTIMIZATION)
# =====================================================================
def log_input_artifacts(run, yolo_cfg, dataset_cfg):
    """Register the YOLO weights and test dataset as W&B *reference* artifacts.

    checksum=True records a content hash of each referenced file -- no bytes are
    uploaded, so an unchanged model/dataset produces no new version while any
    modification mints one. That version bump is the "was this input touched?"
    signal for the cross-commit report.
    """
    model_art = wandb.Artifact("yolo-detector", type="model",
                               metadata={"hf_repo": yolo_cfg.repo, "hf_file": yolo_cfg.file})
    model_art.add_reference(f"file://{os.path.abspath(yolo_cfg.local_path)}", checksum=True)
    run.log_artifact(model_art)

    # dataset_cfg.path is the HF builder name ("parquet"), not a filesystem path;
    # the real data are the shards matching test_glob.
    dataset_art = wandb.Artifact("test-dataset", type="dataset",
                                 metadata={"test_glob": dataset_cfg.test_glob})
    for shard in sorted(glob.glob(dataset_cfg.test_glob)):
        dataset_art.add_reference(f"file://{os.path.abspath(shard)}", checksum=True)
    run.log_artifact(dataset_art)


def build_pareto_figure(pareto, dominated, param_names, study_name):
    """Build the interactive Pareto-front scatter logged to W&B at sweep end.

    x = p95 latency (objective 2), y = accuracy loss / 1-AR (objective 1); both are
    minimized, so the bottom-left corner is best. Dominated trials form a recessive
    gray field, Pareto-optimal trials (study.best_trials) are highlighted in blue and
    connected by the frontier line. Every point's hover carries its trial number and
    full hyperparameter set, so any point on the frontier is traceable straight back
    to the iteration and config that produced it -- which a bare wandb.plot.scatter
    (fixed tooltip, x/y only) cannot do.
    """
    BLUE, GRAY, GRID, AXIS = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7"
    INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

    # customdata columns, in order: trial_number, *params, average_recall, flip_rate.
    def customdata(trials):
        return [
            [t.number]
            + [t.params.get(name) for name in param_names]
            + [t.user_attrs.get("average_recall"), t.user_attrs.get("flip_rate")]
            for t in trials
        ]

    hover_lines = [
        "<b>Trial %{customdata[0]}</b>",
        "accuracy loss (1−AR): %{y:.4f}",
        "p95 latency: %{x:.3f}s",
    ]
    for i, name in enumerate(param_names, start=1):
        hover_lines.append(f"{name}: %{{customdata[{i}]}}")
    hover_lines.append(f"average_recall: %{{customdata[{len(param_names) + 1}]:.3f}}")
    hover_lines.append(f"flip_rate: %{{customdata[{len(param_names) + 2}]:.3f}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure()

    # Dominated trials -- recessive gray field, drawn first (underneath).
    fig.add_trace(go.Scatter(
        x=[t.values[1] for t in dominated], y=[t.values[0] for t in dominated],
        mode="markers", name="Dominated trials",
        marker=dict(color=GRAY, size=8, opacity=0.55),
        customdata=customdata(dominated), hovertemplate=hovertemplate,
    ))

    # Pareto frontier -- straight line through the optimal trials sorted by latency.
    pareto_sorted = sorted(pareto, key=lambda t: t.values[1])
    fig.add_trace(go.Scatter(
        x=[t.values[1] for t in pareto_sorted], y=[t.values[0] for t in pareto_sorted],
        mode="lines", name="Pareto frontier",
        line=dict(color=BLUE, width=2), hoverinfo="skip",
    ))

    # Pareto-optimal trials -- highlighted, drawn on top.
    fig.add_trace(go.Scatter(
        x=[t.values[1] for t in pareto], y=[t.values[0] for t in pareto],
        mode="markers", name="Pareto-optimal",
        marker=dict(color=BLUE, size=12, line=dict(color=SURFACE, width=1.5)),
        customdata=customdata(pareto), hovertemplate=hovertemplate,
    ))

    fig.update_layout(
        title=dict(text=f"Pareto Front — {study_name}", font=dict(size=18, color=INK)),
        template="plotly_white",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=13),
        xaxis=dict(title="p95 latency (s)  →  slower", gridcolor=GRID, zeroline=False,
                   linecolor=AXIS, ticks="outside", tickcolor=AXIS),
        yaxis=dict(title="accuracy loss (1 − AR)  →  worse", gridcolor=GRID, zeroline=False,
                   linecolor=AXIS, ticks="outside", tickcolor=AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        margin=dict(l=70, r=30, t=70, b=90),
        annotations=[dict(text="← better (fast & accurate)", x=0.01, y=0.02,
                          xref="paper", yref="paper", showarrow=False,
                          font=dict(color=GRAY, size=12))],
    )
    return fig


def run_parameter_sweep(
    dataset, model, camera, study_name, estimator_cls: type[BasePoseEstimator],
    sweep_size: int, n_trials: int, meshes: dict[str, o3d.geometry.TriangleMesh],
    yolo_cfg, dataset_cfg,
    extrinsic: np.ndarray = None, seed: int = None
):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.
    """
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
        directions=["minimize", "minimize"],  # Minimize error rate AND execution time
        load_if_exists=True
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
        rng = np.random.default_rng(seed)
        sweep_indices = rng.choice(total_samples, min(sweep_size, total_samples), replace=False).tolist()
        
        # Persist attributes in the study so future runs can resume with the same setup
        study.set_user_attr("seed", seed)
        study.set_user_attr("eval_size", sweep_size)
        study.set_user_attr("sweep_indices", sweep_indices)
        print(f"Created new study. Seed={seed}, eval_size={sweep_size}")
        
    print(f"Sweep validation indices: {sweep_indices}\n")

    # Seed global random number generators
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

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

            # 2. Dynamically suggest model-specific parameters
            suggested_params = estimator_cls.suggest_params(trial)

            # 3. Instantiate model with trial parameters
            trial_estimator = estimator_cls(params=suggested_params, extrinsic=extrinsic)

            # Offline CAD mesh preparation (voxelization, normals, and FPFH/PPF database generation).
            # We prepare the model CAD representations outside the timed evaluation loop so that
            # offline preparation costs are not charged to the online pose estimation latency metric.
            for cart_type, mesh in meshes.items():
                trial_estimator.prepare(mesh, cart_type)

            error_metrics, times, det_failed, pose_failed = evaluate_pipeline(
                dataset, model, camera, trial_estimator, sweep_indices, meshes, depth_trunc=depth_trunc
            )

            total_failed = det_failed + pose_failed
            total_matched = len(error_metrics)
            total_evaluated = total_matched + total_failed

            # Calculate Average Recall
            ar = compute_average_recall(error_metrics, total_evaluated)
            accuracy_score = 1.0 - ar

            # Calculate p95 execution time over actual successful runs only.
            # Failures are already counted as misses in Obj 1 (AR); penalizing them
            # again with fabricated latencies would corrupt the latency objective.
            p95_time = float(np.percentile(times, 95)) if times else float('inf')

            # Save trial diagnostics (Optuna's own record, independent of W&B)
            trial.set_user_attr("average_recall", ar)
            trial.set_user_attr("p95_time", p95_time)
            trial.set_user_attr("detection_failures", det_failed)
            trial.set_user_attr("pose_failures", pose_failed)

            if total_matched > 0:
                flip_rate = sum(1 for e in error_metrics if abs(e.yaw) > 90.0) / total_matched
            else:
                flip_rate = 0.0
            trial.set_user_attr("flip_rate", flip_rate)

            # Log params + metrics together, indexed by trial number -- this is
            # what makes each key's W&B history a real per-trial trend line
            # instead of a single point.
            run.log({
                **suggested_params,
                "depth_trunc": depth_trunc,
                "accuracy_score": accuracy_score,
                "p95_time": p95_time,
                "average_recall": ar,
                "detection_failures": det_failed,
                "pose_failures": pose_failed,
                "flip_rate": flip_rate,
            }, step=trial.number)

            return accuracy_score, p95_time

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
                columns = ["trial_number", *param_names, "accuracy_score", "p95_time", "average_recall", "flip_rate"]
                rows = [
                    [
                        t.number,
                        *[t.params.get(name) for name in param_names],
                        t.values[0],
                        t.values[1],
                        t.user_attrs.get("average_recall"),
                        t.user_attrs.get("flip_rate"),
                    ]
                    for t in completed
                ]

                # Split trials into Pareto-optimal (study.best_trials, the
                # non-dominated set) vs. everything else (dominated).
                best_numbers = {t.number for t in study.best_trials}
                pareto = [t for t in completed if t.number in best_numbers]
                dominated = [t for t in completed if t.number not in best_numbers]

                run.log({
                    "pareto_front": build_pareto_figure(pareto, dominated, param_names, study_name),
                    "pareto_table": wandb.Table(columns=columns, data=rows),
                })

    print("\n" + "=" * 50)
    print("SWEEP COMPLETE (PARETO FRONT FINDINGS)")
    print("=" * 50)
    print(f"Found {len(study.best_trials)} optimal trade-off trials on the Pareto Front:")
    for t_idx, trial in enumerate(study.best_trials):
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
        local_model_path=args.yolo.local_path,
        repo_id=args.yolo.repo,
        filename=args.yolo.file
    )
    camera = Camera(
        fx=args.camera.fx, fy=args.camera.fy,
        cx=args.camera.cx, cy=args.camera.cy
    )
    dataset = load_parquet_dataset(
        dataset_path=args.dataset.path,
        test_glob=args.dataset.test_glob
    )

    estimator_cls = args.model.ESTIMATOR_CLS

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
            seed=args.seed
        )

    else:
        # Default Evaluation mode
        seed = args.seed
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

        np.random.seed(seed)
        o3d.utility.random.seed(seed)

        total_samples = len(dataset)
        eval_indices = np.random.choice(total_samples, min(args.eval_size, total_samples), replace=False).tolist()

        # Directly construct the chosen preset's estimator -- no _target_
        # string resolution needed, args.model.ESTIMATOR_CLS is already the
        # concrete class.
        estimator = estimator_cls(params=args.model.profile.params, extrinsic=extrinsic)

        # Pre-prepare all meshes on estimator
        for cart_type, mesh in meshes.items():
            estimator.prepare(mesh, cart_type)

        print(f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples...")
        print(f"Seed: {seed}")
        print(f"Indices: {eval_indices}\n")

        wandb_config = {
            **dataclasses.asdict(args.model.profile.params),
            "depth_trunc": args.model.profile.depth_trunc,
            "eval_size": args.eval_size,
            "seed": seed,
        }
        with wandb.init(
            project="6dpose",
            name=args.name,
            group=estimator_cls.__name__,
            job_type="benchmark",
            config=wandb_config,
        ) as run:
            log_input_artifacts(run, args.yolo, args.dataset)

            error_metrics, times, det_failed, pose_failed = evaluate_pipeline(
                dataset, model, camera, estimator, eval_indices, meshes, depth_trunc=args.model.profile.depth_trunc
            )

            successful = len(error_metrics)
            total_failed = det_failed + pose_failed
            total = successful + total_failed

            print("\n" + "=" * 50)
            print("BENCHMARK REPORT (Default Parameters)")
            print("=" * 50)
            print(f"Detections & Matches: {successful} / {total} (Success rate: {successful/total*100:.1f}%)")
            print(f"  - YOLO detection failures: {det_failed}")
            print(f"  - Pose estimation failures: {pose_failed}")
            run.log({
                "success_rate": successful / total * 100 if total else 0.0,
                "detection_failures": det_failed,
                "pose_failures": pose_failed,
            })

            if successful > 0:
                ar = compute_average_recall(error_metrics, total)
                print(f"Average Recall (BOP-style AR): {ar:.4f}")

                p95_latency = float(np.percentile(times, 95)) if times else float('inf')
                print(f"p95 Latency: {p95_latency:.4f}s")

                # Decompose errors
                trans_xy_errs = [e.trans_xy for e in error_metrics]
                trans_z_errs = [e.trans_z for e in error_metrics]
                yaw_errs = [e.yaw for e in error_metrics]
                pitch_errs = [e.pitch for e in error_metrics]
                roll_errs = [e.roll for e in error_metrics]
                geodesic_errs = [e.geodesic_rot for e in error_metrics]

                print(f"Translation Error (XY in meters):")
                print(f"  - Mean:   {np.mean(trans_xy_errs):.4f}")
                print(f"  - Median: {np.median(trans_xy_errs):.4f}")
                print(f"Translation Error (Z in meters):")
                print(f"  - Bias (signed mean): {np.mean(trans_z_errs):+.4f}")
                print(f"  - MAE:                {np.mean(np.abs(trans_z_errs)):.4f}")
                print(f"  - Median (abs):       {np.median(np.abs(trans_z_errs)):.4f}")

                print(f"Yaw Rotation Error (degrees):")
                print(f"  - Mean:   {np.mean(np.abs(yaw_errs)):.2f}°")
                print(f"  - Median: {np.median(np.abs(yaw_errs)):.2f}°")
                print(f"Pitch Rotation Error (degrees):")
                print(f"  - Mean:   {np.mean(np.abs(pitch_errs)):.2f}°")
                print(f"  - Median: {np.median(np.abs(pitch_errs)):.2f}°")
                print(f"Roll Rotation Error (degrees):")
                print(f"  - Mean:   {np.mean(np.abs(roll_errs)):.2f}°")
                print(f"  - Median: {np.median(np.abs(roll_errs)):.2f}°")
                print(f"Geodesic Rotation Error (degrees):")
                print(f"  - Mean:   {np.mean(geodesic_errs):.2f}°")
                print(f"  - Median: {np.median(geodesic_errs):.2f}°")

                # Flip rate
                flips = sum(1 for e in error_metrics if abs(e.yaw) > 90.0)
                print(f"Flip Rate (among successful matches): {flips/successful*100:.1f}% ({flips}/{successful})")

                run.log({
                    "average_recall": ar,
                    "p95_latency": p95_latency,
                    "trans_xy_mean": float(np.mean(trans_xy_errs)),
                    "trans_xy_median": float(np.median(trans_xy_errs)),
                    "trans_z_bias": float(np.mean(trans_z_errs)),
                    "trans_z_mae": float(np.mean(np.abs(trans_z_errs))),
                    "trans_z_median_abs": float(np.median(np.abs(trans_z_errs))),
                    "yaw_mean": float(np.mean(np.abs(yaw_errs))),
                    "yaw_median": float(np.median(np.abs(yaw_errs))),
                    "pitch_mean": float(np.mean(np.abs(pitch_errs))),
                    "pitch_median": float(np.median(np.abs(pitch_errs))),
                    "roll_mean": float(np.mean(np.abs(roll_errs))),
                    "roll_median": float(np.median(np.abs(roll_errs))),
                    "geodesic_mean": float(np.mean(geodesic_errs)),
                    "geodesic_median": float(np.median(geodesic_errs)),
                    "flip_rate": flips / successful * 100,
                })

                # Median error on non-flipped samples
                non_flipped_metrics = [e for e in error_metrics if abs(e.yaw) <= 90.0]
                if non_flipped_metrics:
                    non_flipped_xy = [e.trans_xy for e in non_flipped_metrics]
                    non_flipped_yaw = [e.yaw for e in non_flipped_metrics]
                    print(f"Median errors on non-flipped samples:")
                    print(f"  - Translation XY: {np.median(non_flipped_xy):.4f}m")
                    print(f"  - Yaw Rotation:   {np.median(np.abs(non_flipped_yaw)):.2f}°")
                    run.log({
                        "non_flipped_trans_xy_median": float(np.median(non_flipped_xy)),
                        "non_flipped_yaw_median": float(np.median(np.abs(non_flipped_yaw))),
                    })
                else:
                    print("All successful matches were flipped.")
            else:
                print("No samples were successfully matched.")
            print("=" * 50)

if __name__ == "__main__":
    main()
