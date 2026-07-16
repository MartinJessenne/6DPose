"""
6D Pose Estimation Benchmark & Hyperparameter Tuning Suite.

This script evaluates and optimizes 6D Pose Estimation algorithms (such as PPF and RANSAC)
against ground truth labels from Isaac Sim. It runs in two distinct modes:

Usage Modes:
------------
1. Default Benchmark Evaluation (sweep=false)
   Evaluates a chosen method on a random subset of the test split using its default/optimized
   hyperparameters, printing a detailed performance report (Mean/Median Translation & Rotation
   errors, Match Success Rate, and Mean Execution time).
   
   Example (PPF):
     uv run benchmark.py model=ppf eval_size=30
     
   Example (RANSAC):
     uv run benchmark.py model=ransac eval_size=30

2. Hyperparameter Sweep Optimization (sweep=true)
   Launches a Multi-Objective Bayesian Optimization sweep using Optuna to find the Pareto Front
   of optimal accuracy vs. speed trade-offs. The search space is dynamically configured based
   on the selected method, and results are persisted in a local SQLite database for visualization.
   
   Example (PPF Sweep):
     uv run benchmark.py sweep=true model=ppf name=PPF_Sweep trials=50 eval_size=30
     
   Example (RANSAC Sweep):
     uv run benchmark.py sweep=true model=ransac name=RANSAC_Sweep trials=50 eval_size=30

CLI Configuration Overrides:
----------------------------
  model=ppf|ransac    The 6D pose estimation method to run (default: 'ppf').
  sweep=true|false    Flag to execute the Optuna hyperparameter sweep (default: false).
  name=NAME           Custom name for the sweep (creates 'optuna_<NAME>.db' storage).
  trials=NUM          Number of optimization trials to execute (default: 30).
  eval_size=NUM       Number of validation samples to evaluate per trial/benchmark (default: 20).
  seed=NUM            Optional fixed seed to ensure reproducibility.
"""

import os
import time

import numpy as np
import torch
import open3d as o3d
import optuna
import hydra
from omegaconf import DictConfig
from datasets import Dataset

import logging
from pydantic import BaseModel, Field

from main import (
    Camera, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, compute_ground_truth_pose,
    instance_detected
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
# 1. EVALUATION METRIC FUNCTIONS
# =====================================================================
def compute_translation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """
    Computes the translation error (Euclidean distance) in meters.
    
    Args:
        T_est (np.ndarray): 4x4 estimated transformation matrix.
        T_gt (np.ndarray): 4x4 ground truth transformation matrix.
        
    Returns:
        float: Translation error in meters.
    """
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    return float(np.linalg.norm(t_est - t_gt))


def compute_rotation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """
    Computes the geodesic rotation error (angle in degrees).
    
    Args:
        T_est (np.ndarray): 4x4 estimated transformation matrix.
        T_gt (np.ndarray): 4x4 ground truth transformation matrix.
        
    Returns:
        float: Rotation error in degrees.
    """
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    
    # Compute trace of R_est^T @ R_gt
    trace_val = np.trace(R_est.T @ R_gt)
    
    # Clip trace_val to [-1, 1] to avoid domain error in arccos due to float precision limits
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Convert from radians to degrees
    return float(np.degrees(theta))


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
def run_parameter_sweep(
    dataset, model, camera, study_name, estimator_cls: type[BasePoseEstimator], 
    sweep_size: int, n_trials: int, meshes: dict[str, o3d.geometry.TriangleMesh],
    extrinsic: np.ndarray = None, seed: int = None
):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.
    """
    db_name = f"optuna_{study_name}.db"
    db_url = f"sqlite:///{db_name}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        directions=["minimize", "minimize"],  # Minimize error rate AND execution time
        load_if_exists=True
    )
    
    existing_seed = study.user_attrs.get("seed")
    existing_eval_size = study.user_attrs.get("eval_size")
    existing_indices = study.user_attrs.get("sweep_indices")
    
    total_samples = len(dataset)
    
    if existing_seed is not None:
        # Integrity Guard Check
        if existing_eval_size != sweep_size:
            raise ValueError(
                f"Validation size mismatch: DB has eval_size={existing_eval_size}, "
                f"but sweep requested eval_size={sweep_size}."
            )
        if seed is not None and seed != existing_seed:
            raise ValueError(
                f"Seed mismatch: DB has seed={existing_seed}, but sweep requested seed={seed}."
            )
        seed = existing_seed
        sweep_indices = existing_indices
        print(f"Resuming existing study. Loaded seed={seed}, eval_size={sweep_size}")
    else:
        # Create fresh attributes and store
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        rng = np.random.default_rng(seed)
        sweep_indices = rng.choice(total_samples, min(sweep_size, total_samples), replace=False).tolist()
        
        study.set_user_attr("seed", seed)
        study.set_user_attr("eval_size", sweep_size)
        study.set_user_attr("sweep_indices", sweep_indices)
        print(f"Created new study. Seed={seed}, eval_size={sweep_size}")
        
    print(f"Sweep validation indices: {sweep_indices}\n")
    
    # Seed global random number generators
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # 1. Suggest global parameters
        depth_trunc = trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)
        
        # 2. Dynamically suggest model-specific parameters
        suggested_params = estimator_cls.suggest_params(trial)
        
        # 3. Instantiate model with trial parameters
        trial_estimator = estimator_cls(params=suggested_params, extrinsic=extrinsic)
        
        # Pre-prepare all meshes on this trial estimator
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
        
        # Calculate p95 execution time with 5s penalty for failures
        all_times = times + [5.0] * total_failed
        p95_time = float(np.percentile(all_times, 95)) if all_times else 5.0
        
        # Save trial diagnostics
        trial.set_user_attr("average_recall", ar)
        trial.set_user_attr("p95_time", p95_time)
        trial.set_user_attr("detection_failures", det_failed)
        trial.set_user_attr("pose_failures", pose_failed)
        
        if total_matched > 0:
            flips = sum(1 for e in error_metrics if abs(e.yaw) > 90.0)
            trial.set_user_attr("flip_rate", flips / total_matched)
        else:
            trial.set_user_attr("flip_rate", 0.0)
            
        return accuracy_score, p95_time

    print(f"Sweep results are being saved to SQLite database: '{db_name}'")
    study.optimize(objective, n_trials=n_trials)
    
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
@hydra.main(config_path="config", config_name="benchmark_config", version_base=None)
def main(cfg: DictConfig):
    # Load model, camera, and dataset
    print("Loading pipeline assets...")
    model = load_hf_model(
        local_model_path=cfg.yolo.local_path,
        repo_id=cfg.yolo.repo,
        filename=cfg.yolo.file
    )
    camera = Camera(
        fx=cfg.camera.fx, fy=cfg.camera.fy,
        cx=cfg.camera.cx, cy=cfg.camera.cy
    )
    dataset = load_parquet_dataset(
        dataset_path=cfg.dataset.path,
        test_glob=cfg.dataset.test_glob
    )
    
    # Resolve estimator class dynamically from Hydra configuration target
    estimator_cls = hydra.utils.get_class(cfg.model._target_)
    
    # Hoist CAD mesh loading (relative to script directory)
    import glob
    project_root = os.path.dirname(os.path.abspath(__file__))
    mesh_path_pattern = os.path.join(project_root, "meshes", "*.ply")
    mesh_files = glob.glob(mesh_path_pattern)
    meshes = {}
    for f in mesh_files:
        name = os.path.splitext(os.path.basename(f))[0]
        meshes[name] = o3d.io.read_triangle_mesh(f)
    if not meshes:
        raise RuntimeError(f"No meshes found matching pattern: {mesh_path_pattern}. Ensure the meshes/ directory is populated.")
    
    # Retrieve extrinsic matrix for sweep
    extrinsic_list = cfg.camera.get("extrinsic", None)
    extrinsic = np.array(extrinsic_list, dtype=np.float64) if extrinsic_list is not None else None
    
    if cfg.get("sweep", False):
        study_name = cfg.get("name", "6DPoseOptimization")
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            study_name=study_name,
            estimator_cls=estimator_cls,
            sweep_size=cfg.eval_size,
            n_trials=cfg.trials,
            meshes=meshes,
            extrinsic=extrinsic,
            seed=cfg.get("seed", None)
        )

    else:
        # Default Evaluation mode
        seed = cfg.get("seed", None)
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        
        np.random.seed(seed)
        o3d.utility.random.seed(seed)
        
        total_samples = len(dataset)
        eval_indices = np.random.choice(total_samples, min(cfg.eval_size, total_samples), replace=False).tolist()
        
        # Instantiate estimator dynamically via Hydra
        estimator = hydra.utils.instantiate(cfg.model)
        
        # Pre-prepare all meshes on estimator
        for cart_type, mesh in meshes.items():
            estimator.prepare(mesh, cart_type)
            
        print(f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples...")
        print(f"Seed: {seed}")
        print(f"Indices: {eval_indices}\n")
        
        error_metrics, times, det_failed, pose_failed = evaluate_pipeline(
            dataset, model, camera, estimator, eval_indices, meshes, depth_trunc=cfg.depth_trunc
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
        
        if successful > 0:
            ar = compute_average_recall(error_metrics, total)
            print(f"Average Recall (BOP-style AR): {ar:.4f}")
            
            p95_latency = np.percentile(times + [5.0] * total_failed, 95)
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
            print(f"  - Mean:   {np.mean(trans_z_errs):.4f}")
            print(f"  - Median: {np.median(trans_z_errs):.4f}")
            
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
            
            # Median error on non-flipped samples
            non_flipped_metrics = [e for e in error_metrics if abs(e.yaw) <= 90.0]
            if non_flipped_metrics:
                non_flipped_xy = [e.trans_xy for e in non_flipped_metrics]
                non_flipped_yaw = [e.yaw for e in non_flipped_metrics]
                print(f"Median errors on non-flipped samples:")
                print(f"  - Translation XY: {np.median(non_flipped_xy):.4f}m")
                print(f"  - Yaw Rotation:   {np.median(np.abs(non_flipped_yaw)):.2f}°")
            else:
                print("All successful matches were flipped.")
        else:
            print("No samples were successfully matched.")
        print("=" * 50)

if __name__ == "__main__":
    main()
