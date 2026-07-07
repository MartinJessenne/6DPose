"""
6D Pose Estimation Benchmark & Hyperparameter Tuning Suite.

This script evaluates and optimizes 6D Pose Estimation algorithms (such as PPF and RANSAC)
against ground truth labels from Isaac Sim. It runs in two distinct modes:

Usage Modes:
------------
1. Default Benchmark Evaluation
   Evaluates a chosen method on a random subset of the test split using its default/optimized
   hyperparameters, printing a detailed performance report (Mean/Median Translation & Rotation
   errors, Match Success Rate, and Mean Execution time).
   
   Example (PPF):
     uv run benchmark.py --method ppf_icp --eval-size 30
     
   Example (RANSAC):
     uv run benchmark.py --method ransac --eval-size 30

2. Hyperparameter Sweep Optimization (--sweep)
   Launches a Multi-Objective Bayesian Optimization sweep using Optuna to find the Pareto Front
   of optimal accuracy vs. speed trade-offs. The search space is dynamically configured based
   on the selected method, and results are persisted in a local SQLite database for visualization.
   
   Example (PPF Sweep):
     uv run benchmark.py --sweep --method ppf_icp --name PPF_Sweep --trials 50 --eval-size 30
     
   Example (RANSAC Sweep):
     uv run benchmark.py --sweep --method ransac --name RANSAC_Sweep --trials 50 --eval-size 30

CLI Arguments:
--------------
  --method {ppf_icp,ransac}  The 6D pose estimation method to run (default: 'ppf_icp').
  --sweep                    Flag to execute the Optuna hyperparameter sweep.
  --name NAME                Custom name for the sweep (creates 'optuna_<NAME>.db' storage).
  --trials TRIALS            Number of optimization trials to execute (default: 30).
  --eval-size SIZE           Number of validation samples to evaluate per trial/benchmark (default: 20).
"""

import os
import argparse
import time

import numpy as np
import torch
import open3d as o3d
import optuna
from datasets import Dataset

# Import Config, classes, and helper functions from main.py
from main import (
    Config, Camera, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, compute_ground_truth_pose,
    instance_detected
)
from methods.base import BasePoseEstimator

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
    depth_trunc: float = Config.DEFAULT_DEPTH_TRUNC
) -> tuple[list[float], list[float], list[float], int]:

    """
    Evaluates the pose estimation pipeline on a set of sample indices.
    
    Returns:
        tuple containing:
            - trans_errors (list[float]): Translation error for each successfully matched sample.
            - rot_errors (list[float]): Rotation error for each successfully matched sample.
            - times (list[float]): Pose estimation duration in seconds for each successful sample.
            - failed_count (int): Number of samples that failed to detect or estimate pose.
    """
    trans_errors = []
    rot_errors = []
    times = []
    failed_count = 0
    
    for sample_idx in sample_indices:
        img = dataset["rgb"][sample_idx]
        depth_bytes = dataset["depth"][sample_idx]
        
        # 1. Run YOLO detection
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            failed_count += 1
            continue
            
        # 2. Segment and Reconstruct Point Cloud
        try:
            cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera, depth_trunc=depth_trunc)
        except Exception:
            failed_count += 1
            continue

            
        # Verify CAD mesh exists
        mesh_file = f"meshes/{cart_type}.ply"
        if not os.path.exists(mesh_file):
            failed_count += 1
            continue
        cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # 3. Perform 6D Pose Estimation with timing
        start_time = time.time()
        try:
            T_final = estimator.estimate_pose(pcd, cad_mesh)
            if T_final is None:
                failed_count += 1
                continue
        except Exception:
            failed_count += 1
            continue
        elapsed_time = time.time() - start_time
            
        # 4. Calculate Ground Truth pose and compare
        T_ground_truth = compute_ground_truth_pose(dataset, sample_idx)
        
        err_trans = compute_translation_error(T_final, T_ground_truth)
        err_rot = compute_rotation_error(T_final, T_ground_truth)
        
        trans_errors.append(err_trans)
        rot_errors.append(err_rot)
        times.append(elapsed_time)
        
    return trans_errors, rot_errors, times, failed_count


# =====================================================================
# 3. OPTUNA SWEEP STUDY (MULTI-OBJECTIVE OPTIMIZATION)
# =====================================================================
def run_parameter_sweep(dataset, model, camera, study_name, method_name: str, sweep_size: int, n_trials: int):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.
    """
    # Select a fixed validation subset for the sweep to ensure consistent comparisons
    total_samples = len(dataset)
    sweep_indices = np.random.choice(total_samples, min(sweep_size, total_samples), replace=False)
    print(f"Running Multi-Objective sweep for '{method_name}' over {len(sweep_indices)} samples for {n_trials} trials...")
    print(f"Sweep validation indices: {sweep_indices}\n")
    
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Define hyperparameter search space dynamically based on the method name
        depth_trunc = trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)
        
        if method_name == "ppf_icp":
            ppf_sampling_step = trial.suggest_float("ppf_sampling_step", 0.02, 0.10, step=0.01)
            ppf_distance_step = trial.suggest_float("ppf_distance_step", 0.02, 0.10, step=0.01)
            ppf_match_threshold = trial.suggest_float("ppf_match_threshold", 0.02, 0.10, step=0.01)
            ppf_match_tolerance = trial.suggest_float("ppf_match_tolerance", 0.01, 0.08, step=0.01)
            icp_max_correspondence_distance = trial.suggest_float("icp_max_correspondence_distance", 0.02, 0.20)
            icp_max_iterations = trial.suggest_int("icp_max_iterations", 10, 100, step=10)
            
            from methods import PPFICPEstimator, PPFICPParams
            params = PPFICPParams(
                ppf_sampling_step=ppf_sampling_step,
                ppf_distance_step=ppf_distance_step,
                ppf_match_threshold=ppf_match_threshold,
                ppf_match_tolerance=ppf_match_tolerance,
                icp_max_correspondence_distance=icp_max_correspondence_distance,
                icp_max_iterations=icp_max_iterations
            )
            estimator = PPFICPEstimator(params=params)
            
        elif method_name == "ransac":
            voxel_size = trial.suggest_float("voxel_size", 0.02, 0.10, step=0.01)
            icp_max_correspondence_distance = trial.suggest_float("icp_max_correspondence_distance", 0.05, 0.25)
            icp_max_iterations = trial.suggest_int("icp_max_iterations", 10, 100, step=10)
            
            from methods import RansacEstimator, RansacParams
            params = RansacParams(
                voxel_size=voxel_size,
                icp_max_correspondence_distance=icp_max_correspondence_distance,
                icp_max_iterations=icp_max_iterations
            )
            estimator = RansacEstimator(params=params)
            
        else:
            raise ValueError(f"Unknown method for sweep parameters: '{method_name}'")
            
        trans_errs, rot_errs, times, failed = evaluate_pipeline(
            dataset, model, camera, estimator, sweep_indices, depth_trunc=depth_trunc
        )


        
        # Penalize failures heavily
        penalty = 5.0 * failed
        
        # Objective 1: Accuracy (Combined mean translation and rotation errors)
        mean_trans = np.mean(trans_errs) if trans_errs else 1.5
        mean_rot = np.mean(rot_errs) if rot_errs else 180.0
        accuracy_score = mean_trans + (mean_rot / 180.0) + penalty
        
        # Objective 2: Average Execution Time per sample (seconds)
        mean_time = np.mean(times) if times else 5.0  # Penalty time if failed
        
        return accuracy_score, mean_time

    db_name = f"optuna_{study_name}.db"
    db_url = f"sqlite:///{db_name}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        directions=["minimize", "minimize"],  # Minimize error AND execution time
        load_if_exists=True
    )
    print(f"Sweep results are being saved to SQLite database: '{db_name}'")
    
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 50)
    print("SWEEP COMPLETE (PARETO FRONT FINDINGS)")
    print("=" * 50)
    print(f"Found {len(study.best_trials)} optimal trade-off trials on the Pareto Front:")
    for t_idx, trial in enumerate(study.best_trials):
        print(f"\n[Trial {trial.number}]")
        print(f"  - Accuracy Loss Value:   {trial.values[0]:.4f}")
        print(f"  - Avg Execution Time:     {trial.values[1]:.4f}s")
        print("  - Hyperparameters:")
        for name, val in trial.params.items():
            print(f"    * {name}: {val}")
    print("=" * 50)


# =====================================================================
# 4. CLI ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="6D Pose Estimation Benchmark & Parameter Sweep Utility")
    parser.add_argument("--name", type=str, help="Sweep's name")
    parser.add_argument("--sweep", action="store_true", help="Run a hyperparameter sweep using Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for the parameter sweep")
    parser.add_argument("--eval-size", type=int, default=20, help="Number of samples to evaluate on")
    parser.add_argument(
        "--method",
        type=str,
        default="ppf_icp",
        choices=["ppf_icp", "ransac"],
        help="The 6D pose estimation method to benchmark (default: 'ppf_icp')"
    )
    args = parser.parse_args()
    
    # Load model, camera, and dataset
    print("Loading pipeline assets...")
    model = load_hf_model()
    camera = Camera(
        fx=Config.CAMERA_FX, fy=Config.CAMERA_FY,
        cx=Config.CAMERA_CX, cy=Config.CAMERA_CY
    )
    dataset = load_parquet_dataset()
    
    if args.sweep:
        study_name = args.name if args.name else "6DPoseOptimization"
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            study_name=study_name,
            method_name=args.method,
            sweep_size=args.eval_size,
            n_trials=args.trials
        )

    else:
        # Default Evaluation mode
        total_samples = len(dataset)
        eval_indices = np.random.choice(total_samples, min(args.eval_size, total_samples), replace=False)
        
        # Initialize estimator based on selected method
        from methods import get_estimator
        if args.method == "ppf_icp":
            from methods import PPFICPParams
            default_params = PPFICPParams()
            estimator = get_estimator("ppf_icp", params=default_params)
        elif args.method == "ransac":
            from methods import RansacParams
            default_params = RansacParams()
            estimator = get_estimator("ransac", params=default_params)
            
        print(f"Evaluating '{args.method}' parameters on {len(eval_indices)} test samples...")
        print(f"Indices: {eval_indices}\n")
        
        trans_errs, rot_errs, times, failed = evaluate_pipeline(
            dataset, model, camera, estimator, eval_indices
        )
        
        successful = len(trans_errs)
        total = successful + failed
        
        print("\n" + "=" * 50)
        print("BENCHMARK REPORT (Default Parameters)")
        print("=" * 50)
        print(f"Detections & Matches: {successful} / {total} (Success rate: {successful/total*100:.1f}%)")
        if successful > 0:
            print(f"Translation Error (meters):")
            print(f"  - Mean:   {np.mean(trans_errs):.4f}")
            print(f"  - Median: {np.median(trans_errs):.4f}")
            print(f"  - Min:    {np.min(trans_errs):.4f}")
            print(f"  - Max:    {np.max(trans_errs):.4f}")
            print(f"Rotation Error (degrees):")
            print(f"  - Mean:   {np.mean(rot_errs):.2f}°")
            print(f"  - Median: {np.median(rot_errs):.2f}°")
            print(f"  - Min:    {np.min(rot_errs):.2f}°")
            print(f"  - Max:    {np.max(rot_errs):.2f}°")
            print(f"Execution Time (seconds):")
            print(f"  - Mean:   {np.mean(times):.4f}s")
            print(f"  - Median: {np.median(times):.4f}s")
            print(f"  - Min:    {np.min(times):.4f}s")
            print(f"  - Max:    {np.max(times):.4f}s")
        else:
            print("No samples were successfully matched.")
        print("=" * 50)

if __name__ == "__main__":
    main()
