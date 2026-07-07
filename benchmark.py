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
    Config, Camera, PPFICPParams, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, SixDPoseEstimation, compute_ground_truth_pose,
    instance_detected
)

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
    params: PPFICPParams,
    sample_indices: list[int]
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
            cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
        except Exception:
            failed_count += 1
            continue
            
        # Verify CAD mesh exists
        mesh_file = f"meshes/{cart_type}.ply"
        if not os.path.exists(mesh_file):
            failed_count += 1
            continue
        cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # 3. Perform 6D Pose Estimation (PPF + ICP) with timing
        start_time = time.time()
        try:
            T_final = SixDPoseEstimation(pcd, cad_mesh, params=params)
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
def run_parameter_sweep(dataset, model, camera, sweep_size: int, n_trials: int):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.
    """
    # Select a fixed validation subset for the sweep to ensure consistent comparisons
    total_samples = len(dataset)
    sweep_indices = np.random.choice(total_samples, min(sweep_size, total_samples), replace=False)
    print(f"Running Multi-Objective sweep over {len(sweep_indices)} samples for {n_trials} trials...")
    print(f"Sweep validation indices: {sweep_indices}\n")
    
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Define hyperparameter search space
        ppf_sampling_step = trial.suggest_float("ppf_sampling_step", 0.02, 0.10, step=0.01)
        ppf_distance_step = trial.suggest_float("ppf_distance_step", 0.02, 0.10, step=0.01)
        ppf_match_threshold = trial.suggest_float("ppf_match_threshold", 0.02, 0.10, step=0.01)
        ppf_match_tolerance = trial.suggest_float("ppf_match_tolerance", 0.01, 0.08, step=0.01)
        icp_max_correspondence_distance = trial.suggest_float("icp_max_correspondence_distance", 0.02, 0.20)
        icp_max_iterations = trial.suggest_int("icp_max_iterations", 10, 100, step=10)
        
        params = PPFICPParams(
            ppf_sampling_step=ppf_sampling_step,
            ppf_distance_step=ppf_distance_step,
            ppf_match_threshold=ppf_match_threshold,
            ppf_match_tolerance=ppf_match_tolerance,
            icp_max_correspondence_distance=icp_max_correspondence_distance,
            icp_max_iterations=icp_max_iterations
        )
        
        trans_errs, rot_errs, times, failed = evaluate_pipeline(
            dataset, model, camera, params, sweep_indices
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

    db_name = "optuna_pareto_study.db"
    db_url = f"sqlite:///{db_name}"
    
    study = optuna.create_study(
        study_name="6d_pose_optimization",
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
    parser.add_argument("--sweep", action="store_true", help="Run a hyperparameter sweep using Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for the parameter sweep")
    parser.add_argument("--eval-size", type=int, default=20, help="Number of samples to evaluate on")
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
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            sweep_size=args.eval_size,
            n_trials=args.trials
        )
    else:
        # Default Evaluation mode
        total_samples = len(dataset)
        eval_indices = np.random.choice(total_samples, min(args.eval_size, total_samples), replace=False)
        default_params = PPFICPParams()
        
        print(f"Evaluating default parameters on {len(eval_indices)} test samples...")
        print(f"Indices: {eval_indices}\n")
        
        trans_errs, rot_errs, times, failed = evaluate_pipeline(
            dataset, model, camera, default_params, eval_indices
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
