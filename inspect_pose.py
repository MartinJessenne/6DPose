import os
import argparse
import numpy as np
import open3d as o3d

# Import configurations, classes, and pipeline modules from main.py
from main import (
    Config, Camera, PPFICPParams, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, SixDPoseEstimation, compute_ground_truth_pose,
    export_debug_scene, instance_detected
)

# Import metrics from benchmark.py
from benchmark import compute_translation_error, compute_rotation_error

# =====================================================================
# CONFIGURE YOUR TUNED HYPERPARAMETERS HERE
# =====================================================================
# Modify these fields directly to inspect how different configurations behave.
TUNED_PARAMS = PPFICPParams(
    ppf_sampling_step=0.04,
    ppf_distance_step=0.03,
    ppf_match_threshold=0.03,
    ppf_match_tolerance=0.01,
    icp_max_correspondence_distance=0.09915124703931394,
    icp_max_iterations=50
)

# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visual Inspection Debugger: Run specific or random test samples using custom hyperparameters."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--samples",
        type=int,
        nargs="+",
        help="Space-separated list of sample indices to visually inspect (e.g. --samples 47 or --samples 12 35 77)"
    )
    group.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Number of random samples to draw from the test set to inspect visually (e.g. --random 3)"
    )
    
    args = parser.parse_args()
    
    # 1. Load pipeline assets
    print("Loading pipeline assets...")
    model = load_hf_model()
    camera = Camera(
        fx=Config.CAMERA_FX, fy=Config.CAMERA_FY,
        cx=Config.CAMERA_CX, cy=Config.CAMERA_CY
    )
    local_dataset = load_parquet_dataset()
    
    total_samples = len(local_dataset)
    
    # 2. Resolve sample indices based on user request
    if args.random is not None:
        num_random = min(args.random, total_samples)
        sample_indices = np.random.choice(total_samples, num_random, replace=False).tolist()
        print(f"Mode: Random Selection | Selecting {num_random} samples: {sample_indices}")
    else:
        sample_indices = []
        for idx in args.samples:
            if idx < 0 or idx >= total_samples:
                print(f"Warning: Index {idx} is out of bounds (max size: {total_samples}). Skipping.")
            else:
                sample_indices.append(idx)
        print(f"Mode: Focused Selection | Selected sample indices: {sample_indices}")
        
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 3. Process each sample
    for loop_idx, sample_idx in enumerate(sample_indices):
        print(f"\n" + "=" * 60)
        print(f"Processing Visual Inspection for Sample {loop_idx + 1}/{len(sample_indices)} (Index {sample_idx})")
        print("=" * 60)
        
        # Load sample data
        img = local_dataset["rgb"][sample_idx]
        depth_bytes = local_dataset["depth"][sample_idx]
        
        # Run YOLO inference
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            print(f"Skipping Sample {sample_idx}: No cart instance detected by YOLO.")
            continue
            
        # Reconstruct point cloud
        try:
            cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
            print(f"Recognized Cart Class: {cart_type}")
        except Exception as e:
            print(f"Skipping Sample {sample_idx}: Failed during point cloud reconstruction. Error: {e}")
            continue
            
        # Load CAD mesh
        mesh_file = f"meshes/{cart_type}.ply"
        if not os.path.exists(mesh_file):
            print(f"Skipping Sample {sample_idx}: CAD file '{mesh_file}' not found.")
            continue
        cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # Run 6D Pose Estimation with TUNED_PARAMS configured in this file
        try:
            T_final = SixDPoseEstimation(pcd, cad_mesh, params=TUNED_PARAMS)
            if T_final is None:
                print(f"Skipping Sample {sample_idx}: Pose estimation returned None.")
                continue
        except Exception as e:
            print(f"Skipping Sample {sample_idx}: Pose estimation crashed. Error: {e}")
            continue
            
        # Calculate Ground Truth pose
        T_ground_truth = compute_ground_truth_pose(local_dataset, sample_idx)
        
        # Compute metrics
        err_trans = compute_translation_error(T_final, T_ground_truth)
        err_rot = compute_rotation_error(T_final, T_ground_truth)
        
        print("\nPose Estimation Complete:")
        print(f"  - Translation Error: {err_trans:.4f} meters")
        print(f"  - Rotation Error:    {err_rot:.2f}°")
        
        # Save output scene
        output_file = os.path.join(Config.OUTPUT_DIR, f"inspect_scene_sample_{sample_idx}.glb")
        export_debug_scene(pcd, cad_mesh, T_final, T_ground_truth, output_file)
        print(f"Visualized scene exported to: {output_file}")
        
    print(f"\nAll inspections complete. Output files saved in '{Config.OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
