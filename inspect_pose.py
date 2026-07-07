"""
Unified 6D Pose Validation & Inspection Utility.

This script acts as the main execution and debugging interface for the 6D Pose Estimation
pipeline. It loads the streaming Parquet dataset (test split), runs YOLO segmentation,
reconstructs 3D point clouds from RGB-D inputs, aligns reference CAD models using PPF or RANSAC,
and compares predictions against Isaac Sim ground truths.

Usage Modes:
------------
1. Random Validation Mode (--random)
   Selects a set of random test samples, runs the full estimation pipeline, and exports
   the output alignment scenes as .glb files to the `debug_output/` folder.
   By default, it wipes the previous `debug_output/` directory so only the current
   run's results are present.
   
   Example (PPF):
     uv run inspect_pose.py --random 10 --method ppf_icp
     
   Example (RANSAC):
     uv run inspect_pose.py --random 10 --method ransac

2. Targeted Debugging Mode (--indices)
   Takes specific sample indices from the test set split (0-1481) and performs a deep
   dive debug. It exports three files per index to the `debug_failures/` folder:
     - yolo_prediction_{idx}.png : 2D bounding boxes and masks plotted on the RGB image.
     - reconstructed_pcd_{idx}.ply : Reconstructed 3D point cloud of the segmented target.
     - combined_scene_idx_{idx}.glb : 3D visualizer containing the point cloud, predicted CAD model
                                      (green), and ground truth CAD model (blue).
   By default, it wipes the previous `debug_failures/` directory so only the current
   debug run's outputs are present.
   
   Example:
     uv run inspect_pose.py --indices 37 52 88 --method ppf_icp

CLI Arguments:
--------------
  --random [NUM]           Triggers the random validation inspection mode, optionally specifying the number of samples (default: 10).
  --indices IDX [IDX ...]  Triggers targeted debugging mode on one or more sample indices.
  --method {ppf_icp,ransac}  The 6D pose estimation method to use (default: 'ppf_icp').
"""


import os
import argparse
import shutil
import numpy as np
import cv2

import open3d as o3d
from datasets import Dataset

# Import utility classes and functions from main.py
from main import (
    Config, Camera, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, compute_ground_truth_pose,
    export_debug_scene, instance_detected
)
from methods.base import BasePoseEstimator


# =====================================================================
# 1. RANDOM VALIDATION INSPECTION RUNNER
# =====================================================================
def run_random_inspection(
    num_samples: int,
    model,
    camera: Camera,
    dataset: Dataset,
    estimator: BasePoseEstimator
) -> None:
    """
    Runs the 6D Pose estimation pipeline on random test split samples,
    exporting the combined 3D scenes (PCD + CAD meshes) as GLB files to Config.OUTPUT_DIR.
    
    Args:
        num_samples (int): Number of random test samples to evaluate.
        model: The initialized YOLO segmentation model.
        camera (Camera): The pinhole camera model.
        dataset (Dataset): The Hugging Face dataset containing test samples.
        params (PPFICPParams): Optimized hyperparameters for PPF + ICP matching.
        
    Example:
        >>> # To evaluate 5 random samples using the CLI:
        >>> # python inspect_pose.py --random --num-samples 5
        >>> run_random_inspection(5, model, camera, dataset, params)
    """
    output_dir = Config.OUTPUT_DIR
    
    # Overwrite the old debug folder to ensure we don't mix results from previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Select unique random indices without replacement so we don't inspect the same sample twice
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    print(f"Starting random inspection on {num_samples} test samples: {random_indices}")
    
    for loop_idx, sample_idx in enumerate(random_indices):
        print(f"\n--- Processing Sample {loop_idx + 1}/{num_samples} (Index {sample_idx}) ---")
        
        # Load raw PIL images and binary depth buffers
        img = dataset["rgb"][sample_idx]
        depth_bytes = dataset["depth"][sample_idx]
        
        # 1. Run YOLO detection (with retina_masks=True to get high-resolution masks)
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            print(f"Skipping Index {sample_idx}: No cart instance detected.")
            continue
            
        # 2. Segment and Reconstruct 3D Point Cloud (with new 20.0m depth threshold)
        # This isolates the cart points and projects them to camera-frame 3D coordinates.
        cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
        print(f"Recognized class: {cart_type}")
        
        # Verify CAD mesh file exists before running registration
        mesh_file = f"meshes/{cart_type}.ply"
        if not os.path.exists(mesh_file):
            print(f"Skipping Index {sample_idx}: CAD file {mesh_file} not found.")
            continue
        cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # 3. Perform 6D Pose Estimation
        # This executes the selected estimation method (e.g. PPF+ICP or RANSAC+ICP)
        T_final = estimator.estimate_pose(pcd, cad_mesh)
        if T_final is None:
            print(f"Skipping Index {sample_idx}: Pose estimation failed.")
            continue
        print("Final 6D Pose Matrix (Refined via ICP):\n", T_final)
        
        # 4. Compute Ground Truth Pose
        # This applies the chain: T_robot = T_robot_cam @ T_usd_to_cv @ T_world_cam @ T_world_cart
        T_ground_truth = compute_ground_truth_pose(dataset, sample_idx)
        print("Ground Truth 6D Pose Matrix:\n", T_ground_truth)
        
        # 5. Export Combined Scene to GLB
        # Saves the scene as combined_scene_sample_{sample_idx}.glb using the actual dataset index
        output_file = os.path.join(output_dir, f"combined_scene_sample_{sample_idx}.glb")
        export_debug_scene(pcd, cad_mesh, T_final, T_ground_truth, output_file)
        
    print(f"\nAll operations completed. GLB scenes saved to: '{output_dir}/'")


# =====================================================================
# 2. TARGETED DEBUGGING RUNNER
# =====================================================================
def run_targeted_inspection(
    indices: list[int],
    model,
    camera: Camera,
    dataset: Dataset,
    estimator: BasePoseEstimator
) -> None:
    """
    Performs targeted debugging on specific sample indices, exporting
    the 2D YOLO overlays (PNG), the 3D target point cloud (PLY), and
    the aligned scene visualization (GLB).
    
    Args:
        indices (list[int]): List of specific dataset indices to debug.
        model: The initialized YOLO segmentation model.
        camera (Camera): The pinhole camera model.
        dataset (Dataset): The Hugging Face dataset.
        params (PPFICPParams): Optimized hyperparameters.
        
    Example:
        >>> # To debug indices 37 and 52 using the CLI:
        >>> # python inspect_pose.py --indices 37 52
        >>> run_targeted_inspection([37, 52], model, camera, dataset, params)
    """
    output_dir = "debug_failures"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    
    print(f"Running targeted inspection on specific indices: {indices}")
    
    for idx in indices:
        print(f"\n--- Targeted Inspection on Index {idx} ---")
        if idx >= len(dataset):
            print(f"Skipping Index {idx}: Index is out of range for the loaded dataset split.")
            continue
            
        img = dataset["rgb"][idx]
        depth_bytes = dataset["depth"][idx]
        
        # 1. Run YOLO and save 2D overlays
        # Plots bounding boxes and instance segmentation boundaries for 2D mask validation
        result = model(img, retina_masks=True, verbose=False)
        plotted_img = result[0].plot()
        
        yolo_path = os.path.join(output_dir, f"yolo_prediction_{idx}.png")
        cv2.imwrite(yolo_path, plotted_img)
        print(f"Saved 2D YOLO prediction plot to: {yolo_path}")
        
        if not instance_detected(result):
            print(f"No instance detected on Index {idx}. Skipping 3D inspection steps.")
            continue
            
        # 2. Segment and Reconstruct Point Cloud
        # Generates a 3D point cloud using the crop parameters and depth buffer
        cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
        print(f"Recognized class: {cart_type}")
        
        # Save point cloud to PLY format (includes position, colors, and surface normals)
        # This PLY file can be opened directly in Blender or Meshlab to check for floor/obstacle leakage
        pcd_path = os.path.join(output_dir, f"reconstructed_pcd_{idx}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Saved reconstructed 3D point cloud to: {pcd_path}")
        
        # 3. Load reference CAD mesh and run Pose Estimation
        mesh_file = f"meshes/{cart_type}.ply"
        if os.path.exists(mesh_file):
            cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
            T_final = estimator.estimate_pose(pcd, cad_mesh)
            
            if T_final is not None:
                # Retrieve ground truth pose matrix in base_link coordinates
                T_ground_truth = compute_ground_truth_pose(dataset, idx)
                output_glb = os.path.join(output_dir, f"combined_scene_idx_{idx}.glb")
                # Save combined scene (Scene point cloud + Predicted CAD [Green] + Ground Truth CAD [Blue])
                export_debug_scene(pcd, cad_mesh, T_final, T_ground_truth, output_glb)
                print(f"Saved combined 6D Pose scene to: {output_glb}")
            else:
                print("Pose estimation failed. Cannot generate combined GLB scene.")
        else:
            print(f"CAD file {mesh_file} not found. Skipping GLB scene export.")
            
    print(f"\nAll debug outputs for targeted indices successfully saved to: '{output_dir}/'")


# =====================================================================
# 3. CLI ARGUMENT PARSER AND MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 6D Pose Validation & Inspection Utility")
    
    # Establish mutually exclusive execution modes (cannot run both at the same time)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--random", 
        type=int, 
        nargs="?",
        const=10,
        help="Run random validation mode, optionally specifying the number of samples (default: 10)"
    )
    group.add_argument(
        "--indices", 
        type=int, 
        nargs="+", 
        help="Run targeted debugging mode, exporting PNG, PLY, and GLB files for specific sample indices"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ppf_icp",
        choices=["ppf_icp", "ransac"],
        help="The 6D pose estimation method to use (default: 'ppf_icp')"
    )

    
    args = parser.parse_args()
    
    # Load pipeline assets
    print("Loading pipeline assets...")
    model = load_hf_model()
    camera = Camera(
        fx=Config.CAMERA_FX, fy=Config.CAMERA_FY,
        cx=Config.CAMERA_CX, cy=Config.CAMERA_CY
    )
    dataset = load_parquet_dataset()
    
    # Instantiate chosen estimator and parameters
    from methods import get_estimator
    if args.method == "ppf_icp":
        from methods import PPFICPParams
        method_params = PPFICPParams()
        estimator = get_estimator("ppf_icp", params=method_params)
    elif args.method == "ransac":
        from methods import RansacParams
        method_params = RansacParams()
        estimator = get_estimator("ransac", params=method_params)
    
    if args.random is not None:
        run_random_inspection(
            num_samples=args.random,
            model=model,
            camera=camera,
            dataset=dataset,
            estimator=estimator
        )
    elif args.indices is not None:
        run_targeted_inspection(
            indices=args.indices,
            model=model,
            camera=camera,
            dataset=dataset,
            estimator=estimator
        )

if __name__ == "__main__":
    main()
