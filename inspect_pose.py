"""
Unified 6D Pose Validation & Inspection Utility.

This script acts as the main execution and debugging interface for the 6D Pose Estimation
pipeline. It loads the streaming Parquet dataset (test split), runs YOLO segmentation,
reconstructs 3D point clouds from RGB-D inputs, aligns reference CAD models using PPF or RANSAC,
and compares predictions against Isaac Sim ground truths.

Usage Modes:
------------
1. Random Validation Mode (--mode random)
   Selects a set of random test samples, runs the full estimation pipeline, and exports
   the output alignment scenes as .glb files to the `debug_output/` folder.
   By default, it wipes the previous `debug_output/` directory so only the current
   run's results are present.

   Example (PPF):
     uv run inspect_pose.py --mode random --random-samples 10 model:ppf model.profile:default

   Example (RANSAC):
     uv run inspect_pose.py --mode random --random-samples 10 model:ransac model.profile:default

2. Targeted Debugging Mode (--mode indices)
   Takes specific sample indices from the test set split (0-1481) and performs a deep
   dive debug. It exports three files per index to the `debug_failures/` folder:
     - yolo_prediction_{idx}.png : 2D bounding boxes and masks plotted on the RGB image.
     - reconstructed_pcd_{idx}.ply : Reconstructed 3D point cloud of the segmented target.
     - combined_scene_idx_{idx}.glb : 3D visualizer containing the point cloud, predicted CAD model
                                      (green), and ground truth CAD model (blue).
   By default, it wipes the previous `debug_failures/` directory so only the current
   debug run's outputs are present.

   Example:
     uv run inspect_pose.py --mode indices --indices 37 52 88 model:ppf model.profile:default

CLI Configuration Overrides:
----------------------------
  --mode random|indices          The validation mode to run (required).
  --random-samples NUM           Number of random samples to run in random validation mode (default: 10).
  --indices IDX [IDX ...]        Specific test sample indices to debug in targeted mode.
  model:ppf|ransac|ransac3dof    The 6D pose estimation method to use (required, no default).
  model.profile:<name>           Tuning profile for the chosen method (required).
  --output-dir PATH              Directory path to save GLB outputs.
"""


import os
import shutil
from datetime import datetime
import numpy as np
import cv2
import copy
import trimesh
import tyro

import open3d as o3d
from datasets import Dataset

from cli_config import InspectArgs
# Import utility classes and functions from pipeline.py
from pipeline import (
    Camera, load_hf_model, load_parquet_dataset,
    process_and_reconstruct, compute_ground_truth_pose,
    instance_detected, load_cad_meshes
)
from methods.base import BasePoseEstimator


# =====================================================================
# 0. DEBUGGING VISUALIZATION UTILITIES
# =====================================================================
def export_debug_scene(
    pcd: o3d.geometry.PointCloud,
    cad_mesh: o3d.geometry.TriangleMesh,
    T_final: np.ndarray,
    T_ground_truth: np.ndarray,
    output_path: str
) -> None:
    """
    Constructs a debugging 3D scene containing the reconstructed point cloud, the 
    predicted pose mesh (green), the ground truth pose mesh (blue), and the origin 
    coordinate frame. Exports the combined geometries as a GLB file.

    Coordinate Frames & Reference System:
    --------------------------------------
    The debugging view is plotted in the Robot Base Frame ('base_link'). The origin [0, 0, 0] 
    corresponds to the physical center of the robot base. The X-axis (Red) points forward 
    from the robot, the Y-axis (Green) points left, and the Z-axis (Blue) points straight up.
    
    Setting the robot's base frame as the reference frame makes it intuitive to verify if 
    the cart rests correctly on the floor (Z ~= 0) and is in front of the robot.
    """
    # 1. Create origin coordinate axes (Red=X, Green=Y, Blue=Z)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 2. Paint the point cloud a neutral gray. We use deepcopy here to avoid modifying
    # the original point cloud object in memory, preserving its original colors for
    # any callers or subsequent tasks in the pipeline.
    pcd_vis = copy.deepcopy(pcd)
    pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])
    
    # 3. Transform CAD mesh for predicted pose (GREEN)
    predicted_mesh = copy.deepcopy(cad_mesh)
    predicted_mesh.transform(T_final)
    predicted_mesh.paint_uniform_color([0.0, 1.0, 0.0]) # GREEN
    
    # 4. Transform CAD mesh for ground truth pose (BLUE)
    gt_mesh = copy.deepcopy(cad_mesh)
    gt_mesh.transform(T_ground_truth)
    gt_mesh.paint_uniform_color([0.0, 0.0, 1.0]) # BLUE
    
    # 5. Build trimesh scene and export
    # Trimesh expects vertex colors as uint8 [0..255], whereas Open3D specifies them as
    # float32 [0..1]. We convert between the two by multiplying by 255. We iterate over
    # the geometries, wrap them in trimesh objects, and add them to a trimesh.Scene.
    scene = trimesh.Scene()
    for name, m in [("frame", world_frame), ("pred", predicted_mesh), ("gt", gt_mesh)]:
        tm = trimesh.Trimesh(
            vertices=np.asarray(m.vertices),
            faces=np.asarray(m.triangles),
            vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8)
                          if m.has_vertex_colors() else None
        )
        scene.add_geometry(tm, geom_name=name)
        
    pts = np.asarray(pcd_vis.points)
    cols = (np.asarray(pcd_vis.colors) * 255).astype(np.uint8) if pcd_vis.has_colors() else None
    scene.add_geometry(trimesh.PointCloud(vertices=pts, colors=cols), geom_name="scene")
    
    scene.export(output_path)
    print(f"Saved GLB scene to {output_path}")


# =====================================================================
# 1. RANDOM VALIDATION INSPECTION RUNNER
# =====================================================================
def run_random_inspection(
    num_samples: int,
    model,
    camera: Camera,
    dataset: Dataset,
    estimator: BasePoseEstimator,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    output_dir: str,
    depth_trunc: float
) -> None:
    """
    Runs the 6D Pose estimation pipeline on random test split samples,
    exporting the combined 3D scenes (PCD + CAD meshes) as GLB files to the output directory.
    """
    # Overwrite the old debug folder to ensure we don't mix results from previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total samples from map-style dataset (O(1) lookup)
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Select unique random indices without replacement so we don't inspect the same sample twice
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    print(f"Starting random inspection on {num_samples} test samples: {random_indices}")
    
    for loop_idx, sample_idx in enumerate(random_indices):
        print(f"\n--- Processing Sample {loop_idx + 1}/{num_samples} (Index {sample_idx}) ---")
        
        # Load raw PIL images and binary depth buffers
        row = dataset[int(sample_idx)]
        img = row["rgb"]
        depth_bytes = row["depth"]
        
        # 1. Run YOLO detection (with retina_masks=True to get high-resolution masks)
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            print(f"Skipping Index {sample_idx}: No cart instance detected.")
            continue
            
        # 2. Segment and Reconstruct 3D Point Cloud
        # This isolates the cart points and projects them to camera-frame 3D coordinates.
        cart_type, pcd, frame = process_and_reconstruct(
            img, depth_bytes, result, camera, depth_trunc=depth_trunc, return_frame=True
        )
        print(f"Recognized class: {cart_type}")
        
        # Retrieve preloaded mesh
        cad_mesh = meshes.get(cart_type)
        if cad_mesh is None:
            print(f"Skipping Index {sample_idx}: CAD file for class {cart_type} not found.")
            continue
        
        # 3. Perform 6D Pose Estimation
        # This executes the selected estimation method (e.g. PPF+ICP or RANSAC+ICP)
        T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=cart_type, frame=frame)
        if T_final is None:
            print(f"Skipping Index {sample_idx}: Pose estimation failed.")
            continue
        print("Final 6D Pose Matrix (Refined via ICP):\n", T_final)
        
        # 4. Compute Ground Truth Pose
        T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
        T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
        T_ground_truth = compute_ground_truth_pose(T_world_camera, T_world_cart, T_robot_camera=estimator.extrinsic)
        print("Ground Truth 6D Pose Matrix:\n", T_ground_truth)
        
        # 5. Export Combined Scene to GLB (Robot Frame)
        pcd_robot = copy.deepcopy(pcd)
        pcd_robot.transform(estimator.extrinsic)
        output_file = os.path.join(output_dir, f"combined_scene_sample_{sample_idx}.glb")
        export_debug_scene(pcd_robot, cad_mesh, T_final, T_ground_truth, output_file)
        
    print(f"\nAll operations completed. GLB scenes saved to: '{output_dir}/'")


# =====================================================================
# 2. TARGETED DEBUGGING RUNNER
# =====================================================================
def run_targeted_inspection(
    indices: list[int],
    model,
    camera: Camera,
    dataset: Dataset,
    estimator: BasePoseEstimator,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    output_dir: str,
    depth_trunc: float
) -> None:
    """
    Performs targeted debugging on specific sample indices, exporting
    the 2D YOLO overlays (PNG), the 3D target point cloud (PLY), and
    the aligned scene visualization (GLB).
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running targeted inspection on specific indices: {indices}")
    
    for idx in indices:
        print(f"\n--- Targeted Inspection on Index {idx} ---")
        if idx >= len(dataset):
            print(f"Skipping Index {idx}: Index is out of range for the loaded dataset split.")
            continue
            
        row = dataset[int(idx)]
        img = row["rgb"]
        depth_bytes = row["depth"]

        # 1. Run YOLO and save 2D overlays
        result = model(img, retina_masks=True, verbose=False)
        plotted_img = result[0].plot()
        
        yolo_path = os.path.join(output_dir, f"yolo_prediction_{idx}.png")
        cv2.imwrite(yolo_path, plotted_img)
        print(f"Saved 2D YOLO prediction plot to: {yolo_path}")
        
        if not instance_detected(result):
            print(f"No instance detected on Index {idx}. Skipping 3D inspection steps.")
            continue
            
        # 2. Segment and Reconstruct Point Cloud
        cart_type, pcd, frame = process_and_reconstruct(
            img, depth_bytes, result, camera, depth_trunc=depth_trunc, return_frame=True
        )
        print(f"Recognized class: {cart_type}")
        
        # Transform the point cloud to the robot frame using the estimator's extrinsic
        pcd_robot = copy.deepcopy(pcd)
        pcd_robot.transform(estimator.extrinsic)
        
        pcd_path = os.path.join(output_dir, f"reconstructed_pcd_{idx}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd_robot)
        print(f"Saved reconstructed 3D point cloud (Robot Frame) to: {pcd_path}")
        
        # 3. Load reference CAD mesh and run Pose Estimation
        cad_mesh = meshes.get(cart_type)
        if cad_mesh is not None:
            T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=cart_type, frame=frame)
            
            if T_final is not None:
                # Retrieve ground truth pose matrix in base_link coordinates
                T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
                T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
                T_ground_truth = compute_ground_truth_pose(T_world_camera, T_world_cart, T_robot_camera=estimator.extrinsic)
                output_glb = os.path.join(output_dir, f"combined_scene_idx_{idx}.glb")
                export_debug_scene(pcd_robot, cad_mesh, T_final, T_ground_truth, output_glb)
                print(f"Saved combined 6D Pose scene to: {output_glb}")
            else:
                print("Pose estimation failed. Cannot generate combined GLB scene.")
        else:
            print(f"CAD mesh for {cart_type} not found. Skipping GLB scene export.")
            
    print(f"\nAll debug outputs for targeted indices successfully saved to: '{output_dir}/'")


# =====================================================================
# 3. CLI ARGUMENT PARSER AND MAIN ENTRY POINT
# =====================================================================
def main():
    args = tyro.cli(InspectArgs)

    # Hydra used to auto-manage a per-run timestamped working directory
    # (HydraConfig.get().runtime.output_dir); tyro has no equivalent feature,
    # so we roll our own. This directory is disposable -- wiped by
    # run_random_inspection/run_targeted_inspection on every run and already
    # gitignored -- so a plain timestamp is enough, no extra enrichment.
    debug_root = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load pipeline assets
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

    # Hoist CAD mesh loading
    meshes = load_cad_meshes()

    # Directly construct the chosen preset's estimator -- no _target_ string
    # resolution needed, args.model.ESTIMATOR_CLS is already the concrete class.
    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    estimator = args.model.ESTIMATOR_CLS(params=args.model.profile.params, extrinsic=extrinsic)

    # Pre-prepare all meshes on estimator
    for cart_type, mesh in meshes.items():
        estimator.prepare(mesh, cart_type)

    # args.mode is a Literal["random", "indices"] -- tyro already validated it
    # up front, so there's no invalid-mode branch to handle here anymore.
    if args.mode == "random":
        run_random_inspection(
            num_samples=args.random_samples,
            model=model,
            camera=camera,
            dataset=dataset,
            estimator=estimator,
            meshes=meshes,
            output_dir=os.path.join(debug_root, args.output_dir),
            depth_trunc=args.model.profile.depth_trunc
        )
    else:
        run_targeted_inspection(
            indices=list(args.indices),
            model=model,
            camera=camera,
            dataset=dataset,
            estimator=estimator,
            meshes=meshes,
            output_dir=os.path.join(debug_root, "debug_failures"),
            depth_trunc=args.model.profile.depth_trunc
        )

if __name__ == "__main__":
    main()
