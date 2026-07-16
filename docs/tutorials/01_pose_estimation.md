# Tutorial: Visualizing the 3D Pose Estimation Pipeline

This tutorial provides a complete code walkthrough to run the 6D pose estimation pipeline on a single test dataset sample. You will learn how to load data, run 2D YOLO segmentation, reconstruct a 3D point cloud, and align it with a CAD mesh.

---

## 1. Setup and Initialization

First, initialize the environment and load the model, dataset, and camera intrinsics:

```python
import numpy as np
import open3d as o3d
import torch
from main import Camera, load_hf_model, load_parquet_dataset, process_and_reconstruct, compute_ground_truth_pose
from methods.base import prepare_scene_point_cloud
from methods.ransac import RansacEstimator, RansacParams

# 1. Load pipeline assets
print("Loading model and dataset...")
model = load_hf_model()
dataset = load_parquet_dataset()

# 2. Setup camera intrinsics
camera = Camera(
    fx=639.99768, fy=639.99768,
    cx=640.0, cy=400.0
)
```

---

## 2. Segment and Reconstruct Point Cloud

We select sample index `315` from the dataset, run YOLO segmentation, and reconstruct the segmented target's 3D point cloud in the camera's local coordinate frame:

```python
sample_idx = 315
row = dataset[int(sample_idx)]
img = row["rgb"]
depth_bytes = row["depth"]

# 1. Run YOLO inference
print("Running YOLO segmentation...")
result = model(img, retina_masks=True, verbose=False)

# 2. Extract segment mask and reconstruct the 3D Point Cloud in camera frame
cart_type, pcd_cam = process_and_reconstruct(img, depth_bytes, result, camera)
print(f"Recognized Cart Class: {cart_type}")
print(f"Number of reconstructed points: {len(pcd_cam.points)}")
```

---

## 3. Preprocess and Align Coordinate Frames

Next, we estimate surface normals for the point cloud (essential for Point-to-Plane ICP matching), orient them consistently towards the camera, and transform the coordinates into the robot base frame (`base_link`):

```python
# Extrinsic camera-to-robot transform matrix
T_robot_camera = np.array([
    [0.5, 0.0,  0.866, 0.439],
    [0.0, 1.0, -0.0,   0.0  ],
    [-0.866, 0.0, 0.5, 0.304],
    [0.0, 0.0,  0.0,   1.0  ]
])

# Compute normals and project point cloud to base_link frame
pcd_robot = prepare_scene_point_cloud(pcd_cam, T_robot_camera)
```

---

## 4. Run RANSAC Pose Registration

Now we load the target CAD model mesh, downsample the point clouds, compute FPFH features, and run RANSAC global registration followed by dual-hypothesis ICP refinement:

```python
# Load CAD model mesh
mesh_path = f"meshes/{cart_type}.ply"
cad_mesh = o3d.io.read_triangle_mesh(mesh_path)

# Initialize RANSAC Estimator with custom parameters
params = RansacParams(
    voxel_size=0.06,
    ransac_max_iterations=100000,
    icp_max_correspondence_distance=0.15,
    icp_max_iterations=100
)
estimator = RansacEstimator(params=params, extrinsic=T_robot_camera)

# Run 6D pose estimation
print("Running global registration and ICP refinement...")
T_estimated = estimator.estimate_pose(pcd_cam, cad_mesh)
print("Estimated 6D Pose Matrix:\n", T_estimated)
```

---

## 5. Verify against Ground Truth

Finally, load the ground truth pose matrix from Isaac Sim and compute the translation (in meters) and geodesic rotation (in degrees) errors:

```python
from benchmark import compute_translation_error, compute_rotation_error

# Retrieve ground truth pose relative to base_link
T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
T_gt = compute_ground_truth_pose(T_world_camera, T_world_cart, T_robot_camera=T_robot_camera)
print("Ground Truth 6D Pose Matrix:\n", T_gt)

# Calculate errors
t_err = compute_translation_error(T_estimated, T_gt)
r_err = compute_rotation_error(T_estimated, T_gt)

print(f"\nPose Estimation Error metrics:")
print(f"  - Translation Error: {t_err:.4f} meters")
print(f"  - Rotation Error:    {r_err:.2f} degrees")
```
