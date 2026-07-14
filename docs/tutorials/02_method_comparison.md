# Tutorial: Comparing RANSAC and PPF Pose Estimators

This tutorial demonstrates how to run both `RansacEstimator` and `PPFEstimator` side-by-side on the same input observation to compare their execution speed and accuracy.

---

## 1. Loader Setup

We load a sample image from the dataset, YOLO segmentation, and setup both estimators with their default configurations:

```python
import time
import numpy as np
import open3d as o3d
from main import Camera, load_hf_model, load_parquet_dataset, process_and_reconstruct, compute_ground_truth_pose
from methods.ransac import RansacEstimator
from methods.ppf import PPFEstimator
from benchmark import compute_translation_error, compute_rotation_error

# 1. Load assets
model = load_hf_model()
dataset = load_parquet_dataset()
camera = Camera(fx=639.99768, fy=639.99768, cx=640.0, cy=400.0)
T_robot_camera = np.array([
    [0.5, 0.0,  0.866, 0.439],
    [0.0, 1.0, -0.0,   0.0  ],
    [-0.866, 0.0, 0.5, 0.304],
    [0.0, 0.0,  0.0,   1.0  ]
])

# 2. Select a target sample (e.g. sample 1248)
sample_idx = 1248
img = dataset["rgb"][sample_idx]
depth_bytes = dataset["depth"][sample_idx]
result = model(img, retina_masks=True, verbose=False)
cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)

# Load ground truth
T_gt = compute_ground_truth_pose(dataset, sample_idx, T_robot_camera=T_robot_camera)

# Load CAD mesh
cad_mesh = o3d.io.read_triangle_mesh(f"meshes/{cart_type}.ply")
```

---

## 2. Initialize Estimators

Create both estimators, passing the camera extrinsic matrix during initialization:

```python
# Initialize RANSAC estimator
ransac_estimator = RansacEstimator(extrinsic=T_robot_camera)

# Initialize PPF estimator
ppf_estimator = PPFEstimator(extrinsic=T_robot_camera)
```

---

## 3. Run and Measure RANSAC

Run RANSAC pose estimation and record its duration:

```python
print("\n--- Running RANSAC Pose Estimation ---")
start_time = time.time()
T_ransac = ransac_estimator.estimate_pose(pcd, cad_mesh)
ransac_time = time.time() - start_time

if T_ransac is not None:
    t_err_ransac = compute_translation_error(T_ransac, T_gt)
    r_err_ransac = compute_rotation_error(T_ransac, T_gt)
    print(f"RANSAC finished in {ransac_time:.4f}s")
    print(f"  - Translation Error: {t_err_ransac:.4f} m")
    print(f"  - Rotation Error:    {r_err_ransac:.2f}°")
else:
    print("RANSAC alignment failed.")
```

---

## 4. Run and Measure PPF-ICP

Run Point Pair Features matching and record its duration:

```python
print("\n--- Running PPF-ICP Pose Estimation ---")
start_time = time.time()
T_ppf = ppf_estimator.estimate_pose(pcd, cad_mesh)
ppf_time = time.time() - start_time

if T_ppf is not None:
    t_err_ppf = compute_translation_error(T_ppf, T_gt)
    r_err_ppf = compute_rotation_error(T_ppf, T_gt)
    print(f"PPF-ICP finished in {ppf_time:.4f}s")
    print(f"  - Translation Error: {t_err_ppf:.4f} m")
    print(f"  - Rotation Error:    {r_err_ppf:.2f}°")
else:
    print("PPF-ICP alignment failed.")
```

---

## 5. Result Comparison

Typically, **RANSAC** is much faster because it operates on voxel-downsampled features (FPFH), whereas **PPF** trains a full hash table on the CAD model and matches point pairs. Compare the trade-offs on your machine to select the best option for your robot's real-time loops.
