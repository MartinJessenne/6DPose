# Coding Agent Guide for 6DPose Project

Welcome! This guide is written specifically for AI Coding Agents and developer tooling to quickly understand the project's architecture, code conventions, environment settings, and documentation structure.

---

## 1. System Architecture

The project implements a 6D Pose Estimation pipeline for industrial towing carts. It has two main sections:
1. **Front-End (YOLO Segmentation)**: Runs 2D YOLOv8 instance segmentation (`ultralytics`), crops the depth map using the predicted bounding box, and creates a masked RGB-D frame (`MaskedImageFrame` in `main.py`).
2. **Back-End (3D Registration)**: Back-projects the depth crop into a 3D point cloud (`open3d`), estimates surface normals, transforms coordinates into the robot base frame (`base_link`), and solves the registration pose relative to the CAD mesh target using `BasePoseEstimator` subclasses.

### Peer Models (`BasePoseEstimator`):
- **`RansacEstimator`** (`methods/ransac.py`): Extract FPFH feature descriptors on downsampled point clouds, run RANSAC global registration, and refine using point-to-plane ICP.
- **`PPFICPEstimator`** (`methods/ppf_icp.py`): Training a PPF (Point Pair Feature) match database using OpenCV's `ppf_match_3d` module and refining via ICP.
- **Refinement Helper**: Both estimators invoke `refine_pose_dual_hypothesis` (in `methods/base.py`) which computes Point-to-Plane ICP on two orientations (original and rotated 180° around the local Z-axis) to handle cart y-axis symmetry.

---

## 2. Configuration & Parameter Management

- **Hydra Configs**: All parameters (dataset globs, camera intrinsics, model choices, RANSAC thresholds, sweep limits) are managed via YAML files in the `config/` directory.
- **Main Config Entrypoint**: `config/config.yaml`.
- **Interpolations**: Intrinsics/extrinsics are interpolated into the model params using Hydra (e.g. `${camera.extrinsic}`).
- **CLI Overrides**: Override configuration parameters directly when running python scripts:
  - Run inspection: `uv run inspect_pose.py mode=random random_samples=2 model=ransac`
  - Run benchmark: `uv run benchmark.py model=ppf_icp eval_size=5`
  - Run Optuna sweep: `uv run benchmark.py model=ransac sweep=true trials=5`

---

## 3. Package Dependencies & Virtual Env

- **Installer**: `uv` package manager (`pyproject.toml` and `uv.lock`).
- **Python**: Python `3.12` is required.
- **Dependency Groups**: Optional dependency group `docs` manages site generation libraries.
- **Dependency Exclusions**: `opencv-python` conflicts with `opencv-contrib-python`. It is explicitly excluded on all platforms via a platform marker `sys_platform == 'never'` in the `tool.uv` override-dependencies block inside `pyproject.toml`. Do not reinstall `opencv-python` directly.

---

## 4. Documentation System (MkDocs)

The project documentation is built using **MkDocs** and the **Material theme**, following the **Diátaxis** framework layout in the `docs/` directory.

### Directories:
- `docs/index.md`: Getting started guide.
- `docs/tutorials/`: Copy-pasteable Markdown pipeline tutorials.
- `docs/how-to/`: Step-by-step guides (e.g. adding new estimators).
- `docs/explanation/`: Conceptual overviews (symmetries, transform math).
- `docs/reference/`: API reference templates resolved dynamically using `mkdocstrings`.

### Command to build documentation:
```bash
# Sync dependencies including doc group
uv sync --extra docs

# Build documentation strictly (warnings treated as errors)
uv run mkdocs build --strict

# Run local preview server
uv run mkdocs serve
```
