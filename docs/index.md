# Getting Started with 6DPose

Welcome to the **6DPose** project! This repository provides a complete pipeline to estimate the 6D pose of industrial towing carts from an RGB-D camera feed. It combines a 2D deep learning segmentation front-end (Ultralytics YOLO) with a 3D geometric registration back-end (RANSAC or Point Pair Features + local ICP refinement) evaluated in the robot's base coordinate frame (`base_link`).

---

## Installation

### Prerequisites
- **Python 3.12**
- **uv**: A fast Python package installer and resolver. Install it via:
  ```bash
  curl -LsSf https://astral-sh/uv/install.sh | sh
  ```

### Step 1: Clone the Repository
```bash
git clone https://github.com/MartinJessenne/6DPose.git
cd 6DPose
```

### Step 2: Sync Virtual Environment
This project uses `pyproject.toml` to manage dependencies. Run the following command to automatically create a virtual environment (`.venv`) and install all required libraries:
```bash
uv sync
```

---

## Downloading Pipeline Assets

Before running pose estimation, you must download the pre-trained YOLO segment model weights (`best.pt`) and the parquet dataset splits. We provide a helper initialization script that downloads them automatically from Hugging Face:

```bash
uv run initialize_project.py
```

This script performs the following tasks:
- Downloads `best.pt` from the `UItraviolet/yolo_multicart` repository and places it in the root.
- Downloads the `.parquet` files from the `UItraviolet/industrial_cart` dataset and saves them under `dataset/data/data/`.

---

## Running Pose Estimation in <10 Minutes

The project uses [tyro](https://brentyi.github.io/tyro/) for its CLI: every option is a typed
dataclass field in `cli_config.py`, so you get `--help` and validation for free. See the
[Configuration tutorial](tutorials/03_config_system.md) for the full picture.

### 1. Run Random Pose Inspection
Run the pipeline on 2 random samples from the test set using RANSAC registration, and export the output alignment scenes as `.glb` files into the `debug_output/` folder:
```bash
uv run inspect_pose.py --mode random --random-samples 2 model:ransac model.profile:default
```

### 2. Run Pipeline Benchmarks
Evaluate the PPF-ICP pose estimator performance metrics (Translation/Rotation error, execution speed, match success rate) over 5 validation samples:
```bash
uv run benchmark.py --eval-size 5 model:ppf model.profile:default
```

### 3. Run Hyperparameter Tuning
Optimize matching thresholds, voxel sizes, or ICP correspondence distances using multi-objective Bayesian optimization via Optuna:
```bash
uv run benchmark.py --sweep --trials 5 --eval-size 2 model:ransac model.profile:default
```
Outputs and sweep trials are persisted in a local sqlite database `sweeps/optuna_Sweep.db`.
