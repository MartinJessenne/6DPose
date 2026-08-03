# Coding Agent Guide for 6DPose Project

Welcome! This guide is written specifically for AI Coding Agents and developer tooling to quickly understand the project's architecture, code conventions, environment settings, and documentation structure.

---

## 1. System Architecture

The project implements a 6D Pose Estimation pipeline for industrial towing carts. The pipeline consists of two main stages:

1. **Front-End (2D Detection & Masking)**:
   - Runs a 2D **YOLO26n-seg** instance segmentation model (a custom-trained nano variant with an image segmentation head from `ultralytics`) to locate the target cart and predict its class and segmentation mask.
   - Extracts the 2D bounding box and segmentation mask for the largest cart instance.
   - Crops both the RGB image and raw depth map using the bounding box, and blacks out the background by applying the segmentation mask.
   - Wraps the cropped RGB, cropped depth, crop coordinates (bounding box offset), and camera model into a `MaskedImageFrame` to prevent coordinate alignment issues.

2. **Back-End (3D Point Cloud Registration)**:
   - Back-projects the masked depth crop to rebuild the 3D scene point cloud in the camera's local frame. To preserve spatial accuracy, the camera principal point is dynamically shifted based on the crop coordinates.
   - Computes surface normals for the point cloud and orients them consistently towards the camera.
   - Transforms the point cloud coordinates from the OpenCV camera frame to the robot base frame (`base_link`) using the camera-to-robot extrinsics matrix.
   - Solves the 6D registration pose relative to the target CAD mesh using concrete subclasses of `BasePoseEstimator`.

### Modular Sub-step Estimators (`BasePoseEstimator`)
The estimation algorithms are designed using a Strategy pattern so that individual components (downsampling, feature extraction, voting, and registration) are modular and can be independently swapped or customized:
- **`RansacEstimator`** (`methods/ransac.py`): Performs voxel downsampling, extracts Fast Point Feature Histograms (FPFH) descriptors on the downsampled point clouds, runs RANSAC global registration to find a coarse initial alignment, and refines it via Point-to-Plane ICP.
- **`PPFEstimator`** (`methods/ppf.py`): Trains a Point Pair Feature (PPF) match database from the target CAD mesh using OpenCV's `ppf_match_3d` module, runs a voting-based global alignment on the scene points, and refines the pose using Point-to-Point/Plane ICP.
- **Dual-Hypothesis ICP Refinement**: To handle the physical 180° Y-axis symmetry of the towing carts (where they look identical from the front and back), both estimators call `refine_pose_dual_hypothesis` in `methods/base.py`. This runs Point-to-Plane ICP on two initial alignments (the coarse pose and the pose rotated 180° around the cart's local Z-axis) and retains the orientation with the highest fitness score.


---

## 2. Configuration & Parameter Management

- **tyro CLI**: All parameters (dataset globs, camera intrinsics, model choices, RANSAC thresholds, sweep limits) are plain Python dataclasses in `cli_config.py` at the repo root -- no YAML config tree.
- **Main entrypoint dataclasses**: `BenchmarkArgs` and `InspectArgs` in `cli_config.py`, built via `tyro.cli(...)` at the top of each script's `main()`.
- **Model selection**: a 2-level subcommand choice, `model:<algorithm>` then `model.profile:<tuning>` (e.g. `model:ransac3dof model.profile:acc-opt`) -- both required, no implicit default. See `docs/explanation/tyro_cli_config.md` for why.
- **CLI Overrides**: Override configuration parameters directly when running python scripts (scalar options before the subcommand tokens):
  - Run inspection: `uv run inspect_pose.py --mode random --random-samples 2 model:ransac model.profile:default`
  - Run benchmark: `uv run benchmark.py --eval-size 5 model:ppf model.profile:default`
  - Run Optuna sweep: `uv run benchmark.py --sweep --trials 5 model:ransac model.profile:default`

---

## 3. Package Dependencies & Virtual Env

- **Installer**: `uv` package manager (`pyproject.toml` and `uv.lock`).
- **Python**: Python `3.12` is required.
- **Dependency Groups**: Optional dependency group `docs` manages site generation libraries.
- **Opencv Dependency Override**: 
  - `opencv-contrib-python` is a complete superset of `opencv-python` and contains the extra registration modules we need (like `ppf_match_3d`). We only need `opencv-contrib-python` installed.
  - However, third-party libraries like `ultralytics` list `opencv-python` as a direct dependency in their package metadata.
  - To prevent `uv` from installing both (which would cause conflicting namespace files in `site-packages/cv2` and strip the contrib modules), we use the `override-dependencies` field under `[tool.uv]` in `pyproject.toml`. We override `opencv-python` with a platform marker that is never true: `"opencv-python; sys_platform == 'never'"`.
  - This informs `uv` that `opencv-python` is not needed on any platform, successfully forcing the environment to default to `opencv-contrib-python` as the single, complete OpenCV package. Do not install `opencv-python` manually.

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

---

## 5. Running Tests (pytest)

- **Single-Threaded Execution Mandatory**: `open3d`, `opencv-contrib-python`, `torch`, and `scipy`/OpenMP spawn multi-threaded worker pools scaled to all available CPU cores by default. Running tests without limiting thread pools causes severe thread contention and high RAM usage that can crash the container/instance.
- **Always set single-threaded environment variables when executing pytest**:
  ```bash
  OMP_NUM_THREADS=1 OPEN3D_CPU_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest
  ```

---

## 6. Collaboration Workflow (owner writes the code)

**The authoritative rules live in `CLAUDE.md` at the repo root. Read it before making any
change.** It is deliberately not duplicated here, so the two cannot drift apart.

In one sentence: the project owner types every line that lands in git, so agents never
create, edit, or delete a tracked file — experiments go in `scratch/` as monkey-patches,
and the deliverable is a collaborative teaching note in `dev-notes/<branch>/` that the
owner reads, questions inline, and implements by hand.

This binds subagents and skills too; `/simplify` and `/code-review` must report rather
than apply. Read-only work — investigation, debugging discussion, running tests and
benchmarks — is unaffected.

