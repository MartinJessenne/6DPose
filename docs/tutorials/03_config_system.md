# Tutorial: Managing Configurations with Hydra

This tutorial walks you through how to use the Hydra configuration system to customize pose estimation runs, select model presets, and override hyperparameters directly from the command line.

---

## 1. Config Directory Structure

The configuration tree is located in `config/` and organized into logical groups:

```text
config/
├── config.yaml               # Global main configuration entrypoint
├── camera/
│   └── default.yaml          # Camera intrinsic and extrinsic matrices
├── dataset/
│   └── default.yaml          # Parquet dataset paths and file globs
└── model/
    ├── ppf_icp.yaml          # Default presets for PPF-ICP Estimator
    └── ransac.yaml           # Default presets for RANSAC Estimator
```

The main entrypoint config (`config.yaml`) sets the default presets using the `defaults` list:

```yaml
defaults:
  - camera: default
  - dataset: default
  - model: ransac             # Default model is RANSAC
  - _self_
```

---

## 2. Choosing and Overriding Config Presets

You can swap presets or override parameters dynamically on the command line without modifying any source files.

### Swap the Estimator Model Preset:
```bash
# Swaps the default model (ransac) to the ppf_icp preset
uv run inspect_pose.py model=ppf_icp
```

### Override Hyperparameters Dynamically:
You can dig into nested YAML properties using dot-notation:
```bash
# Override voxel size and ICP correspondence distance for RANSAC
uv run inspect_pose.py model=ransac model.params.voxel_size=0.08 model.params.icp_max_correspondence_distance=0.1
```

---

## 3. Dynamic Configuration Interpolation

A powerful feature of Hydra is parameter interpolation. For example, both estimators need the camera extrinsic transform matrix to project coordinates into the robot base frame (`base_link`).

Instead of defining the extrinsics twice, the model configurations use a dynamic reference (`${camera.extrinsic}`):

```yaml
# config/model/ransac.yaml
_target_: methods.ransac.RansacEstimator
params:
  voxel_size: 0.06
extrinsic: ${camera.extrinsic}  # Resolves dynamically to the extrinsic matrix defined in camera/default.yaml
```

When you change the camera preset, the model's extrinsics automatically update.

---

## 4. Creating Custom Config Presets

To save a specific set of hyperparameters as a named preset (e.g. for a custom experiment):

1. Create a new YAML file under `config/model/ransac_custom.yaml`:
   ```yaml
   _target_: methods.ransac.RansacEstimator
   params:
     voxel_size: 0.08
     ransac_max_iterations: 200000
     icp_max_correspondence_distance: 0.12
     icp_max_iterations: 80
   extrinsic: ${camera.extrinsic}
   ```
2. Select it when running pose estimation or benchmarks:
   ```bash
   uv run benchmark.py model=ransac_custom eval_size=10
   ```
