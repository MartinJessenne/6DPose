# How to Add a New Pose Estimator Model

This guide provides step-by-step instructions on how to implement, register, and configure a new 6D pose estimation model in the pipeline.

---

## Step 1: Create the Estimator Class

All estimators must inherit from `BasePoseEstimator` defined in `methods/base.py`. Create a new file under `methods/` (e.g., `methods/my_new_model.py`) and define your estimator:

```python
import numpy as np
import open3d as o3d
from methods.base import BasePoseEstimator, prepare_scene_point_cloud, refine_pose_dual_hypothesis

class MyNewModelParams:
    """Hyperparameters specific to your new method."""
    def __init__(self, my_threshold: float = 0.05, icp_max_iterations: int = 50):
        self.my_threshold = my_threshold
        self.icp_max_iterations = icp_max_iterations

class MyNewModelEstimator(BasePoseEstimator):
    """Implementation of your new 6D Pose Estimation method."""

    def __init__(self, params: MyNewModelParams | dict = None, extrinsic: list | np.ndarray = None):
        # Handle dict input from Hydra configuration instantiations
        if isinstance(params, dict):
            self.params = MyNewModelParams(**params)
        else:
            self.params = params if params is not None else MyNewModelParams()
            
        # Parse extrinsic camera parameter
        if extrinsic is not None:
            self.extrinsic = np.asarray(extrinsic, dtype=np.float64)
        else:
            self.extrinsic = np.eye(4)

    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        **kwargs
    ) -> np.ndarray | None:
        # 1. Preprocess the point cloud (estimating normals and aligning frames)
        pcd = prepare_scene_point_cloud(pcd, self.extrinsic)

        # 2. Implement your alignment registration logic
        # ...
        # T_init = your_coarse_registration_algorithm(pcd, cad_mesh)
        T_init = np.eye(4)  # Placeholder

        # 3. Refine the registration using dual-hypothesis ICP
        T_refined = refine_pose_dual_hypothesis(
            model_pc=cad_mesh.sample_points_uniformly(number_of_points=1000),
            scene_pcd=pcd,
            T_init=T_init,
            icp_max_correspondence_distance=self.params.my_threshold * 2.0,
            icp_max_iterations=self.params.icp_max_iterations
        )
        
        return T_refined
```

---

## Step 2: Register in the Factory Function

To allow the script runners to query your new estimator, register it in the `get_estimator` factory function inside `methods/__init__.py`:

```python
# methods/__init__.py

from methods.my_new_model import MyNewModelEstimator

def get_estimator(method_name: str, **kwargs) -> BasePoseEstimator:
    name_clean = method_name.lower().strip()
    if name_clean == "ppf":
        return PPFEstimator(**kwargs)
    elif name_clean == "ransac":
        return RansacEstimator(**kwargs)
    elif name_clean == "my_new_model":
        return MyNewModelEstimator(**kwargs)
    else:
        raise ValueError(f"Unrecognized pose estimation method name: '{method_name}'")
```

---

## Step 3: Create the Hydra Configuration

To enable command-line overrides and automatic instantiation, add a YAML config file for your model under `config/model/my_new_model.yaml`:

```yaml
# config/model/my_new_model.yaml
_target_: methods.my_new_model.MyNewModelEstimator
params:
  my_threshold: 0.05
  icp_max_iterations: 50
extrinsic: ${camera.extrinsic}
```

---

## Step 4: Run and Verify

You can now select and run your new model using standard Hydra CLI overrides:

```bash
# Run random pose inspection
uv run inspect_pose.py mode=random model=my_new_model

# Run pipeline benchmark
uv run benchmark.py model=my_new_model
```
