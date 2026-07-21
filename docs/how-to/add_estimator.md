# How to Add a New Pose Estimator Model

This guide provides step-by-step instructions on how to implement, register, and configure a new 6D pose estimation model in the pipeline.

---

## Step 1: Create the Estimator Class

All estimators must inherit from `BasePoseEstimator` defined in `methods/base.py`. Create a new file under `methods/` (e.g., `methods/my_new_model.py`) and define your estimator:

```python
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from methods.base import BasePoseEstimator, prepare_scene_point_cloud, refine_pose_dual_hypothesis

@dataclass(frozen=True)
class MyNewModelParams:
    """Hyperparameters specific to your new method."""
    my_threshold: float = 0.05
    icp_max_iterations: int = 50

class MyNewModelEstimator(BasePoseEstimator):
    """Implementation of your new 6D Pose Estimation method."""

    def __init__(self, params: MyNewModelParams | dict = None, extrinsic: list | np.ndarray = None):
        # Handle dict input (e.g. from Optuna's suggest_params during a sweep)
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

> [!NOTE]
> There is no `get_estimator` factory function or Hydra `_target_` string resolution to wire up. Registering a new algorithm is one addition to `cli_config.py`, described below -- see [Configuration with tyro](../explanation/tyro_cli_config.md) for the full picture of how `ModelPreset` is built.

## Step 2: Register It in cli_config.py

Add a `*Profile` dataclass, a `*ProfileSelect` union (even with just one "default" entry -- you can add more tuned profiles later, see [the config tutorial](../tutorials/03_config_system.md)), a `*Preset` dataclass naming your estimator class, and one new entry in the top-level `ModelPreset` union:

```python
# In cli_config.py

@dataclass(frozen=True)
class MyNewModelProfile:
    params: MyNewModelParams = field(default_factory=MyNewModelParams)
    depth_trunc: float = 3.0

MyNewModelProfileSelect = Union[
    Annotated[MyNewModelProfile, tyro.conf.subcommand(name="default")],
]

@dataclass(frozen=True)
class MyNewModelPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = MyNewModelEstimator
    profile: MyNewModelProfileSelect

ModelPreset = Union[
    Annotated[PPFPreset, tyro.conf.subcommand(name="ppf")],
    Annotated[RansacPreset, tyro.conf.subcommand(name="ransac")],
    Annotated[Ransac3DoFPreset, tyro.conf.subcommand(name="ransac3dof")],
    Annotated[MyNewModelPreset, tyro.conf.subcommand(name="my_new_model")],  # <-- new
]
```

---

## Step 3: Run and Verify

You can now select and run your new model from the CLI:

```bash
# Run random pose inspection
uv run inspect_pose.py --mode random model:my-new-model model.profile:default

# Run pipeline benchmark
uv run benchmark.py model:my-new-model model.profile:default
```
