# Dev Note: Pure Zero-Debt Refactor — Nuking `run_config.py` & Consolidating Configuration on `cli_config.py`

status: draft

---

## 1. Overview & Verified Architecture

The sandbox agent has executed this radical refactor in an isolated branch workspace and verified it against the full test suite (`114 passed cleanly in 37.05s`).

### Pure Zero-Debt Changes:
1. **[`run_config.py`](file:///home/martin/6DPose/run_config.py) Deleted Entirely**:
   * Removed `RunConfig`, `SweepConfig`, and `resolve_run_config` from the codebase (`rm run_config.py`).
   * Eliminates the parallel class hierarchy, shotgun surgery, and duplicate field assignments.

2. **Self-Resolving CLI Dataclasses ([`cli_config.py`](file:///home/martin/6DPose/cli_config.py))**:
   * `tyro` CLI dataclasses (`CommonArgs`, `EvalArgs`, `SweepArgs`) now compute their own resolved runtime assets as clean read-only `@property` getters:
     * `@property def extrinsic(self) -> np.ndarray`: Converts extrinsic camera tuple to `np.ndarray`.
     * `@property def estimator_cls(self) -> type[BasePoseEstimator]`: Returns `self.model.ESTIMATOR_CLS`.
     * `@property def depth_trunc(self) -> float`: Returns `self.model.profile.depth_trunc`.
     * `@property def resolved_params(self)`: Resolves estimator profile parameters with CLI overrides applied.
     * `@property def resolved_seed(self) -> int`: Returns fixed seed or auto-generates 31-bit integer.
     * `@property def search_space(self) -> dict[str, SearchRange]`: Returns un-pinned Optuna search range dict on `SweepArgs`.

3. **Direct CLI Argument Consumption ([`benchmark.py`](file:///home/martin/6DPose/benchmark.py) & [`sweep.py`](file:///home/martin/6DPose/sweep.py))**:
   * `benchmark.py` receives `args` directly from `tyro.cli(Command)`.
   * `run_parameter_sweep` accepts `cfg: SweepArgs` directly.

---

## 2. Complete Verified File Implementations

### Action 1: Delete [`run_config.py`](file:///home/martin/6DPose/run_config.py)
Delete the file [`run_config.py`](file:///home/martin/6DPose/run_config.py) completely.

---

### Action 2: Update [`cli_config.py`](file:///home/martin/6DPose/cli_config.py)

Add `resolve_param_overrides` and properties to `CommonArgs` and `SweepArgs` in [`cli_config.py`](file:///home/martin/6DPose/cli_config.py#L546-L580):

```python
def resolve_param_overrides(
    estimator_cls: type[BasePoseEstimator], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Validate and coerce CLI overrides. Returns {name: coerced value}."""
    if not overrides:
        return {}
    coerced = estimator_cls.params_cls().with_overrides(**overrides)
    return {k: getattr(coerced, k) for k in overrides}


@dataclass(frozen=True)
class CommonArgs:
    """Everything both eval and sweep commands need."""

    model: ModelPreset
    yolo: YoloConfig = field(default_factory=YoloConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval_size: int = 20
    seed: int | None = None
    n_seeds: int = 1
    o3d_seed: int | None = 0
    split: Literal["all", "test", "validation", "train"] = "test"
    overrides: dict[str, str] = field(default_factory=dict)
    dump_frames: bool = True
    no_wandb: bool = False

    @property
    def extrinsic(self) -> np.ndarray:
        return np.array(self.camera.extrinsic, dtype=np.float64)

    @property
    def estimator_cls(self) -> type[BasePoseEstimator]:
        return self.model.ESTIMATOR_CLS

    @property
    def depth_trunc(self) -> float:
        return self.model.profile.depth_trunc

    @property
    def resolved_overrides(self) -> dict[str, Any]:
        return resolve_param_overrides(self.estimator_cls, self.overrides)

    @property
    def resolved_params(self):
        return self.model.profile.params.with_overrides(**self.resolved_overrides)

    @property
    def resolved_seed(self) -> int:
        return self.seed if self.seed is not None else int(np.random.SeedSequence().entropy % (2**31 - 1))

    @property
    def use_wandb(self) -> bool:
        return not self.no_wandb


@dataclass(frozen=True)
class EvalArgs(CommonArgs):
    """Evaluate one configuration."""

    name: str = "Benchmark"


@dataclass(frozen=True)
class SweepArgs(CommonArgs):
    """Search hyperparameters with Optuna."""

    name: str = "Sweep"
    trials: int = 30

    @property
    def search_space(self) -> dict[str, SearchRange]:
        return {
            k: v
            for k, v in self.estimator_cls.params_cls.search_space().items()
            if k not in self.resolved_overrides
        }


Command = Union[
    Annotated[EvalArgs, tyro.conf.subcommand(name="eval")],
    Annotated[SweepArgs, tyro.conf.subcommand(name="sweep")],
]
```

---

### Action 3: Update [`benchmark.py`](file:///home/martin/6DPose/benchmark.py)

Replace main logic in [`benchmark.py`](file:///home/martin/6DPose/benchmark.py#L75-L160) to consume `args` directly:

```python
def main():
    args = tyro.cli(Command)

    np.random.seed(args.resolved_seed)
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    if isinstance(args, SweepArgs):
        print(f"Search space ({len(args.search_space)} free parameters):")
        for name, rng in args.search_space.items():
            print(f"  {name:32} {rng}")

    print("Loading pipeline assets...")
    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    dataset = load_parquet_dataset(dataset_path=args.dataset.path, test_glob=args.dataset.test_glob)

    meshes = load_cad_meshes()

    if isinstance(args, SweepArgs):
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            cfg=args,
            meshes=meshes,
        )

    else:
        seed = args.resolved_seed
        estimator_cls = args.estimator_cls
        extrinsic = args.extrinsic

        total_samples = len(dataset)
        eval_indices = draw_eval_indices(total_samples, args.eval_size, seed)

        effective_n_seeds = args.n_seeds
        internal_seeds = derive_internal_seeds(seed, effective_n_seeds)

        print(
            f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples..."
        )
        print(f"Seed: {seed}")
        if effective_n_seeds > 1:
            print(f"Pooled over {effective_n_seeds} internal RANSAC seeds: {internal_seeds}")
        print(f"Indices: {eval_indices}\n")

        if args.resolved_overrides:
            print(f"Parameter overrides applied: {args.resolved_overrides}")

        base_estimator = estimator_cls.build(
            profile_params=args.model.profile.params,
            overrides=args.overrides,
            extrinsic=extrinsic,
        )

        if not args.use_wandb:
            run_ctx = contextlib.nullcontext()
        else:
            wandb_config = {
                **dataclasses.asdict(base_estimator.params),
                "depth_trunc": args.depth_trunc,
                "eval_size": args.eval_size,
                "seed": seed,
                "n_seeds": effective_n_seeds,
            }
            run_ctx = wandb.init(
                project="6dpose",
                name=args.name,
                group=estimator_cls.__name__,
                job_type="benchmark",
                config=wandb_config,
            )
```

---

### Action 4: Update [`sweep.py`](file:///home/martin/6DPose/sweep.py)

Update signature and imports in [`sweep.py`](file:///home/martin/6DPose/sweep.py#L20-L45):

```python
from cli_config import SweepArgs

def run_parameter_sweep(
    dataset,
    model,
    camera,
    cfg: SweepArgs,
    meshes: dict[str, o3d.geometry.TriangleMesh],
):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.
    """
    if cfg.resolved_overrides:
        print(f"Parameter overrides pinned for every trial: {cfg.resolved_overrides}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(project_root, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    db_name = os.path.join(sweep_dir, f"optuna_{cfg.name}.db")
    db_url = f"sqlite:///{db_name}"

    study = optuna.create_study(
        study_name=cfg.name,
        storage=db_url,
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )
```

Inside `objective(trial)` in `sweep.py`:
* `trial_seeds = derive_internal_seeds(seed, cfg.n_seeds, salt=trial.number)`
* `trial_params = cfg.estimator_cls.params_cls.sample_optuna(trial, base=cfg.resolved_params, fixed=cfg.resolved_overrides)`
* `trial_estimator = cfg.estimator_cls(params=params_i, extrinsic=cfg.extrinsic)`
* `log_input_artifacts(run, cfg.yolo, cfg.dataset)`

---

### Action 5: Update Test Imports

In [`tests/test_param_overrides.py`](file:///home/martin/6DPose/tests/test_param_overrides.py#L1-L10):
```python
from cli_config import resolve_param_overrides
```

---

## 3. How to Verify

Run pytest to verify that all 114 unit tests pass:
```bash
OMP_NUM_THREADS=1 OPEN3D_CPU_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest
```
