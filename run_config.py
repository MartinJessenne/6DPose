import dataclasses
from typing import Any

import numpy as np

from cli_config import SweepArgs
from methods.base import BasePoseEstimator, SearchRange


def resolve_param_overrides(estimator_cls, overrides):
    """Validate and coerce CLI overrides. Returns {name: coerced value}."""
    if not overrides:
        return {}
    coerced = estimator_cls.params_cls().with_overrides(**overrides)
    return {k: getattr(coerced, k) for k in overrides}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything decided BEFORE any compute. Assembled once, then read-only."""

    estimator_cls: type[BasePoseEstimator]
    params: Any  # fully resolved estimator params
    depth_trunc: float
    extrinsic: np.ndarray
    resolved_overrides: dict[str, Any]
    eval_size: int
    seed: int
    n_seeds: int
    o3d_seed: int | None
    dataset_path: str
    dataset_glob: str
    name: str
    use_wandb: bool
    dump_frames: bool = True


@dataclasses.dataclass(frozen=True)
class SweepConfig(RunConfig):
    """Configuration specific to hyperparameter sweep mode."""

    n_trials: int = 30
    study_name: str = "Sweep"
    search_space: dict[str, SearchRange] = dataclasses.field(default_factory=dict)


def resolve_run_config(args) -> RunConfig:
    """The single entry point. Everything else reads the RunConfig it returns."""
    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    estimator_cls = args.model.ESTIMATOR_CLS
    overrides = resolve_param_overrides(estimator_cls, args.overrides)

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

    search_space = {
        k: v for k, v in estimator_cls.params_cls.search_space().items() if k not in overrides
    }

    if isinstance(args, SweepArgs) or getattr(args, "sweep", False):
        return SweepConfig(
            estimator_cls=estimator_cls,
            params=args.model.profile.params.with_overrides(**overrides),
            depth_trunc=args.model.profile.depth_trunc,
            extrinsic=extrinsic,
            resolved_overrides=overrides,
            eval_size=args.eval_size,
            seed=seed,
            n_seeds=args.n_seeds,
            o3d_seed=args.o3d_seed,
            dataset_path=args.dataset.path,
            dataset_glob=args.dataset.test_glob,
            name=getattr(args, "name", "Sweep"),
            use_wandb=not args.no_wandb,
            dump_frames=args.dump_frames,
            n_trials=getattr(args, "trials", 30),
            study_name=getattr(args, "study_name", getattr(args, "name", "Sweep")),
            search_space=search_space,
        )

    return RunConfig(
        estimator_cls=estimator_cls,
        params=args.model.profile.params.with_overrides(**overrides),
        depth_trunc=args.model.profile.depth_trunc,
        extrinsic=extrinsic,
        resolved_overrides=overrides,
        eval_size=args.eval_size,
        seed=seed,
        n_seeds=args.n_seeds,
        o3d_seed=args.o3d_seed,
        dataset_path=args.dataset.path,
        dataset_glob=args.dataset.test_glob,
        name=getattr(args, "name", "Benchmark"),
        use_wandb=not args.no_wandb,
        dump_frames=args.dump_frames,
    )
