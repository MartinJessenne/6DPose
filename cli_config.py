"""CLI/config dataclasses for benchmark.py and inspect_pose.py.

This module is the tyro-based successor to config/*.yaml (Hydra). It holds
everything that used to live in the Hydra config tree: the yolo/camera/dataset
settings, the per-estimator hyperparameter presets, and the two top-level CLI
argument dataclasses (BenchmarkArgs, InspectArgs).

See docs/explanation/tyro_cli_config.md for the full walkthrough of how this
maps onto the old Hydra config groups, and why it's structured this way.
"""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal, Union

import tyro

from methods.base import BasePoseEstimator
from methods.ppf import PPFEstimator, PPFParams
from methods.ransac import RansacEstimator, RansacParams
from methods.ransac3dof import (
    Ransac3DoFEstimator,
    Ransac3DoFFullMeshEstimator,
    Ransac3DoFParams,
)


# =====================================================================
# 1. SHARED SETTINGS (mirrors the old config/camera, config/dataset, and
#    the yolo: block of config/config.yaml)
# =====================================================================
@dataclass(frozen=True)
class YoloConfig:
    """Where to find the YOLO segmentation weights (repo, file, local cache path)."""
    repo: str = "UItraviolet/yolo_multicart"
    file: str = "runs/segment/train-2/weights/best.pt"
    local_path: str = "best.pt"


@dataclass(frozen=True)
class CameraConfig:
    """Intrinsics + camera-to-robot-base extrinsic for the depth sensor."""
    fx: float = 639.99768
    fy: float = 639.99768
    cx: float = 640.0
    cy: float = 400.0
    extrinsic: tuple[tuple[float, float, float, float], ...] = (
        (0.5, 0.0, 0.8660254037844386, 0.439),
        (0.0, 1.0, -0.0, 0.0),
        (-0.8660254037844386, 0.0, 0.5, 0.304),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Parquet dataset location and glob patterns."""
    path: str = "parquet"
    # train_glob/val_glob are not currently read by benchmark.py or inspect_pose.py
    # (only `path` and `test_glob` are) — kept for 1:1 parity with the old YAML.
    train_glob: str = "dataset/data/train-*-of-00127.parquet"
    val_glob: str = "dataset/data/validation-*-of-00016.parquet"
    test_glob: str = "dataset/data/test-*-of-00016.parquet"


# =====================================================================
# 2. MODEL PRESETS — replaces config/model/*.yaml (9 files -> 3 algorithms,
#    each with 2-4 tuning profiles).
#
#    Each old YAML file bundled TWO things atomically: a set of estimator
#    hyperparameters, and (for 6 of the 9 files) a global depth_trunc
#    override. A `*Profile` dataclass below reproduces that same atomic
#    bundling as {params, depth_trunc}, so selecting a profile changes both
#    at once -- exactly like selecting the old `model=ransac3dof_acc_opt`
#    did. See docs/explanation/tyro_cli_config.md for why this needs its own
#    wrapper instead of putting depth_trunc on the outer *Preset.
#
#    Selection is a 2-level tyro subcommand hierarchy:
#        model:<algorithm> profile:<tuning>
#    e.g. `model:ransac3dof profile:acc_opt --model.profile.params.voxel-size 0.08`
# =====================================================================
@dataclass(frozen=True)
class PPFProfile:
    """One tuned (params, depth_trunc) bundle for PPFEstimator."""
    params: PPFParams = field(default_factory=PPFParams)
    depth_trunc: float = 3.0


PPFProfileSelect = Union[
    Annotated[PPFProfile, tyro.conf.subcommand(name="default")],
    Annotated[
        PPFProfile,
        tyro.conf.subcommand(
            name="rt_opt",
            # Optuna sweep PPFNewSweep Trial #32: near-real-time deployment
            # (p95 latency ~0.19s, BOP-style AR ~1.2%).
            default=PPFProfile(
                params=PPFParams(
                    ppf_sampling_step=0.03,
                    ppf_distance_step=0.06,
                    ppf_match_threshold=0.04,
                    ppf_match_tolerance=0.08,
                    icp_max_correspondence_distance=0.1724415146443418,
                    icp_max_iterations=60,
                ),
                depth_trunc=3.8,
            ),
        ),
    ],
]


@dataclass(frozen=True)
class RansacProfile:
    """One tuned (params, depth_trunc) bundle for RansacEstimator."""
    params: RansacParams = field(default_factory=RansacParams)
    depth_trunc: float = 3.0


RansacProfileSelect = Union[
    Annotated[RansacProfile, tyro.conf.subcommand(name="default")],
    Annotated[
        RansacProfile,
        tyro.conf.subcommand(
            name="pareto1",
            # Optuna sweep, Pareto front trial #14. ransac_max_iterations not
            # tuned here -- falls back to RansacParams' own class default.
            default=RansacProfile(
                params=RansacParams(
                    voxel_size=0.08,
                    icp_max_correspondence_distance=0.1200659534,
                    icp_max_iterations=90,
                ),
                depth_trunc=6.2,
            ),
        ),
    ],
    Annotated[
        RansacProfile,
        tyro.conf.subcommand(
            name="realtime",
            # Optuna sweep, Pareto front trial #36: real-time deployment
            # (p95 latency ~0.29s).
            default=RansacProfile(
                params=RansacParams(
                    voxel_size=0.09,
                    icp_max_correspondence_distance=0.1292972835,
                    icp_max_iterations=20,
                ),
                depth_trunc=3.6,
            ),
        ),
    ],
    Annotated[
        RansacProfile,
        tyro.conf.subcommand(
            name="rt_opt",
            # Optuna sweep RansacNewSweep Trial #27: real-time deployment
            # (p95 latency ~0.25s, BOP-style AR ~5.5%).
            default=RansacProfile(
                params=RansacParams(
                    voxel_size=0.09,
                    icp_max_correspondence_distance=0.09794024637923043,
                    icp_max_iterations=20,
                ),
                depth_trunc=2.1,
            ),
        ),
    ],
]


@dataclass(frozen=True)
class Ransac3DoFProfile:
    """One tuned (params, depth_trunc) bundle for Ransac3DoFEstimator."""
    params: Ransac3DoFParams = field(default_factory=Ransac3DoFParams)
    depth_trunc: float = 3.0


Ransac3DoFProfileSelect = Union[
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="default",
            # NOTE: this is NOT just Ransac3DoFParams()'s bare class defaults --
            # config/model/ransac3dof.yaml pins concrete measured/tuned values
            # for z_offset and front_crop_depth rather than leaving them at the
            # class's None/"auto-derive" placeholders. Caught by
            # tests/test_cli_config.py during the tyro migration.
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(z_offset=0.01, front_crop_depth=0.35),
                depth_trunc=3.0,
            ),
        ),
    ],
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="acc_opt",
            # Optuna sweep Ransac3dofZgate (+ later front-crop sweep): best
            # accuracy trade-off found. front_crop_depth kills 180-degree
            # flips (A/B on 40 samples: AR 0.069 -> 0.154, flip 45.9% -> 21.4%).
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.02,
                    ransac_max_iterations=14029,
                    icp_max_correspondence_distance=0.07280260785158235,
                    icp_max_iterations=80,
                    z_offset=0.01,
                    z_gate_threshold=0.24838996628120097,
                    edge_length_threshold=0.8315299309347685,
                    front_crop_depth=0.32839186866723463,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=3.8,
            ),
        ),
    ],
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="rt_opt",
            # Optuna sweep: real-time deployment trade-off.
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.04,
                    ransac_max_iterations=2772,
                    icp_max_correspondence_distance=0.06404478356487074,
                    icp_max_iterations=50,
                    z_offset=0.01,
                    z_gate_threshold=0.12938125580720045,
                    edge_length_threshold=0.831518113921687,
                    front_crop_depth=0.47964349266741707,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=2.2,
            ),
        ),
    ],
]


def _ransac3dof_fullmesh_default_profile() -> Ransac3DoFProfile:
    # Deliberately does NOT reuse Ransac3DoFProfileSelect's "default" -- that
    # one bakes in front_crop_depth=0.35, which would silently defeat the
    # point of this no-crop ablation baseline in eval mode. z_offset is kept
    # (a sensor-rig measurement, unrelated to cropping).
    return Ransac3DoFProfile(params=Ransac3DoFParams(z_offset=0.01), depth_trunc=3.0)


@dataclass(frozen=True)
class PPFPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = PPFEstimator
    # No outer default here -- tyro's subcommand dispatch for a nested Union
    # field only works if the field has NO default of its own (confirmed by
    # testing against tyro 1.0.15: a field-level default short-circuits
    # subcommand selection entirely, silently ignoring the CLI token). The
    # trade-off, also confirmed: `profile:<name>` must always be given
    # explicitly -- there's no implicit "no token = first/default arm".
    profile: PPFProfileSelect


@dataclass(frozen=True)
class RansacPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = RansacEstimator
    profile: RansacProfileSelect


@dataclass(frozen=True)
class Ransac3DoFPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = Ransac3DoFEstimator
    profile: Ransac3DoFProfileSelect


@dataclass(frozen=True)
class Ransac3DoFFullMeshPreset:
    """No-crop ablation baseline -- see Ransac3DoFFullMeshEstimator."""
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = Ransac3DoFFullMeshEstimator
    profile: Ransac3DoFProfile = field(default_factory=_ransac3dof_fullmesh_default_profile)


ModelPreset = Union[
    Annotated[
        PPFPreset,
        tyro.conf.subcommand(
            name="ppf",
            description="profiles: model.profile:default, model.profile:rt-opt",
        ),
    ],
    Annotated[
        RansacPreset,
        tyro.conf.subcommand(
            name="ransac",
            description=(
                "profiles: model.profile:default, model.profile:pareto1, "
                "model.profile:realtime, model.profile:rt-opt"
            ),
        ),
    ],
    Annotated[
        Ransac3DoFPreset,
        tyro.conf.subcommand(
            name="ransac3dof",
            description="profiles: model.profile:default, model.profile:acc-opt, model.profile:rt-opt",
        ),
    ],
    Annotated[
        Ransac3DoFFullMeshPreset,
        tyro.conf.subcommand(
            name="ransac3dof-fullmesh",
            description=(
                "SE(2) no-crop ablation baseline (historical, pre-front-crop). "
                "No profile subcommand -- single fixed default, override via "
                "--model.profile.params.<field>."
            ),
        ),
    ],
]


# =====================================================================
# 3. TOP-LEVEL CLI ARGS — replaces config/config.yaml (shared) +
#    config/benchmark_config.yaml / config/inspect_config.yaml (per entry
#    point). yolo/camera/dataset are shared; the rest are each script's own
#    runner parameters.
#
#    `model` has no default (see the note on *Preset.profile above -- the
#    same "outer default breaks subcommand dispatch" tyro behavior applies
#    here too), so `model:<algo> profile:<tuning>` must always be given
#    explicitly. No more implicit "if you don't say model=X you get ppf".
# =====================================================================
@dataclass(frozen=True)
class BenchmarkArgs:
    """Example: uv run benchmark.py --eval-size 30 model:ransac3dof model.profile:acc-opt

    TWO subcommand tokens are ALWAYS required together: model:<algo> AND
    model.profile:<tuning>. Forgetting model.profile:<tuning> is the single
    most common mistake here -- if you see "Missing subcommand: Expected one
    of {model.profile:default, model.profile:acc-opt, ...}", that's it, you
    picked model:<algo> but forgot to also pick model.profile:<tuning>.

    Available algorithms: model:ppf, model:ransac, model:ransac3dof.
    To see an algorithm's profile choices before running: model:<algo> --help
    """
    model: ModelPreset
    yolo: YoloConfig = field(default_factory=YoloConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval_size: int = 20
    sweep: bool = False
    trials: int = 30
    name: str = "Sweep"
    seed: int | None = None


@dataclass(frozen=True)
class InspectArgs:
    """Example: uv run inspect_pose.py --mode random model:ransac3dof model.profile:acc-opt

    TWO subcommand tokens are ALWAYS required together: model:<algo> AND
    model.profile:<tuning>. Forgetting model.profile:<tuning> is the single
    most common mistake here -- if you see "Missing subcommand: Expected one
    of {model.profile:default, model.profile:acc-opt, ...}", that's it, you
    picked model:<algo> but forgot to also pick model.profile:<tuning>.

    Available algorithms: model:ppf, model:ransac, model:ransac3dof.
    To see an algorithm's profile choices before running: model:<algo> --help
    """
    model: ModelPreset
    mode: Literal["random", "indices"]  # tyro validates this choice up front -- no null/"" sentinel needed.
    yolo: YoloConfig = field(default_factory=YoloConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    random_samples: int = 10
    indices: tuple[int, ...] = ()
    output_dir: str = "debug_output/"
