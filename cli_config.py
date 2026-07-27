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
from methods.vsac_se2 import VSACSe2Estimator, VSACSe2Params


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
    Annotated[
        PPFProfile,
        tyro.conf.subcommand(
            name="trial28",
            # Optuna sweep Trial #28: accuracy ~93.0%, p95 latency ~0.56s, AR ~7.0%, flip rate ~42.5%.
            default=PPFProfile(
                params=PPFParams(
                    ppf_sampling_step=0.04,
                    ppf_distance_step=0.05,
                    ppf_match_threshold=0.05,
                    ppf_match_tolerance=0.04,
                    icp_max_correspondence_distance=0.124003893980807,
                    icp_max_iterations=20,
                ),
                depth_trunc=2.6,
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
    Annotated[
        RansacProfile,
        tyro.conf.subcommand(
            name="trial15",
            # Optuna sweep Trial #15: accuracy ~94.8%, p95 latency ~0.40s, AR ~5.2%, flip rate ~56.4%.
            default=RansacProfile(
                params=RansacParams(
                    voxel_size=0.1,
                    icp_max_correspondence_distance=0.222396429505729,
                    icp_max_iterations=50,
                ),
                depth_trunc=5.6,
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
            # Optuna sweep optuna_Ransac3DofCrop_Sweep Trial #1: highest average recall (~0.251, p95 latency ~1.67s).
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.02,
                    ransac_max_iterations=8192,
                    icp_max_correspondence_distance=0.10100435818212444,
                    icp_max_iterations=70,
                    z_offset=0.01,
                    z_gate_threshold=0.30986258444115694,
                    edge_length_threshold=0.85427888273254,
                    front_crop_depth=0.8092762136127303,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=3.2,
            ),
        ),
    ],
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="rt_opt",
            # Optuna sweep optuna_Ransac3DofCrop_Sweep Trial #35: real-time profile (p95 latency ~0.295s, flip rate ~17.6%).
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.07,
                    ransac_max_iterations=2572,
                    icp_max_correspondence_distance=0.053439448281393,
                    icp_max_iterations=10,
                    z_offset=0.01,
                    z_gate_threshold=0.1343018655763445,
                    edge_length_threshold=0.826763881996128,
                    front_crop_depth=1.5094694353528446,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=2.6,
            ),
        ),
    ],
]


@dataclass(frozen=True)
class VSACSe2Profile:
    """One tuned (params, depth_trunc) bundle for VSACSe2Estimator."""

    params: VSACSe2Params = field(default_factory=VSACSe2Params)
    depth_trunc: float = 3.0


VSACSe2ProfileSelect = Union[
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="default",
            # Mirrors Ransac3DoFProfileSelect's "default" arm (same measured
            # z_offset/front_crop_depth) plus VSACSe2Params.rho at its own
            # class default -- not yet re-tuned by a dedicated sweep.
            default=VSACSe2Profile(
                params=VSACSe2Params(z_offset=0.01, front_crop_depth=0.35),
                depth_trunc=3.0,
            ),
        ),
    ],
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="bare",
            # Raw VSACSe2Params() class defaults (auto-derived z_offset, full
            # mesh, no front-slab crop) -- an ablation counterpart to
            # "default", and NOT redundant filler: a Union needs >= 2 members
            # or Python's typing collapses Union[X] to bare X, which silently
            # breaks tyro's subcommand dispatch (confirmed by testing; see
            # docs/explanation/tyro_cli_config.md's "real tyro gotcha" section
            # for the sibling issue this rhymes with).
            default=VSACSe2Profile(),
        ),
    ],
]


Ransac3DoFFullMeshProfileSelect = Union[
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="default",
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(z_offset=0.01),
                depth_trunc=3.0,
            ),
        ),
    ],
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="acc_opt",
            # Optuna sweep optuna_Ransac3DofFullMesh_Sweep Trial #25: accuracy-focused full mesh profile
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.04,
                    ransac_max_iterations=39267,
                    icp_max_correspondence_distance=0.07851111384977721,
                    icp_max_iterations=40,
                    z_offset=0.01,
                    z_gate_threshold=0.3186998846185683,
                    edge_length_threshold=0.8389466396574985,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=3.0,
            ),
        ),
    ],
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="rt_opt",
            # Optuna sweep optuna_Ransac3DofFullMesh_Sweep Trial #34: real-time full mesh profile
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.06,
                    ransac_max_iterations=2847,
                    icp_max_correspondence_distance=0.08274659221521605,
                    icp_max_iterations=50,
                    z_offset=0.01,
                    z_gate_threshold=0.17182370500647015,
                    edge_length_threshold=0.8556119078087416,
                    ransac_confidence=0.999,
                    seed=0,
                ),
                depth_trunc=4.1,
            ),
        ),
    ],
]


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
    profile: Ransac3DoFFullMeshProfileSelect


@dataclass(frozen=True)
class VSACSe2Preset:
    """PROSAC/MSAC + independent-inlier-tiebreak variant of Ransac3DoFPreset --
    see VSACSe2Estimator, VSAC_Implementation_Plan.md."""

    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = VSACSe2Estimator
    profile: VSACSe2ProfileSelect


ModelPreset = Union[
    Annotated[
        PPFPreset,
        tyro.conf.subcommand(
            name="ppf",
            description="profiles: model.profile:default, model.profile:rt-opt, model.profile:trial28",
        ),
    ],
    Annotated[
        RansacPreset,
        tyro.conf.subcommand(
            name="ransac",
            description=(
                "profiles: model.profile:default, model.profile:pareto1, "
                "model.profile:realtime, model.profile:rt-opt, model.profile:trial15"
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
            description="profiles: model.profile:default, model.profile:acc-opt, model.profile:rt-opt",
        ),
    ],
    Annotated[
        VSACSe2Preset,
        tyro.conf.subcommand(
            name="vsac3dof",
            description="profiles: model.profile:default",
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
    # Sweep-only: each trial's estimator-internal RANSAC seed is otherwise
    # pinned at the estimator params' class default (e.g. Ransac3DoFParams.seed
    # = 0) for every trial, so a "best" trial could just be a lucky seed draw
    # rather than a genuinely better hyperparameter setting. n_seeds > 1
    # evaluates each trial across that many internal seeds and optimizes the
    # mean, at roughly an n_seeds-fold increase in per-trial cost -- dial down
    # to 1 for quick iteration, use 3+ for a trustworthy search.
    n_seeds: int = 1
    # Per-frame CSV (sample_idx, outcome, abstention cause, flip-disambiguation
    # diagnostics where available) written next to the run, for offline auditing
    # (e.g. "is a persistent set of frames driving gross_yaw_rate?", or "did
    # this config starve at the FPFH stage or reject its own candidates?").
    # Cheap relative to the pose-estimation compute itself; --no-dump-frames opts out.
    dump_frames: bool = True
    # Sweep-only: estimator params forced to a fixed value in EVERY trial,
    # overriding whatever suggest_params proposed. This is how an A/B arm is
    # declared -- e.g. --param-overrides free_space_gate=true.
    #
    # It exists because a sweep trial's params come from suggest_params alone
    # (see benchmark.py's objective); model.profile.params is consulted only on
    # the single-evaluation path. Without this, naming a flag on the command
    # line of a --sweep run is silently a no-op and the "treatment" arm runs the
    # control a second time while its name and logs claim otherwise.
    #
    # Values are parsed as Python literals where possible (true/false/1/0.5/None),
    # falling back to the raw string. Unknown parameter names are a hard error,
    # not a warning: a typo here would silently produce exactly the phantom
    # comparison this field exists to prevent.
    param_overrides: dict[str, str] = field(default_factory=dict)


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
    mode: Literal[
        "random", "indices"
    ]  # tyro validates this choice up front -- no null/"" sentinel needed.
    yolo: YoloConfig = field(default_factory=YoloConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    random_samples: int = 10
    indices: tuple[int, ...] = ()
    output_dir: str = "debug_output/"
