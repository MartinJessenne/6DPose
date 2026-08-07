"""CLI/config dataclasses for benchmark.py and inspect_pose.py.

This module is the tyro-based successor to config/*.yaml (Hydra). It holds
everything that used to live in the Hydra config tree: the yolo/camera/dataset
settings, the per-estimator hyperparameter presets, and the two top-level CLI
argument dataclasses (BenchmarkArgs, InspectArgs).

See docs/explanation/tyro_cli_config.md for the full walkthrough of how this
maps onto the old Hydra config groups, and why it's structured this way.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal, Union

import numpy as np
import tyro

from methods.base import BasePoseEstimator, SearchRange
from methods.ppf import PPFEstimator, PPFParams
from methods.ransac import RansacEstimator, RansacParams
from methods.ransac3dof import (
    Ransac3DoFEstimator,
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
            # Optuna sweep optuna_Ransac3DofCrop_Sweep Trial #1: highest average recall (~0.251, p95 latency ~1.67s).
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(
                    voxel_size=0.02,
                    ransac_max_iterations=8192,
                    icp_max_correspondence_distance=0.10100435818212444,
                    icp_max_iterations=70,
                    z_offset=0.01,
                    z_gate_threshold=0.30986258444115694,
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
                params=VSACSe2Params(z_offset=0.01),
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
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="tuned",
            # The best configuration measured on this project: 70 frames, seed
            # 2144065271, --o3d-seed 0, n_seeds 1.
            #
            #   good_rate        0.5143 -> 0.9429   (McNemar +30, p < 0.0001)
            #   gross_yaw_rate   0.4783 -> 0.0435
            #   pose_ar          0.2050 -> 0.3620
            #   p95              19.07s -> 7.36s    (the cull and veto prune the search)
            #
            # against the same profile with hoppe_normal_orientation,
            # icp_visibility_cull, normal_consistency and the front-face veto all
            # off. 32 frames fixed, 2 broken (both leanflow). hoppe/cull now come
            # from Ransac3DoFParams' defaults -- see the comment there for why
            # they are corrections rather than arms.
            #
            # The continuous values below are Optuna study VSAC_NullOff_M2 (W&B
            # s3zi4564) trial #33, found BEFORE those four landed, so they are
            # stale by construction and are what the next sweep re-tunes:
            # z_gate_threshold, rho, icp_max_correspondence_distance are the
            # three remaining free parameters.
            #
            # ransac_max_iterations 46940 is an upper bound, not a tuned value:
            # 150000 changes literally zero frames of 70 (McNemar b=0, c=0) while
            # doubling p95, so the curve has saturated somewhere below it. The
            # downward ladder has not been run yet.
            default=VSACSe2Profile(
                params=VSACSe2Params(
                    voxel_size=0.02,
                    z_gate_threshold=0.34896831026780845,
                    ransac_max_iterations=46940,
                    icp_max_iterations=100,
                    icp_max_correspondence_distance=0.13768813892484938,
                    rho=0.108351160884956,
                    z_offset=0.01,
                    front_face_max_angle_deg=60.0,
                    normal_consistency=True,
                ),
                depth_trunc=5.5,
            ),
        ),
    ],
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="tuned-cheap",
            # Same sweep (s3zi4564), trial #28. Reaches a HIGHER good_rate than
            # #33 -- 0.7440 vs 0.7246 -- at 2575 iterations instead of 46940 and
            # 2.03s p95 instead of 5.65s. #33 wins only on pose_ar (0.2833 vs
            # 0.2789), which is inside that sweep's best-of-N inflation (measured
            # at +0.021 when #33 was re-run standalone as W&B y4ctgo32).
            #
            # This exists because within the sweep's voxel 0.02 bucket,
            # Spearman(ransac_max_iterations, good_rate) = -0.006 and
            # Spearman(ransac_max_iterations, p95) = +0.930. Sampling effort buys
            # latency and nothing else, so the 18x iteration budget of "tuned" is
            # pure cost. Deployment needs 200ms (>= 5 Hz on a Jetson Orin Nano,
            # AGENTS.md); 2.03s is 10x over, 5.65s is 28x over.
            #
            # Untested with hoppe_normal_orientation / normal_consistency, which
            # is exactly the run this profile exists to make a one-liner.
            default=VSACSe2Profile(
                params=VSACSe2Params(
                    voxel_size=0.02,
                    z_gate_threshold=0.10155507475760123,
                    ransac_max_iterations=2575,
                    icp_max_iterations=50,
                    icp_max_correspondence_distance=0.07362195960602905,
                    rho=0.49565427444240406,
                    z_offset=0.01,
                ),
                depth_trunc=5.5,
            ),
        ),
    ],
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="tuned-vis",
            # "tuned" plus the T0 refinement: visibility-culled, tightened,
            # yaw-guarded ICP. Identical in every other parameter, so
            # `tuned` vs `tuned-vis` is a one-token A/B.
            #
            # Local, 18 fixtures, --o3d-seed 0, against tuned:
            #   pose_ar      0.1971 -> 0.4101
            #   good_rate    0.6111 -> 0.6111  (bit-identical, 0 abstentions)
            #   trans p50    2.24cm -> 0.81cm
            #   yaw p50      1.02deg -> 0.10deg
            #   p95          7.05s  -> 7.35s
            #
            # The mechanism is measured RNG-free by initialising ICP at ground
            # truth and watching it walk away 2.2cm; see the vault note
            # "30.06 - T0 Translation Error and the Visibility Cull". Local
            # fixtures overstate effects ~2.5x -- only the ranking transfers.
            default=VSACSe2Profile(
                params=VSACSe2Params(
                    voxel_size=0.02,
                    z_gate_threshold=0.34896831026780845,
                    ransac_max_iterations=46940,
                    icp_max_iterations=100,
                    icp_max_correspondence_distance=0.13768813892484938,
                    rho=0.108351160884956,
                    z_offset=0.01,
                    icp_visibility_cull=True,
                ),
                depth_trunc=5.5,
            ),
        ),
    ],
    Annotated[
        VSACSe2Profile,
        tyro.conf.subcommand(
            name="tuned-vis-gate",
            # "tuned-vis" plus the front-face orientation veto at 60 degrees.
            # E03 measured that veto NEGATIVE before the model-normal fix (flips
            # became abstentions, good_rate flat); re-measured on the fixed
            # pipeline it converts flips into correct poses instead, which is
            # what the 29-07 report predicted and listed as its step 3.
            #
            # Local, 18 fixtures, --o3d-seed 0, against tuned:
            #   pose_ar      0.1971 -> 0.4511
            #   good_rate    0.6111 -> 0.7778
            #   abstention   0.0000 -> 0.0000
            #   trans p50    2.24cm -> 0.82cm
            #   p95          7.05s  -> 4.70s   (the veto prunes the search)
            #
            # 60 rather than 45: ground-truth bearings reach 48.79 deg measured
            # from the towing face, so a 45 deg veto rejects 3.2% of correct
            # poses. See Ransac3DoFParams.front_face_max_angle_deg.
            default=VSACSe2Profile(
                params=VSACSe2Params(
                    voxel_size=0.02,
                    z_gate_threshold=0.34896831026780845,
                    ransac_max_iterations=46940,
                    icp_max_iterations=100,
                    icp_max_correspondence_distance=0.13768813892484938,
                    rho=0.108351160884956,
                    z_offset=0.01,
                    icp_visibility_cull=True,
                    front_face_max_angle_deg=60.0,
                ),
                depth_trunc=5.5,
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
        VSACSe2Preset,
        tyro.conf.subcommand(
            name="vsac3dof",
            description=(
                "profiles: model.profile:default, model.profile:bare, "
                "model.profile:tuned, model.profile:tuned-cheap, "
                "model.profile:tuned-vis, model.profile:tuned-vis-gate"
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
