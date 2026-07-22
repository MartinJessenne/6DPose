# Configuration with tyro

This explains *why* the project moved from Hydra+OmegaConf to [tyro](https://brentyi.github.io/tyro/)
for its CLI/config system, and how the pieces in `cli_config.py` fit together. For hands-on usage,
see the [Configuration tutorial](../tutorials/03_config_system.md).

---

## The core idea: a CLI is just a typed function signature

Hydra builds its config tree from YAML files plus a `defaults:` composition list, then resolves
`_target_` strings to dynamically import and construct classes. tyro does the equivalent job
starting from the opposite end: you write a plain Python
[`dataclass`](https://docs.python.org/3/library/dataclasses.html), and `tyro.cli(YourDataclass)`
inspects its type hints to build an `argparse`-style parser automatically. Every field becomes a
`--flag`, nested dataclasses become dotted flag groups (`--camera.fx`), and there is no string
that has to match a class path anywhere -- the type hint *is* the class reference.

```python
@dataclass(frozen=True)
class CameraConfig:
    fx: float = 639.99768
    ...

args = tyro.cli(BenchmarkArgs)   # args.camera.fx is already a float, validated
```

This is why `hydra.utils.get_class(cfg.model._target_)` (a runtime string → class lookup) has no
tyro equivalent and needs none: `args.model.ESTIMATOR_CLS` is already the concrete class object,
because the dataclass field itself was typed to be that class's config.

## Why `frozen=True` and `ClassVar`

Every config/params dataclass in this project (`RansacParams`, `Ransac3DoFParams`, `PPFParams` in
`methods/*.py`, and everything in `cli_config.py`) is declared `@dataclass(frozen=True)`. This is
a deliberate, project-wide convention, not a tyro requirement: these objects are settings, read
once after parsing and never mutated, so making them immutable documents that intent and rules
out an entire class of accidental-mutation bugs. (A repo-wide grep confirmed nothing ever wrote to
`.params.<field>` after construction, so this cost nothing in practice.)

`ESTIMATOR_CLS` on each `*Preset` dataclass is annotated `ClassVar[type[BasePoseEstimator]]`
rather than a regular field:

```python
@dataclass(frozen=True)
class Ransac3DoFPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = Ransac3DoFEstimator
    profile: Ransac3DoFProfileSelect
```

`ClassVar` tells both Python's `dataclasses` module and tyro "this is not an instance field" --
it's invisible to `tyro.cli()`'s parser generation entirely. Without it, tyro would try to expose
a nonsensical `--model.estimator-cls` flag letting you type in a class name; with it,
`ESTIMATOR_CLS` is just a fixed piece of Python-level metadata you read as
`args.model.ESTIMATOR_CLS`.

## The nested preset hierarchy: `ModelPreset` → `*Preset` → `*Profile`

The old `config/model/*.yaml` had 9 files across 3 algorithms (PPF, RANSAC, RANSAC-3DoF), each
file being one YAML-level "preset." Structurally that's a **flat list**, not a tree -- Hydra's
config groups are just directories of alternative files.

Under tyro, model selection is a genuine **2-level subcommand hierarchy** instead:

1. `model:<algorithm>` -- exactly 3 choices, the real invariant (there really are only 3
   estimator classes).
2. `model.profile:<tuning>` -- 2 to 4 choices *per algorithm*, each a `{params, depth_trunc}`
   bundle transcribed from one of the old YAML files.

```python
Ransac3DoFProfileSelect = Union[
    Annotated[Ransac3DoFProfile, tyro.conf.subcommand(name="default", default=...)],
    Annotated[Ransac3DoFProfile, tyro.conf.subcommand(name="acc_opt", default=...)],
    Annotated[Ransac3DoFProfile, tyro.conf.subcommand(name="rt_opt", default=...)],
]

@dataclass(frozen=True)
class Ransac3DoFPreset:
    ESTIMATOR_CLS: ClassVar[type[BasePoseEstimator]] = Ransac3DoFEstimator
    profile: Ransac3DoFProfileSelect

ModelPreset = Union[
    Annotated[PPFPreset, tyro.conf.subcommand(name="ppf")],
    Annotated[RansacPreset, tyro.conf.subcommand(name="ransac")],
    Annotated[Ransac3DoFPreset, tyro.conf.subcommand(name="ransac3dof")],
]
```

Why 2 levels instead of a flat 9-entry union: adding a new tuned profile to, say,
RANSAC-3DoF only ever touches `Ransac3DoFProfileSelect` -- the top-level `ModelPreset` union
(exactly 3 entries, one per real algorithm) never grows. The cost is one extra CLI token per
invocation (`model:ransac3dof model.profile:acc-opt` instead of a single
`model:ransac3dof_acc_opt`).

### Why `depth_trunc` lives on the *profile*, not the *preset*

In the old YAML tree, 6 of the 9 model files were `# @package _global_` rewrites that set the
*global* `depth_trunc` alongside the estimator params, while the other 3 didn't touch it at all --
two different mechanisms for what was conceptually one atomic choice ("here is a tuned setup").
`Ransac3DoFProfile` (and its PPF/RANSAC siblings) bundles `params` and `depth_trunc` together in
one dataclass specifically so selecting a profile can't leave `depth_trunc` at a mismatched value
-- there's exactly one mechanism now, and it can't desync.

## A real tyro gotcha, found by testing, not documentation

While wiring this up, giving the outer `*Preset.profile` field its own default
(`field(default_factory=Ransac3DoFProfile)`) turned out to silently break subcommand dispatch:
tyro would just use that concrete default value regardless of which `model.profile:<name>` token
was passed on the CLI, with no error. Minimal repro:

```python
@dataclass(frozen=True)
class P:
    x: int = 1
    y: int = 2

Sel = Union[
    Annotated[P, tyro.conf.subcommand(name="default")],
    Annotated[P, tyro.conf.subcommand(name="tuned", default=P(x=99, y=100))],
]

@dataclass(frozen=True)
class Outer:
    profile: Sel = field(default_factory=P)   # <-- this default wins, always

tyro.cli(Outer, args=["profile:tuned"])   # returns P(x=1, y=2), NOT P(x=99, y=100)!
```

Removing the outer default fixes it -- but the trade-off (confirmed by testing every
combination) is that a subcommand-typed field is *either* always required and fully switchable
via CLI token, *or* has an implicit default but can never be switched via CLI token at all. There
is no middle ground in this version of tyro. We chose **always required**: `model:<algo>` and
`model.profile:<tuning>` must be given explicitly on every invocation now -- there's no more
implicit "if you don't say anything you get `ppf`" the way Hydra's `defaults:` list gave for free.

Two more sharp edges worth knowing, also found by running the actual CLI rather than assuming:

- **Option ordering matters.** Top-level scalar flags (`--eval-size`, `--sweep`, ...) must come
  *before* the subcommand tokens (`model:ransac3dof model.profile:acc-opt`), not after --
  tyro's own error message ("Arguments are applied to the directly preceding subcommand") explains
  why once you hit it.
- **Subcommand name tokens always render and require hyphens**, even when written with
  underscores in `cli_config.py` (`tyro.conf.subcommand(name="acc_opt")` → you type
  `model.profile:acc-opt`). This matches `--flag` naming, but unlike flags (which accept
  `--foo_bar` and `--foo-bar` interchangeably), subcommand tokens only match the hyphenated form.

## What Hydra could do that tyro can't: run-directory management

`inspect_pose.py` used `HydraConfig.get().runtime.output_dir` to get Hydra's auto-generated,
timestamped per-run working directory for debug output. tyro has no equivalent feature -- it's a
CLI parser, not a run-orchestration framework. The replacement is a one-line, hand-rolled
`datetime.now().strftime(...)` directory name. This is fine here because that directory is
disposable: wiped on every run and already gitignored, so no enrichment (git commit hash, etc.)
was worth adding.

## What needed *no* replacement at all: `${camera.extrinsic}`

Hydra's `${camera.extrinsic}` interpolation let `config/model/*.yaml` reference the camera preset's
extrinsic matrix without duplicating it. Under plain dataclasses there's simply nothing to
resolve -- `args.camera.extrinsic` is already a real Python tuple, read directly wherever it's
needed (`ESTIMATOR_CLS(params=..., extrinsic=args.camera.extrinsic)`). No lazy resolution, no
circular-reference risk, no "what if the camera preset changes after the model preset is built"
ordering question -- the interpolation wasn't a feature that got dropped, it was solved by not
needing a config-composition step at all.

## Adding a new algorithm or profile

- **New tuning profile for an existing algorithm**: add one entry to that algorithm's
  `*ProfileSelect` union in `cli_config.py`. See the
  [Configuration tutorial](../tutorials/03_config_system.md#4-adding-a-new-tuning-profile).
- **New algorithm entirely** (a new estimator class): see
  [How to Add a New Model](../how-to/add_estimator.md) -- it's one `*Profile` + `*ProfileSelect` +
  `*Preset` block, plus one new entry in the top-level `ModelPreset` union. No YAML file, no
  `_target_` string, no factory function registration.
