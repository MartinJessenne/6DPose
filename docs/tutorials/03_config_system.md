# Tutorial: Managing Configuration with tyro

This tutorial walks you through how to use [tyro](https://brentyi.github.io/tyro/) to customize
pose estimation runs, select model presets, and override hyperparameters directly from the
command line. For the "why" behind this design (and a comparison with the Hydra system it
replaced), see [Configuration with tyro](../explanation/tyro_cli_config.md) in Explanation.

---

## 1. Where the configuration lives

There is no `config/` YAML tree anymore. Every setting is a plain Python
[`dataclass`](https://docs.python.org/3/library/dataclasses.html) in `cli_config.py` at the repo
root:

```text
cli_config.py
├── YoloConfig, CameraConfig, DatasetConfig   # shared settings
├── PPFParams / RansacParams / Ransac3DoFParams  # in methods/*.py -- one per algorithm
├── PPFProfile / RansacProfile / Ransac3DoFProfile   # {params, depth_trunc} bundles
├── PPFPreset / RansacPreset / Ransac3DoFPreset      # {ESTIMATOR_CLS, profile}
├── ModelPreset                                # the algorithm-level choice
├── BenchmarkArgs                              # top-level CLI args for benchmark.py
└── InspectArgs                                # top-level CLI args for inspect_pose.py
```

`tyro.cli(BenchmarkArgs)` (or `InspectArgs`) reads this dataclass's type hints and builds the
whole CLI parser automatically — every field becomes a `--flag`, with type checking and
`--help` text for free.

---

## 2. Choosing an algorithm and a tuning profile

Selecting a model is a **two-step choice**, because each of the 3 algorithms (PPF, RANSAC,
RANSAC-3DoF) has 2-4 tuned profiles (the old `config/model/*.yaml` files):

```bash
# Step 1: model:<algorithm>   Step 2: model.profile:<tuning>
uv run inspect_pose.py --mode random model:ransac model.profile:default
```

To see which profiles exist for an algorithm:
```bash
uv run benchmark.py model:ransac3dof --help
```
```text
usage: benchmark.py model:ransac3dof [-h] {model.profile:default,model.profile:acc-opt,model.profile:rt-opt}
```

There is no implicit default anymore — `model:<algo>` and `model.profile:<tuning>` must always be
given explicitly (see [Configuration with tyro](../explanation/tyro_cli_config.md) for why tyro
works this way). Subcommand names always use hyphens on the command line
(`model.profile:acc-opt`), even though they're written with underscores in `cli_config.py`
(`name="acc_opt"`) — same convention as `--flag` options, but strictly enforced for subcommand
tokens.

**Ordering matters**: top-level scalar options (`--eval-size`, `--sweep`, etc.) must come
*before* the subcommand tokens:

```bash
# Correct
uv run benchmark.py --eval-size 30 model:ransac3dof model.profile:acc-opt

# Wrong -- tyro will reject --eval-size here with "Unrecognized options"
uv run benchmark.py model:ransac3dof model.profile:acc-opt --eval-size 30
```

---

## 3. Overriding a single hyperparameter

Any field nested inside the chosen profile can be overridden with its dotted flag path:

```bash
uv run inspect_pose.py --mode random \
    model:ransac model.profile:default \
    --model.profile.params.voxel-size 0.08 \
    --model.profile.params.icp-max-correspondence-distance 0.1
```

The `depth_trunc` that used to be a separate top-level Hydra override now lives on the profile
too, since it was always tuned together with the params (see the explanation doc):

```bash
uv run benchmark.py --eval-size 30 model:ransac3dof model.profile:acc-opt \
    --model.profile.depth-trunc 4.0
```

---

## 4. Adding a new tuning profile

To save a new set of hyperparameters as a named profile (e.g. for a custom experiment), add one
entry to the relevant `*ProfileSelect` union in `cli_config.py` — no new file needed:

```python
# In cli_config.py, inside Ransac3DoFProfileSelect's Union[...]
(
    Annotated[
        Ransac3DoFProfile,
        tyro.conf.subcommand(
            name="my_custom_profile",
            default=Ransac3DoFProfile(
                params=Ransac3DoFParams(voxel_size=0.08, ransac_max_iterations=200000),
                depth_trunc=3.0,
            ),
        ),
    ],
)
```

Then select it the same way as any other profile:
```bash
uv run benchmark.py --eval-size 10 model:ransac3dof model.profile:my-custom-profile
```

See [How to Add a New Model](../how-to/add_estimator.md) for adding an entirely new *algorithm*
(a new estimator class, not just a new profile of an existing one).
