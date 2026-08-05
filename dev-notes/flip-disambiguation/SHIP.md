# Ship it — the last implementation guide

`status: draft`

Five phases. Each ends with a green tree you could stop at. Phase 3 is what makes the final
sweep worth running; Phase 5 is the sweep.

**Where you are:** `ruff` clean, 110 tests passing, 52 subtests. The `Annotated` fix landed,
`TrialPruned` landed, `supports_seed` is gone from `benchmark.py`. What remains is one
structural change (Phase 2), one experimental-design change (Phase 3), and cleanup.

---

# Phase 1 — finish the cleanup (30 min, mechanical)

## 1.1 `supports_seed` remnant in `sweep.py`

One reference left. `seed` lives on `BaseParams` now, so every params class has it and the
flag can never be `False`. Delete the parameter, its use, and the `[None]` fallback;
`effective_n_seeds` becomes `n_seeds` unconditionally.

## 1.2 Extrinsic must be an array, and mandatory

`PPFEstimator.__init__` and `RansacEstimator.__init__` currently store whatever they are
handed:

```python
self.extrinsic = extrinsic          # a list stays a list
```

A Python list reaching `np.asarray(self.extrinsic)[:3, 3][:2]` further down is a `TypeError`
waiting for one specific call site. Restore the coercion:

```python
self.extrinsic = np.asarray(extrinsic, dtype=np.float64)
```

And in `BasePoseEstimator.build`, make `extrinsic` a **required keyword**:

```python
@classmethod
def build(cls, *, profile_params=None, overrides=None, extrinsic: np.ndarray) -> Self:
```

`Ransac3DoFEstimator` already raises if it gets `None`, so the refusal belongs in the
signature where a caller sees it, not 40 lines into a constructor.

The `*` in the signature makes everything after it keyword-only. Worth using deliberately:
it means nobody can ever call `build(params, overrides, extrinsic)` positionally and get the
argument order wrong.

## 1.3 `build()` uses `or` where it means `is None`

```python
base = profile_params or cls.params_cls()          # wrong
base = cls.params_cls() if profile_params is None else profile_params
```

**A frozen dataclass is always truthy** unless it defines `__bool__` or `__len__`. So
`x or default` *looks* like a None check and is not one. It happens to work here — but it
will silently pick the default the first time someone writes a params class with `__len__`,
and `x or y` on an empty list, empty dict, `0`, or `0.0` is the same bug in a form you will
meet constantly.

## 1.4 Confirm the `__post_init__` type check is complete

It should reject a value whose *type* is wrong, not only one whose range is wrong — that is
what let `ransac_max_iterations = 38000.0` through. Verify it covers both:

```python
    def __post_init__(self):
        hints = typing.get_type_hints(type(self), include_extras=True)
        for name, rng in self.search_space().items():
            v = getattr(self, name)
            if v is None:
                continue
            declared = _unwrap_optional(hints[name])
            if not isinstance(v, declared) or isinstance(v, bool) is not (declared is bool):
                raise TypeError(f"{type(self).__name__}.{name} = {v!r} is not {declared.__name__}")
            if not (rng.min <= v <= rng.max):
                raise ValueError(
                    f"{type(self).__name__}.{name} = {v} outside [{rng.min}, {rng.max}]"
                )
```

**Two Python facts this depends on**, both of which will catch you again:

- **`bool` is a subclass of `int`.** `isinstance(True, int)` is `True`, `True == 1`. Every
  numeric type check needs an explicit `bool` case or booleans sail through int fields.
- **`isinstance(1, float)` is `False`.** Python's numeric tower does *not* make `int` a
  subclass of `float`. So a `float` field handed the integer `1` raises here, while
  `coerce_override` promotes `int → float` for CLI values. Make them agree — either promote
  in both places or reject in both. Two paths that disagree about one question is the exact
  shape of every bug this refactor has chased.

**Gate:** `uv run ruff check . && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q`

---

# Phase 2 — one entry point, two subcommands

This is the structural change. It deletes a whole category of bug rather than fixing an
instance of one.

## 2.1 What is wrong

`sweep: bool = False` is a **mode selector wearing a flag's costume**. The proof is in
`cli_config.py` itself: three fields carry comments beginning *"Sweep-only:"*
(`trials`, `n_seeds`, `overrides`). When a third of an argument class documents itself as
inapplicable, it is two argument classes.

The consequences are live today:

| value | `benchmark.py` | `sweep.py` | `run_config.py` |
| :--- | :--- | :--- | :--- |
| seed fallback modulus | `2**31 - 1` | `2**31 - 1` | `2**32 - 1` |
| `o3d_seed` | never read | never read | read |

`--o3d-seed` is declared, its docstring records why it is necessary (unseeded, Open3D swings
`gross_yaw_rate` between 0.556 and 0.833 across identical runs), and **nothing on the main
path reads it**. The evaluation branch seeds Open3D with the *main* seed; the sweep path
never seeds it at all. The only code that honours the flag is `resolve_run_config`, which
nothing calls.

## 2.2 The shape to build

```
sixdpose eval  --eval-size 30 model:vsac3dof model.profile:tuned
sixdpose sweep --trials 100 --overrides icp_visibility_cull true model:vsac3dof model.profile:tuned
```

Three dataclasses instead of one:

```python
@dataclass(frozen=True)
class CommonArgs:
    """Everything both commands need."""
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


@dataclass(frozen=True)
class EvalArgs(CommonArgs):
    """Evaluate one configuration."""
    name: str = "Benchmark"


@dataclass(frozen=True)
class SweepArgs(CommonArgs):
    """Search hyperparameters with Optuna."""
    trials: int = 30
    study_name: str = "Sweep"
```

and the entry point becomes a `Union`, which is how tyro spells subcommands — the same
mechanism already giving you `model:vsac3dof`:

```python
Command = Union[
    Annotated[EvalArgs, tyro.conf.subcommand(name="eval")],
    Annotated[SweepArgs, tyro.conf.subcommand(name="sweep")],
]
```

**What this deletes permanently:** `RunConfig.sweep`, the `if cfg.sweep:` branch (tyro
dispatches on the type — you never write the test), `--trials` being silently accepted and
ignored in eval mode, and `search_space` being computed where it means nothing.

`RunConfig` splits the same way: `RunConfig` for the shared fields, `SweepConfig(RunConfig)`
adding `n_trials`, `study_name`, `search_space`. Every field meaningful in every object —
which is what a dataclass is for.

## 2.3 The entry point does four things

```python
def main():
    args = tyro.cli(Command)
    cfg = resolve_config(args)              # pure: no I/O, no network
    assets = load_assets(cfg)               # YOLO, camera, dataset, meshes
    run(cfg, assets)                        # dispatch on type(cfg)
```

The ordering is the invariant worth protecting: **all configuration is decided before any
I/O**. That is what makes the config object printable, freezable, and testable without
downloading YOLO weights.

`run_config.py` stays a **leaf** — it imports nothing from the other root modules, and that
is deliberate. An entry point must import the runners; if `run_config` were the entry point
it would have to import `sweep.py`, which puts `optuna` and `wandb` back within reach of
configuration assembly and lets a second set of config semantics grow there again. Keep the
arrows pointing one way:

```
entry point -> sweep -> evaluation -> run_config / metrics
```

Rename `benchmark.py` while you are here. It is a lie whenever the sweep runs.

## 2.4 The edits that follow

**a. Seed both RNGs once, before dispatch:**

```python
    np.random.seed(cfg.seed)
    if cfg.o3d_seed is not None:
        o3d.utility.random.seed(cfg.o3d_seed)
```

**Why two:** Open3D keeps its **own** global RNG, completely separate from numpy's.
`prepare()` calls `sample_points_uniformly` to build the model cloud, and that draws from
Open3D's generator — `np.random.seed(...)` does not touch it however many times you call it.
Unseeded, the model cloud differs every run, so FPFH differs, so the result differs.

When a result is irreproducible, **count the generators before suspecting the algorithm.**

**b. One seed fallback.** Standardise on `2**31 - 1` (what `derive_internal_seeds` already
draws from) and delete the other two computations.

**c. `run_parameter_sweep(cfg, assets)`** — sixteen parameters down to two. Ten of the
current ones are configuration `cfg` already holds.

**Delete the `resolve_param_overrides` call inside it.** `cfg.resolved_overrides` is already
the answer. Resolving overrides in two places is exactly how the two paths became free to
disagree about one flag.

**d. Extract `run_evaluation(cfg, assets)`** into `evaluation.py` so both branches are one
call. A 130-line inline branch opposite a one-line call is not two branches reading `cfg`.

**e. Print the search space at startup on the sweep path:**

```python
    print(f"Search space ({len(cfg.search_space)} free parameters):")
    for name, rng in cfg.search_space.items():
        print(f"  {name:32} {rng}")
```

`cfg.search_space` is the declared ranges minus anything pinned by `--overrides`. Three lines
that answer "is Optuna being asked the right questions?" before any compute. Both of this
project's silent search-space collapses would have been visible on trial 0.

**Gate — the two checks that prove Phase 2 landed:**

1. `--overrides voxel_size 0.04` on the sweep path: `voxel_size` must be **absent** from the
   printed search space.
2. Run the same eval command twice: `good_rate` must match **exactly**. Before the
   `o3d_seed` fix it could differ by the 0.556–0.833 spread.

---

# Phase 3 — shrink the search space from 8 free parameters to 3

This is what makes the final sweep meaningful rather than decorative.

## 3.1 Why

Optuna's TPE sampler models "which values produced good results" against "which produced
bad", and needs to have seen enough of the space to build those models. To resolve each axis
into `k` bins in `d` dimensions costs on the order of `k^d` samples. At a crude `k = 3`:

- `d = 8` → `3⁸ = 6561` trials
- `d = 3` → `3³ = 27` trials

You run ~100. At eight dimensions the winner is substantially a lucky draw that then gets
shipped as a "tuned profile". At three it is a real search.

Every parameter below stops being searched because **something other than the objective
already determines it** — workcell geometry, the sensor datasheet, or the CAD.

## 3.2 `depth_trunc = 5.5` — workcell geometry

The generation script places the camera at `d ~ U(0.8, 3.0)` m from the cart's front face
with a 30° upward tilt. The farthest point of the longest cart (colruyt, `L_x = 2.575` m)
sits at image-plane depth

```
cos(30°)·(3.0 + 2.575) + (0.757 − 0.304)·sin(30°) = 5.06 m
```

where `0.757` is the cart height `L_z` and `0.304` the camera height. **5.5 clears that with
margin and stays inside the D455's 0.6–6 m ideal range.**

`depth_trunc` should bound the *working volume* and never shape the object. At 5.5 it is
inert for the cart; its only remaining job is discarding warehouse background that survived
mask dilation. Below 5.06 it silently truncates the rear of the longest cart on far frames —
which is a data corruption, not a tuning choice.

**Edit.** It lives in two places and both need changing: `VSACSe2Profile.depth_trunc` in
`cli_config.py` (set `5.5` in `tuned`, `tuned-cheap`, `tuned-vis`, `tuned-vis-gate`; leave
`default` and `bare`, they are ablation baselines reproducing historical numbers), and the
`trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)` in the sweep objective, which
becomes a module constant:

```python
# Image-plane depth beyond which returns are discarded. Workcell geometry, NOT a
# tunable: generation places the camera at d ~ U(0.8, 3.0) m with a 30 deg tilt, so
# the far end of the longest cart sits at 5.06 m. Below that this silently truncates
# the cart itself.
DEPTH_TRUNC_M = 5.5
```

## 3.3 `voxel_size = 0.02` — sensor noise

Stereo depth noise grows with the square of range:

```
σ_z = z² · σ_d / (f·B)
```

where `σ_d` is disparity-matching noise in pixels, `f` the focal length in pixels and `B` the
stereo baseline. For the D455, `f·B = 639.99768 px × 0.095 m = 60.80`, and `σ_d` is
0.1–0.3 px (the upper end being Intel's "<2% at 4 m" spec, which folds in calibration bias).

Over the front slab's working range (0.69–2.60 m) that gives `σ_z` between 0.08 cm and
3.3 cm.

**Bracketed on both sides.** From below: a voxel under 0.02 m is *sub-noise* over the far
half of the range — it preserves noise instead of averaging it, and FPFH becomes less
repeatable. From above: `min_sample_distance = 3 × voxel` is already a quarter of leanflow's
0.243 m slab at 0.02.

So the next person to widen this has to argue against `σ_z`, not against a number. Remove it
from the search space entirely — not pinned by override, *removed*, so nothing spends a TPE
dimension on it.

## 3.4 `front_crop_depth` → an aspect-ratio formula

**The defect:** the crop depth is an absolute distance in metres, applied to three carts of
very different lengths. The slab's plan-view aspect ratio `R = L_y / depth` is what
disambiguates the 180° flip — a near-square slab is symmetric and the flip is undecidable.

With `depth = f · L_x` (a fixed fraction of cart length):

| cart | `L_y/L_x` | `R` at `f=0.25` | `R` at `f=0.35` |
| :--- | ---: | ---: | ---: |
| colruyt | 0.325 | 1.30 | **0.93** |
| picanol | 0.522 | 2.09 | 1.49 |
| leanflow | 0.724 | 2.90 | 2.07 |

colruyt is the binding constraint at every `f` — longest body, squarest slab. At `f = 0.35`
it is back to 0.93, square again.

**Take the better formula.** Target the aspect ratio directly instead of by proxy:

```
depth = L_y / R_target,     R_target = 2.0
```

giving colruyt 0.419 m, picanol 0.398 m, leanflow 0.352 m. **Every cart gets aspect exactly
2.0**, and the slab depths come out nearly uniform (0.35–0.42 m) — so `min_sample_distance`,
voxel quantization and FPFH support behave comparably across the fleet, instead of
leanflow's slab being 2.6× thinner than colruyt's.

It is the same amount of code as the fraction version and controls the failure mode directly.

**Edit.** Replace the field with `front_crop_aspect: float | None = 2.0` in
`Ransac3DoFParams`. Keep `crop_front_face(mesh, depth, min_height)` exactly as it is — it is
a correct pure geometric function that should keep taking metres so it stays testable in
isolation. Do the conversion in `Ransac3DoFEstimator.prepare`, where the mesh is in hand.

**Risk to watch:** this cuts leanflow's slab from 0.735 m to 0.352 m. Less geometry means
fewer FPFH keypoints and weaker retrieval. If `good_rate` drops specifically on leanflow,
that is the mechanism, and `--dump-frames` will show it by cart type.

## 3.5 `edge_length_threshold` → an absolute tolerance

**The parameterisation is wrong.** It compares two point-pair distances as a *ratio*, but the
error being tolerated is *absolute* sensor noise. Under a ratio test, the same physical
error passes at long baselines and fails at short ones — so the check is stricter exactly
where the geometry is weakest.

Replace with an absolute tolerance:

```python
if abs(len_p - len_q) > params.edge_length_tolerance:
    continue
```

with `tolerance = k · σ · √2`, `k = 3`, and `σ` the scene-point position noise. Two
independent points each carry `σ`, so their difference carries `σ√2`. Taking the worst case
over the working range (`σ = 0.033` m at `σ_d = 0.3`):

```
tolerance = 3 × 0.033 × 1.414 ≈ 0.14 m
```

That is generous. The principled refinement computes `σ` per-correspondence from each scene
point's own depth — you have `z` for every point and `σ_z = z²σ_d/(fB)` is one multiply.
Worth building eventually; the constant is fine to start.

Remove `edge_length_threshold` from the search space.

## 3.6 `ransac_max_iterations` → a per-profile budget, not a search dimension

**This one is not flat, and an earlier claim that it "buys latency and nothing else" was
wrong.** That came from a correlation computed inside a sweep bucket where iterations
co-varied with eight other parameters. A clean single-variable ladder (commit `8c6893c`,
seed 5829, `eval_size 70`, `n_seeds 3`, Hoppe on, normal-consistency off) says otherwise:

| iterations | `good_rate` | `pose_ar` | p95 |
| ---: | ---: | ---: | ---: |
| 1 000 | 0.6184 | 0.2138 | 1.93 s |
| 3 000 | 0.6522 | 0.2350 | 2.02 s |
| 10 000 | 0.7488 | 0.2825 | 2.56 s |
| 46 940 | **0.8164** | **0.3137** | 5.59 s |

**Monotone in both accuracy metrics across a 47× range.** Capping at 10 000 costs 0.068
`good_rate` and 0.031 `pose_ar` to save 3.0 s of p95. A real trade, not a free win.

So it *is* a latency budget — but the budget must be chosen against the curve, and **the
curve has moved**: the front-face gate cut p95 from 5.55 s to 3.08 s by pruning hypotheses,
so the relationship above is not the one on the gated pipeline.

**Edit.** Remove it from the search space and set it per profile. Then re-measure the ladder
on top of the gate — `{2 000, 10 000, 46 940, 100 000}`, full test set, paired seed — and
pick the knee for `tuned-fast` and the plateau for `tuned-accurate`. Two profiles, one
honest trade-off, no search.

**Be clear-eyed about latency:** nothing here reaches the 0.2 s / 5 Hz Orin Nano budget. The
fastest arm above is 1.93 s, still 10× over, and that has held across every configuration
measured on this project. The iteration knob chooses where on a bad curve to sit; it does
not get you onto a good one.

## 3.7 What stays swept

| parameter | why it survives |
| :--- | :--- |
| `z_gate_threshold` | Floor derivable from sensor noise, ceiling needs a CAD measurement nobody has taken. See below. |
| `rho` | Genuinely empirical — the spatial-independence radius for flip disambiguation. |
| `icp_max_correspondence_distance` | Its job changed when the visibility cull landed; needs re-finding. |

**The `z_gate_threshold` measurement, if you want it down to two dimensions.** In `scratch/`:
load each cart's `model_down` at ground-truth pose, transform to the base frame, histogram
the `z` coordinates at 1 cm bins. The modes are the cart's horizontal structures; the
ceiling is half the minimum gap between adjacent modes. Then set
`z_gate_threshold = min(ceiling, 0.06)` and run it as an arm against the current 0.349.

If the ceiling comes out *below* the noise floor the two bounds have crossed — which would
be a real finding, not a tuning problem: it would mean the gate cannot simultaneously admit
true matches and reject structural confusions at this sensor's precision.

**And one warning about the tuned profiles:** the old `icp_max_correspondence_distance`
search ceiling of 0.10 sat *below every tuned optimum on record* (0.101 to 0.222). The search
was stopping exactly where the answer was, so every value inherited from those sweeps is an
artefact of the bound, not a property of the data.

---

# Phase 4 — package it

There is no package. `[project]` names `6dpose` but the modules sit loose at the repo root,
and `uv run benchmark.py` works only because `pythonpath = ["."]` happens to cover it. That
is a dev convenience, not something you can deploy to an Orin Nano.

- Move the root modules under `src/sixdpose/`.
- Add the console entry point:
  ```toml
  [project.scripts]
  sixdpose = "sixdpose.__main__:main"
  ```
- Fix `[tool.ruff.lint.isort] known-first-party` — it currently lists `utils`, which does not
  exist, and omits `run_config`, `evaluation`, `metrics`, `sweep`, `pipeline`, `reporting`.
  Six of your own modules have been sorted as third-party this whole time.

Then `pip install .` gives you `sixdpose eval ...` and `sixdpose sweep ...` on the target.

---

# Phase 5 — the runs

## 5.1 Guard the pipeline first

Not a tracing framework — **assertions at the stage boundaries**, which is the part that
actually catches anything. Cheap enough to leave on:

| stage | assert |
| :--- | :--- |
| scene reconstruction | cloud non-empty; all coordinates finite; no point beyond `depth_trunc` |
| model preparation | `len(normals) == len(points)`; every normal unit-length to 1e-6; FPFH shape `(N, 33)` |
| global registration | `‖RᵀR − I‖_F < 1e-6` and `det(R) > 0` |
| SE(2) projection | `R[2,0]`, `R[2,1]` ≈ 0 and `R[2,2]` ≈ 1; `z == _active_z_offset` |
| metrics | `good + gross + abstain == 1.000` |

Three of these deserve their reasoning:

**`det(R) = +1`.** A rotation satisfies `RᵀR = I`, which forces `det(R) = ±1`. Only `+1` is a
rotation; `−1` is a rotation composed with a **reflection** — a mirror image. A cart is
roughly box-shaped, so a mirrored model still lands plenty of points near scene points and
the fitness looks healthy, while `yaw = arctan2(R[1,0], R[0,0])` returns a perfectly ordinary
number. The error reaches your headline metric with nothing flagging it.

**Unit normals.** FPFH describes a point by the *angles* between its normal and its
neighbours', and an angle comes from a dot product of vectors assumed unit length. A normal
of length 0.5 does not give a slightly wrong angle, it gives a different histogram. There is
a live cause here: Open3D's `voxel_down_sample` averages normals in a voxel **without
renormalising**, and on thin tubular frames the two walls of a tube fall in one voxel and
their normals partly cancel.

**The partition invariant.** Every frame that reached the estimator lands in exactly one of
good / gross-yaw / abstained, all three over the same denominator, so they sum to 1 by
construction. It earns a permanent place because **a rate can be improved by emptying its
denominator** — make the estimator abstain more and `gross_yaw_rate` falls while nothing got
better. That has shipped here three times.

## 5.2 Discard the old sweep databases

Every trial in `sweeps/optuna_*.db` is untrustworthy for compounding reasons: some recorded
parameter values that were sampled and then thrown away, all of them sampled
`ransac_max_iterations` as a float, and all of them searched `icp_max_correspondence_distance`
against a ceiling below its own optimum. They cannot be re-analysed, only re-run.

## 5.3 The final runs

```bash
# Sanity: the local gate, both paths.
uv run sixdpose eval --no-wandb --split all --dataset.path tests/fixtures \
  model:vsac3dof model.profile:tuned

# The sweep. 3 free dimensions, so ~60 trials is now a real search.
uv run sixdpose sweep --trials 60 --eval-size 70 --n-seeds 3 \
  model:vsac3dof model.profile:tuned

# The arms, paired on one seed so McNemar applies.
uv run sixdpose eval --seed 2144065271 --eval-size 70 --n-seeds 3 \
  --overrides normal_consistency true model:vsac3dof model.profile:tuned
```

**Good:** the search space prints 3 free parameters before trial 0; the partition line sums
to 1.000; two identical commands give identical `good_rate`.

**Bad:** a pinned parameter still listed in the search space; a recorded param the estimator
never received; `gross_yaw_rate_per_seed` entries all equal (the seed is not varying).

---

# Tricks worth keeping

1. **`hasattr` / `getattr(x, n, default)` / `dict.get()` fail silently on a typo.** They
   answer "no" instead of raising, so a misspelled name disables a whole branch while the
   code still reads as though the case is handled. Assert on the effect, not the presence.
2. **`typing.get_origin(Annotated[int, x])` returns `Annotated`, not `int`.** Peel
   `__metadata__` first, always. And `get_type_hints(cls)` strips `Annotated` while
   `get_type_hints(cls, include_extras=True)` keeps it — you need both, for different jobs.
3. **`typing.Union` and `types.UnionType` are different objects.** `Optional[X]` gives the
   first, `X | None` the second. Check for both or half your annotations slip through.
4. **`bool` is a subclass of `int`; `int` is not a subclass of `float`.** Both bite in any
   type dispatch over numbers.
5. **A frozen dataclass is always truthy.** `x or default` is not a None check — and neither
   is it for `0`, `0.0`, `[]` or `{}`.
6. **Libraries carry their own RNGs.** Open3D's is separate from numpy's. When a result is
   irreproducible, count the generators before suspecting the algorithm.
7. **A validator that checks a range has not checked a type.** Different questions, both
   need asking.
8. **`dataclasses.replace(obj, **changes)` re-runs `__post_init__`** — so validation fires on
   every derived object, not only at construction.
9. **Optuna: `study.ask()` samples for real; `FixedTrial(values)` returns what you hand it
   and raises for any name you did not supply.** Use the first to test the sampler, the
   second to test precedence. `create_study()` with no `storage=` is in-memory and fast
   enough for a unit test.
10. **`optuna.TrialPruned` keeps a trial in the history but out of `best_trials`** — the
    right way to exclude a degenerate run without pretending it never happened.
11. **A search bound that sits on the optimum is invisible in the results.** If tuned values
    cluster at a range's edge, the range is wrong, not the values.
