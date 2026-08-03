# 01 — The configuration refactor: implementation guide

`status: draft`

## Why

Four bugs this week, all the same shape: each component was correct, the **seams between
them** were not. `--param-overrides` silently ignored on one path; `fixed` threaded
everywhere but never passed; `ransac_max_iterations` suggested twice; tyro's `=` parsing.

No test of a single function catches any of them, because the problem was never inside a
function. It was that **three places assemble configuration, and they disagree**:

| path | params from | `--param-overrides` means |
| :--- | :--- | :--- |
| `benchmark.py` plain | `args.model.profile.params` | rejected |
| `benchmark.py --sweep` | `{**suggest_params(…), **overrides}` | the only mechanism |
| `scripts/local_eval.py` | `type(p)(**{**vars(p), **overrides})` | merged onto the profile |

And you cannot see any of it without downloading YOLO weights first, because configuration
assembly is welded to execution.

**After this refactor:** one place decides configuration, one entry point runs it, and a test
can run the exact command you would type while asserting on what every stage received.

---

## The one rule to hold onto

```
benchmark.py  ->  sweep.py  ->  evaluation.py  ->  run_config.py
                                              ->  metrics.py
```

**Arrows point one way, and nothing at or below `evaluation.py` may import `optuna` or
`wandb`.**

That single constraint is what prevents this bug class returning. A second set of
configuration semantics can only grow inside the sweep path if the sweep path is allowed to
assemble configuration — and it won't be, because assembly lives in a module that cannot see
Optuna.

Put that sentence in each module's docstring. It is the rule a future reader needs.

---

## Commit 0 — clear the dangling references

You deleted `Ransac3DoFFullMeshEstimator`, but `cli_config.py` still imports it, so
`import cli_config` raises and everything downstream is dead.

Remove from `cli_config.py`: the import, `Ransac3DoFFullMeshProfileSelect` and its two
profiles, the `Ransac3DoFFullMeshPreset` dataclass, and its member + description string in the
`ModelPreset` union. Then delete `TestRansac3DoFFullMeshProfiles` from
`tests/test_cli_config.py`.

```bash
uv run python -c "import cli_config"   # names the next one until there are none
uv run pytest -q
```

---

## Commit 1 — split `benchmark.py`

**Pure moves. Change no logic.** If your diff contains anything other than relocated
functions and adjusted imports, split the commit — a behaviour change hidden inside a
1200-line move is unreviewable.

Where every existing symbol goes:

| new file | takes | may import |
| :--- | :--- | :--- |
| `metrics.py` | `PoseErrorMetrics`, `extract_pose_errors`, `compute_average_recall`, `GROSS_YAW_DEG`, `TrialMetrics`, `compute_trial_metrics`, `finite_or_none`, `FrameRecord`, `write_frame_records_csv` | numpy, pydantic, csv |
| `run_config.py` | `_BOOLS`, `resolve_param_overrides` | numpy, cli_config, methods |
| `evaluation.py` | `evaluate_pipeline`, `draw_eval_indices`, `derive_internal_seeds` | metrics, pipeline, methods, o3d |
| `sweep.py` | `run_parameter_sweep`, `build_pareto_figure` | optuna, wandb, plotly, evaluation |
| `reporting.py` | `log_input_artifacts` | wandb, metrics |
| `benchmark.py` | `main()` only | everything above |

Delete `reject_param_overrides_outside_sweep` — commit 2 makes the flag work on both paths
instead of banning it on one.

### How to do the move without losing your place

1. Create the file with its docstring and imports.
2. In `benchmark.py`, fold the function (`zc`), select the folded line (`V`), cut (`d`), paste
   into the new file.
3. Add the import back in `benchmark.py` — for now, `from metrics import compute_trial_metrics`
   etc. Let the failing test tell you what you missed.

```bash
uv run pytest -q      # must be green with ZERO test edits
```

Green with no test changes is the whole proof that this commit changed nothing.

---

## Commit 2 — `run_config.py`: one configuration path

This is the heart of the refactor. Read it before pasting — everything after depends on it.

### Errors that say where they came from

```python
class ConfigError(Exception):
    """Base for every configuration-resolution failure.

    Having a base class means a caller can catch every failure from this module
    with one `except ConfigError`, without also swallowing unrelated ValueErrors.
    """


class UnknownOverrideError(ConfigError):
    def __init__(self, name: str, valid: set[str]) -> None:
        self.name = name
        self.valid = valid
        super().__init__(
            f"Unknown parameter override {name!r}. Available: {sorted(valid)}"
        )
```

Storing `name` and `valid` on the exception — not only in the message — lets a test assert on
`err.name` instead of grepping a string that will be reworded one day.

### Resolving the overrides

This is your existing `resolve_param_overrides`, with the raise swapped for the typed error:

```python
def resolve_param_overrides(estimator_cls, extrinsic, overrides) -> dict[str, Any]:
    if not overrides:
        return {}

    valid = {f.name for f in dataclasses.fields(estimator_cls(extrinsic=extrinsic).params)}
    resolved: dict[str, Any] = {}
    for name, raw in overrides.items():
        if name not in valid:
            raise UnknownOverrideError(name, valid)
        if isinstance(raw, str):
            try:
                resolved[name] = ast.literal_eval(raw.capitalize() if raw in _BOOLS else raw)
            except (ValueError, SyntaxError):
                resolved[name] = raw
        else:
            resolved[name] = raw
    return resolved
```

### The two builders — one meaning for "override"

```python
def build_estimator_params(profile_params, overrides):
    """Profile params, with overrides applied LAST so they win.

    Used by the plain evaluation path.
    """
    return type(profile_params)(**{**vars(profile_params), **overrides})


def build_trial_params(estimator_cls, trial, overrides) -> dict[str, Any]:
    """One sweep trial's params: suggested, with overrides applied LAST so they win.

    `fixed` stops the parameter being SUGGESTED (so Optuna never models a dimension
    that cannot affect the objective); the merge supplies its VALUE, including for
    parameters suggest_params never proposes at all.
    """
    return {
        **estimator_cls.suggest_params(trial, fixed=frozenset(overrides)),
        **overrides,
    }
```

Both apply overrides last. That is the unification: **"override" means "applied last, wins"**,
everywhere, and there is no longer a path where it means "error" or "ignored".

### Seeing the search space without running a study

```python
class RecordingTrial:
    """Stands in for optuna.Trial; answers with `low` and records what was asked.

    Lives here rather than in the tests because the program itself needs it (see
    search_space_of) -- and one definition cannot drift from another.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    def suggest_float(self, name, low, high, step=None, log=False):
        self.asked.append(name)
        return low

    def suggest_int(self, name, low, high, step=1, log=False):
        self.asked.append(name)
        return low


def search_space_of(estimator_cls, overrides) -> list[str]:
    """The parameter names this estimator would ask Optuna for, given these overrides."""
    trial = RecordingTrial()
    estimator_cls.suggest_params(trial, fixed=frozenset(overrides))
    return sorted(trial.asked)
```

`search_space_of` is the answer to "is Optuna being asked the right questions?", computed
without a study, a sampler, or a single frame of data.

### The config object

```python
@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything decided BEFORE any compute. Assembled once, then read-only."""

    estimator_cls: type[BasePoseEstimator]
    params: Any                    # fully resolved estimator params
    depth_trunc: float
    extrinsic: np.ndarray
    resolved_overrides: dict[str, Any]
    search_space: list[str]
    sweep: bool
    n_trials: int
    eval_size: int
    seed: int
    n_seeds: int
    o3d_seed: int | None
    dataset_path: str
    dataset_glob: str
    name: str
    use_wandb: bool


def resolve_run_config(args) -> RunConfig:
    """The single entry point. Everything else reads the RunConfig it returns."""
    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    estimator_cls = args.model.ESTIMATOR_CLS
    overrides = resolve_param_overrides(estimator_cls, extrinsic, args.param_overrides)

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

    return RunConfig(
        estimator_cls=estimator_cls,
        params=build_estimator_params(args.model.profile.params, overrides),
        depth_trunc=args.model.profile.depth_trunc,
        extrinsic=extrinsic,
        resolved_overrides=overrides,
        search_space=search_space_of(estimator_cls, overrides) if args.sweep else [],
        sweep=args.sweep,
        n_trials=args.trials,
        eval_size=args.eval_size,
        seed=seed,
        n_seeds=args.n_seeds,
        o3d_seed=args.o3d_seed,
        dataset_path=args.dataset.path,
        dataset_glob=args.dataset.test_glob,
        name=args.name,
        use_wandb=not args.no_wandb,
    )
```

`frozen=True` means it cannot be mutated after assembly. That matters: it makes "the config
changed halfway through the run" impossible by construction, rather than by discipline.

### Wire it in

In `sweep.py`'s objective, replace the inline dict with the builder:

```python
suggested_params = build_trial_params(cfg.estimator_cls, trial, cfg.resolved_overrides)
```

```bash
uv run pytest -q
```

---

## Commit 3 — absorb `scripts/local_eval.py`, then delete it

`local_eval.py` is agent-written scaffolding you have never run. It carries a
`sys.path.insert(0, REPO_ROOT)` (forbidden by `CLAUDE.md` §2, and unnecessary —
`pyproject.toml` already sets `pythonpath = ["."]`) and its docstring documents the broken
`key=value` override syntax, so it was never run with overrides either.

So: **salvage the ideas, delete the file.** Five things are worth keeping, each on its own
merit:

1. **The partition-invariant line** — `good + gross + abstain = 1.000`. This is the guard
   against the failure that shipped three times: a rate improving because its denominator was
   emptied. Port it verbatim into `report_terminal`.
2. **`--o3d-seed`** — measured: unseeded Open3D swings `gross_yaw_rate` 0.556–0.833 across
   identical runs. Belongs on every path, not just a local script.
3. **Pre-preparing every mesh before the timed region**, so `p95_latency_s` is comparable
   between a local run and a sweep.
4. **Printing `gross_yaw_rate` and `abstention_rate` adjacent**, so they are read together.
5. **Split pooling** over the fixture splits.

### New `BenchmarkArgs` fields

```python
    # Terminal-only reporting: the local pre-push gate, replacing scripts/local_eval.py.
    no_wandb: bool = False
    # Open3D keeps its OWN global RNG, and prepare() calls sample_points_uniformly to
    # build the model cloud. Left unseeded that cloud -- and therefore FPFH and the whole
    # result -- differs every run: measured gross_yaw_rate swinging 0.556-0.833 across
    # three identical invocations. Seed it, or two arms are not comparable.
    o3d_seed: int | None = 0
    # Which dataset split(s) to evaluate. "all" pools them.
    split: Literal["all", "test", "validation", "train"] = "test"
```

### `evaluation.py` gains the orchestration

```python
def run_evaluation(cfg: RunConfig, model, camera, meshes, dataset) -> TrialMetrics:
    """Prepare, evaluate, score. The sequence local_eval.py and main() both duplicated."""
    estimator = cfg.estimator_cls(params=cfg.params, extrinsic=cfg.extrinsic)

    # Pre-prepare every mesh BEFORE the timed region: otherwise the first frame of each
    # cart type pays FPFH preparation inside the measurement and p95_latency_s is not
    # comparable across runs.
    for cart_type, mesh in meshes.items():
        estimator.prepare(mesh, cart_type)

    errors, times, det_failures, pose_failures, records = evaluate_pipeline(
        dataset=dataset,
        model=model,
        camera=camera,
        estimator=estimator,
        sample_indices=draw_eval_indices(len(dataset), cfg.eval_size, cfg.seed),
        meshes=meshes,
        depth_trunc=cfg.depth_trunc,
    )
    return compute_trial_metrics(errors, times, det_failures, pose_failures), records
```

### `reporting.py`

```python
def report_terminal(m: TrialMetrics) -> None:
    print(f"{'pose_ar':<22} {m.pose_ar:.4f}      (higher is better)")
    print(f"{'gross_yaw_rate':<22} {m.gross_yaw_rate:.4f}")
    print(f"{'abstention_rate':<22} {m.abstention_rate:.4f}   <-- read WITH gross_yaw")
    print(f"{'good_rate':<22} {m.good_rate:.4f}")
    # The partition invariant. A rate that improved by emptying its denominator shows
    # up here and nowhere else -- three mechanisms shipped before this line existed.
    total = m.good_rate + m.gross_yaw_rate + m.abstention_rate
    print(
        f"partition: good + gross + abstain = "
        f"{m.good_rate:.3f} + {m.gross_yaw_rate:.3f} + {m.abstention_rate:.3f} = {total:.3f}"
    )
```

Then delete `scripts/local_eval.py` and `LocalEvalArgs`, and update the references in
`AGENTS.md` and `CLAUDE.md` §1 that name it as the local gate.

```bash
# before
uv run scripts/local_eval.py model:vsac3dof model.profile:tuned
# after
uv run benchmark.py --no-wandb --split all --dataset.path tests/fixtures \
  model:vsac3dof model.profile:tuned
```

---

## Commit 4 — `tracing.py`: the program instruments itself

The idea: **the test runs the byte-identical command you would type**, plus one environment
variable, and the program records what each stage received.

`sys.setprofile(fn)` asks Python to call `fn` on every function call and return. The callback
receives a *frame*, which carries `f_code.co_name` (the function's name), `f_locals` (its
arguments) and `f_back` (the caller). That is a debugger's machinery without breakpoints, and
it is all stdlib.

```python
"""
Opt-in call tracing for tests.

Off unless SIXDPOSE_TRACE=<path> is set, so a traced command is byte-identical to the
real one -- the test runs exactly what you would type, and the tracer rides along.
"""

import atexit
import json
import os
import sys

_RECORDS: list[dict] = []
_MODULES: tuple[str, ...] = ()
_SEQ = 0


def _safe(value) -> str:
    """repr() that cannot itself blow up the program being traced."""
    try:
        text = repr(value)
    except Exception:
        return "<unrepr-able>"
    return text if len(text) <= 300 else text[:300] + "..."


def _probe(frame, event, arg):
    global _SEQ
    if event != "call":
        return
    module = frame.f_globals.get("__name__", "")
    # The filter is NOT optional: unfiltered this records every numpy internal and
    # produces a gigabyte of trace for a two-frame run.
    if not module.startswith(_MODULES):
        return
    _SEQ += 1
    _RECORDS.append({
        "seq": _SEQ,
        "event": "call",
        "fn": frame.f_code.co_name,
        "module": module,
        "args": {k: _safe(v) for k, v in frame.f_locals.items()},
        "caller": frame.f_back.f_code.co_name if frame.f_back else None,
    })


def emit(stage: str, **payload) -> None:
    """A structured marker at a stage boundary.

    The raw call log answers "was this called, and which branch ran". emit() answers
    "with exactly which values", in a schema that survives renaming a function.
    """
    global _SEQ
    if not os.environ.get("SIXDPOSE_TRACE"):
        return
    _SEQ += 1
    _RECORDS.append({
        "seq": _SEQ,
        "event": "stage",
        "stage": stage,
        "payload": {k: _safe(v) for k, v in payload.items()},
    })


def install(modules=("benchmark", "run_config", "evaluation", "sweep", "methods")) -> None:
    global _MODULES
    path = os.environ.get("SIXDPOSE_TRACE")
    if not path:
        return
    _MODULES = tuple(modules)
    sys.setprofile(_probe)
    atexit.register(_flush, path)


def _flush(path: str) -> None:
    sys.setprofile(None)
    with open(path, "w") as handle:
        json.dump(_RECORDS, handle, indent=2)
```

Two lines at the top of `main()`:

```python
def main():
    tracing.install()
    args = tyro.cli(BenchmarkArgs)
```

and one `emit` where the config is decided, in `resolve_run_config`'s caller:

```python
    cfg = resolve_run_config(args)
    tracing.emit(
        "config_resolved",
        estimator=cfg.estimator_cls.__name__,
        overrides=cfg.resolved_overrides,
        search_space=cfg.search_space,
        sweep=cfg.sweep,
    )
```

**Three things to know about `sys.setprofile`:**

- It applies to the **current thread only**. If work moves to threads later, they need
  `threading.setprofile` too.
- It does not fire for frames already on the stack when you install it — hence installing on
  the first line of `main()`.
- It is disabled inside your callback, so `_probe` cannot trace itself into infinite recursion.

---

## Commit 5 — the tests

Scope of `tests/test_param_overrides.py` is the **override mechanism, end to end**. Pose
quality belongs in a separate regression file.

### The helper

```python
# tests/helpers/trace.py
import json, os, shlex, subprocess, tempfile


class Trace:
    def __init__(self, records, proc):
        self.records = records
        self.proc = proc

    def calls(self, fn):
        return [r for r in self.records if r.get("fn") == fn]

    def stage(self, name):
        for r in self.records:
            if r.get("stage") == name:
                return r["payload"]
        raise AssertionError(f"stage {name!r} never reached. Stages: "
                             f"{[r.get('stage') for r in self.records if 'stage' in r]}")

    def assert_called(self, fn):
        assert self.calls(fn), f"{fn} was never called"

    def assert_not_called(self, fn):
        assert not self.calls(fn), f"{fn} was called {len(self.calls(fn))}x"


def run_traced(cmd: str, expect_success: bool = True) -> Trace:
    path = tempfile.mktemp(suffix=".json")
    env = {**os.environ, "SIXDPOSE_TRACE": path}
    proc = subprocess.run(shlex.split(cmd), env=env, capture_output=True, text=True)
    if expect_success and proc.returncode != 0:
        raise AssertionError(f"command failed:\n{proc.stderr[-2000:]}")
    records = json.load(open(path)) if os.path.exists(path) else []
    return Trace(records, proc)
```

### Fast tests, no subprocess

Follow the existing `_select()` pattern in `tests/test_cli_config.py`, which already does
`tyro.cli(X, args=[...])`.

```python
    def test_unknown_name_raises(self):
        with self.assertRaises(UnknownOverrideError) as ctx:
            resolve_param_overrides(VSACSe2Estimator, EXTRINSIC, {"icp_visibility_culll": "true"})
        self.assertEqual(ctx.exception.name, "icp_visibility_culll")

    def test_bools_become_real_bools(self):
        out = resolve_param_overrides(VSACSe2Estimator, EXTRINSIC,
                                      {"icp_visibility_cull": "true"})
        self.assertIs(out["icp_visibility_cull"], True)

    def test_pinned_param_leaves_the_search_space(self):
        full = search_space_of(VSACSe2Estimator, {})
        pinned = search_space_of(VSACSe2Estimator, {"voxel_size": 0.02})
        self.assertIn("voxel_size", full)
        self.assertNotIn("voxel_size", pinned)

    def test_no_parameter_suggested_twice(self):
        trial = RecordingTrial()
        VSACSe2Estimator.suggest_params(trial)
        dupes = {n for n in trial.asked if trial.asked.count(n) > 1}
        self.assertEqual(dupes, set(), f"suggested twice: {dupes}")
```

`assertIs`, never `assertTrue` — `assertTrue` passes when the value is the *string* `"true"`,
which is exactly the bug it is meant to catch.

### Slow tests, the real command

```python
BASE = ("uv run benchmark.py --no-wandb --eval-size 2 "
        "--dataset.path tests/fixtures --split test "
        "model:vsac3dof model.profile:tuned")


@pytest.mark.slow
class OverridesEndToEnd(unittest.TestCase):

    def test_sweep_pins_and_removes_from_search_space(self):
        t = run_traced(f"uv run benchmark.py --sweep --trials 1 --eval-size 2 --no-wandb "
                       f"--dataset.path tests/fixtures "
                       f"--param-overrides icp_visibility_cull true "
                       f"model:vsac3dof model.profile:tuned")
        cfg = t.stage("config_resolved")
        self.assertIn("icp_visibility_cull", cfg["overrides"])
        self.assertNotIn("icp_visibility_cull", cfg["search_space"])
        t.assert_called("run_parameter_sweep")          # the right branch ran

    def test_plain_path_applies_the_override(self):
        t = run_traced(f"{BASE} --param-overrides icp_visibility_cull true")
        self.assertIn("icp_visibility_cull", t.stage("config_resolved")["overrides"])
        t.assert_not_called("run_parameter_sweep")      # and NOT the sweep branch

    def test_equals_syntax_is_rejected(self):
        t = run_traced(f"{BASE} --param-overrides icp_visibility_cull=true",
                       expect_success=False)
        self.assertNotEqual(t.proc.returncode, 0)
        self.assertIn("icp_visibility_cull=true", t.proc.stderr)
```

Register the marker so the fast layer stays your save-hook:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
markers = ["slow: runs a real subprocess; use `-m 'not slow'` to skip"]
```

```bash
uv run pytest -m "not slow" -q   # inner loop
uv run pytest -q                 # pre-push
```

**Make each test fail before you make it pass.** Comment out the guard it targets, watch red,
restore. A test you have never seen fail is a test you have not verified.

---

## Verify the whole thing

```bash
uv run python -c "import cli_config"
uv run ruff check . && uv run ruff format --check .
uv run pytest -q

# the local gate still works
uv run benchmark.py --no-wandb --split all --dataset.path tests/fixtures \
  model:vsac3dof model.profile:tuned

# the tracer, on the real command
SIXDPOSE_TRACE=/tmp/t.json uv run benchmark.py --sweep --trials 1 --eval-size 2 \
  --no-wandb --dataset.path tests/fixtures \
  --param-overrides icp_visibility_cull true \
  model:vsac3dof model.profile:tuned
python -m json.tool /tmp/t.json | head -40
```

**Good:** the trace shows `resolve_param_overrides` receiving the raw dict, `suggest_params`
receiving `fixed={'icp_visibility_cull'}`, `icp_visibility_cull` absent from `search_space`,
and `run_parameter_sweep` in the call log.

**Bad:** an empty `fixed`, or `icp_visibility_cull` present in `search_space` — the wiring
regressed.

**The check that proves the unification:** the same
`--param-overrides icp_visibility_cull true` must produce the same resolved value on the plain
path and the sweep path. Assert it rather than trusting it.

---

## APIs you will meet

**`ast.literal_eval`** parses a string containing a Python *literal* — number, string, tuple,
list, dict, `True`/`False`/`None`. Unlike `eval` it never executes code. Raises `ValueError`
or `SyntaxError` on anything else, which is why the fall-through to the raw string exists.

**`dataclasses.fields(obj)`** lists a dataclass's declared fields; `{f.name for f in …}` is how
`resolve_param_overrides` knows which override names are legal without hard-coding a list.

**`vars(obj)`** returns an object's `__dict__` — for a dataclass instance, its field values. So
`type(p)(**{**vars(p), **overrides})` means "rebuild this params object with these fields
replaced". (`dataclasses.replace(p, **overrides)` does the same and is more idiomatic; either
is fine, `replace` validates field names for you.)

**`frozen=True` on a dataclass** blocks attribute assignment after construction. Used on
`RunConfig` so "the config changed halfway through the run" is impossible by construction.

**`subprocess.run(..., capture_output=True, text=True)`** runs a command and returns an object
with `.returncode`, `.stdout`, `.stderr` as strings. `shlex.split` turns a command *string*
into the argument *list* it expects, respecting quotes.

**`atexit.register(fn, arg)`** runs `fn(arg)` when the interpreter exits normally — how the
tracer flushes its log without every code path having to remember to.

---

## Related

`02-deriving-the-fixed-parameters.md` — the physics derivations (depth_trunc, voxel_size, the
crop fraction, the edge-length tolerance). Unrelated to this refactor and still pending; do it
after.