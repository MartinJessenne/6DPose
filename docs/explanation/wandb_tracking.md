# Experiment Tracking with Weights & Biases

This explains why W&B was added on top of the existing print-statement reports and Optuna SQLite
studies, and how its concepts map onto `benchmark.py`. For the CLI/config side of this codebase,
see [Configuration with tyro](tyro_cli_config.md).

---

## The problem this solves

Before this, a benchmark run's results existed only as terminal output, copy-pasted by hand into
notes if you wanted to keep them. Sweep results fared better (Optuna's own SQLite study), but
that's per-study and isn't easily browsed alongside *other* studies, other algorithms, or
one-off benchmark runs. W&B gives every run -- whether a single fixed-parameter benchmark or one
trial of a sweep -- a permanent, queryable, cloud-hosted record: what commit it ran on, what
hyperparameters it used, and what it measured.

## W&B vocabulary, mapped directly onto this codebase

- **Project**: the top-level container. This project uses **one shared project, `"6dpose"`**,
  across all three algorithms (rather than one project per algorithm) specifically because the
  whole point of this benchmark suite is comparing PPF vs. RANSAC vs. RANSAC-3DoF against each
  other -- splitting them into separate projects would fragment that exact comparison.
- **Group** / **Job type**: the finer-grained facets *within* the shared project.
  `group=estimator_cls.__name__` (e.g. `"Ransac3DoFEstimator"`) groups runs by algorithm;
  `job_type="benchmark"` vs. `job_type="sweep"` distinguishes a single fixed-parameter evaluation
  from one trial of a hyperparameter sweep.
- **Run**: one `wandb.init()` → ... → `run.finish()` lifecycle. In `benchmark.py`, the
  default-eval branch opens exactly one run for the whole evaluation; the sweep branch opens one
  run *per Optuna trial*.
- **Config**: hyperparameters logged once, at `wandb.init(config=...)` time. For a benchmark run,
  this is `dataclasses.asdict(args.model.profile.params)` plus `depth_trunc`/`eval_size`/`seed` --
  literally the same frozen dataclass you already have from the tyro CLI, no re-encoding needed.
- **Metrics**: `run.log({...})`, called as many times as you like within a run. Every number the
  benchmark report already prints (success rate, AR, p95 latency, translation/rotation error
  mean+median per axis, flip rate, non-flipped-sample medians) gets logged this way -- see
  `benchmark.py`'s default-eval branch for the exact 1:1 mapping from `print(...)` to `run.log(...)`.
- **Tags**: free-form labels for filtering in the UI. The sweep branch tags each trial's run with
  the Optuna `study_name`, so you can filter the W&B UI down to one sweep's worth of trials.

## Why manual logging, not `WeightsAndBiasesCallback`

`optuna-integration` ships an official `WeightsAndBiasesCallback` that can be passed to
`study.optimize(objective, callbacks=[wandbc])`. We deliberately did **not** use it, for two
reasons:

1. **It only captures what Optuna itself knows about a trial** -- the suggested params and the
   objective's return values. It has no idea about `trial.set_user_attr(...)` diagnostics
   (`average_recall`, `detection_failures`, `pose_failures`, `flip_rate` in this codebase's
   `objective()`), so those would need manual `wandb.log()` calls inside the objective regardless
   -- at which point the callback isn't saving much.
2. **It's genuinely marked deprecated in the version installed here.** Both
   `WeightsAndBiasesCallback` and `MLflowCallback` carry an explicit
   `@deprecated_class("4.9.0", "6.0.0")` decorator in `optuna-integration==4.9.0` (the current
   release, confirmed by reading the installed source directly, not inferred from the separate
   fact that Optuna moved third-party integrations into their own package a while back -- that
   package move itself isn't a deprecation signal, but this specific decorator on this specific
   class is). Instantiating either class emits a real runtime warning: *"Deprecated in v4.9.0
   ... removal ... currently scheduled for v6.0.0, but this schedule is subject to change."*
   Since we do the three lines of `wandb.init()`/`run.log()`/`finish()` ourselves, this code path
   never touches that deprecated class at all -- unaffected by whatever happens in v6.0.0.

The manual approach is exactly as much code as the callback would have needed anyway (one
`wandb.init()` block, one `run.log()` call), and it's easier to see and understand exactly what
gets logged, which mattered more here than saving a few lines.

## A real gotcha found by testing: `run.log()`, not `wandb.log()`

Because `study.optimize()` calls the sweep's `objective(trial)` function repeatedly, in the same
Python process, across every trial, each trial needs its *own* W&B run -- not one run accumulating
data for the whole sweep. That's what `reinit="create_new"` on `wandb.init()` is for: it forces a
genuinely new run every call, even though a previous run may have just been active in this same
process.

The gotcha, found by directly testing this rather than assuming it would just work: with
`reinit="create_new"`, the module-level `wandb.log(...)` shorthand (which normally logs to
whatever run is the "current" ambient one) silently fails to find an active run and raises
`wandb.errors.errors.Error: You must call wandb.init() before wandb.log()`. The fix is to always
call `.log(...)` on the `run` object returned by `wandb.init()` directly:

```python
with wandb.init(..., reinit="create_new") as run:
    ...
    run.log({...})   # correct -- explicitly tied to this run
    # wandb.log({...})  # would silently fail to attach under reinit="create_new"
```

This codebase uses `run.log(...)` consistently in both the benchmark branch and the sweep branch,
even though the benchmark branch doesn't strictly need `reinit="create_new"` (only one run ever
opens there) -- one convention, not two, for one less thing to remember.

## Git commit capture is automatic

`wandb.init()` records the current commit SHA (and an uncommitted-diff patch, if any) on the run
page with zero extra code, as long as it's run inside a git working directory. No manual tagging
needed to answer "what code produced this result."

## Setup prerequisite

Cloud-hosted W&B needs network access and an authenticated API key on whatever machine actually
runs `benchmark.py` -- run `wandb login` once on that machine (the free educational tier is what
this project uses). This can't be done from a machine that never runs the pipeline itself.
