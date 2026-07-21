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
- **Run**: one `wandb.init()` → ... → `run.finish()` lifecycle. **One CLI execution is always
  exactly one run** -- the default-eval branch opens one run for the whole evaluation; the sweep
  branch opens *one run for the entire sweep*, not one per trial (see below for why, and how each
  trial still gets its own point in that run's history).
- **Config**: hyperparameters logged once, at `wandb.init(config=...)` time. For a benchmark run,
  this is `dataclasses.asdict(args.model.profile.params)` plus `depth_trunc`/`eval_size`/`seed` --
  literally the same frozen dataclass you already have from the tyro CLI, no re-encoding needed.
  For a sweep run, config is the sweep-level settings (`eval_size`, `n_trials`, `seed`) since no
  single set of hyperparameters describes the whole sweep -- each trial's own params are logged
  as metrics instead (see below).
- **Metrics**: `run.log({...})`, called as many times as you like within a run, optionally with an
  explicit `step=`. Every number the benchmark report already prints (success rate, AR, p95
  latency, translation/rotation error mean+median per axis, flip rate, non-flipped-sample medians)
  gets logged this way in the benchmark branch. In the sweep branch, `step=trial.number` is what
  turns "log once per trial" into a real per-trial trend line instead of a single point -- see
  [Run history vs. Run summary](#run-history-vs-run-summary) below.
- **Tags**: free-form labels for filtering in the UI. The sweep run is tagged with the Optuna
  `study_name` (redundant with its `name=` here, but convenient if you ever rename runs later).

## Run history vs. Run summary

Two different views of the same logged data, easy to conflate from the console output alone:

- **History** is the full time series: every `run.log(..., step=N)` call appends a row. If the
  same metric key is logged at multiple steps, its history is a real multi-point trend -- this is
  what the sweep branch relies on, logging every trial's params + metrics at `step=trial.number`.
- **Summary** is just a snapshot: the most recently logged value per key. It's what shows up in
  W&B's run-comparison tables.

In the benchmark branch, every metric is logged exactly once per run, so history and summary
trivially coincide (a single point). In the sweep branch, with N trials logged at N different
steps, history is a genuine N-point trend per metric/parameter -- which is also exactly the data
`wandb.Table` draws on for the Pareto-front scatter described next.

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

## One run per sweep, not one per trial

The first version of this integration opened a fresh W&B run per Optuna trial (via
`wandb.init(..., reinit="create_new")` inside `objective()`), mirroring the "multirun" pattern
`WeightsAndBiasesCallback` uses. That works, but a 200-trial sweep means 200 separate run pages
cluttering the workspace's run list -- not what you want when browsing.

The fix: open **one run for the entire sweep**, before `study.optimize()` starts, and have every
trial log into that same run at `step=trial.number` instead of creating its own run:

```python
with wandb.init(project="6dpose", name=study_name, ...) as run:
    def objective(trial):
        ...
        run.log({**suggested_params, "depth_trunc": depth_trunc, "accuracy_score": ..., ...},
                 step=trial.number)
        return accuracy_score, p95_time

    try:
        study.optimize(objective, n_trials=remaining)
    finally:
        # build + log the Pareto-front scatter here (see below) -- in a
        # `finally` so it still runs even if the sweep is Ctrl+C'd partway
```

One real gotcha found while iterating on the per-trial-run version (worth knowing even though the
final design doesn't hit it): that version used `reinit="create_new"` to force a genuinely new run
on every `wandb.init()` call in the same process, and with that setting, the module-level
`wandb.log(...)` shorthand silently failed to find an active run
(`wandb.errors.errors.Error: You must call wandb.init() before wandb.log()`) -- only calling
`.log(...)` on the `run` object returned by `wandb.init()` worked. The current one-run-per-sweep
design only calls `wandb.init()` once, so this specific trap no longer applies, but `run.log(...)`
(never the bare `wandb.log(...)`) is still used everywhere for consistency -- one convention,
not "whichever happens to work in this branch."

(Note for completeness: `WeightsAndBiasesCallback`'s *default* mode, `as_multirun=False`, already
behaves similarly -- one run for the whole study, trial-indexed logging via `step=`. It just
doesn't know about `trial.set_user_attr(...)` diagnostics or build a Pareto-front chart, and per
the deprecation note above, this codebase doesn't depend on that class either way.)

## Seeing the Pareto front in W&B: `wandb.Table` + `wandb.plot.scatter`

Optuna Dashboard's Pareto-front view was the target to match. After `study.optimize()` finishes
(or is interrupted -- the `finally` block runs either way), the sweep branch reads every
`COMPLETE` trial directly from `study.trials` (Optuna's own persistent record, not just whatever
ran in this process -- so a resumed sweep's chart always reflects the complete study, not just the
newly-added trials), builds a `wandb.Table` with one row per trial (params + `accuracy_score` +
`p95_time` + `average_recall` + `flip_rate`), and logs a scatter of it:

```python
table = wandb.Table(columns=[...], data=rows)
run.log({"pareto_front": wandb.plot.scatter(table, "p95_time", "accuracy_score", title=...)})
```

This renders as a native scatter panel on the run page -- speed on one axis, accuracy on the
other, one point per trial, exactly the Pareto-front comparison view `study.best_trials` gives you
textually in the console output already. Verified directly (not assumed): a small offline-mode
script confirmed both the `Run history` becoming a genuine multi-point trend under `step=`
logging, and the scatter chart materializing as a real logged artifact
(`files/media/table/*.table.json` on disk) rather than silently no-op-ing.

## Git commit capture is automatic

`wandb.init()` records the current commit SHA (and an uncommitted-diff patch, if any) on the run
page with zero extra code, as long as it's run inside a git working directory. No manual tagging
needed to answer "what code produced this result."

## Setup prerequisite

Cloud-hosted W&B needs network access and an authenticated API key on whatever machine actually
runs `benchmark.py` -- run `wandb login` once on that machine (the free educational tier is what
this project uses). This can't be done from a machine that never runs the pipeline itself.
