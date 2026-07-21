# How to Run Parameter Sweeps

This guide explains how to execute multi-objective hyperparameter optimization sweeps using Optuna, with model/algorithm selection via tyro CLI overrides.

---

## Overview

Parameter sweeps optimize matching thresholds, downsampling parameters, and solver limits to find the Pareto-optimal frontier between **Accuracy** (low translation and rotation error) and **Execution Time**. The optimization results are logged directly to a local SQLite database file.

The sweep mechanics themselves (Optuna's `suggest_params`/multi-objective study/Pareto front) are unchanged by the tyro migration -- only the CLI syntax for picking a starting model/profile changed. See [Configuration with tyro](../explanation/tyro_cli_config.md) for the full CLI picture.

---

## Step 1: Run a Parameter Sweep

To start a sweep, pass `--sweep` and choose your algorithm + a starting profile on the CLI (the sweep explores its own hyperparameter search space regardless of which profile you pick -- see `suggest_params` in each `methods/*.py` file -- but `model:<algo>` still selects *which* estimator class is being tuned).

### Basic Command:
```bash
uv run benchmark.py --sweep model:ransac model.profile:default
```

### Config Customizations:
You can specify the number of Optuna trials, the size of the validation evaluation slice per trial, and a unique name for the Optuna study. Remember: scalar options come *before* the subcommand tokens.
```bash
uv run benchmark.py --sweep --trials 50 --eval-size 30 --name "PPF_Tuning" model:ppf model.profile:default
```

---

## Step 2: Locate and Inspect the Optuna Database

- **File Output**: The sweep results are saved to an SQLite database file matching the structure `sweeps/optuna_<name>.db`.
  - E.g., `--name PPF_Tuning` generates `sweeps/optuna_PPF_Tuning.db`.
- **Git Ignore**: New `.db` files under `sweeps/` are ignored by `.gitignore` to prevent committing experimental databases to source control (a few already-committed ones predate this and remain tracked deliberately).

### Programmatic Inspection in Python

You can load the study database to analyze the results or extract the best trials programmatically:

```python
import optuna

# Re-load the study from the SQLite database
study = optuna.load_study(
    study_name="PPF_Tuning",
    storage="sqlite:///sweeps/optuna_PPF_Tuning.db"
)

# Print best trials on the Pareto Front
print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
for trial in study.best_trials:
    print(f"Trial {trial.number}:")
    print(f"  - Accuracy Loss: {trial.values[0]:.4f}")
    print(f"  - Duration:      {trial.values[1]:.4f}s")
    print("  - Params:", trial.params)
```

---

## Step 3: The Same Sweep in Weights & Biases

Every trial also opens its own W&B run (project `6dpose`, tagged with the study name), logging
the trial's suggested hyperparameters plus the diagnostics also stored as Optuna user-attrs
(`average_recall`, `detection_failures`, `pose_failures`, `flip_rate`). The Optuna SQLite study
above remains the source of truth for resuming/inspecting a sweep programmatically; W&B is a
parallel, browsable record of the same trials -- useful for the parallel-coordinates/comparison
views its UI gives you for free. See [Experiment Tracking with W&B](../explanation/wandb_tracking.md)
for why it's wired this way (manual logging, not the `WeightsAndBiasesCallback` helper).
