# How to Run Parameter Sweeps

This guide explains how to execute multi-objective hyperparameter optimization sweeps using Optuna and Hydra configuration overrides.

---

## Overview

Parameter sweeps optimize matching thresholds, downsampling parameters, and solver limits to find the Pareto-optimal frontier between **Accuracy** (low translation and rotation error) and **Execution Time**. The optimization results are logged directly to a local SQLite database file.

---

## Step 1: Run a Parameter Sweep

To start a sweep, set `sweep=true` and choose your model on the CLI. 

### Basic Command:
```bash
uv run benchmark.py model=ransac sweep=true
```

### Config Customizations:
You can specify the number of Optuna trials, the size of the validation evaluation slice per trial, and a unique name for the Optuna study:
```bash
uv run benchmark.py model=ppf sweep=true trials=50 eval_size=30 name="PPF_Tuning"
```

---

## Step 2: Locate and Inspect the Optuna Database

- **File Output**: The sweep results are saved to an SQLite database file matching the structure `optuna_<name>.db` in the workspace root directory.
  - E.g., `--name PPF_Tuning` generates `optuna_PPF_Tuning.db`.
- **Git Ignore**: These `.db` files are ignored by `.gitignore` to prevent committing experimental databases to source control.

### Programmatic Inspection in Python

You can load the study database to analyze the results or extract the best trials programmatically:

```python
import optuna

# Re-load the study from the SQLite database
study = optuna.load_study(
    study_name="PPF_Tuning",
    storage="sqlite:///optuna_PPF_Tuning.db"
)

# Print best trials on the Pareto Front
print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
for trial in study.best_trials:
    print(f"Trial {trial.number}:")
    print(f"  - Accuracy Loss: {trial.values[0]:.4f}")
    print(f"  - Duration:      {trial.values[1]:.4f}s")
    print("  - Params:", trial.params)
```
