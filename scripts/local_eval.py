"""
scripts/local_eval.py has been absorbed into benchmark.py (--no-wandb).

Use instead:
  uv run benchmark.py --no-wandb --split all --dataset.path tests/fixtures model:vsac3dof model.profile:tuned
"""

import sys

print("scripts/local_eval.py has been absorbed into benchmark.py (--no-wandb).")
print("Run instead:")
print(
    "  uv run benchmark.py --no-wandb --split all --dataset.path tests/fixtures model:<algo> model.profile:<tuning>"
)
sys.exit(1)
