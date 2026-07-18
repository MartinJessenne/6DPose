#!/usr/bin/env bash
# Launches a crash-resumable Optuna sweep on an ephemeral container:
#   1. restores the study DB from the 'sweep-backups' branch if not present
#      (fresh container after a wipe),
#   2. starts the periodic DB backup pusher in the background,
#   3. runs the sweep under nohup in a retry loop, so it survives VS Code
#      tunnel/terminal drops and Python crashes. trials= is a TOTAL target
#      (benchmark.py only runs the remaining trials on resume).
#
# Usage: run_resumable_sweep.sh <STUDY_NAME> <TRIALS> <EVAL_SIZE> [MODEL=ransac3dof]
# Monitor:  tail -f sweeps/sweep_<NAME>.log
set -u

NAME=${1:?usage: run_resumable_sweep.sh <STUDY_NAME> <TRIALS> <EVAL_SIZE> [MODEL]}
TRIALS=${2:?missing TRIALS}
EVAL=${3:?missing EVAL_SIZE}
MODEL=${4:-ransac3dof}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/sweeps/optuna_${NAME}.db"

mkdir -p "$ROOT/sweeps"

# Restore the study from the backup branch after a container wipe
if [ ! -f "$DB" ]; then
    URL="$(git -C "$ROOT" remote get-url origin)"
    if git clone --depth 1 --branch sweep-backups "$URL" /tmp/sweep-restore 2>/dev/null; then
        if [ -f "/tmp/sweep-restore/optuna_${NAME}.db" ]; then
            cp "/tmp/sweep-restore/optuna_${NAME}.db" "$DB"
            echo "[sweep] restored study DB from origin/sweep-backups"
        fi
        rm -rf /tmp/sweep-restore
    fi
fi

nohup bash "$ROOT/scripts/backup_sweep_db.sh" "$NAME" 600 \
    > "$ROOT/sweeps/backup_${NAME}.log" 2>&1 &
echo "[sweep] backup pusher started (pid $!, log: sweeps/backup_${NAME}.log)"

nohup bash -c "cd '$ROOT' && until uv run benchmark.py sweep=true model=$MODEL trials=$TRIALS eval_size=$EVAL name=$NAME; do
    echo \"[sweep] process died (\$(date -u +%T)), resuming in 15s\"; sleep 15
done; echo '[sweep] target reached, done.'" \
    > "$ROOT/sweeps/sweep_${NAME}.log" 2>&1 &
echo "[sweep] sweep started (pid $!). Monitor with: tail -f sweeps/sweep_${NAME}.log"
