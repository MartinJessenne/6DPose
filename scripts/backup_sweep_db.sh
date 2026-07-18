#!/usr/bin/env bash
# Periodically snapshots a sweep DB and pushes it to the 'sweep-backups'
# branch on origin. Designed for ephemeral containers: GitHub is the only
# storage that survives a container wipe, and the container already has
# push credentials (it pushes code commits).
#
# Usage: backup_sweep_db.sh <STUDY_NAME> [INTERVAL_SECONDS=600]
#
# The snapshot uses SQLite's backup API (crash-consistent even while Optuna
# is writing), so the pushed file is always a valid database.
set -u

NAME=${1:?usage: backup_sweep_db.sh <STUDY_NAME> [INTERVAL_SECONDS]}
INTERVAL=${2:-600}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/sweeps/optuna_${NAME}.db"
BRANCH="sweep-backups"
BK="/tmp/sweep-backup-repo"
URL="$(git -C "$ROOT" remote get-url origin)"

if [ ! -d "$BK/.git" ]; then
    git clone --depth 1 --branch "$BRANCH" "$URL" "$BK" 2>/dev/null || {
        rm -rf "$BK"
        mkdir -p "$BK"
        git -C "$BK" init -q
        git -C "$BK" remote add origin "$URL"
        git -C "$BK" checkout -q --orphan "$BRANCH"
    }
fi

echo "[backup] pushing '$DB' to origin/$BRANCH every ${INTERVAL}s"
while true; do
    if [ -f "$DB" ]; then
        python3 - "$DB" "$BK/optuna_${NAME}.db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
PY
        git -C "$BK" add "optuna_${NAME}.db"
        if git -C "$BK" commit -qm "backup ${NAME} $(date -u +%FT%TZ)"; then
            git -C "$BK" push -q origin "HEAD:$BRANCH" \
                && echo "[backup] $(date -u +%T) pushed" \
                || echo "[backup] $(date -u +%T) PUSH FAILED (will retry next cycle)"
        fi
    else
        echo "[backup] waiting for $DB to appear"
    fi
    sleep "$INTERVAL"
done
