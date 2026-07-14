#!/usr/bin/env bash
# initialize_project.sh — 6D Pose session bootstrap for molab (v4)
#
# Run at the start of EVERY molab session (nothing persists here):
#   export HF_TOKEN=hf_...
#   nohup bash ~/6DPose/initialize_project.sh > /tmp/init.log 2>&1 &
#   tail -f /tmp/init.log     # safe to lose — molab's terminal UI crashes; the job won't
#
# Design principle, learned the hard way: molab runs under gVisor with a
# *userspace* TCP stack that is intermittently flaky. High-concurrency network
# work either hangs a socket with no timeout, or takes the whole instance down
# with no error message (the kill happens outside the sandbox, so there is
# nothing in dmesg — don't go looking).
#
# So every network step here is: LOW CONCURRENCY + HARD TIMEOUT + RETRY +
# RESUMABLE. Failure costs one shard / one wheel, never the whole run. Every
# stage is idempotent — re-run this script after any crash and it picks up where
# it stopped.
set -uo pipefail

export HOME=/home/marimo
PROJECT="$HOME/6DPose"
REPO_URL="https://github.com/MartinJessenne/6DPose.git"
DATASET="UItraviolet/industrial_cart"
MODEL="UItraviolet/yolo_multicart"
DEST="$PROJECT/dataset/data"
PY_VERSION="3.12"

# ---- Environment --------------------------------------------------------------
export HF_HOME="$HOME/.cache/huggingface"
export UV_CACHE_DIR="$HOME/.cache/uv"
export TMPDIR="$HOME/tmp"

# HF: keep the network calm.
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # each byte written once; dedup cache buys nothing
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_HUB_DOWNLOAD_TIMEOUT=30        # fail a dead socket instead of waiting forever
unset HF_XET_HIGH_PERFORMANCE            # documented for >=64 GB RAM; wrong tool here

# uv: molab injects VIRTUAL_ENV=/tmp/uv-venv into the environment. It is not ours,
# and uv will warn and ignore .venv because of it. Unset it and be explicit.
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$PROJECT/.venv"
export UV_CONCURRENT_DOWNLOADS=4   # default is 50. torch wheels are GB-sized; 50 kills the box.
export UV_CONCURRENT_INSTALLS=2
export UV_CONCURRENT_BUILDS=1
export UV_HTTP_TIMEOUT=60          # don't hang forever on a dead socket

MAX_WORKERS=2       # per-shard HF download concurrency. ~15 min / 46 GB. Do not raise.
SHARD_TIMEOUT=300
ATTEMPTS=3

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

echo "===================================================="
echo "6D Pose bootstrap (ephemeral) — $(date +%T)"
echo "===================================================="
# NOTE: no `df` check here. Under gVisor `df` reports a fake `none 8.0E` on every
# mount, so the old check ALWAYS passed — right up until the instance died. It was
# worse than useless. Disk is not the constraint anyway: a 60 GB dd staircase
# wrote at ~1.8 GB/s with no ENOSPC and zero memory growth.

# ---- 1. Repo ------------------------------------------------------------------
if [ ! -d "$PROJECT/.git" ]; then
  echo "[repo] cloning..."
  git clone "$REPO_URL" "$PROJECT" || { echo "FATAL: clone failed." >&2; exit 1; }
else
  echo "[repo] already present."
fi
cd "$PROJECT" || exit 1

# ---- 2. HF auth ---------------------------------------------------------------
# Token from the environment ONLY. Never commit it, never paste it into a log.
if hf auth whoami >/dev/null 2>&1; then
  echo "[auth] already authenticated."
elif [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" >/dev/null && echo "[auth] logged in."
elif [ -t 0 ]; then
  read -rsp "Hugging Face token: " T; echo ""
  [ -n "$T" ] || { echo "FATAL: token required (private repo)." >&2; exit 1; }
  hf auth login --token "$T" >/dev/null
else
  echo "FATAL: no token. Run: export HF_TOKEN=hf_...  then re-run." >&2
  exit 1
fi

# ---- 3. YOLO weights ----------------------------------------------------------
if [ -s "$PROJECT/best.pt" ]; then
  echo "[yolo] best.pt already present."
else
  echo "[yolo] downloading best.pt..."
  for attempt in $(seq 1 $ATTEMPTS); do
    if timeout 300 hf download "$MODEL" runs/segment/train-2/weights/best.pt \
         --local-dir "$TMPDIR/w" >/dev/null 2>&1; then
      mv "$TMPDIR/w/runs/segment/train-2/weights/best.pt" "$PROJECT/best.pt"
      rm -rf "$TMPDIR/w"
      echo "[yolo] ok."
      break
    fi
    echo "[yolo] retry $attempt/$ATTEMPTS"
    sleep 5
  done
  [ -s "$PROJECT/best.pt" ] || { echo "FATAL: could not fetch best.pt." >&2; exit 1; }
fi

# ---- 4. Dataset: one shard at a time, verified by size, resumable -------------
echo "[dataset] fetching manifest..."
mkdir -p "$DEST"
python3 - "$DATASET" > "$TMPDIR/manifest.tsv" <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().repo_info(sys.argv[1], repo_type="dataset", files_metadata=True)
for s in info.siblings:
    if s.rfilename.endswith(".parquet"):
        print(f"{s.rfilename}\t{s.size}")
PY
TOTAL=$(wc -l < "$TMPDIR/manifest.tsv")
[ "$TOTAL" -gt 0 ] || { echo "FATAL: empty manifest — auth or repo name wrong." >&2; exit 1; }
echo "[dataset] $TOTAL parquet shards (~46 GB)"

START=$SECONDS
ok=0; skipped=0; retried=0; failed=0

while IFS=$'\t' read -r f size; do
  # Skip ONLY on an exact size match. A plain -s test would accept a truncated
  # file left behind by a stalled download — and stalls are the norm here.
  if [ -f "$DEST/$f" ] && [ "$(stat -c %s "$DEST/$f" 2>/dev/null)" = "$size" ]; then
    skipped=$((skipped + 1)); continue
  fi
  rm -f "$DEST/$f"

  got=0
  for attempt in $(seq 1 $ATTEMPTS); do
    if timeout "$SHARD_TIMEOUT" hf download "$DATASET" "$f" \
         --repo-type dataset --local-dir "$DEST" --max-workers "$MAX_WORKERS" \
         >/dev/null 2>&1; then
      got=1; break
    fi
    retried=$((retried + 1))
    echo "$(date +%T)  [dataset] retry $attempt/$ATTEMPTS  $f"
    sleep 5
  done

  if [ "$got" = 1 ]; then
    ok=$((ok + 1))
    printf '%s  [dataset] %3d/%3d  %s\n' "$(date +%T)" "$((ok + skipped))" "$TOTAL" "$f"
  else
    failed=$((failed + 1))
    echo "$(date +%T)  [dataset] FAILED: $f"
  fi
done < "$TMPDIR/manifest.tsv"

# Verify every shard byte-exact against the Hub.
bad=0
while IFS=$'\t' read -r f size; do
  [ "$(stat -c %s "$DEST/$f" 2>/dev/null || echo -1)" = "$size" ] || {
    bad=$((bad + 1)); echo "   MISMATCH  $f"; }
done < "$TMPDIR/manifest.tsv"

printf '[dataset] downloaded=%d skipped=%d retries=%d failed=%d bad=%d  (%dm %ds, %s)\n' \
  "$ok" "$skipped" "$retried" "$failed" "$bad" \
  "$(((SECONDS - START) / 60))" "$(((SECONDS - START) % 60))" "$(du -sh "$DEST" | cut -f1)"

if [ "$bad" -ne 0 ] || [ "$failed" -ne 0 ]; then
  echo "FATAL: dataset incomplete. Re-run this script — it resumes." >&2
  exit 1
fi
# --local-dir already wrote the real files; $HF_HOME/hub would be a 2nd full copy.
rm -rf "$HF_HOME/hub"

# ---- 5. Python interpreter ----------------------------------------------------
# Without this, uv falls back to the system CPython 3.13 and pyproject's
# `requires-python = "==3.12.*"` rejects it. This is a network fetch too, so retry.
if uv python find "$PY_VERSION" >/dev/null 2>&1; then
  echo "[python] $PY_VERSION already installed."
else
  echo "[python] installing CPython $PY_VERSION..."
  for attempt in $(seq 1 $ATTEMPTS); do
    timeout 300 uv python install "$PY_VERSION" && break
    echo "[python] retry $attempt/$ATTEMPTS"
    uv python install "$PY_VERSION" --reinstall >/dev/null 2>&1 || true  # clear a corrupt partial
    sleep 5
  done
  uv python find "$PY_VERSION" >/dev/null 2>&1 || {
    echo "FATAL: could not install Python $PY_VERSION." >&2; exit 1; }
fi

# ---- 6. uv sync: throttled and retried ---------------------------------------
# This is the step that has taken the instance down before. Cause is the same as
# everything else: uv's default 50 concurrent downloads, against multi-GB wheels
# (torch et al), on a fragile userspace TCP stack. Concurrency is capped to 2
# above. The retry loop is what makes it safe: uv's cache ($UV_CACHE_DIR) keeps
# every wheel it has already fetched, so each attempt resumes rather than
# restarting. A crash costs one wheel, not the environment.
echo "[env] uv sync (downloads=$UV_CONCURRENT_DOWNLOADS, installs=$UV_CONCURRENT_INSTALLS)..."
synced=0
for attempt in $(seq 1 5); do
  if uv sync -p "$PY_VERSION"; then
    synced=1; break
  fi
  echo "[env] uv sync attempt $attempt failed — retrying (cache is preserved, so this resumes)"
  sleep 10
done

if [ "$synced" -ne 1 ]; then
  cat >&2 <<'EOF'
FATAL: uv sync failed after 5 attempts.

If the INSTANCE died rather than uv erroring, drop concurrency to the floor and
re-run — the cache means you keep all progress:
    export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
    uv sync -p 3.12
EOF
  exit 1
fi

echo ""
echo "===================================================="
echo "Ready.  cd $PROJECT && uv run inspect_pose.py --random 5 --method ransac"
echo "===================================================="