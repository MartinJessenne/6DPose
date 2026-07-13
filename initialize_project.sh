#!/usr/bin/env bash
# initialize_project.sh — 6D Pose session bootstrap
# Run at the start of EVERY molab session (nothing big persists here):
#   cd /marimo && bash 6DPose/initialize_project.sh
#
# v2 changes (the "uv sync crashes the instance" fix):
#   - uv concurrency capped: downloads 8 (default is 50!), installs/builds 2.
#   - Everything uv touches hot — cache, tmp, and the venv itself — moved to
#     the EPHEMERAL local disk. /marimo is FUSE/network-backed: no hardlinks
#     (uv falls back to copying GBs) and buffered writes that can OOM the
#     instance mid-sync. Only the repo + dataset stay on /marimo.
#   - The venv location is exported via UV_PROJECT_ENVIRONMENT and appended to
#     ~/.bashrc_persist so `uv run` finds it in every future terminal.
#   - free/df snapshots before the sync so a crash leaves evidence.
set -euo pipefail

PERSIST=/marimo
LOCAL=/home/marimo                      # ephemeral local disk, fast, dies on reboot
PROJECT="$PERSIST/6DPose"
DATASET="UItraviolet/industrial_cart"
MODEL="UItraviolet/yolo_multicart"
DEST="$PROJECT/dataset/data"

export HOME="$PERSIST"

# ---- Placement policy ---------------------------------------------------------
# PERSISTENT (/marimo): repo, dataset, HF cache (xet chunks are resumable and
#   worth keeping across reboots), tunnel auth.
# EPHEMERAL ($LOCAL):   uv cache, uv tmp, the venv. Rebuilt each boot; with
#   capped concurrency a cold sync is a few minutes, and it can't corrupt or
#   OOM the way FUSE-backed installs do.
export HF_HOME="$PERSIST/.cache/huggingface"

export UV_CACHE_DIR="$LOCAL/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$LOCAL/venvs/6DPose"
export UV_PYTHON_INSTALL_DIR="$LOCAL/uv-pythons"
export UV_CONCURRENT_DOWNLOADS=8        # default 50 — the actual "massive concurrent requests"
export UV_CONCURRENT_INSTALLS=2
export UV_CONCURRENT_BUILDS=2
export UV_LINK_MODE=hardlink            # cache and venv now share the local disk, so this works

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$LOCAL/tmp" "$LOCAL/venvs"

# Make future VS Code terminals see the same uv layout (your HOME-redirect
# bashrc sources this file).
RC="$PERSIST/.bashrc_persist"
UVMARK="# >>> 6dpose-uv-env >>>"
if ! grep -qF "$UVMARK" "$RC" 2>/dev/null; then
  cat >> "$RC" <<EOF

$UVMARK
export UV_CACHE_DIR=$LOCAL/.cache/uv
export UV_PROJECT_ENVIRONMENT=$LOCAL/venvs/6DPose
export UV_PYTHON_INSTALL_DIR=$LOCAL/uv-pythons
export UV_CONCURRENT_DOWNLOADS=8
export UV_CONCURRENT_INSTALLS=2
export UV_CONCURRENT_BUILDS=2
export UV_LINK_MODE=hardlink
# <<< 6dpose-uv-env <
EOF
fi

pip install -q --upgrade huggingface_hub hf_xet --root-user-action=ignore

echo "===================================================="
echo "6D Pose Session Bootstrap"
echo "===================================================="

# ---- 1. Repo -------------------------------------------------------------------
if [ ! -d "$PROJECT/.git" ]; then
  echo "[Repo] Cloning..."
  git clone https://github.com/MartinJessenne/6DPose.git "$PROJECT"
fi
cd "$PROJECT"

# ---- 2. Auth --------------------------------------------------------------------
# Token comes from the environment ONLY. Never commit it, never paste it in
# chats/logs. Set it once per session:   export HF_TOKEN=hf_...
if hf auth whoami >/dev/null 2>&1; then
  echo "[HF] Authenticated."
elif [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN"
elif [ -t 0 ]; then
  read -rsp "Hugging Face token: " T; echo ""
  [ -n "$T" ] || { echo "FATAL: token required (private dataset)." >&2; exit 1; }
  hf auth login --token "$T"
else
  echo "FATAL: no token. Run: export HF_TOKEN=hf_...  then re-run." >&2
  exit 1
fi

# ---- 3. YOLO weights --------------------------------------------------------------
if [ ! -f best.pt ]; then
  echo "[YOLO] Downloading best.pt..."
  hf download "$MODEL" runs/segment/train-2/weights/best.pt --local-dir /tmp/w
  mv /tmp/w/runs/segment/train-2/weights/best.pt best.pt
  rm -rf /tmp/w
fi

# ---- 4. Dataset: one shot, full speed ----------------------------------------------
# HF downloads keep TMPDIR on the persistent volume: 29 GB of staging must NOT
# hit the small local disk.
if [ -d "$DEST" ] && [ "$(find "$DEST" -name '*.parquet' | wc -l)" -ge 159 ]; then
  echo "[Dataset] Already present ($(du -sh "$DEST" | cut -f1)). Skipping."
else
  echo "[Dataset] Downloading (~29 GB, ~2 min at full speed)..."
  mkdir -p "$DEST" "$PERSIST/tmp"
  time TMPDIR="$PERSIST/tmp" hf download "$DATASET" --repo-type dataset \
      --include "*.parquet" --local-dir "$DEST" --max-workers 16
  echo "[Dataset] $(du -sh "$DEST" | cut -f1) in $DEST"
fi

# ---- 5. Python env: local disk, capped concurrency ---------------------------------
# uv's tmp goes to LOCAL disk (wheel extraction is the memory/IO hot path).
echo "[Env] Pre-sync resources:"
free -h | sed 's/^/   /'
df -h "$LOCAL" "$PERSIST" | sed 's/^/   /'

export TMPDIR="$LOCAL/tmp"
uv python install -q 3.12
uv sync -p 3.12

echo ""
echo "[Env] venv at $UV_PROJECT_ENVIRONMENT (local disk — rebuilt each boot)"
echo "Ready:  cd $PROJECT && uv run inspect_pose.py --random 5 --method ransac"
