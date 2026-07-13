#!/usr/bin/env bash
# initialize_project.sh — 6D Pose session bootstrap
# Run at the start of EVERY molab session (nothing big persists here):
#   cd /marimo && bash 6DPose/initialize_project.sh
# ~2-3 minutes total at observed 300 MB/s.
set -euo pipefail

PERSIST=/marimo
PROJECT="$PERSIST/6DPose"
DATASET="UItraviolet/industrial_cart"
MODEL="UItraviolet/yolo_multicart"
DEST="$PROJECT/dataset/data"

export HOME="$PERSIST"
export HF_HOME="$PERSIST/.cache/huggingface"
export UV_CACHE_DIR="$PERSIST/.cache/uv"
export TMPDIR="$PERSIST/tmp"
mkdir -p "$TMPDIR" "$HF_HOME"

# Fast path ON: current hf_xet, parallel workers. (The old script disabled
# these for crash diagnosis; at 300 MB/s the dataset lands in ~100 s.)
pip install -q --upgrade huggingface_hub hf_xet --root-user-action=ignore

echo "===================================================="
echo "6D Pose Session Bootstrap"
echo "===================================================="

# ---- 1. Repo ----------------------------------------------------------------
if [ ! -d "$PROJECT/.git" ]; then
  echo "[Repo] Cloning..."
  git clone https://github.com/MartinJessenne/6DPose.git "$PROJECT"
fi
cd "$PROJECT"

# ---- 2. Auth ----------------------------------------------------------------
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

# ---- 3. YOLO weights ---------------------------------------------------------
if [ ! -f best.pt ]; then
  echo "[YOLO] Downloading best.pt..."
  hf download "$MODEL" runs/segment/train-2/weights/best.pt --local-dir /tmp/w
  mv /tmp/w/runs/segment/train-2/weights/best.pt best.pt
  rm -rf /tmp/w
fi

# ---- 4. Dataset: one shot, full speed ----------------------------------------
if [ -d "$DEST" ] && [ "$(find "$DEST" -name '*.parquet' | wc -l)" -ge 159 ]; then
  echo "[Dataset] Already present ($(du -sh "$DEST" | cut -f1)). Skipping."
else
  echo "[Dataset] Downloading (~29 GB, ~2 min at full speed)..."
  mkdir -p "$DEST"
  time hf download "$DATASET" --repo-type dataset \
      --include "*.parquet" --local-dir "$DEST" --max-workers 16
  echo "[Dataset] $(du -sh "$DEST" | cut -f1) in $DEST"
fi

# ---- 5. Python env ------------------------------------------------------------
uv python install -q 3.12
uv sync -p 3.12

echo ""
echo "Ready:  cd $PROJECT && uv run inspect_pose.py --random 5 --method ransac"
