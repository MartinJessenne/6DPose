#!/usr/bin/env bash
# initialize_project.sh — 6D Pose session bootstrap (fully ephemeral)
# Run at the start of EVERY molab session:
#   cd /home/marimo && bash 6DPose/initialize_project.sh   (or curl | bash)
#
# v3: nothing persists on molab, so everything is fetched fresh each session
# into /home/marimo. Repo clone + best.pt + 29 GB dataset + uv sync.
# uv concurrency stays capped (default is 50 concurrent downloads) and its
# tmp/cache/venv are all local anyway now — no FUSE, no hardlink fallback.
set -euo pipefail

export HOME=/home/marimo
PROJECT="$HOME/6DPose"
DATASET="UItraviolet/industrial_cart"
MODEL="UItraviolet/yolo_multicart"
DEST="$PROJECT/dataset/data"

export HF_HOME="$HOME/.cache/huggingface"
export UV_CACHE_DIR="$HOME/.cache/uv"
export UV_CONCURRENT_DOWNLOADS=8        # default 50 — keep it civil
export UV_CONCURRENT_INSTALLS=2
export UV_CONCURRENT_BUILDS=2
export TMPDIR="$HOME/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd "$HOME"

pip install -q --upgrade huggingface_hub hf_xet --root-user-action=ignore

echo "===================================================="
echo "6D Pose Session Bootstrap (ephemeral)"
echo "===================================================="

# ---- 0. Disk sanity: dataset (29 GB) + venv (~6 GB) + caches must all fit ----
echo "[Disk]"
df -h "$HOME" /tmp | sed 's/^/   /'
avail_kb=$(df -k --output=avail "$HOME" | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt $((40 * 1024 * 1024)) ]; then   # < 40 GiB
  echo "   WARNING: less than 40 GiB free on $HOME — dataset + env may not fit."
fi

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

# ---- 4. Dataset -----------------------------------------------------------------
if [ -d "$DEST" ] && [ "$(find "$DEST" -name '*.parquet' | wc -l)" -ge 159 ]; then
  echo "[Dataset] Already present ($(du -sh "$DEST" | cut -f1)). Skipping."
else
  echo "[Dataset] Downloading (~29 GB)..."
  mkdir -p "$DEST"
  time hf download "$DATASET" --repo-type dataset \
      --include "*.parquet" --local-dir "$DEST" --max-workers 16
  echo "[Dataset] $(du -sh "$DEST" | cut -f1) in $DEST"
fi

# ---- 5. Python env ----------------------------------------------------------------
echo "[Env] Pre-sync resources:"
free -h | sed 's/^/   /'
df -h "$HOME" | sed 's/^/   /'

uv python install -q 3.12
uv sync -p 3.12

echo ""
echo "Ready:  cd $PROJECT && uv run inspect_pose.py --random 5 --method ransac"
