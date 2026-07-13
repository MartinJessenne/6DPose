#!/usr/bin/env bash
# initialize_project.sh — 6D Pose asset downloader (auth + resumable dataset)
#
# Foreground (prompts for token if needed):
#   cd /marimo/6DPose && bash initialize_project.sh 2>&1 | tee /marimo/dl.log
#
# Detached (survives a tunnel drop; requires the token to already be seeded):
#   cd /marimo/6DPose
#   setsid nohup bash initialize_project.sh > /marimo/dl.log 2>&1 < /dev/null &
#   tail -f /marimo/dl.log
#
# Seed the token once (persists in /marimo, so once per volume, not per container):
#   read -rsp 'HF token: ' T && echo "HF_TOKEN=$T" >> /marimo/.env && chmod 600 /marimo/.env && unset T
set -uo pipefail

PERSIST=/marimo
PROJECT="$PERSIST/6DPose"
DATASET="UItraviolet/industrial_cart"
MODEL="UItraviolet/yolo_multicart"
DEST="$PROJECT/dataset/data"
BATCH=10

export HOME="$PERSIST"
export XDG_CACHE_HOME="$PERSIST/.cache"
export HF_HOME="$PERSIST/.cache/huggingface"
export UV_CACHE_DIR="$PERSIST/.cache/uv"
export TMPDIR="$PERSIST/tmp"
mkdir -p "$TMPDIR" "$HF_HOME" "$DEST"

# Xet's in-memory reconstruction ("Reconstructing...") was running when the
# download died at ~26.7 GB. Disk was later proven fine (30 GB dd, no crash),
# so the downloader itself is the remaining suspect. Both accelerators off.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$PROJECT" || { echo "FATAL: $PROJECT not found. Clone it first." >&2; exit 1; }

echo "===================================================="
echo "6D Pose Project Initialization & Asset Downloader"
echo "===================================================="

# ---- 1. Hugging Face auth --------------------------------------------------
[ -f "$PERSIST/.env" ] && { set -a; . "$PERSIST/.env"; set +a; }
[ -f .env ]            && { set -a; . ./.env;         set +a; }

if hf auth whoami >/dev/null 2>&1; then
  echo "[HF] Already authenticated (token cached in $HF_HOME)."

elif [ -n "${HF_TOKEN:-}" ]; then
  echo "[HF] Logging in with HF_TOKEN from environment..."
  hf auth login --token "$HF_TOKEN" || { echo "FATAL: token rejected." >&2; exit 1; }

else
  # A detached run has stdin on /dev/null, so `read` would return EOF instantly
  # and we'd "fail" while the user types into a shell that isn't listening.
  # Only prompt when stdin is actually a terminal.
  if [ -t 0 ]; then
    echo ""
    echo "[HF] No cached token found."
    read -rsp "Please paste your Hugging Face write token: " HF_TOKEN_INPUT
    echo ""
    if [ -z "$HF_TOKEN_INPUT" ]; then
      echo "FATAL: this dataset is private — a token is required." >&2
      exit 1
    fi
    hf auth login --token "$HF_TOKEN_INPUT" || { echo "FATAL: token rejected." >&2; exit 1; }
    printf 'HF_TOKEN=%s\n' "$HF_TOKEN_INPUT" >> "$PERSIST/.env"
    chmod 600 "$PERSIST/.env"
    echo "[HF] Token saved to $PERSIST/.env — you won't be asked again."
  else
    echo "FATAL: no token available and stdin is not a terminal." >&2
    echo "" >&2
    echo "  Seed it once, then re-run detached:" >&2
    echo "    read -rsp 'HF token: ' T && echo \"HF_TOKEN=\$T\" >> $PERSIST/.env && chmod 600 $PERSIST/.env && unset T" >&2
    exit 1
  fi
fi

# ---- 2. Black-box recorder -------------------------------------------------
# Writes to the persistent layer, so it survives a container restart and tells
# us what the process looked like at the moment of death.
: > "$PERSIST/dl_monitor.log"
(
  while true; do
    printf '%s rss_kb=%s disk_gb=%s\n' \
      "$(date +%T)" \
      "$(ps -eo rss,comm --sort=-rss | awk '/hf|python/ {s+=$1} END {print s+0}')" \
      "$(df -BG --output=used "$PERSIST" | tail -1 | tr -dc '0-9')" \
      >> "$PERSIST/dl_monitor.log"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

# ---- 3. YOLO weights -------------------------------------------------------
if [ -f best.pt ]; then
  echo "[YOLO] Found locally at 'best.pt'. Skipping."
else
  echo "[YOLO] Downloading best.pt..."
  hf download "$MODEL" runs/segment/train-2/weights/best.pt --local-dir temp_weights \
    || { echo "FATAL: YOLO download failed." >&2; exit 1; }
  mv temp_weights/runs/segment/train-2/weights/best.pt best.pt
  rm -rf temp_weights
  echo "[YOLO] Saved to 'best.pt'."
fi

# ---- 4. Dataset: batched and resumable -------------------------------------
# Each batch is a FRESH `hf` process, so no download state accumulates across
# the full 29 GB, and a crash costs at most $BATCH files — re-running resumes.
mapfile -t FILES < <(python3 - <<PY
from huggingface_hub import HfApi
for f in HfApi().list_repo_files("$DATASET", repo_type="dataset"):
    if f.endswith(".parquet"):
        print(f)
PY
)
total=${#FILES[@]}
if [ "$total" -eq 0 ]; then
  echo "FATAL: listed 0 parquet files — does your token have access to $DATASET?" >&2
  exit 1
fi
echo "[Dataset] $total parquet files in repo."

last=$(( (total + BATCH - 1) / BATCH ))
for ((i=0; i<total; i+=BATCH)); do
  batch=("${FILES[@]:i:BATCH}")
  n=$(( i / BATCH + 1 ))
  echo "== Batch $n/$last (files $((i+1))-$((i+${#batch[@]})) of $total) =="
  hf download "$DATASET" --repo-type dataset \
      --local-dir "$DEST" --max-workers 2 "${batch[@]}" \
    || { echo "!! Batch $n failed. Re-run this script to resume." >&2; exit 1; }
  sync
  echo "   $(du -sh "$DEST" | cut -f1) downloaded, $(df -BG --output=avail "$PERSIST" | tail -1 | tr -dc '0-9') GB free"
done

echo ""
echo "[Dataset] Complete: $(du -sh "$DEST" | cut -f1) in $DEST"

# ---- 5. Python environment -------------------------------------------------
echo ""
echo "[Python] Setting up 3.12 environment..."
uv python install 3.12
uv sync -p 3.12

echo ""
echo "Initialization finished. Run the project with:"
echo "  cd $PROJECT && uv run inspect_pose.py --random 5 --method ransac"