#!/usr/bin/env bash
# download_dataset.sh
set -uo pipefail

PERSIST=/marimo
PROJECT="$PERSIST/6DPose"
DATASET="UItraviolet/industrial_cart"
DEST="$PROJECT/dataset/data"

export HOME="$PERSIST"
export HF_HOME="$PERSIST/.cache/huggingface"
export TMPDIR="$PERSIST/tmp"
mkdir -p "$TMPDIR" "$DEST"

# The two suspects: Xet reconstruction and hf_transfer's parallel buffers.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$PROJECT"

# --- black box recorder: survives the crash, tells us what died -------------
(
  while true; do
    printf '%s rss_kb=%s disk=%s\n' \
      "$(date +%T)" \
      "$(ps -eo rss,comm --sort=-rss | awk '/hf|python/ {s+=$1} END {print s+0}')" \
      "$(df -BG --output=used "$PERSIST" | tail -1 | tr -dc '0-9')" \
      >> "$PERSIST/dl_monitor.log"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

mapfile -t FILES < <(python3 - <<PY
from huggingface_hub import HfApi
for f in HfApi().list_repo_files("$DATASET", repo_type="dataset"):
    if f.endswith(".parquet"):
        print(f)
PY
)
total=${#FILES[@]}
echo "[Dataset] $total parquet files."

BATCH=10
for ((i=0; i<total; i+=BATCH)); do
  batch=("${FILES[@]:i:BATCH}")
  n=$((i/BATCH + 1))
  echo "== Batch $n ($((i+1))-$((i+${#batch[@]})) of $total) =="
  hf download "$DATASET" --repo-type dataset \
      --local-dir "$DEST" --max-workers 2 "${batch[@]}" || {
        echo "!! batch $n failed; re-run to resume" >&2; exit 1; }
  sync
  echo "   $(du -sh "$DEST" | cut -f1) on disk"
done

echo "[Dataset] Complete: $(du -sh "$DEST" | cut -f1)"
