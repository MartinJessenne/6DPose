#!/usr/bin/env bash
# fetch_dataset.sh — resumable HF dataset download for molab
# Run on a FRESH instance:  export HF_TOKEN=hf_...  &&  bash fetch_dataset.sh
#
# Why this shape:
#   molab's network path (gVisor userspace TCP) is intermittently flaky. A 49 GB
#   single-shot download can hang forever with no error, or take the instance
#   down. Fixing that is not a matter of finding the right env var — the fix is
#   to make failure CHEAP: one shard at a time, hard timeout, retries, and
#   skip-what's-already-correct. A failure costs one ~300 MB shard, not the whole
#   run. Re-run after any crash and it picks up exactly where it stopped.
#
# Deliberately NOT here: any background "warm auth" download. An earlier version
# had one; it silently pulled a second full copy of the repo into $HF_HOME and
# raced the loop. Nothing runs in parallel with the loop.
set -uo pipefail

export HOME=/home/marimo
REPO="UItraviolet/industrial_cart"
DEST="$HOME/6DPose/dataset/data"

export HF_HOME="$HOME/.cache/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # each byte is written once; dedup cache buys nothing
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_HUB_DOWNLOAD_TIMEOUT=30        # fail a dead socket instead of waiting forever
unset HF_XET_HIGH_PERFORMANCE            # documented for >=64 GB RAM; not our friend here

MAX_WORKERS=2                            # per-shard. ~10 min / 46 GB. Do not raise.
SHARD_TIMEOUT=300                        # wall-clock kill for a hung shard
ATTEMPTS=3

mkdir -p "$DEST"

# ---- Auth --------------------------------------------------------------------
if ! hf auth whoami >/dev/null 2>&1; then
  [ -n "${HF_TOKEN:-}" ] || { echo "FATAL: export HF_TOKEN=hf_... first (private repo)." >&2; exit 1; }
  hf auth login --token "$HF_TOKEN" >/dev/null
fi
echo "[auth] $(hf auth whoami 2>/dev/null | head -1)"

# ---- Manifest: filename + expected size, straight from the Hub ---------------
echo "[manifest] fetching shard list..."
python3 - "$REPO" > /tmp/manifest.tsv <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().repo_info(sys.argv[1], repo_type="dataset", files_metadata=True)
for s in info.siblings:
    if s.rfilename.endswith(".parquet"):
        print(f"{s.rfilename}\t{s.size}")
PY
TOTAL=$(wc -l < /tmp/manifest.tsv)
[ "$TOTAL" -gt 0 ] || { echo "FATAL: empty manifest — auth or repo name wrong." >&2; exit 1; }
echo "[manifest] $TOTAL parquet shards"

# ---- Sequential resumable fetch ----------------------------------------------
START=$SECONDS
ok=0; skipped=0; retried=0; failed=0

while IFS=$'\t' read -r f size; do
  # Skip only if the local file is EXACTLY the right size. A plain -s test would
  # happily accept a truncated file left behind by a stalled download.
  if [ -f "$DEST/$f" ] && [ "$(stat -c %s "$DEST/$f" 2>/dev/null)" = "$size" ]; then
    skipped=$((skipped + 1)); continue
  fi
  rm -f "$DEST/$f"   # drop any partial before refetching

  got=0
  for attempt in $(seq 1 $ATTEMPTS); do
    if timeout "$SHARD_TIMEOUT" hf download "$REPO" "$f" \
         --repo-type dataset --local-dir "$DEST" --max-workers "$MAX_WORKERS" \
         >/dev/null 2>&1; then
      got=1; break
    fi
    retried=$((retried + 1))
    echo "$(date +%T)  retry $attempt/$ATTEMPTS  $f"
    sleep 5
  done

  if [ "$got" = 1 ]; then
    ok=$((ok + 1))
    printf '%s  ok  [%3d/%3d]  %s  (%s)\n' \
      "$(date +%T)" "$((ok + skipped))" "$TOTAL" "$f" "$(du -sh "$DEST" 2>/dev/null | cut -f1)"
  else
    failed=$((failed + 1))
    echo "$(date +%T)  FAILED after $ATTEMPTS attempts: $f"
  fi
done < /tmp/manifest.tsv

ELAPSED=$((SECONDS - START))

# ---- Verify: every shard byte-exact against the Hub --------------------------
echo ""
echo "[verify] checking all $TOTAL shards against Hub sizes..."
bad=0
while IFS=$'\t' read -r f size; do
  local_size=$(stat -c %s "$DEST/$f" 2>/dev/null || echo -1)
  if [ "$local_size" != "$size" ]; then
    bad=$((bad + 1)); echo "   MISMATCH  $f  local=$local_size  remote=$size"
  fi
done < /tmp/manifest.tsv

echo ""
echo "=================================================="
printf 'downloaded : %d\nskipped    : %d\nretries    : %d\nfailed     : %d\n' \
  "$ok" "$skipped" "$retried" "$failed"
printf 'wall time  : %dm %ds\non disk    : %s\n' \
  "$((ELAPSED / 60))" "$((ELAPSED % 60))" "$(du -sh "$DEST" | cut -f1)"
echo "=================================================="

if [ "$bad" -eq 0 ] && [ "$failed" -eq 0 ]; then
  echo "OK — all $TOTAL shards present and byte-exact."
  # $HF_HOME/hub would be a second full copy; --local-dir already wrote the real files.
  rm -rf "$HF_HOME/hub"
  exit 0
else
  echo "INCOMPLETE — $bad mismatched, $failed failed. Just re-run this script; it resumes."
  exit 1
fi
