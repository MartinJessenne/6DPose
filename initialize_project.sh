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
#
# v4 UPDATE — the bandwidth-burst fix: the instance dies seconds after a
# 1-second download burst clears ~300 MB/s. Average throughput is fine (~75 MB/s);
# the killer is transient spikes to 300–350 MB/s made of many xet range-get
# sockets ramping in lockstep. hf_xet can't be throttled (Rust binary, own HTTP
# stack — ignores LD_PRELOAD shapers like trickle AND, per HF's own tracker,
# http(s)_proxy env vars), and tc/cgroup shaping is a no-op under gVisor. So the
# bulk transfers below bypass hf_xet entirely.
#
# First attempt was a single rate-capped curl per file — it never bursts, but it
# capped effective speed at ~6 MB/s, not the intended ~200. Reason: gVisor's
# userspace TCP keeps a small per-socket receive window, so ONE connection has a
# hard throughput ceiling regardless of --limit-rate (bandwidth-delay product).
# This also retroactively explains the old ~75 MB/s xet average: ~16 concurrent
# range-get sockets x ~6 MB/s/socket ~= 75 MB/s. The bursts were never about
# per-socket excess capacity — they were ~16 sockets each hitting that ceiling in
# the same instant.
#
# Fix: aria2c. It segments one file across N connections (recovers the
# concurrency you need for real throughput) but enforces a single AGGREGATE
# rate limit across all of them, re-checked at sub-second granularity — unlike
# curl, where each stream only knows its own cap. Even if two segments start
# together, the aggregate throttle reins the pair in before it can stack into a
# spike. See the step 4 note for tuning knobs.
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

# HF metadata calls only (the repo_info manifest fetch). The BULK transfers no
# longer go through hf_xet at all — they're done with segmented, aggregate-rate-
# capped aria2c (step 4), which is what reliably prevents the >300 MB/s bursts
# that crash the box. These xet knobs are kept as guardrails for any accidental
# hf download fallback; they do NOT govern the dataset/weights transfers anymore.
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

# Bulk-download tuning (aria2c). DL_RATE is the AGGREGATE ceiling across every
# segment of a file combined — this is what stops the >300 MB/s burst, and it
# holds regardless of DL_STREAMS. DL_STREAMS is how many segments aria2 splits
# each file into; since one gVisor socket tops out near ~6 MB/s, you need several
# to actually reach DL_RATE (8 streams x ~6 MB/s/socket ceiling ~= plenty of
# headroom under a 200 MB/s aggregate cap). Both are overridable so you can sweep
# them (start DL_STREAMS low, raise it) against the marimo burst tracker to find
# the max concurrency that still never shows a burst above threshold.
DL_RATE="${DL_RATE:-200M}"
DL_STREAMS="${DL_STREAMS:-8}"

SHARD_TIMEOUT=300   # wall-clock kill for a hung shard
ATTEMPTS=3          # per-file retry budget (yolo weights + each dataset shard)

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

echo "===================================================="
echo "6D Pose bootstrap (ephemeral) — $(date +%T)"
echo "  bulk download rate cap: $DL_RATE"
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

# ---- 2b. Recover the raw token for aria2 --------------------------------------
# The bulk downloads (steps 3 & 4) use aria2c, which needs the token in an
# Authorization header. `hf auth login` above wrote it to $HF_HOME/token; prefer
# the env var if we still have it, else read it back from there. This bearer is
# only needed for the initial huggingface.co request; the /resolve/ redirect to
# the CDN is presigned, and aria2 correctly drops the header on the cross-host hop.
DL_TOKEN="${HF_TOKEN:-}"
[ -n "$DL_TOKEN" ] || DL_TOKEN="$(tr -d '\r\n' < "$HF_HOME/token" 2>/dev/null || true)"
[ -n "$DL_TOKEN" ] || { echo "FATAL: no HF token available for downloads." >&2; exit 1; }

# ---- 2c. aria2 ------------------------------------------------------------
# Segmented, aggregate-rate-limited downloads (see the v4 note up top). Not
# preinstalled on a fresh molab image.
if ! command -v aria2c >/dev/null 2>&1; then
  echo "[deps] installing aria2..."
  export DEBIAN_FRONTEND=noninteractive
  apt update -qq && apt install -y -qq aria2 \
    || { echo "FATAL: could not install aria2." >&2; exit 1; }
fi

# ---- 3. YOLO weights (aria2, same reasoning as the dataset in step 4) --------
if [ -s "$PROJECT/best.pt" ]; then
  echo "[yolo] best.pt already present."
else
  echo "[yolo] downloading best.pt..."
  YOLO_PATH="runs/segment/train-2/weights/best.pt"
  for attempt in $(seq 1 $ATTEMPTS); do
    if timeout 300 aria2c \
         --header="Authorization: Bearer $DL_TOKEN" \
         -x "$DL_STREAMS" -s "$DL_STREAMS" -k 1M \
         --max-overall-download-limit="$DL_RATE" \
         --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
         -d "$PROJECT" -o "best.pt.part" \
         "https://huggingface.co/$MODEL/resolve/main/$YOLO_PATH" \
         >/dev/null 2>&1 \
       && mv "$PROJECT/best.pt.part" "$PROJECT/best.pt" \
       && [ -s "$PROJECT/best.pt" ]; then
      echo "[yolo] ok."
      break
    fi
    echo "[yolo] retry $attempt/$ATTEMPTS"
    rm -f "$PROJECT/best.pt.part" "$PROJECT/best.pt.part.aria2" "$PROJECT/best.pt"
    sleep 5
  done
  [ -s "$PROJECT/best.pt" ] || { echo "FATAL: could not fetch best.pt." >&2; exit 1; }
fi

# ---- 4. Dataset: one shard at a time, aria2 segmented+capped, verified, resumable
# WHY NOT `hf download`: the crash is bandwidth bursts, not average throughput.
# net_io_counters during a run showed the average ~75 MB/s but 1-second bursts
# spiking to 300–350 MB/s — and the instance dies seconds after a burst clears
# 300. Those bursts are N xet range-get sockets all ramping in lockstep and
# stacking their startup transients at the interface. hf_xet can't be throttled
# (Rust binary with its own HTTP stack: ignores LD_PRELOAD shapers and
# http(s)_proxy env), and tc/cgroup shaping is a no-op under gVisor's userspace
# netstack (same reason free/df/ss lie in here).
#
# WHY NOT PLAIN CURL EITHER: a single curl --limit-rate stream never bursts, but
# it never exceeded ~6 MB/s regardless of the cap — gVisor's per-socket receive
# window is small, so ONE connection has a hard throughput ceiling independent
# of any rate limit (bandwidth-delay product). That also explains the old
# hf_xet average: ~16 sockets x ~6 MB/s/socket ~= ~75 MB/s. The bursts were never
# excess per-socket capacity — they were many sockets hitting that ceiling at
# once.
#
# THE FIX: aria2c splits each shard into $DL_STREAMS segments (recovers real
# throughput — several sockets, each near its ~6 MB/s ceiling) while enforcing
# ONE aggregate rate limit ($DL_RATE) across all of them together, re-checked
# well under a second. Even if two segments ramp up together, the aggregate
# throttle reins the pair in before it can stack into a burst — this is the
# piece plain curl couldn't do (each stream there only knew its own cap).
#
# The /resolve/ endpoint serves the materialized file over plain HTTPS for Xet
# repos too (same path HF_HUB_DISABLE_XET forces internally); the byte-exact
# check below is the backstop against any endpoint quirk that truncates.
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
  rm -f "$DEST/$f" "$DEST/$f.aria2"
  # aria2 (like curl) won't create parent dirs the way hf download did; shard
  # paths include a leading subdir (e.g. data/train-00000-...parquet).
  mkdir -p "$(dirname "$DEST/$f")"

  got=0
  for attempt in $(seq 1 $ATTEMPTS); do
    if timeout "$SHARD_TIMEOUT" aria2c \
         --header="Authorization: Bearer $DL_TOKEN" \
         -x "$DL_STREAMS" -s "$DL_STREAMS" -k 1M \
         --max-overall-download-limit="$DL_RATE" \
         --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
         -d "$(dirname "$DEST/$f")" -o "$(basename "$f")" \
         "https://huggingface.co/datasets/$DATASET/resolve/main/$f?download=true" \
         >/dev/null 2>&1; then
      got=1; break
    fi
    retried=$((retried + 1))
    echo "$(date +%T)  [dataset] retry $attempt/$ATTEMPTS  $f"
    rm -f "$DEST/$f" "$DEST/$f.aria2"   # drop the partial a timed-out/hung fetch left behind
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
# aria2 wrote the real files directly into $DEST; no $HF_HOME/hub copy is
# created by this path, but clean it anyway in case a fallback ever populated it.
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
# (torch et al), on a fragile userspace TCP stack. NOTE: uv has no --limit-rate,
# so the DL_RATE cap above does NOT apply here — the lever for uv is concurrency.
# Many parallel wheel streams are exactly the synchronized-ramp burst pattern
# that crashes the dataset pull, so if a *crash* (not a plain uv error) happens
# here, drop UV_CONCURRENT_DOWNLOADS to 1 (see the FATAL block). Concurrency is
# capped to 4 above; the retry loop makes it safe: uv's cache ($UV_CACHE_DIR)
# keeps every wheel already fetched, so each attempt resumes rather than
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
