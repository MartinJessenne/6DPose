#!/usr/bin/env bash
# initialize_project.sh — 6D Pose session bootstrap for molab (v5)
#
# Run at the start of EVERY molab session (nothing persists here):
#   export HF_TOKEN=hf_...
#   nohup bash ~/6DPose/initialize_project.sh > /tmp/init.log 2>&1 &
#   tail -f /tmp/init.log     # safe to lose — molab's terminal UI crashes; the job won't
#
# Design principle, learned the hard way: molab runs under gVisor with a
# *userspace* TCP stack. High-bandwidth-burst network work takes the whole
# instance down with no error message (the kill happens outside the sandbox,
# so there is nothing in dmesg — don't go looking). The kill threshold is a
# 1-second burst clearing ~300 MB/s.
#
# So every network step here is: RATE-CAPPED (or inherently slow) + HARD
# TIMEOUT + RETRY + RESUMABLE. Failure costs one shard / one wheel, never the
# whole run. Every stage is idempotent — re-run this script after any crash
# and it picks up where it stopped.
#
# v5 UPDATE — the real bottleneck was the TCP receive window, and it's tunable.
# The v4 model said: gVisor caps ONE socket at ~6 MB/s (bandwidth-delay product
# against a small receive window), so you need many sockets for throughput, and
# bursts are many sockets ramping in lockstep. That model was CORRECT — but only
# for the default window. The default tcp_rmem here is "4096 1048576 4194304"
# (1 MB default buffer), and gVisor's netstack HONORS a sysctl write to it.
# Raising the default to 4 MB was measured to lift a SINGLE socket from ~6 MB/s
# to ~227 MiB/s (probe on a 292 MiB train shard, 07/2026).
#
# That inverts the whole risk model:
#   OLD (1 MB window):  sockets slow  -> need many streams; bursts = stacking.
#   NEW (4 MB window):  every socket is a firehose -> ANY unthrottled transfer
#                       can blow past the ~300 MB/s kill threshold ON ITS OWN.
#
# Consequences, encoded below:
#   * The window is raised ONLY for the aria2 bulk-download phase (steps 3-4),
#     where every transfer runs under aria2's aggregate rate cap (verified to
#     hold: 8 segments under a 200M cap averaged 174 MiB/s in the same probe).
#   * With fast sockets we now want FEW streams, not many: DL_STREAMS=2 (2, not
#     1, only so one stalled segment can't stall the shard), and a lower cap
#     (DL_RATE=150M) for margin — a fresh aria2c process warms up its limiter
#     on every one of the 159 shards, and startup transients are bigger when
#     sockets are fast.
#   * The window is RESTORED to the small default before anything unthrottled
#     touches the network again — uv (python install + sync) has NO rate
#     limiter, so it gets the old regime back, where 4 concurrent wheel streams
#     x ~6 MB/s/socket ~= 24 MB/s aggregate is safe by construction.
#   * If the sysctl turns out not to be writable (gVisor config change), the
#     script falls back to the v4 regime automatically: DL_STREAMS=8 slow
#     sockets under the same aggregate cap.
#   * NEVER run an uncapped transfer while the window is raised. An uncapped
#     single-socket probe hit ~238 MB/s — one second of that near the threshold
#     is how the box dies.
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

# HF metadata calls only (the repo_info manifest fetch). The BULK transfers do
# not go through hf_xet at all — they're done with segmented, aggregate-rate-
# capped aria2c (steps 3-4). These xet knobs are kept as guardrails for any
# accidental hf download fallback; they matter MORE in v5: with the raised TCP
# window an unthrottled hf_xet pull would burst far harder than it used to.
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # each byte written once; dedup cache buys nothing
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_HUB_DOWNLOAD_TIMEOUT=30        # fail a dead socket instead of waiting forever
unset HF_XET_HIGH_PERFORMANCE            # documented for >=64 GB RAM; wrong tool here

# uv: molab injects VIRTUAL_ENV=/tmp/uv-venv into the environment. It is not ours,
# and uv will warn and ignore .venv because of it. Unset it and be explicit.
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$PROJECT/.venv"
export UV_CONCURRENT_DOWNLOADS=4   # 4 x ~6 MB/s socket ceiling ~= 24 MB/s — but ONLY
export UV_CONCURRENT_INSTALLS=2    # safe because the TCP window is restored to the
export UV_CONCURRENT_BUILDS=1      # small default before uv ever runs (step 4b).
export UV_HTTP_TIMEOUT=60          # don't hang forever on a dead socket

SHARD_TIMEOUT=300   # wall-clock kill for a hung shard
ATTEMPTS=3          # per-file retry budget (yolo weights + each dataset shard)

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

echo "===================================================="
echo "6D Pose bootstrap (ephemeral) — $(date +%T)"
echo "===================================================="
# NOTE: no `df` check here. Under gVisor `df` reports a fake `none 8.0E` on every
# mount, so the old check ALWAYS passed — right up until the instance died. It was
# worse than useless. Disk is not the constraint anyway: a 60 GB dd staircase
# wrote at ~1.8 GB/s with no ENOSPC and zero memory growth.

# ---- 1. Repo ------------------------------------------------------------------
# (Unthrottled git fetch, but the repo is small and the TCP window is still at
# its slow default here — raising it happens in step 2d, after everything tiny.)
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
# Segmented, aggregate-rate-limited downloads (see the v5 note up top). Not
# preinstalled on a fresh molab image. Done BEFORE raising the TCP window so
# the apt fetch (unthrottled, but tiny) runs against the slow default.
if ! command -v aria2c >/dev/null 2>&1; then
  echo "[deps] installing aria2..."
  export DEBIAN_FRONTEND=noninteractive
  apt update -qq && apt install -y -qq aria2 \
    || { echo "FATAL: could not install aria2." >&2; exit 1; }
fi

# ---- 2d. Raise the TCP receive window for the capped bulk phase ----------------
# gVisor's default tcp_rmem ("4096 1048576 4194304", 1 MB default buffer) caps a
# single socket at ~6 MB/s (bandwidth-delay product). gVisor honors a sysctl
# write here; a 4 MB default lifted one socket to ~227 MiB/s in testing. We
# raise it ONLY for the aria2 phase (everything under an aggregate cap) and
# restore it in step 4b before uv (which has no rate limiter) touches the net.
# Doesn't persist across molab sessions — must run every time.
TCP_RMEM_OLD="$(tr -s '\t' ' ' < /proc/sys/net/ipv4/tcp_rmem)"
if sysctl -qw net.ipv4.tcp_rmem="4096 4194304 16777216" 2>/dev/null; then
  echo "[net] tcp_rmem raised for bulk phase (was: $TCP_RMEM_OLD)"
  RMEM_RAISED=1
else
  echo "[net] tcp_rmem not writable — falling back to many-slow-sockets regime"
  RMEM_RAISED=0
fi

restore_rmem() {
  # Idempotent; also registered on EXIT so a crash/ctrl-C can't leave the box
  # with a raised window and some later unthrottled tool running.
  if [ "${RMEM_RAISED:-0}" = 1 ]; then
    sysctl -qw net.ipv4.tcp_rmem="$TCP_RMEM_OLD" 2>/dev/null \
      && echo "[net] tcp_rmem restored to default"
    RMEM_RAISED=0
  fi
}
trap restore_rmem EXIT

# Bulk-download tuning (aria2c). DL_RATE is the AGGREGATE ceiling across every
# segment of a file combined — it holds regardless of DL_STREAMS (verified:
# 8 segments under 200M averaged 174 MiB/s). DL_STREAMS is how many segments
# aria2 splits each file into. The right values depend on which regime we're in:
#
#   raised window: one socket alone can exceed the cap, so fewer streams =
#     smaller synchronized-ramp transients. 2 streams (not 1: a single stalled
#     segment shouldn't stall the shard). Cap at 150M for margin — each of the
#     159 shards is a fresh aria2c process with fresh limiter state, and
#     warm-up overshoot is bigger when sockets are fast. If the marimo burst
#     tracker still shows shard-boundary spikes near 200+, drop DL_RATE to 100M
#     (46 GB still finishes in <8 min).
#
#   default window: v4 math applies — ~6 MB/s per-socket ceiling, so 8 streams
#     x ~6 ~= 48 MB/s real throughput, aggregate cap mostly a formality.
#
# Both overridable from the environment for sweeps.
if [ "$RMEM_RAISED" = 1 ]; then
  DL_RATE="${DL_RATE:-150M}"
  DL_STREAMS="${DL_STREAMS:-2}"
else
  DL_RATE="${DL_RATE:-200M}"
  DL_STREAMS="${DL_STREAMS:-8}"
fi
echo "[net] bulk config: rate cap $DL_RATE, $DL_STREAMS segment(s)/file"

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
# WHY NOT `hf download`: hf_xet can't be throttled (Rust binary with its own
# HTTP stack: ignores LD_PRELOAD shapers and http(s)_proxy env), and tc/cgroup
# shaping is a no-op under gVisor's userspace netstack (same reason free/df/ss
# lie in here). Under the OLD 1 MB window it burst to 300-350 MB/s as ~16
# range-get sockets ramped in lockstep; under the raised window it would be
# strictly worse. aria2c is the only path with a real aggregate limiter.
#
# WHY THE RAISED WINDOW + FEW STREAMS (v5): with the default window, one socket
# tops out ~6 MB/s (bandwidth-delay product), which is what capped v4 at
# 8 x 6 ~= 48 MB/s regardless of DL_RATE. Raising tcp_rmem lifts a single
# socket to 200+ MB/s (measured), so throughput no longer needs many sockets —
# and fewer sockets means smaller startup transients under the same aggregate
# cap. The cap itself was verified to hold in the fast-socket regime.
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
  exit 1   # trap restores tcp_rmem on the way out
fi
# aria2 wrote the real files directly into $DEST; no $HF_HOME/hub copy is
# created by this path, but clean it anyway in case a fallback ever populated it.
rm -rf "$HF_HOME/hub"

# ---- 4b. Restore the small TCP window BEFORE any unthrottled tool runs ---------
# Everything past this point (uv python install, uv sync) has NO rate limiter.
# With the raised window, uv's 4 concurrent wheel streams could stack toward
# 4 x 200+ MB/s — instant kill. The small default window is itself a ~6 MB/s
# per-socket throttle: restoring it makes uv safe by construction
# (4 x ~6 ~= 24 MB/s aggregate). Torch wheels at 24 MB/s cost a couple of
# minutes; a dead instance costs everything.
restore_rmem

# ---- 5. Python interpreter ----------------------------------------------------
# Without this, uv falls back to the system CPython 3.13 and pyproject's
# `requires-python = "==3.12.*"` rejects it. This is a network fetch too
# (unthrottled — hence it runs AFTER the window restore), so retry.
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
# This is the step that has taken the instance down before. uv has no
# --limit-rate, so DL_RATE does NOT apply here — the levers for uv are
# (a) concurrency, capped to 4 above, and (b) the restored small TCP window
# (step 4b), which caps each stream near ~6 MB/s at the transport layer. Do NOT
# run uv sync while the window is raised. The retry loop makes failures cheap:
# uv's cache ($UV_CACHE_DIR) keeps every wheel already fetched, so each attempt
# resumes rather than restarting. A crash costs one wheel, not the environment.
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
