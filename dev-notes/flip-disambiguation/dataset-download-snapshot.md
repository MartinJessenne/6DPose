# Delete the hand-rolled downloader, use `snapshot_download`

`status: draft`

**The measurement that settles it:** the custom asyncio/httpx downloader was moving
**5.39 MB/s aggregate** on a box with an 806 MB/s link. Pointing `huggingface_hub`'s own
`snapshot_download` (with the `hf_xet` accelerator) at the same dataset got
**25-54 MB/s per file**, immediately, with zero tuning. That's not "our constants were
off" — it's "the whole approach was wrong for this backend." `UItraviolet/industrial_cart`
is Xet-backed, and `hf_xet` is HF's purpose-built client for that storage layer; a plain
HTTP `Range` GET against `/resolve/main/...` was fighting a protocol it wasn't designed
for. Delete the custom downloader, don't tune it further.

---

## The replacement, in full

Everything below replaces the entire current contents of `initialize_project.py`:

```python
# /// script
# dependencies = [
#   "huggingface-hub[hf_xet]",
# ]
# ///

"""
Usage:
    uv run initialize_project.py
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset")


def check_auth():
    if os.getenv("HF_TOKEN"):
        return
    if Path("~/.cache/huggingface/token").expanduser().exists():
        return
    print("Error: HF_TOKEN environment variable or cached token not found.", file=sys.stderr)
    sys.exit(1)


def download_with_retries(attempts=5, backoff_s=5):
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=DATASET,
                repo_type="dataset",
                local_dir=DEST_DIR,
                allow_patterns=["*.parquet"],
                max_workers=8,
            )
            return
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"snapshot_download failed (attempt {attempt}/{attempts}): {e}", file=sys.stderr)
            time.sleep(backoff_s)


if __name__ == "__main__":
    check_auth()
    download_with_retries()
```

That's the whole file — down from ~200 lines to ~55.

## What's gone, and why it's safe

Everything below is deleted outright, not kept "just in case":

- **`AsyncTokenBucket`, `download_segment`, `compute_segments`, `download_file`** — the
  entire custom parallel-segmented-download machinery. `snapshot_download` does file
  discovery, parallel fetch, and resume itself.
- **`httpx`, the manual `socket`/`httpx.Limits` tuning, `tqdm`** — `snapshot_download`
  brings its own transport and its own progress bars (you saw them in the crash log:
  `Fetching 159 files`, `Downloading (incomplete total...)`).
- **`CONCURRENT_FILES` / `SEGMENTS_PER_FILE` / `CHUNK_SIZE` / `MIN_SEGMENT_SIZE`** — all
  were tuning knobs for the deleted machinery. `max_workers=8` is the one knob that
  survives, and it means something different now (parallel *files*, not raw segments).

## The `molab` bandwidth cap doesn't have a home here anymore

`snapshot_download`/`hf_xet` has no bandwidth-throttle parameter — rate-limiting isn't
part of what it's built to do. My recommendation: **drop it**, don't try to rebuild a
token-bucket around `snapshot_download`. If you ever need to cap this specific script's
bandwidth on a shared connection again, do it at the layer rate-limiting actually
belongs to — the OS/network layer, not application code — e.g. `trickle -d 25600 uv run
initialize_project.py` (caps at 25 MB/s) or a `tc`/wondershaper rule on the interface.
That also means the cap would apply to *anything* you run, not just this one script,
which is arguably what you wanted in the first place. Worth revisiting only if you
actually hit the molab case again — no need to build it speculatively now.

## The retry wrapper is justified by a real crash, not speculation

The `RuntimeError: Cannot send a request, as the client has been closed` you hit is a
race in `huggingface_hub`'s own httpx retry path (visible in the traceback:
`_httpx_follow_relative_redirects_with_backoff`), triggered by a transient DNS blip on
the container. It's not something your code controls, and it's not deterministic.
`download_with_retries` exists because this **already happened once, on this exact
call** — that's the bar CLAUDE.md sets for adding error handling: don't guard against
things that can't happen, and this demonstrably can. It's safe to retry blindly because
`local_dir` downloads are idempotent — a file already fully written is skipped, so a
retry only re-fetches whatever was mid-flight when it died.

## One-time cleanup on the current box

The old script's `.tmp` partial files are harmless orphans now (the new code never
touches them), but worth clearing so they don't confuse a later `du -sh dataset/`:

```bash
find dataset -name '*.tmp' -delete
```

## The test to write

There's exactly one piece of this file that's pure logic rather than network I/O:
`check_auth`. Test it directly rather than trying to mock `snapshot_download` (mocking
a whole HF client for a bootstrap script isn't worth the effort it'd take to keep in
sync):

```python
def test_check_auth_exits_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(SystemExit):
        check_auth()

def test_check_auth_passes_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    check_auth()  # should not raise
```

## How to verify

```bash
uv run initialize_project.py
```

**Good:** `snapshot_download`'s own progress bars show per-file rates in the tens of
MB/s (matching what you already saw), and it exits 0 once all 159 shards are present
under `dataset/data/`. **Bad:** if `check_auth` exits immediately, `HF_TOKEN` isn't set
in this shell — re-check the vast.ai template's environment variables.
