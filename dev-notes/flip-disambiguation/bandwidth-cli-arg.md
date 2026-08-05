# `initialize_project.py` — optional bandwidth cap via a tyro arg

`status: draft`

**What changes.** Today `initialize_project.py` always caps the aggregate download at
`MAX_BANDWIDTH = 200 * 1024 * 1024` (200 MB/s) — the token-bucket limiter is built at
module import time (`initialize_project.py:71`) and every segment download consults it
unconditionally. You want that cap to be opt-in: full-speed by default, capped only when
you pass a flag for the shared connection ("molab"). You asked for `tyro` specifically
instead of `argparse` — right call, it's what every other entry point in this repo
(`benchmark.py`, `inspect_pose.py`, `cli_config.py`) already standardises on, and this
script currently has zero CLI surface, so there's no argparse code being replaced —
just a new dataclass in the house style.

---

## 1. Where the cap actually lives

Two things happen today that need separating:

1. `rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)` (`initialize_project.py:71`) — a
   module-level global, built once, unconditionally, before `main()` even runs.
2. `await rate_limiter.consume(len(chunk))` (`initialize_project.py:94`) — every
   `download_segment` call reaches out to that same global to throttle itself.

To make the cap optional you cannot just branch on a flag *inside* `AsyncTokenBucket` —
the cleanest state to be in is "no limiter object exists", not "a limiter object exists
but is configured to do nothing". Building it unconditionally and then trying to make it
a no-op forces you to invent a "rate = infinity" sentinel and prove the token bucket
math still behaves at that limit. Skip that: make `rate_limiter` an `AsyncTokenBucket |
None`, and only call `.consume()` when it isn't `None`.

That in turn means `rate_limiter` can no longer be a name mutated at module scope with
`global` inside `main()` — CLAUDE.md's stance on implicit state applies here too: a
`global rate_limiter` reassignment means `download_segment` reads a name whose value
depends on what ran earlier in the same process, invisible from its own signature. Pass
it as a parameter instead, all the way down: `main()` → `download_file()` →
`download_segment()`. Explicit data flow, and `None` is a real, first-class value the
type checker knows about (`AsyncTokenBucket | None`), not a magic sentinel.

---

## 2. The tyro API you need here

You've used `tyro.cli()` already via `cli_config.py`, but this script's needs are the
simplest possible case, so it's worth isolating exactly what's required:

- **A `tyro.cli(SomeDataclass)` call parses `sys.argv` into an instance of
  `SomeDataclass`.** `inspect_pose.py:309` (`args = tyro.cli(InspectArgs)`) is the
  closest precedent in this repo — a single flat dataclass, no subcommand `Union` like
  `benchmark.py`'s `Command` needs. You don't need subcommands here; you have exactly
  one dataclass with one field.
- **Every field becomes a `--flag-name value` CLI argument**, named after the field with
  underscores turned into hyphens. A field typed `Literal["default", "molab"]` becomes a
  CLI argument that tyro validates against those exact strings up front — passing
  anything else is a usage error before your code runs at all, which is why
  `cli_config.py:550` uses the same pattern for `split: Literal["all", "test", ...]`
  instead of a bare `str`.
- **The trap:** tyro does *not* read a trailing `# comment` as help text, and this repo
  doesn't put per-field docstrings under each field either — check `cli_config.py:592`
  (`mode: Literal[...]  # tyro validates this choice up front...`), the comment is there
  for the *human reader of the source*, not for `--help`. If you want the choice
  self-documenting on the CLI, the `Literal` values themselves have to be the whole
  explanation (`"default"` / `"molab"` already are). Don't spend effort on a docstring
  tyro won't surface — match the existing convention, a short trailing comment.
- **`tyro.cli(cls, args=[...])` accepts an explicit argument list**, bypassing
  `sys.argv` entirely. That's the hook you need for a test (§5) — same idea as trick #9
  in `SHIP.md`: `study.ask()` samples for real, `FixedTrial` returns exactly what you
  hand it; here, no `args=` reads real `sys.argv`, `args=["--profile", "molab"]` lets a
  test assert dispatch without a subprocess.

---

## 3. The edits

**a. Declare the dependency.** `uv run` builds the script's throwaway venv from the
`# /// script` header at the top of the file — `tyro` has to be listed there, exactly
like `httpx`, `huggingface-hub`, and `tqdm` already are, or the `import tyro` on the
next run fails inside a venv that was never told to install it:

```python
# /// script
# dependencies = [
#   "httpx",
#   "huggingface-hub",
#   "tqdm",
#   "tyro",
# ]
# ///
```

**b. Add the dataclass**, near the other imports (`initialize_project.py:9-18`):

```python
from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass(frozen=True)
class Args:
    profile: Literal["default", "molab"] = "default"
    # "molab": cap aggregate bandwidth at MAX_BANDWIDTH for the shared connection.
    # "default": no cap, download at full link speed.
```

`frozen=True` matches every other CLI dataclass in this repo (`cli_config.py`'s
`CommonArgs`, `EvalArgs`, `SweepArgs` are all frozen) — there's no reason parsed CLI
args should be mutable once constructed, and freezing them means nothing downstream can
silently rebind `args.profile` mid-run.

**c. Delete the module-level limiter** (`initialize_project.py:71`) —
`rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)` goes away completely. It's built inside
`main()` now, conditionally.

**d. Thread `rate_limiter` through both download functions:**

```python
async def download_segment(client, url, start, end, fd, progress_bar, rate_limiter):
    """Download one byte range and pwrite() it at its offset. Resumable retries."""
    for attempt in range(5):
        offset = start
        segment_headers = {**AUTH_HEADERS, "Range": f"bytes={offset}-{end}"}
        try:
            async with client.stream(
                "GET",
                url,
                headers=segment_headers,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=60.0),
            ) as r:
                if r.status_code != 206:
                    raise httpx.HTTPStatusError(
                        f"Expected 206 Partial Content, got {r.status_code}",
                        request=r.request,
                        response=r,
                    )
                async for chunk in r.aiter_bytes(chunk_size=CHUNK_SIZE):
                    if rate_limiter is not None:
                        await rate_limiter.consume(len(chunk))
                    os.pwrite(fd, chunk, offset)
                    offset += len(chunk)
                    progress_bar.update(len(chunk))
            return
        except Exception:
            if attempt == 4:
                raise
            start = offset
            await asyncio.sleep(2**attempt)
```

```python
async def download_file(client, filename, size, progress_bar, sem, rate_limiter):
    """Download a single file (segmented), gated by the concurrency semaphore."""
    dest_path = DEST_DIR / filename

    if dest_path.exists() and dest_path.stat().st_size == size:
        progress_bar.update(size)
        return

    async with sem:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(".tmp")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"

        fd = os.open(temp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, size)
            await asyncio.gather(
                *[
                    download_segment(client, url, s, e, fd, progress_bar, rate_limiter)
                    for s, e in compute_segments(size)
                ]
            )
            os.fsync(fd)
        finally:
            os.close(fd)

        temp_path.rename(dest_path)
```

**e. Parse the args and build the limiter at the top of `main()`:**

```python
async def main():
    args = tyro.cli(Args)
    capped = args.profile == "molab"
    rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH) if capped else None

    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]

    total_size = sum(s.size for s in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")
    cap_msg = f"capped at {MAX_BANDWIDTH / 1024**2:.0f} MB/s aggregate" if capped else "uncapped"
    print(f"{CONCURRENT_FILES} files in flight x {SEGMENTS_PER_FILE} segments, {cap_msg}")

    max_conns = CONCURRENT_FILES * SEGMENTS_PER_FILE
    limits = httpx.Limits(
        max_keepalive_connections=max_conns + 2,
        max_connections=max_conns * 2,
    )
    custom_socket_options = [
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024),
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
    ]
    transport = httpx.AsyncHTTPTransport(limits=limits, socket_options=custom_socket_options)

    sem = asyncio.Semaphore(CONCURRENT_FILES)
    async with httpx.AsyncClient(transport=transport) as client:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            await asyncio.gather(
                *[
                    download_file(client, s.rfilename, s.size, pbar, sem, rate_limiter)
                    for s in shards
                ]
            )
```

Usage after this lands:

```
uv run initialize_project.py                    # full speed, no cap
uv run initialize_project.py --profile molab     # capped at 200 MB/s
```

`AsyncTokenBucket` itself (`initialize_project.py:45-71`) needs no change — it already
does exactly one job, rate-limit whatever `.consume()` is called on it, and now it's
simply not instantiated in the uncapped path.

---

## 4. One thing this doesn't fix — flag, don't build yet

`CONCURRENT_FILES = 5` and `SEGMENTS_PER_FILE = 2` (10 connections total) were sized for
the *capped* case: the comment at `initialize_project.py:23-25` says cold Xet-bridge
objects serve ~30-40 MB/s each, so 10 connections give roughly `10 x 35 MB/s ≈ 350 MB/s`
of raw capacity against a 200 MB/s ceiling — enough headroom that the token bucket, not
connection count, is what's actually limiting. Once `--profile` defaults to uncapped,
connection count becomes the *only* ceiling, and 350 MB/s may be well under what your
full-speed link can do. Don't change `CONCURRENT_FILES` / `SEGMENTS_PER_FILE` in this
edit — measure uncapped throughput first (§6), and only bump concurrency if the download
plateaus below your actual link speed. Changing two variables (the cap and the
concurrency) in one commit means a slow uncapped run won't tell you which knob to turn.

---

## 5. The test to write

Not a network test — this only needs to prove the dispatch logic, which is pure and
synchronous. Two cases:

```python
def test_default_profile_is_uncapped():
    args = tyro.cli(Args, args=[])
    assert args.profile == "default"

def test_molab_profile_selected():
    args = tyro.cli(Args, args=["--profile", "molab"])
    assert args.profile == "molab"
```

The `args=` keyword is what makes this a real unit test instead of something that needs
a subprocess and a captured `sys.argv` — see the tyro primer above. If you want to also
prove the wiring (not just the parsing), assert directly on the `capped` computation
that follows: `capped = args.profile == "molab"` should be `False` for `test_..._is_uncapped`
and `True` for `test_molab_profile_selected`. That line is one-character-fragile (a
typo'd `"molab"` string anywhere would silently mean the flag never engages the cap) and
otherwise has no test coverage at all.

Since `initialize_project.py` isn't part of the package under `pythonpath = ["."]"` in
the same way the root modules are (it's a standalone PEP 723 script), check where you
want this test to live — either add it to `tests/` if pytest can already import a
top-level script by path in this repo's config, or keep it as a small `if __name__ ==
"__main__":` block at the bottom of the script itself if not. Either is fine; the
content of the two asserts above is what matters.

---

## 6. How to verify

No dataset re-download needed to check the dispatch — `--help` alone proves tyro parsed
the dataclass correctly:

```bash
uv run initialize_project.py --help
```

**Good:** usage text lists `--profile {default,molab}` with default `default`. Passing
`--profile shared` (not a valid choice) exits with a tyro usage error before any network
call — that's the "reject a typo instead of silently downloading uncapped" property
mentioned in §1.

To confirm the cap is actually wired to the right branch, run against a dataset (or a
small test bucket) and watch the printed banner: the "uncapped" line versus the
"capped at 200 MB/s aggregate" line should flip depending on `--profile`, and — the
real proof — the transfer rate `tqdm` reports should visibly exceed 200 MB/s only in
the uncapped run, assuming your link and the Xet backend can sustain that (§4).
