# /// script
# dependencies = [
#   "httpx",
#   "huggingface-hub",
#   "tqdm"
# ]
# ///

import os
import sys
import socket
import asyncio
import time
from pathlib import Path
import httpx
from huggingface_hub import HfApi
from tqdm import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
SEGMENTS_PER_FILE = 8              # Parallel range requests per file
CHUNK_SIZE = 128 * 1024            # 128 KB socket read chunks
MAX_BANDWIDTH = 200 * 1024 * 1024  # Strict aggregate ceiling: 200 MB/s
BURST_WINDOW = 0.10                # Token bucket burst window (seconds).
                                   # Smaller = smoother instantaneous rate,
                                   # important since a >300 MB/s spike kills the instance.
MIN_SEGMENT_SIZE = 8 * 1024 * 1024 # Don't split files into segments smaller than 8 MB

# Retrieve HF token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists():
        HF_TOKEN = token_path.read_text().strip()

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable or cached token not found.", file=sys.stderr)
    sys.exit(1)

AUTH_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


class AsyncTokenBucket:
    """Aggregate rate limiter shared by all active segments.

    Waiters sleep OUTSIDE the lock so a single slow consumer never
    blocks token accounting for the others.
    """
    def __init__(self, rate_bytes_per_sec, burst_window=BURST_WINDOW):
        self.rate = rate_bytes_per_sec
        self.capacity = rate_bytes_per_sec * burst_window
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount):
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= amount:
                    self.tokens -= amount
                    return

                wait_time = (amount - self.tokens) / self.rate
            # Sleep with the lock RELEASED, then re-check.
            await asyncio.sleep(wait_time)


rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)


async def download_segment(client, url, start, end, fd, progress_bar):
    """Download one byte range and pwrite() it into the pre-allocated file.

    No file lock: segments own disjoint byte ranges and os.pwrite is a
    positional write that doesn't touch a shared file offset, so concurrent
    writes to the same fd are safe. This is what lets all 8 sockets drain
    simultaneously instead of one at a time.
    """
    for attempt in range(5):
        offset = start
        segment_headers = {**AUTH_HEADERS, "Range": f"bytes={offset}-{end}"}
        try:
            async with client.stream(
                "GET", url, headers=segment_headers,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=30.0),
            ) as r:
                # 206 Partial Content is the expected status for Range requests
                if r.status_code != 206:
                    raise httpx.HTTPStatusError(
                        f"Expected 206 Partial Content, got {r.status_code}",
                        request=r.request, response=r,
                    )
                async for chunk in r.aiter_bytes(chunk_size=CHUNK_SIZE):
                    await rate_limiter.consume(len(chunk))
                    os.pwrite(fd, chunk, offset)
                    offset += len(chunk)
                    progress_bar.update(len(chunk))
            return
        except Exception:
            if attempt == 4:
                raise
            # Resume from the last byte successfully written: no re-download,
            # and the progress bar stays accurate.
            start = offset
            await asyncio.sleep(2 ** attempt)


def compute_segments(size):
    """Split `size` bytes into up to SEGMENTS_PER_FILE ranges, but never
    create tiny segments (small files download as a single stream)."""
    n = min(SEGMENTS_PER_FILE, max(1, size // MIN_SEGMENT_SIZE))
    base = size // n
    ranges = []
    for i in range(n):
        s = i * base
        e = size - 1 if i == n - 1 else s + base - 1
        ranges.append((s, e))
    return ranges


async def download_file_segmented(client, filename, size, progress_bar):
    """Coordinates parallel segment tasks for a single file."""
    dest_path = DEST_DIR / filename

    # Idempotent skip
    if dest_path.exists() and dest_path.stat().st_size == size:
        progress_bar.update(size)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"

    # Pre-allocate so segments can pwrite at their byte boundaries
    fd = os.open(temp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)
        tasks = [
            download_segment(client, url, s, e, fd, progress_bar)
            for s, e in compute_segments(size)
        ]
        await asyncio.gather(*tasks)
        os.fsync(fd)
    finally:
        os.close(fd)

    temp_path.rename(dest_path)


async def main():
    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]

    total_size = sum(s.size for s in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")
    print(f"Downloading shards sequentially using up to {SEGMENTS_PER_FILE} parallel segments per file...")

    limits = httpx.Limits(
        max_keepalive_connections=SEGMENTS_PER_FILE + 2,
        max_connections=SEGMENTS_PER_FILE * 2,
    )

    custom_socket_options = [
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024),
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
    ]

    transport = httpx.AsyncHTTPTransport(limits=limits, socket_options=custom_socket_options)

    async with httpx.AsyncClient(transport=transport) as client:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for shard in shards:
                await download_file_segmented(client, shard.rfilename, shard.size, pbar)


if __name__ == "__main__":
    asyncio.run(main())
