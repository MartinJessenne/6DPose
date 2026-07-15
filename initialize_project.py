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
DEST_DIR = Path("./dataset")       # rfilenames already start with "data/", avoids data/data/
CONCURRENT_FILES = 5               # Cold Xet-bridge objects serve ~30-40 MB/s each:
                                   # parallelize ACROSS files to reach the aggregate cap.
SEGMENTS_PER_FILE = 2              # Light intra-file splitting (10 connections total)
CHUNK_SIZE = 128 * 1024            # 128 KB socket read chunks
MAX_BANDWIDTH = 200 * 1024 * 1024  # Strict aggregate ceiling: 200 MB/s
BURST_WINDOW = 0.10                # Token bucket burst window (seconds)
MIN_SEGMENT_SIZE = 8 * 1024 * 1024 # Never create segments smaller than 8 MB

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
    """Aggregate rate limiter shared by all active segments across all files."""
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
            await asyncio.sleep(wait_time)


rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)


async def download_segment(client, url, start, end, fd, progress_bar):
    """Download one byte range and pwrite() it at its offset. Resumable retries."""
    for attempt in range(5):
        offset = start
        segment_headers = {**AUTH_HEADERS, "Range": f"bytes={offset}-{end}"}
        try:
            async with client.stream(
                "GET", url, headers=segment_headers,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=60.0),
            ) as r:
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
            start = offset  # resume from last byte written
            await asyncio.sleep(2 ** attempt)


def compute_segments(size):
    n = min(SEGMENTS_PER_FILE, max(1, size // MIN_SEGMENT_SIZE))
    base = size // n
    return [
        (i * base, size - 1 if i == n - 1 else (i + 1) * base - 1)
        for i in range(n)
    ]


async def download_file(client, filename, size, progress_bar, sem):
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
            await asyncio.gather(*[
                download_segment(client, url, s, e, fd, progress_bar)
                for s, e in compute_segments(size)
            ])
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
    print(f"{CONCURRENT_FILES} files in flight x {SEGMENTS_PER_FILE} segments, "
          f"capped at {MAX_BANDWIDTH / 1024**2:.0f} MB/s aggregate")

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
            await asyncio.gather(*[
                download_file(client, s.rfilename, s.size, pbar, sem)
                for s in shards
            ])


if __name__ == "__main__":
    asyncio.run(main())
