# /// script
# dependencies = [
#   "httpx",
#   "huggingface-hub",
#   "tqdm"
# ]
# ///

import os
import sys
import asyncio
import time
from pathlib import Path
import httpx
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
SEGMENTS_PER_FILE = 8              # Bypasses the CDN throttle by fetching 8 chunks of the file at once
CHUNK_SIZE = 128 * 1024            # 128 KB socket read chunks
MAX_BANDWIDTH = 200 * 1024 * 1024  # Strict aggregate safety ceiling: 200 MB/s

# Retrieve HF token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists():
        HF_TOKEN = token_path.read_text().strip()

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable or cached token not found.", file=sys.stderr)
    sys.exit(1)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


class AsyncTokenBucket:
    """Strict aggregate rate limiter to govern total bandwidth across all active segments."""
    def __init__(self, rate_bytes_per_sec):
        self.rate = rate_bytes_per_sec
        self.capacity = rate_bytes_per_sec * 0.25  # 250ms burst safety window
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount):
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                
                needed = amount - self.tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)


rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)


async def download_segment(client, url, start, end, temp_file_path, file_lock, progress_bar):
    """Downloads a specific byte range of a file."""
    segment_headers = {**headers, "Range": f"bytes={start}-{end}"}
    
    for attempt in range(5):
        try:
            async with client.stream("GET", url, headers=segment_headers, follow_redirects=True, timeout=30.0) as r:
                # 206 Partial Content is the expected successful HTTP status for Range requests
                if r.status_code != 206:
                    raise httpx.HTTPStatusError(f"Expected 206 Partial Content, got {r.status_code}", request=r.request, response=r)
                
                # We open the file and write to the specific offset safely using a lock
                # (Standard disk writes are fast and synchronous, but we lock to keep the offset thread-safe)
                async with file_lock:
                    with open(temp_file_path, "r+b") as f:
                        f.seek(start)
                        async for chunk in r.aiter_bytes(chunk_size=CHUNK_SIZE):
                            chunk_len = len(chunk)
                            await rate_limiter.consume(chunk_len)
                            f.write(chunk)
                            progress_bar.update(chunk_len)
                return
        except Exception as e:
            if attempt == 4:
                raise e
            await asyncio.sleep(2 ** attempt)


async def download_file_segmented(client, filename, size, progress_bar):
    """Coordinates segment tasks for a single file sequentially."""
    dest_path = DEST_DIR / filename
    
    # Idempotent skip
    if dest_path.exists() and dest_path.stat().st_size == size:
        progress_bar.update(size)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    # Pre-allocate file size on disk so workers can seek and write to designated byte boundaries
    with open(temp_path, "wb") as f:
        f.truncate(size)

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"
    
    # Calculate byte offsets for each worker segment
    segment_size = size // SEGMENTS_PER_FILE
    tasks = []
    file_lock = asyncio.Lock()
    
    for i in range(SEGMENTS_PER_FILE):
        start = i * segment_size
        # The last segment gets any remaining bytes from division rounding
        end = size - 1 if i == SEGMENTS_PER_FILE - 1 else (start + segment_size - 1)
        tasks.append(
            download_segment(client, url, start, end, temp_path, file_lock, progress_bar)
        )
        
    await asyncio.gather(*tasks)
    temp_path.rename(dest_path)


async def main():
    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]
    
    total_size = sum(s.size for s in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")
    print(f"Downloading shards sequentially using {SEGMENTS_PER_FILE} parallel segments per file...")

    limits = httpx.Limits(max_keepalive_connections=SEGMENTS_PER_FILE + 2, max_connections=SEGMENTS_PER_FILE * 2)
    
    # Direct 4MB socket window config keeps our individual range streams broad and fast
    custom_socket_options = [
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024),
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
    ]
    
    transport = httpx.AsyncHTTPTransport(limits=limits, socket_options=custom_socket_options)
    
    async with httpx.AsyncClient(transport=transport) as client:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            # We process files sequentially, but download each file using highly parallelized range requests
            for shard in shards:
                await download_file_segmented(client, shard.rfilename, shard.size, pbar)

if __name__ == "__main__":
    import socket
    asyncio.run(main())
