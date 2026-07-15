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
CONCURRENT_DOWNLOADS = 24         # 24 sockets * ~6 MB/s = ~144 MB/s
MAX_BANDWIDTH = 150 * 1024 * 1024  # Remains your hard safety ceiling
CHUNK_SIZE = 128 * 1024         # 128 KB chunks

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
    """A strict token bucket rate limiter to govern aggregate bandwidth."""
    def __init__(self, rate_bytes_per_sec):
        self.rate = rate_bytes_per_sec
        self.capacity = rate_bytes_per_sec  # Max burst size is 1 second of transfer
        self.tokens = rate_bytes_per_sec
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount):
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                
                # Top up tokens
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                
                # Wait for enough tokens to replenish
                needed = amount - self.tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)

# Initialize global rate limiter
rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)

async def download_shard(client, filename, size, sem, progress_bar):
    dest_path = DEST_DIR / filename
    
    # Idempotent skip: check if file is already fully downloaded
    if dest_path.exists() and dest_path.stat().st_size == size:
        progress_bar.update(size)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"
    
    async with sem:
        for attempt in range(5):
            try:
                # Use client.stream to avoid loading the entire file into memory
                async with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as r:
                    r.raise_for_status()
                    
                    with open(temp_path, "wb") as f:
                        # FIX: Changed to aiter_bytes() for async streaming
                        async for chunk in r.aiter_bytes(chunk_size=CHUNK_SIZE):
                            chunk_len = len(chunk)
                            # Strict sleep here guarantees we never violate the global speed limit
                            await rate_limiter.consume(chunk_len)
                            f.write(chunk)
                            progress_bar.update(chunk_len)
                            
                # Swap temp file to final location on success
                temp_path.rename(dest_path)
                return
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                if attempt == 4:
                    print(f"\nFailed to download {filename} after 5 attempts: {e}", file=sys.stderr)
                    raise e
                await asyncio.sleep(2 ** attempt)

async def main():
    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]
    
    total_size = sum(s.size for s in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")

    sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    
    # We use a single client instance to reuse the underlying connection pools
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            tasks = [
                download_shard(client, shard.rfilename, shard.size, sem, pbar)
                for shard in shards
            ]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
