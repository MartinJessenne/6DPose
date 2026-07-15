# /// script
# dependencies = [
#   "aiohttp",
#   "huggingface-hub",
#   "tqdm"
# ]
# ///

import os
import sys
import asyncio
import time
import socket
from pathlib import Path
import aiohttp
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
CONCURRENT_DOWNLOADS = 4            # 4 parallel streams saturate the pipe without CPU choke
CHUNK_SIZE = 512 * 1024              # Large 512 KB reads reduce event loop overhead
MAX_BANDWIDTH = 200 * 1024 * 1024    # Strict aggregate ceiling: 200 MB/s

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
    """A high-performance rate limiter that prevents burst spikes."""
    def __init__(self, rate_bytes_per_sec):
        self.rate = rate_bytes_per_sec
        # Allow up to 0.5s of burst capacity to smooth out execution jitters
        self.capacity = rate_bytes_per_sec * 0.5  
        self.tokens = self.capacity
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
                
                # Calculate sleep time to fulfill the token deficit
                needed = amount - self.tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)

# Initialize global aggregate rate limiter
rate_limiter = AsyncTokenBucket(MAX_BANDWIDTH)

async def download_shard(session, filename, size, sem, progress_bar):
    dest_path = DEST_DIR / filename
    
    # Idempotent skip
    if dest_path.exists() and dest_path.stat().st_size == size:
        progress_bar.update(size)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"
    
    async with sem:
        for attempt in range(5):
            try:
                # aiohttp handles stream reading with significantly lower memory-copy overhead
                async with session.get(url, headers=headers, timeout=120) as r:
                    r.raise_for_status()
                    
                    with open(temp_path, "wb") as f:
                        # Stream the chunks asynchronously
                        async for chunk in r.content.iter_chunked(CHUNK_SIZE):
                            chunk_len = len(chunk)
                            await rate_limiter.consume(chunk_len)
                            f.write(chunk)
                            progress_bar.update(chunk_len)
                            
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
    
    # Customize the TCPConnector to force maximum socket recycling and buffer sizing
    connector = aiohttp.TCPConnector(
        limit=CONCURRENT_DOWNLOADS,
        force_close=False,             # Ensure keep-alive connections are maintained!
        enable_cleanup_closed=True,
        ttl_dns_cache=300
    )
    
    # We use aiohttp ClientSession as it outperforms HTTPX on raw concurrent stream writing
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            tasks = [
                download_shard(session, shard.rfilename, shard.size, sem, pbar)
                for shard in shards
            ]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
