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
import subprocess
from pathlib import Path
import httpx
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
CONCURRENT_DOWNLOADS = 3            # Keeping it low prevents gVisor overhead
CHUNK_SIZE = 256 * 1024              # 256 KB chunk reads balance speed and lock frequency
MAX_BANDWIDTH = 200 * 1024 * 1024    # 200 MB/s Aggregate Target Limit

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
        # Max burst capacity is set to 1 second of data to absorb minor scheduling jitters safely
        self.capacity = rate_bytes_per_sec  
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
                
                # Sleep exactly the time needed to replenish the deficit
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
                async with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as r:
                    r.raise_for_status()
                    
                    with open(temp_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=CHUNK_SIZE):
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
    # ---- Step 1: Raise TCP Window safely ----
    tcp_rmem_path = "/proc/sys/net/ipv4/tcp_rmem"
    old_rmem = None
    raised_rmem = False
    
    try:
        if os.path.exists(tcp_rmem_path):
            with open(tcp_rmem_path, "r") as f:
                old_rmem = f.read().strip()
            
            # Elevate the TCP read memory limits: 4MB default / 16MB max
            subprocess.run(
                ["sysctl", "-qw", "net.ipv4.tcp_rmem=4096 4194304 16777216"], 
                check=True, 
                stderr=subprocess.DEVNULL
            )
            print(f"[net] tcp_rmem raised to 4MB default (was: {old_rmem})")
            raised_rmem = True
    except Exception:
        print("[net] Warning: sysctl write failed. Speed may bottleneck on default window limits.")

    try:
        # ---- Step 2: Download Dataset ----
        print(f"Fetching manifest for {DATASET}...")
        api = HfApi()
        info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
        shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]
        
        total_size = sum(s.size for s in shards)
        print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")

        sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
        
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(limits=limits) as client:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                tasks = [
                    download_shard(client, shard.rfilename, shard.size, sem, pbar)
                    for shard in shards
                ]
                await asyncio.gather(*tasks)
                
    finally:
        # ---- Step 3: Always restore defaults to protect subsequent commands ----
        if raised_rmem and old_rmem:
            try:
                subprocess.run(
                    ["sysctl", "-qw", f"net.ipv4.tcp_rmem={old_rmem}"], 
                    check=True, 
                    stderr=subprocess.DEVNULL
                )
                print("[net] tcp_rmem successfully restored to default.")
            except Exception as e:
                print(f"[net] Warning: Failed to restore tcp_rmem: {e}")

if __name__ == "__main__":
    asyncio.run(main())
