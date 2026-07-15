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
import socket
from pathlib import Path
import httpx
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
CONCURRENT_DOWNLOADS = 3            # Keeping it low prevents gVisor cpu overhead
CHUNK_SIZE = 256 * 1024              # 256 KB chunk reads
MAX_BANDWIDTH = 200 * 1024 * 1024    # 200 MB/s Strict Aggregate Limit
FORCE_SOCKET_BUFFER_SIZE = 4 * 1024 * 1024  # Force a 4MB TCP Receive Window per socket

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

# ---- Socket Tuning Hook ----
def tune_socket(sock: socket.socket):
    """Forcefully injects custom TCP receive buffer sizes directly to the socket descriptor."""
    try:
        # Set receive buffer size
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, FORCE_SOCKET_BUFFER_SIZE)
        
        # Disable Nagle's algorithm for low latency packet processing inside gVisor's stack
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        # Some sandboxed environments block direct socket mutations; degrade gracefully
        pass

class CustomAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Custom HTTPX transport that overrides socket creation to apply low-level gVisor tuning."""
    async def handle_async_request(self, request, *args, **kwargs):
        # We hook into the socket factory mechanism of HTTPX
        original_connect = self._pool._connector.connect
        
        async def tuned_connect(*conn_args, **conn_kwargs):
            connection = await original_connect(*conn_args, **conn_kwargs)
            # Access the raw underlying socket if it exists and apply our tuning
            if hasattr(connection, "_socket") and connection._socket:
                tune_socket(connection._socket)
            return connection
            
        self._pool._connector.connect = tuned_connect
        return await super().handle_async_request(request, *args, **kwargs)


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
    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [s for s in info.siblings if s.rfilename.endswith(".parquet")]
    
    total_size = sum(s.size for s in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")

    sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    
    # Instantiate HTTPX client using our custom socket-tuned transport
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    transport = CustomAsyncHTTPTransport(limits=limits)
    
    async with httpx.AsyncClient(transport=transport) as client:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            tasks = [
                download_shard(client, shard.rfilename, shard.size, sem, pbar)
                for shard in shards
            ]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
