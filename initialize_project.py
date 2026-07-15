# /// script
# dependencies = [
#   "httpx",
#   "huggingface-hub",
#   "tqdm"
# ]
# ///

import os
import sys
import time
import math
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import httpx
from huggingface_hub import HfApi
from tqdm import tqdm

# ---- Configuration ----
DATASET = "UItraviolet/industrial_cart"
DEST_DIR = Path("./dataset/data")
NUM_PROCESSES = 7                  # 7 workers * 28 MB/s = ~196 MB/s average
WORKER_BANDWIDTH_LIMIT = 28 * 1024 * 1024  # Capped at 28 MB/s per process (stays safe under 30MB/s)
CHUNK_SIZE = 256 * 1024            # 256 KB chunks

# Retrieve HF token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists():
        HF_TOKEN = token_path.read_text().strip()

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable or cached token not found.", file=sys.stderr)
    sys.exit(1)


class TokenBucket:
    """Synchronous token bucket for strict per-process rate limiting."""
    def __init__(self, rate_bytes_per_sec):
        self.rate = rate_bytes_per_sec
        self.capacity = rate_bytes_per_sec * 0.5  # 0.5s burst margin
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    def consume(self, amount):
        while True:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens >= amount:
                self.tokens -= amount
                return
            
            needed = amount - self.tokens
            time.sleep(needed / self.rate)


def worker_task(worker_id, task_queue, progress_queue, token_val):
    """The task running inside each isolated process."""
    limiter = TokenBucket(WORKER_BANDWIDTH_LIMIT)
    headers = {"Authorization": f"Bearer {token_val}"}
    
    # Each worker gets its own isolated HTTPX client and socket pool
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        while True:
            item = task_queue.get()
            if item is None:
                break
                
            filename, size = item
            dest_path = DEST_DIR / filename
            
            # Idempotency check
            if dest_path.exists() and dest_path.stat().st_size == size:
                progress_queue.put(size)
                continue
                
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = dest_path.with_suffix(".tmp")
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{filename}?download=true"
            
            success = False
            for attempt in range(5):
                try:
                    with client.stream("GET", url, headers=headers) as r:
                        r.raise_for_status()
                        with open(temp_path, "wb") as f:
                            for chunk in r.iter_bytes(chunk_size=CHUNK_SIZE):
                                chunk_len = len(chunk)
                                limiter.consume(chunk_len)
                                f.write(chunk)
                                progress_queue.put(chunk_len)
                    
                    temp_path.rename(dest_path)
                    success = True
                    break
                except Exception as e:
                    if temp_path.exists():
                        temp_path.unlink()
                    time.sleep(2 ** attempt)
            
            if not success:
                # Signal failure back to main process
                progress_queue.put(-1)


def main():
    print(f"Fetching manifest for {DATASET}...")
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    shards = [(s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith(".parquet")]
    
    total_size = sum(size for _, size in shards)
    print(f"Found {len(shards)} shards. Total size: {total_size / (1024**3):.2f} GB")
    print(f"Starting {NUM_PROCESSES} parallel rate-limited worker processes...")

    task_queue = Queue()
    progress_queue = Queue()

    # Feed tasks into the queue
    for shard in shards:
        task_queue.put(shard)
        
    # Add termination signals for each worker process
    for _ in range(NUM_PROCESSES):
        task_queue.put(None)

    # Spawn independent worker processes
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=worker_task, args=(i, task_queue, progress_queue, HF_TOKEN))
        p.start()
        processes.append(p)

    # Main thread processes progress queue to update global tqdm progress bar
    try:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            completed_bytes = 0
            while completed_bytes < total_size:
                bytes_downloaded = progress_queue.get()
                if bytes_downloaded == -1:
                    print("\n[Error] One of the workers failed a shard download after 5 retries.", file=sys.stderr)
                    continue
                pbar.update(bytes_downloaded)
                completed_bytes += bytes_downloaded
    except KeyboardInterrupt:
        print("\nTerminating workers...")
        for p in processes:
            p.terminate()
    finally:
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
