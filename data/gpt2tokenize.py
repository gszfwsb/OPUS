"""
Convert FineWeb Parquet to GPT-2 .bin format.

Tokenizes text with tiktoken (GPT-2 encoding) and writes .bin shards
with a standard header (magic=20240520, version=1, num_tokens).

Usage:
    python data/gpt2tokenize.py \
        --parquet_dir ./fineweb_parquet \
        --output_dir ./bins/fineweb_train \
        --val_output_dir ./bins/fineweb_val \
        --val_tokens 200000000 \
        --num_workers 32
"""

import os
import sys

# Set tiktoken cache directory BEFORE importing tiktoken
if "TIKTOKEN_CACHE_DIR" not in os.environ:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _cache_dir = os.path.join(_repo_root, ".cache", "tiktoken")
    os.environ["TIKTOKEN_CACHE_DIR"] = _cache_dir

import argparse
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import tiktoken


def tokenize_text(text: str, enc) -> List[int]:
    """Tokenize text to GPT-2 tokens."""
    try:
        return enc.encode(text, allowed_special={'<|endoftext|>'})
    except:
        return []


def parquet_worker_ultra(
    worker_id: int,
    files: List[Path],
    output_queue: Queue,
    max_tokens: int = None,
    val_queue: Queue = None,
    val_tokens_per_file: int = 0,
    batch_size: int = 1000  # Larger batch for I/O optimization
):
    """
    Worker optimized for I/O bottlenecks.
    
    Key changes:
    - Use Arrow native arrays (avoid pandas DataFrame.iterrows())
    - Extract text column as numpy array first (100x faster)
    - Batch processing
    """
    enc = tiktoken.get_encoding("gpt2")
    
    total_train_tokens = 0
    total_val_tokens = 0
    docs_processed = 0
    
    # Batch buffers
    train_batch = []
    val_batch = []
    
    def flush_batches():
        """Send batched data to queues"""
        nonlocal train_batch, val_batch
        if train_batch:
            output_queue.put(('batch', train_batch))
            train_batch = []
        if val_batch and val_queue:
            val_queue.put(('batch', val_batch))
            val_batch = []
    
    for parquet_file in files:
        try:
            parquet_obj = pq.ParquetFile(str(parquet_file))
            file_tokens = 0
            
            # Process row groups
            for row_group_idx in range(parquet_obj.num_row_groups):
                table = parquet_obj.read_row_group(row_group_idx)
                
                # ⚡ KEY OPTIMIZATION: Direct Arrow array access (NO pandas!)
                # This is 100x faster than df.iterrows()
                if 'text' not in table.column_names:
                    continue
                
                text_column = table.column('text')
                
                # Convert to Python list once (much faster than iterating Series)
                try:
                    texts = text_column.to_pylist()
                except:
                    # Fallback for some Arrow types
                    texts = [text_column[i].as_py() for i in range(len(text_column))]
                
                # Process texts in batch
                for text in texts:
                    if not text or not isinstance(text, str):
                        continue
                    
                    # Tokenize
                    tokens = tokenize_text(text, enc)
                    if not tokens:
                        continue
                    
                    token_count = len(tokens) + 1
                    docs_processed += 1
                    
                    # Validation or training?
                    if val_queue and file_tokens < val_tokens_per_file:
                        val_batch.append((tokens, token_count))
                        file_tokens += token_count
                        total_val_tokens += token_count
                    else:
                        train_batch.append((tokens, token_count))
                        total_train_tokens += token_count
                    
                    # Flush when batch is full
                    if len(train_batch) >= batch_size or len(val_batch) >= batch_size:
                        flush_batches()
                    
                    # Check max tokens
                    if max_tokens and total_train_tokens >= max_tokens:
                        flush_batches()
                        output_queue.put(('done', None))
                        if val_queue:
                            val_queue.put(('done', None))
                        print(f"Worker {worker_id}: Hit max_tokens. Processed {docs_processed:,} docs, train={total_train_tokens:,} val={total_val_tokens:,} tokens")
                        return
                        
        except Exception as e:
            print(f"Worker {worker_id}: Error processing {parquet_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Flush remaining
    flush_batches()
    
    # Send done signal
    output_queue.put(('done', None))
    if val_queue:
        val_queue.put(('done', None))
    print(f"Worker {worker_id}: Finished. Processed {docs_processed:,} docs, train={total_train_tokens:,} val={total_val_tokens:,} tokens")


class StreamingBinWriter:
    """Write tokens to .bin shards with minimal fsync."""
    
    def __init__(self, out_dir: Path, tokens_per_shard: int = 100_000_000):
        self.out_dir = out_dir
        self.tokens_per_shard = tokens_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.shard_idx = 0
        self.current_tokens = []
        self.current_token_count = 0
        self.total_tokens_written = 0
    
    def add_tokens(self, tokens: List[int]):
        """Add tokens from one document (will auto-append EOT)."""
        token_count = len(tokens) + 1
        
        # Check if adding this would exceed shard size
        if self.current_token_count + token_count > self.tokens_per_shard and self.current_tokens:
            self._flush_shard()
        
        self.current_tokens.extend(tokens)
        self.current_tokens.append(50256)  # <|endoftext|>
        self.current_token_count += token_count
    
    def _flush_shard(self):
        """Write current shard to disk."""
        if not self.current_tokens:
            return
        
        shard_path = self.out_dir / f"train_{self.shard_idx:06d}.bin"
        
        # Convert to numpy array
        tokens_array = np.array(self.current_tokens, dtype=np.uint16)
        
        # Write with header (magic=20240520, version=1, num_tokens)
        with open(shard_path, 'wb') as f:
            header = np.zeros(256, dtype=np.int32)
            header[0] = 20240520  # magic
            header[1] = 1         # version
            header[2] = len(tokens_array)
            f.write(header.tobytes())
            f.write(tokens_array.tobytes())
            # Fsync ONLY at shard boundaries
            os.fsync(f.fileno())
        
        self.total_tokens_written += len(tokens_array)
        self.shard_idx += 1
        
        # Clear buffer
        self.current_tokens = []
        self.current_token_count = 0
    
    def finalize(self) -> dict:
        """Flush remaining tokens and return stats."""
        self._flush_shard()
        return {
            'total_shards': self.shard_idx,
            'total_tokens': self.total_tokens_written
        }


def convert_parquets_to_bins(
    parquet_dir: Path,
    output_dir: Path,
    tokens_per_shard: int = 100_000_000,
    num_workers: int = 16,
    max_tokens: int = None,
    val_output_dir: Path = None,
    val_tokens_total: int = 0
) -> dict:
    """Main conversion function with I/O optimization."""
    
    print(f"\n{'='*60}")
    print(f"Parquet -> GPT-2 .bin")
    print(f"{'='*60}")
    print(f"Input:      {parquet_dir}")
    print(f"Train out:  {output_dir}")
    if val_output_dir:
        print(f"Val out:    {val_output_dir}")
        print(f"Val tokens: {val_tokens_total:,}")
    print(f"Shard size: {tokens_per_shard:,}")
    print(f"Workers:    {num_workers}")
    if max_tokens:
        print(f"Max tokens: {max_tokens:,}")
    print(f"{'='*60}\n")
    
    # Discover parquet files
    print("Discovering parquet files...")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    # Calculate validation tokens per file (uniform sampling)
    val_tokens_per_file = val_tokens_total // len(parquet_files) if val_tokens_total > 0 else 0
    if val_tokens_per_file > 0:
        print(f"  Validation: ~{val_tokens_per_file:,} tokens per file (uniform sampling)")
    
    # Split files across workers
    files_per_worker = len(parquet_files) // num_workers
    worker_files = []
    for i in range(num_workers):
        start = i * files_per_worker
        end = start + files_per_worker if i < num_workers - 1 else len(parquet_files)
        worker_files.append(parquet_files[start:end])
    
    print(f"  Files per worker: ~{files_per_worker}\n")
    
    # Create queues (larger for I/O optimization)
    train_queue = Queue(maxsize=200)  # Larger queue for better I/O buffering
    val_queue = Queue(maxsize=200) if val_tokens_per_file > 0 else None
    
    # Calculate per-worker max tokens
    worker_max_tokens = max_tokens // num_workers if max_tokens else None
    
    # Start workers
    print(f"Starting {num_workers} worker processes...")
    workers = []
    for worker_id in range(num_workers):
        p = Process(
            target=parquet_worker_ultra,
            args=(
                worker_id,
                worker_files[worker_id],
                train_queue,
                worker_max_tokens,
                val_queue,
                val_tokens_per_file,
                1000  # Large batch_size for I/O optimization
            )
        )
        p.start()
        workers.append(p)
    
    # Create writers
    train_writer = StreamingBinWriter(output_dir, tokens_per_shard)
    val_writer = StreamingBinWriter(val_output_dir, tokens_per_shard) if val_output_dir else None
    
    # Statistics
    total_train_tokens = 0
    total_train_docs = 0
    total_val_tokens = 0
    total_val_docs = 0
    train_workers_done = 0
    val_workers_done = 0
    
    # Progress bars
    pbar_total = max_tokens if max_tokens else None
    with tqdm(total=pbar_total, desc="Training tokens", unit="tok", unit_scale=True, position=0) as train_pbar, \
         tqdm(total=val_tokens_total if val_tokens_total > 0 else None, desc="Validation tokens", unit="tok", unit_scale=True, position=1, disable=val_queue is None) as val_pbar:
        
        while train_workers_done < num_workers or (val_queue and val_workers_done < num_workers):
            # Process training queue
            if train_workers_done < num_workers:
                try:
                    item = train_queue.get(timeout=0.1)
                    
                    if item[0] == 'done':
                        train_workers_done += 1
                    elif item[0] == 'batch':
                        # Process batch of documents
                        batch = item[1]
                        for tokens, token_count in batch:
                            train_writer.add_tokens(tokens)
                            total_train_tokens += token_count
                            total_train_docs += 1
                            train_pbar.update(token_count)
                        
                        # Early exit if max reached
                        if max_tokens and total_train_tokens >= max_tokens:
                            print(f"\nMax training tokens reached: {total_train_tokens:,}")
                            for p in workers:
                                p.terminate()
                            break
                            
                except Empty:
                    pass
            
            # Process validation queue
            if val_queue and val_workers_done < num_workers:
                try:
                    item = val_queue.get(timeout=0.1)
                    
                    if item[0] == 'done':
                        val_workers_done += 1
                    elif item[0] == 'batch':
                        # Process batch of documents
                        batch = item[1]
                        for tokens, token_count in batch:
                            val_writer.add_tokens(tokens)
                            total_val_tokens += token_count
                            total_val_docs += 1
                            val_pbar.update(token_count)
                            
                except Empty:
                    pass
            
            # Exit condition
            if train_workers_done >= num_workers and (not val_queue or val_workers_done >= num_workers):
                break
    
    # Cleanup workers
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    # Finalize writers
    train_stats = train_writer.finalize()
    val_stats = val_writer.finalize() if val_writer else {'total_tokens': 0, 'total_shards': 0}
    
    print(f"\nConversion complete:")
    print(f"  Train: {total_train_docs:,} docs, {total_train_tokens:,} tokens, {train_stats['total_shards']} shards")
    if val_writer:
        print(f"  Val:   {total_val_docs:,} docs, {total_val_tokens:,} tokens, {val_stats['total_shards']} shards")
    
    return {'train': train_stats, 'val': val_stats}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert FineWeb Parquet to GPT-2 .bin format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parquet_dir", type=str, required=True,
                   help="Directory containing .parquet files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for training .bin shards")
    p.add_argument("--val_output_dir", type=str, default=None,
                   help="Output directory for validation .bin shards (optional)")
    p.add_argument("--val_tokens", type=int, default=0,
                   help="Total validation tokens to collect (sampled uniformly from each file, default: 0)")
    p.add_argument("--tokens_per_shard", type=int, default=100_000_000,
                   help="Tokens per .bin shard (default: 100M)")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Number of parallel worker processes (default: 16)")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Maximum TRAINING tokens to process (default: None = process all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    val_output_dir = Path(args.val_output_dir) if args.val_output_dir else None
    
    stats = convert_parquets_to_bins(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        tokens_per_shard=args.tokens_per_shard,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        val_output_dir=val_output_dir,
        val_tokens_total=args.val_tokens
    )
    
    print(f"\nDone. Train: {stats['train']['total_tokens']:,} tokens ({stats['train']['total_shards']} shards)")
    if val_output_dir:
        print(f"     Val:   {stats['val']['total_tokens']:,} tokens ({stats['val']['total_shards']} shards)")


if __name__ == "__main__":
    main()
