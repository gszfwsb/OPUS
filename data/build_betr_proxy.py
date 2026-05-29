"""
Build an OPUS proxy .bin shard by pulling the highest BETR-scored documents.

Workflow:
  1. Scan BETR-scored parquet shards (with `betr_score` column).
  2. Order files by their maximum score so we visit the most promising data first.
  3. Inside each file, sort rows by `betr_score` descending and tokenize text with GPT-2 BPE.
  4. Stream tokens into a GPT-NeoX-compatible .bin (same header as other data) until
     `target_tokens` is reached (default: 30M).

This produces a proxy file that is tightly aligned with the benchmark embeddings used
for BETR scoring, making it ideal for OPUS proxy batches.

Example usage:

python -u data/build_betr_proxy.py \
  --input_root ./fineweb_200B_parquet_betr \
  --output_dir ./bins/fineweb_betr_proxy_top30M \
  --target_tokens 30000000
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm

# Ensure tiktoken cache directory exists before importing
if "TIKTOKEN_CACHE_DIR" not in os.environ:
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["TIKTOKEN_CACHE_DIR"] = str(repo_root / ".cache" / "tiktoken")

import tiktoken

# Add project root to sys.path so `data.*` imports work when invoked as a script.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bisect import bisect_right
from collections import defaultdict

from data.gpt2tokenize import StreamingBinWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select top BETR-scored documents and build a proxy .bin shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing BETR-scored parquets (with betr_score column).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the generated proxy .bin shard(s).",
    )
    p.add_argument(
        "--target_tokens",
        type=int,
        default=30_000_000,
        help="Number of tokens to collect for the proxy file.",
    )
    p.add_argument(
        "--tokens_per_shard",
        type=int,
        default=30_000_000,
        help="Tokens per shard in the output (default: single shard equal to target tokens).",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on how many parquet files to scan.",
    )
    p.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Optional minimum betr_score threshold; documents below are ignored.",
    )
    return p.parse_args()


def list_parquet_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def compute_file_max_score(path: Path) -> float:
    """
    Read only the `betr_score` column and return its max (or -inf if empty).
    """
    try:
        table = pq.read_table(path, columns=["betr_score"])
    except Exception as e:
        print(f"[skip] Unable to read {path}: {e}")
        return float("-inf")

    if table.num_rows == 0:
        return float("-inf")

    score_col = table.column("betr_score")
    # Handle entirely null columns
    if score_col.null_count == score_col.length():
        return float("-inf")

    max_val = pc.max(score_col).as_py()
    if max_val is None:
        return float("-inf")
    return float(max_val)


def tokenize_text(text: str, enc) -> List[int]:
    try:
        return enc.encode(text, allowed_special={"<|endoftext|>"})
    except Exception:
        return []


def build_row_group_offsets(pf: pq.ParquetFile) -> List[int]:
    offsets = []
    pos = 0
    for rg in range(pf.num_row_groups):
        offsets.append(pos)
        rows = pf.metadata.row_group(rg).num_rows
        pos += rows
    offsets.append(pos)
    return offsets


def gather_top_indices(
    pf: pq.ParquetFile,
    min_score: float | None,
    keep_fraction: float = 0.005,
    min_keep: int = 2000,
    max_keep: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores_array, top_indices_sorted_desc)
    """
    total_rows = pf.metadata.num_rows
    scores = np.empty(total_rows, dtype=np.float32)
    pos = 0
    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["betr_score"]).column("betr_score")
        try:
            arr = col.to_numpy(zero_copy_only=False)
        except (pa.ArrowInvalid, TypeError):
            arr = np.array(col.to_pylist(), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=-1e9).astype(np.float32, copy=False)
        length = len(arr)
        scores[pos : pos + length] = arr
        pos += length
    if pos != total_rows:
        scores = scores[:pos]

    if min_score is not None:
        mask = scores >= min_score
        if not np.any(mask):
            return scores, np.array([])
        candidate_idx = np.nonzero(mask)[0]
    else:
        candidate_idx = np.arange(scores.size)

    if candidate_idx.size == 0:
        return scores, np.array([])

    desired = max(int(scores.size * keep_fraction), min_keep)
    desired = min(desired, candidate_idx.size, max_keep)

    if desired <= 0:
        desired = min(candidate_idx.size, max_keep)

    top_subset = candidate_idx[
        np.argpartition(scores[candidate_idx], -desired)[-desired:]
    ]
    top_sorted = top_subset[np.argsort(scores[top_subset])[::-1]]
    return scores, top_sorted


def process_file(
    path: Path,
    writer: StreamingBinWriter,
    enc,
    target_tokens: int,
    token_counter: int,
    min_score: float | None = None,
) -> Tuple[int, int]:
    """
    Stream the highest scoring documents from `path` into `writer`.

    Returns:
        (tokens_written, docs_written)
    """
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        print(f"[skip] Failed to open {path}: {e}")
        return token_counter, 0

    if pf.metadata.num_rows == 0:
        return token_counter, 0

    scores, sorted_indices = gather_top_indices(pf, min_score=min_score)
    if sorted_indices.size == 0:
        return token_counter, 0

    offsets = build_row_group_offsets(pf)
    group_to_entries: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for global_idx in sorted_indices:
        rg = bisect_right(offsets, global_idx) - 1
        if rg < 0:
            continue
        local_idx = global_idx - offsets[rg]
        group_to_entries[rg].append((local_idx, global_idx))

    selected_texts: Dict[int, str] = {}

    for rg, entries in group_to_entries.items():
        table = pf.read_row_group(rg, columns=["text"])
        text_col = table.column("text")
        for local_idx, global_idx in entries:
            if local_idx >= text_col.length():
                continue
            val = text_col[local_idx]
            selected_texts[global_idx] = val.as_py() if val is not None else None

    docs_written = 0

    for idx in sorted_indices:
        if token_counter >= target_tokens:
            break
        text = selected_texts.get(idx)
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = tokenize_text(text, enc)
        if not tokens:
            continue
        writer.add_tokens(tokens)
        token_counter += len(tokens) + 1
        docs_written += 1

    return token_counter, docs_written


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    target_tokens = args.target_tokens

    assert target_tokens > 0, "target_tokens must be positive"

    parquet_files = list_parquet_files(input_root)
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]

    if not parquet_files:
        raise SystemExit(f"No parquet files found under {input_root}")

    print(f"Found {len(parquet_files)} parquet files. Ranking by max betr_score...")
    file_stats = []
    for path in tqdm(parquet_files, desc="Scanning files for max score"):
        max_score = compute_file_max_score(path)
        if max_score == float("-inf"):
            continue
        file_stats.append((path, max_score))

    if not file_stats:
        raise SystemExit("No readable parquet files with betr_score found.")

    file_stats.sort(key=lambda x: x[1], reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = StreamingBinWriter(out_dir=output_dir, tokens_per_shard=args.tokens_per_shard)
    enc = tiktoken.get_encoding("gpt2")

    total_tokens = 0
    total_docs = 0
    files_visited = 0

    for path, max_score in tqdm(file_stats, desc="Building proxy"):
        if total_tokens >= target_tokens:
            break
        total_tokens, docs_written = process_file(
            path=path,
            writer=writer,
            enc=enc,
            target_tokens=target_tokens,
            token_counter=total_tokens,
            min_score=args.min_score,
        )
        if docs_written > 0:
            total_docs += docs_written
            files_visited += 1

    stats = writer.finalize()
    final_tokens = stats["total_tokens"]

    print("\n======================================")
    print("BETR proxy build complete.")
    print(f"Files visited      : {files_visited}")
    print(f"Documents written  : {total_docs}")
    print(f"Tokens requested   : {target_tokens:,}")
    print(f"Tokens generated   : {final_tokens:,}")
    print(f"Output directory   : {output_dir}")
    print("======================================")


if __name__ == "__main__":
    main()

