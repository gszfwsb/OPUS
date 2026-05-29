"""
BETR-style similarity scoring for FineWeb Parquet files.

This script:
  1) Loads benchmark embeddings from `benchmark_embeddings_arctic_l_v2.npz`
  2) Loads a local Arctic-Embed L v2 encoder
  3) Walks over FineWeb parquet files, encodes each document `text`
  4) Computes cosine similarity between each document embedding and all
     benchmark embeddings, then reduces to a single scalar score per document
     (by default: max similarity)
  5) Writes new parquet files that contain all original columns + `betr_score`

Usage (example for your setup, from project root):

  python -u data/betr_score_parquet.py \
    --input_root /path/to/fineweb_parquet/data \
    --output_root /path/to/fineweb_parquet_betr \
    --benchmark_npz benchmark_embeddings_arctic_l_v2.npz \
    --encoder_model Snowflake/snowflake-arctic-embed-l-v2.0 \
    --batch_size 192 \
    --max_length 512 \
    --device cuda \
    --dtype bfloat16 \
    --reduction max

You can later stream top-k by `betr_score` (e.g. to build a 30B-token subset),
then convert the selected parquet shards to .bin with `gpt2tokenize.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_benchmark_embeddings(npz_path: Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load benchmark embeddings and move to device, with L2 normalization."""
    data = np.load(npz_path, allow_pickle=True)
    embs = data["embeddings"].astype("float32")  # [N_bench, D]
    bench = torch.from_numpy(embs).to(device=device, dtype=dtype)
    bench = F.normalize(bench, dim=1)
    return bench


def load_encoder(model_name: str, device: str, dtype: torch.dtype):
    """Load Arctic-Embed encoder + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Encode a batch of texts -> [B, D] embedding tensor."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embs = outputs.pooler_output  # [B, D]
    else:
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
        summed = (last_hidden * mask).sum(dim=1)  # [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        embs = summed / counts

    return embs


def score_single_parquet(
    parquet_path: Path,
    out_path: Path,
    tokenizer,
    model,
    bench_norm: torch.Tensor,
    device: str,
    batch_size: int,
    max_length: int,
    reduction: str = "max",
) -> None:
    """Score a single parquet file and write out with `betr_score` column."""
    try:
        table = pq.read_table(str(parquet_path))
    except Exception as e:
        print(f"  [skip] Failed to read {parquet_path.name}: {e}")
        return

    if "text" not in table.column_names:
        print(f"  [skip] No 'text' column in {parquet_path.name}")
        return

    text_column = table.column("text")
    try:
        texts = text_column.to_pylist()
    except Exception:
        texts = [text_column[i].as_py() for i in range(len(text_column))]

    n_docs = len(texts)
    scores = np.zeros(n_docs, dtype=np.float32)

    # Process in batches
    idxs = np.arange(n_docs)
    for start in tqdm(
        range(0, n_docs, batch_size),
        desc=f"Scoring {parquet_path.name}",
        leave=False,
    ):
        end = min(start + batch_size, n_docs)
        batch_indices = idxs[start:end]
        batch_texts = [texts[i] for i in batch_indices if isinstance(texts[i], str) and texts[i]]

        if not batch_texts:
            continue

        # Map from local batch position to global row index
        local_to_global: List[int] = []
        for i in batch_indices:
            if isinstance(texts[i], str) and texts[i]:
                local_to_global.append(i)

        doc_embs = encode_texts(batch_texts, tokenizer, model, device, max_length)  # [B_valid, D]
        doc_embs = F.normalize(doc_embs, dim=1)

        # Cosine similarity with benchmark embeddings
        sim = torch.matmul(doc_embs, bench_norm.T)  # [B_valid, N_bench]

        if reduction == "max":
            doc_scores, _ = torch.max(sim, dim=1)
        elif reduction == "mean":
            doc_scores = torch.mean(sim, dim=1)
        else:
            raise ValueError(f"Unsupported reduction '{reduction}', use 'max' or 'mean'.")

        doc_scores_np = doc_scores.to(torch.float32).cpu().numpy()
        for local_idx, global_idx in enumerate(local_to_global):
            scores[global_idx] = doc_scores_np[local_idx]

    # Build output table
    arrays = []
    names = []
    for i, name in enumerate(table.column_names):
        arrays.append(table.column(i))
        names.append(name)

    arrays.append(pa.array(scores, type=pa.float32()))
    names.append("betr_score")

    out_table = pa.Table.from_arrays(arrays, names=names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(out_path), compression="snappy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BETR-style similarity scoring for FineWeb parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing FineWeb parquet data (will be walked recursively).",
    )
    p.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for scored parquet files (same relative layout, with betr_score column).",
    )
    p.add_argument(
        "--benchmark_npz",
        type=str,
        default="benchmark_embeddings_arctic_l_v2.npz",
        help="Path to benchmark embeddings .npz.",
    )
    p.add_argument(
        "--encoder_model",
        type=str,
        default="Snowflake/snowflake-arctic-embed-l-v2.0",
        help="HuggingFace model name or local path for Arctic-Embed encoder.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=192,
        help="Batch size for encoding documents.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenizer.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run encoder on (cuda or cpu).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype: float32|bfloat16|float16.",
    )
    p.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="How to reduce benchmark similarities into a single score per document.",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on number of parquet files to process (for testing).",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards to split the file list into (for multi-GPU parallelism).",
    )
    p.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of this shard (0-based). Only effective when num_shards > 1.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    bench_npz = Path(args.benchmark_npz)

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if not bench_npz.exists():
        raise SystemExit(f"Benchmark NPZ not found: {bench_npz}")

    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must be in [0, num_shards)")

    requested_device = args.device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[BETR] Requested CUDA but no GPU is available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype_key = args.dtype.lower()
    if dtype_key not in dtype_map:
        raise SystemExit(f"Unsupported dtype '{args.dtype}'. Choose from {sorted(dtype_map.keys())}.")
    torch_dtype = dtype_map[dtype_key]

    if device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("[BETR] Requested low-precision dtype on CPU; falling back to float32.")
        torch_dtype = torch.float32

    print(f"[BETR] Input root:   {input_root}")
    print(f"[BETR] Output root:  {output_root}")
    print(f"[BETR] Benchmark npz: {bench_npz}")
    print(f"[BETR] Encoder:      {args.encoder_model}")
    print(f"[BETR] Device:       {device}")
    print(f"[BETR] Dtype:        {torch_dtype}")
    print(f"[BETR] Batch size:   {args.batch_size}")
    print(f"[BETR] Max length:   {args.max_length}")
    print(f"[BETR] Reduction:    {args.reduction}")
    print(f"[BETR] Sharding:     shard {args.shard_idx} / {args.num_shards}")

    # Load benchmark embeddings and encoder
    bench_norm = load_benchmark_embeddings(bench_npz, device=device, dtype=torch_dtype)
    tokenizer, model = load_encoder(args.encoder_model, device=device, dtype=torch_dtype)

    # Collect parquet files
    parquet_files: List[Path] = []
    for root, _, files in os.walk(input_root):
        for fname in files:
            if fname.endswith(".parquet"):
                parquet_files.append(Path(root) / fname)

    parquet_files = sorted(parquet_files)
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]

    if args.num_shards > 1:
        total_files = len(parquet_files)
        parquet_files = [
            p for idx, p in enumerate(parquet_files) if idx % args.num_shards == args.shard_idx
        ]
        print(
            f"[BETR] Shard filtering: keeping {len(parquet_files)} / {total_files} files "
            f"(shard {args.shard_idx})"
        )

    print(f"[BETR] Found {len(parquet_files)} parquet files to score.")

    for p_path in tqdm(parquet_files, desc="BETR scoring files"):
        rel = p_path.relative_to(input_root)
        out_path = output_root / rel

        if out_path.exists():
            # Skip already-scored files to allow resume
            continue

        score_single_parquet(
            parquet_path=p_path,
            out_path=out_path,
            tokenizer=tokenizer,
            model=model,
            bench_norm=bench_norm,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            reduction=args.reduction,
        )


if __name__ == "__main__":
    main()


