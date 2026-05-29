"""
Generate benchmark embeddings for OPUS proxy construction.

This script extracts text from evaluation benchmarks (HellaSwag, ARC, PiQA, etc.),
encodes them with a sentence encoder (Arctic-Embed L v2 by default), and saves
the embeddings as a .npz file.  The resulting file is used by `betr_score_parquet.py`
to compute per-document relevance scores against the evaluation distribution.

Pipeline position:
  embed_benchmarks.py  -->  betr_score_parquet.py  -->  build_betr_proxy.py

Usage:
  python -u data/embed_benchmarks.py \
    --data_dir opencompass/data \
    --encoder_model Snowflake/snowflake-arctic-embed-l-v2.0 \
    --output benchmark_embeddings_arctic_l_v2.npz \
    --batch_size 64 \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Data loaders (self-contained; mirrors eval/run_lighteval_offline.py loaders)
# ---------------------------------------------------------------------------

def load_hellaswag(data_dir: Path) -> List[str]:
    path = data_dir / "hellaswag" / "hellaswag.jsonl"
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if "query" in item:
                # OpenCompass format
                query = item.get("query", "")
                for choice in item.get("choices", []):
                    texts.append(f"{query} {choice}")
            else:
                ctx = item.get("ctx_a", "") + " " + item.get("ctx_b", "")
                activity = item.get("activity_label", "")
                for ending in item.get("endings", []):
                    texts.append(f"{activity}: {ctx} {ending}")
    return texts


def load_arc(data_dir: Path, subset: str = "ARC-c") -> List[str]:
    arc_dir = data_dir / "ARC" / subset
    candidates = [
        arc_dir / ("ARC-Challenge-Test.jsonl" if subset == "ARC-c" else "ARC-Easy-Test.jsonl"),
        arc_dir / "test.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(f"[skip] ARC-{subset} not found in {arc_dir}")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            qdata = item.get("question", {})
            stem = qdata.get("stem", "") if isinstance(qdata, dict) else str(qdata)
            choices = qdata.get("choices", []) if isinstance(qdata, dict) else []
            for c in choices:
                texts.append(f"{stem} {c.get('text', '')}")
    return texts


def load_piqa(data_dir: Path) -> List[str]:
    path = data_dir / "piqa" / "dev.jsonl"
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            goal = item.get("goal", "")
            texts.append(f"{goal} {item.get('sol1', '')}")
            texts.append(f"{goal} {item.get('sol2', '')}")
    return texts


def load_siqa(data_dir: Path) -> List[str]:
    path = data_dir / "siqa" / "dev.jsonl"
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            ctx = item.get("context", "")
            q = item.get("question", "")
            for key in ("answerA", "answerB", "answerC"):
                texts.append(f"{ctx} {q} {item.get(key, '')}")
    return texts


def load_winogrande(data_dir: Path) -> List[str]:
    wino_dir = data_dir / "winogrande"
    candidates = [wino_dir / "dev.jsonl", wino_dir / "winogrande_xl" / "validation.jsonl"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(f"[skip] Winogrande not found in {wino_dir}")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            sentence = item.get("sentence", "")
            for opt in (item.get("option1", ""), item.get("option2", "")):
                texts.append(sentence.replace("_", opt))
    return texts


def load_commonsenseqa(data_dir: Path) -> List[str]:
    path = data_dir / "commonsenseqa" / "dev_rand_split.jsonl"
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            stem = item.get("question", {}).get("stem", "")
            for c in item.get("question", {}).get("choices", []):
                texts.append(f"{stem} {c.get('text', '')}")
    return texts


def load_openbookqa(data_dir: Path) -> List[str]:
    path = data_dir / "openbookqa" / "Main" / "test.jsonl"
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            stem = item.get("question", {}).get("stem", "")
            for c in item.get("question", {}).get("choices", []):
                texts.append(f"{stem} {c.get('text', '')}")
    return texts


def load_mmlu(data_dir: Path) -> List[str]:
    mmlu_dir = data_dir / "mmlu" / "test"
    if not mmlu_dir.exists():
        print(f"[skip] MMLU not found at {mmlu_dir}")
        return []
    texts = []
    for csv_file in sorted(mmlu_dir.glob("*_test.csv")):
        subject = csv_file.stem.replace("_test", "").replace("_", " ")
        with open(csv_file) as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                question = row[0]
                choices = row[1:5]
                for choice in choices:
                    texts.append(
                        f"The following is a question about {subject}. "
                        f"Question: {question} Answer: {choice}"
                    )
    return texts


TASK_LOADERS = {
    "hellaswag": load_hellaswag,
    "arc_challenge": lambda d: load_arc(d, "ARC-c"),
    "arc_easy": lambda d: load_arc(d, "ARC-e"),
    "piqa": load_piqa,
    "siqa": load_siqa,
    "winogrande": load_winogrande,
    "commonsenseqa": load_commonsenseqa,
    "openbookqa": load_openbookqa,
    "mmlu": load_mmlu,
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def load_encoder(model_name: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
    model.to(device).eval()
    return tokenizer, model


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Encode a list of texts -> [N, D] L2-normalised embedding tensor."""
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embs = outputs.pooler_output
    else:
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        embs = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    return F.normalize(embs.float(), dim=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def embed_benchmarks(
    data_dir: str,
    encoder_model: str,
    output_path: str,
    tasks: List[str] | None = None,
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> None:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise SystemExit(f"Unsupported dtype '{dtype}'. Choose from {sorted(dtype_map.keys())}.")

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but no GPU available, falling back to CPU.")
        device = "cpu"
        torch_dtype = torch.float32

    if tasks is None:
        tasks = list(TASK_LOADERS.keys())

    # Collect texts from all benchmark tasks
    all_texts: List[str] = []
    task_counts: Dict[str, int] = {}
    for task_name in tasks:
        loader = TASK_LOADERS.get(task_name)
        if loader is None:
            print(f"[skip] Unknown task '{task_name}'")
            continue
        task_texts = loader(data_dir)
        task_counts[task_name] = len(task_texts)
        all_texts.extend(task_texts)

    if not all_texts:
        raise SystemExit("No benchmark texts extracted. Check --data_dir and task data files.")

    print(f"\nBenchmark text extraction summary:")
    for name, count in task_counts.items():
        print(f"  {name:20s}: {count:>6d} texts")
    print(f"  {'TOTAL':20s}: {len(all_texts):>6d} texts\n")

    # Encode
    print(f"Loading encoder: {encoder_model}")
    tokenizer, model = load_encoder(encoder_model, device, torch_dtype)

    all_embeddings = []
    for start in tqdm(range(0, len(all_texts), batch_size), desc="Encoding benchmarks"):
        batch = all_texts[start : start + batch_size]
        embs = encode_texts(batch, tokenizer, model, device, max_length)
        all_embeddings.append(embs.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, D]
    print(f"Final embeddings shape: {embeddings.shape}")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output), embeddings=embeddings)
    print(f"Saved to {output}  ({os.path.getsize(output) / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate benchmark embeddings for OPUS proxy construction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory containing benchmark data (e.g. opencompass/data).")
    p.add_argument("--encoder_model", type=str,
                   default="Snowflake/snowflake-arctic-embed-l-v2.0",
                   help="HuggingFace model name or local path for the sentence encoder.")
    p.add_argument("--output", type=str,
                   default="benchmark_embeddings_arctic_l_v2.npz",
                   help="Output .npz file path.")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   choices=list(TASK_LOADERS.keys()),
                   help="Subset of tasks to embed (default: all).")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Encoding batch size.")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length for the tokenizer.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device (cuda or cpu).")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   help="Model dtype: float32 | bfloat16 | float16.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed_benchmarks(
        data_dir=args.data_dir,
        encoder_model=args.encoder_model,
        output_path=args.output,
        tasks=args.tasks,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
    )
