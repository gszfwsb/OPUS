"""
Build an OPUS proxy .bin shard from raw benchmark validation sets.

Tokenizes the evaluation text (context + each choice) from all target benchmarks
into a GPT-NeoX-compatible .bin file, suitable for use as an OPUS proxy.

Supported benchmarks:
  MMLU, ANLI (R1+R2+R3), HellaSwag, PIQA, SIQA, WinoGrande, ARC-Easy,
  ARC-Challenge, CommonsenseQA, WSC

The text representation for each benchmark mirrors the format used during
log-likelihood evaluation (see eval/run_lighteval_offline.py), so the proxy
gradient is maximally aligned with actual eval-time loss.

Example usage:

  python -u data/build_benchmark_proxy.py \
    --data_dir opencompass/data \
    --output_dir ./bins/fineweb_benchmark_proxy \
    --tasks all
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

if "TIKTOKEN_CACHE_DIR" not in os.environ:
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["TIKTOKEN_CACHE_DIR"] = str(repo_root / ".cache" / "tiktoken")

import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gpt2tokenize import StreamingBinWriter


# ---------------------------------------------------------------------------
# Benchmark text loaders
#
# Each loader returns List[str] — one string per (context, choice) pair,
# matching the text the model scores during evaluation.
# ---------------------------------------------------------------------------

def preprocess_hellaswag(text: str) -> str:
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


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
                query = item.get("query", "")
                for choice in item.get("choices", []):
                    texts.append(f"{query} {choice}")
            else:
                ctx = item.get("ctx_a", "") + " " + item.get("ctx_b", "").capitalize()
                activity = item.get("activity_label", "")
                ctx = preprocess_hellaswag(activity + ": " + ctx)
                for ending in item.get("endings", []):
                    texts.append(f"{ctx} {preprocess_hellaswag(ending)}")
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


def load_anli(data_dir: Path) -> List[str]:
    """Load ANLI R1+R2+R3 dev sets. Text = premise + hypothesis."""
    anli_root = data_dir / "anli"
    if not anli_root.exists():
        print(f"[skip] ANLI not found at {anli_root}")
        return []
    texts = []
    for round_dir in sorted(anli_root.rglob("*/dev.jsonl")):
        with open(round_dir) as f:
            for line in f:
                item = json.loads(line)
                context = item.get("context", item.get("premise", ""))
                hypothesis = item.get("hypothesis", "")
                if context and hypothesis:
                    texts.append(f"{context} {hypothesis}")
    return texts


def load_wsc(data_dir: Path) -> List[str]:
    """Load WSC (SuperGLUE Winograd Schema Challenge). Text = full sentence."""
    path = data_dir / "SuperGLUE" / "WSC" / "val.jsonl"
    if not path.exists():
        print(f"[skip] WSC not found at {path}")
        return []
    texts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "").strip()
            if text:
                texts.append(text)
    return texts


TASK_LOADERS = {
    "mmlu": load_mmlu,
    "anli": load_anli,
    "hellaswag": load_hellaswag,
    "piqa": load_piqa,
    "siqa": load_siqa,
    "winogrande": load_winogrande,
    "arc_easy": lambda d: load_arc(d, "ARC-e"),
    "arc_challenge": lambda d: load_arc(d, "ARC-c"),
    "commonsenseqa": load_commonsenseqa,
    "wsc": load_wsc,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize benchmark validation sets into an OPUS proxy .bin shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing benchmark data (e.g. opencompass/data).",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for the generated proxy .bin shard(s).",
    )
    p.add_argument(
        "--tasks", type=str, nargs="+", default=["all"],
        help="Benchmark tasks to include. Use 'all' for all supported tasks.",
    )
    p.add_argument(
        "--tokens_per_shard", type=int, default=100_000_000,
        help="Max tokens per .bin shard (default large enough for a single shard).",
    )
    p.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat benchmark texts N times to increase proxy token count.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    if "all" in args.tasks:
        tasks = list(TASK_LOADERS.keys())
    else:
        tasks = args.tasks

    enc = tiktoken.get_encoding("gpt2")

    all_texts: list[str] = []
    task_counts: dict[str, int] = {}

    for task_name in tasks:
        loader = TASK_LOADERS.get(task_name)
        if loader is None:
            print(f"[skip] Unknown task '{task_name}', available: {list(TASK_LOADERS.keys())}")
            continue
        task_texts = loader(data_dir)
        task_counts[task_name] = len(task_texts)
        all_texts.extend(task_texts)

    if not all_texts:
        raise SystemExit("No benchmark texts extracted. Check --data_dir and data files.")

    print("\nBenchmark text extraction summary:")
    for name, count in task_counts.items():
        print(f"  {name:20s}: {count:>6d} texts")
    print(f"  {'TOTAL':20s}: {len(all_texts):>6d} texts")

    if args.repeat > 1:
        all_texts = all_texts * args.repeat
        print(f"  After {args.repeat}x repeat : {len(all_texts):>6d} texts")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = StreamingBinWriter(out_dir=output_dir, tokens_per_shard=args.tokens_per_shard)

    total_tokens = 0
    total_docs = 0

    for text in tqdm(all_texts, desc="Tokenizing benchmarks"):
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        if not tokens:
            continue
        writer.add_tokens(tokens)
        total_tokens += len(tokens) + 1  # +1 for EOT
        total_docs += 1

    stats = writer.finalize()
    final_tokens = stats["total_tokens"]

    print("\n======================================")
    print("Benchmark proxy build complete.")
    print(f"Documents written  : {total_docs}")
    print(f"Tokens generated   : {final_tokens:,}")
    print(f"Shards             : {stats['total_shards']}")
    print(f"Output directory   : {output_dir}")
    print("======================================")


if __name__ == "__main__":
    main()
