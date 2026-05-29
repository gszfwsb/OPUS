#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B"
DEFAULT_OUT_DIR = "./models/Qwen2.5-7B"
DEFAULT_HF_HOME = None

ALLOW_PATTERNS = [
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "*.tiktoken",
    "*.model",
    "config.json",
    "generation_config.json",
    "chat_template*",
]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", None))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    _ensure_dir(out_dir)

    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser()
        _ensure_dir(hf_home)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

    # Prefer hub snapshot download (doesn't require tokenizer deps to instantiate).
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[ERROR] import huggingface_hub failed: {e}", file=sys.stderr)
        print("Install one of:", file=sys.stderr)
        print("  pip install -U huggingface_hub", file=sys.stderr)
        print("  pip install -U transformers", file=sys.stderr)
        return 2

    kwargs = dict(
        repo_id=args.model_id,
        local_dir=str(out_dir),
        allow_patterns=ALLOW_PATTERNS,
        revision=args.revision,
        token=args.token,
    )

    # Compatibility across hub versions
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)

    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    manifest = {
        "model_id": args.model_id,
        "revision": args.revision,
        "out_dir": str(out_dir),
        "files": files,
    }
    (out_dir / "_tokenizer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    print("[OK] Tokenizer files downloaded to:", out_dir)
    for f in files:
        print(" -", f)
    print("[OK] Wrote manifest:", out_dir / "_tokenizer_manifest.json")

    # Optional quick offline load check (only if transformers is available)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(out_dir, local_files_only=True)
        _ = tok.encode("hello world")
        print("[OK] Offline AutoTokenizer load test passed.")
    except Exception as e:
        print("[WARN] Offline load test skipped/failed (transformers not available or deps missing):", e)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())