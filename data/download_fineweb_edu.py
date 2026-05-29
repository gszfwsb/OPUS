"""
Download FineWeb-Edu in local batches (no remote streaming), separate by scores
(3/4/5), enforce exact per-score token caps, and resume without duplication.

Per run:
- List repo files (parquet) from Hugging Face
- Download next N unprocessed parquet files to a local temp dir
- Process locally: tokenize with tiktoken (len + 1 EOT), write gz JSONL per score
- Update manifest with processed files and per-score token totals
- Optionally delete local parquet to save disk
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_TIKTOKEN_CACHE_DIR = REPO_ROOT / ".cache" / "tiktoken"
HF_CACHE_ROOT = Path(os.environ.get("HF_CACHE_ROOT", DEFAULT_HF_CACHE_ROOT))
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEFAULT_TIKTOKEN_CACHE_DIR))
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
# Prefer Rust-based downloader for lower memory usage
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")


import gzip
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set

import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import urllib.request
import urllib.error
import time


DEFAULT_REPO_ID = "HuggingFaceFW/fineweb-edu"


@dataclass
class BucketStats:
    total_tokens: int = 0
    total_docs: int = 0
    shards: int = 0


class DocShardWriter:
    def __init__(self, out_dir: Path, shard_docs: int, start_index: int = 0, compresslevel: int = 1, fsync_every_docs: int = 0) -> None:
        self.out_dir = out_dir
        self.shard_docs = int(max(1, shard_docs))
        self.current_docs = 0
        self.current_idx = int(max(0, start_index))
        self.compresslevel = int(max(0, min(9, compresslevel)))
        self.fsync_every_docs = int(max(0, fsync_every_docs))
        self.fh: Optional[gzip.GzipFile] = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_new(self) -> None:
        if self.fh is not None:
            self.fh.close()
        fname = self.out_dir / f"part-{self.current_idx:06d}.jsonl.gz"
        self.fh = gzip.open(fname, mode="at", compresslevel=self.compresslevel, encoding="utf-8")
        self.current_docs = 0
        self.current_idx += 1

    def _fsync(self) -> None:
        try:
            if self.fh is None:
                return
            # Flush Python buffers
            self.fh.flush()
            # Try to reach the underlying raw file descriptor and fsync it
            # gzip.open(..., encoding=...) returns a TextIOWrapper → GzipFile → fileobj
            raw = getattr(self.fh, "buffer", None)
            fileobj = getattr(raw, "fileobj", None)
            if hasattr(raw, "flush"):
                try:
                    raw.flush()
                except Exception:
                    pass
            if hasattr(fileobj, "flush"):
                try:
                    fileobj.flush()
                except Exception:
                    pass
            if hasattr(fileobj, "fileno"):
                import os as _os
                try:
                    _os.fsync(fileobj.fileno())
                except Exception:
                    pass
        except Exception:
            # Best-effort only
            pass

    def write(self, record: dict) -> None:
        if self.fh is None:
            self._open_new()
        if self.current_docs >= self.shard_docs:
            self._open_new()
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.current_docs += 1
        if self.fsync_every_docs and (self.current_docs % self.fsync_every_docs == 0):
            self._fsync()

    def close(self) -> None:
        if self.fh is not None:
            # Ensure data is flushed to disk (important on GPFS)
            self._fsync()
            self.fh.close()
            self.fh = None


def clamp_score(raw: object) -> Optional[int]:
    try:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            s = int(round(float(raw)))
        else:
            s = int(round(float(str(raw))))
        if 1 <= s <= 5:
            return s
        return None
    except Exception:
        return None


def load_manifest(path: Path, wanted_scores: List[int]) -> tuple[Dict[int, BucketStats], Set[str], Set[str], Dict[str, dict]]:
    buckets: Dict[int, BucketStats] = {s: BucketStats() for s in wanted_scores}
    processed_files: Set[str] = set()
    in_progress_files: Set[str] = set()
    preserved_buckets: Dict[str, dict] = {}  # Preserve buckets for scores not in wanted_scores
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            prev_buckets = data.get("buckets", {})
            # Load all previous buckets for preservation
            for key, b in prev_buckets.items():
                try:
                    score_int = int(key)
                    if score_int in wanted_scores:
                        # Load into active buckets
                        buckets[score_int].total_tokens = int(b.get("total_tokens", 0))
                        buckets[score_int].total_docs = int(b.get("total_docs", 0))
                        buckets[score_int].shards = int(b.get("shards", 0))
                    else:
                        # Preserve for scores not being processed
                        preserved_buckets[key] = b
                except (ValueError, TypeError):
                    # Invalid score key, preserve as-is
                    preserved_buckets[key] = b
            processed_files = set(data.get("processed_files", []))
            in_progress_files = set(data.get("in_progress_files", []))
        except Exception:
            pass
    return buckets, processed_files, in_progress_files, preserved_buckets


def save_manifest(path: Path, buckets: Dict[int, BucketStats], processed_files: Set[str], in_progress_files: Set[str] = None, preserved_buckets: Dict[str, dict] = None) -> None:
    # Merge active buckets with preserved ones
    all_buckets = {}
    if preserved_buckets:
        all_buckets.update(preserved_buckets)  # Start with preserved buckets
    # Overlay with active buckets (overwrites if same score)
    all_buckets.update({str(s): asdict(b) for s, b in buckets.items()})
    
    out = {
        "buckets": all_buckets,
        "processed_files": sorted(list(processed_files)),
        "in_progress_files": sorted(list(in_progress_files)) if in_progress_files is not None else [],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def get_start_indices(output_dir: Path, wanted_scores: List[int]) -> Dict[int, int]:
    starts: Dict[int, int] = {}
    for s in wanted_scores:
        d = output_dir / str(s)
        if d.exists():
            existing = sorted([p for p in d.glob("part-*.jsonl.gz")])
            if existing:
                last = existing[-1].stem
                try:
                    starts[s] = int(last.split("-")[1]) + 1
                except Exception:
                    starts[s] = len(existing)
            else:
                starts[s] = 0
        else:
            starts[s] = 0
    return starts


def main() -> None:
    p = argparse.ArgumentParser(description="Local-batch FineWeb-Edu downloader and scorer with resume")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional: sample config name like sample-10BT/sample-100BT (filters to sample/<N>BT/).",
    )
    p.add_argument(
        "--repo_prefix",
        type=str,
        default=None,
        help="Optional: filter parquet files to this repo path prefix (e.g. sample/100BT/). Overrides --config.",
    )
    p.add_argument("--output_dir", type=str, default="finewebedu")
    p.add_argument("--local_dir", type=str, default="fineweb_caching")
    p.add_argument("--batch_files", type=int, default=50)
    p.add_argument("--scores", type=str, default="4,5")
    p.add_argument("--score_columns", type=str, default="score,edu_score")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--target_tokens_b", type=float, default=30.0)
    p.add_argument("--session_tokens_b", type=float, default=0.0)
    p.add_argument("--shard_docs", type=int, default=1_000_000)
    p.add_argument("--compresslevel", type=int, default=1)
    p.add_argument("--delete_after", action="store_true")
    p.add_argument("--skip_train_splits", type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=False,
                   help="Skip large default/train-*.parquet shards to avoid stalls")
    p.add_argument("--reader", type=str, choices=["datasets", "pyarrow"], default="datasets",
                   help="Row reader to iterate parquet: 'datasets' (default) or 'pyarrow' for lower memory")
    p.add_argument("--arrow_batch_rows", type=int, default=8192,
                   help="Row batch size when --reader pyarrow is used")
    p.add_argument("--continuous", action="store_true",
                   help="If set, keep fetching next batches until targets are met or files exhausted")
    p.add_argument("--stage_dir", type=str, default=None,
                   help="Optional: stage HF downloads on fast local disk (e.g., /tmp or local NVMe) to avoid GPFS writeback pressure; files are deleted after processing")
    p.add_argument("--fsync_every_docs", type=int, default=0,
                   help="If > 0, fsync output file every N docs to bound dirty cache on network FS")
    p.add_argument("--safe_download", action="store_true",
                   help="Use internal streaming HTTP downloader with small chunks and periodic fsync to bound GPFS dirty cache")
    p.add_argument("--download_chunk_kb", type=int, default=256,
                   help="Chunk size (KB) for --safe_download streaming")
    p.add_argument("--download_fsync_bytes_mb", type=int, default=4,
                   help="Fsync every N MB during --safe_download streaming; 0 to disable until end")
    p.add_argument("--download_sleep_ms", type=int, default=0,
                   help="Optional sleep between chunks during --safe_download to further smooth writeback")
    p.add_argument("--sync_between_files", action="store_true",
                   help="Call os.sync() after each file is processed/deleted to ensure writeback before next download")
    args = p.parse_args()

    repo_id = args.repo_id
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    local_dir = Path(args.local_dir); local_dir.mkdir(parents=True, exist_ok=True)
    stage_dir: Optional[Path] = None
    if args.stage_dir:
        stage_dir = Path(args.stage_dir); stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "local_batches_manifest.json"

    enc = tiktoken.get_encoding("gpt2")
    wanted_scores: List[int] = [int(s.strip()) for s in args.scores.split(",") if s.strip()]
    score_cols: List[str] = [c.strip() for c in args.score_columns.split(",") if c.strip()]

    buckets, processed_files, in_progress_files, preserved_buckets = load_manifest(manifest_path, wanted_scores)
    # Skip any files that were in-progress from a previous crashed run
    if in_progress_files:
        print(f"Warning: Found {len(in_progress_files)} in-progress files from previous run. Skipping to avoid duplication:")
        for f in sorted(in_progress_files):
            print(f"  - {f}")
        processed_files.update(in_progress_files)
        in_progress_files.clear()
    start_indices = get_start_indices(out_root, wanted_scores)
    # Choose a conservative fsync cadence on network FS to limit dirty cache growth
    fsync_docs = int(args.fsync_every_docs) if (hasattr(args, "fsync_every_docs") and args.fsync_every_docs and args.fsync_every_docs > 0) else 0
    writers: Dict[int, DocShardWriter] = {
        s: DocShardWriter(out_root / str(s), shard_docs=args.shard_docs, start_index=start_indices[s], compresslevel=args.compresslevel, fsync_every_docs=fsync_docs)
        for s in wanted_scores
    }

    global_target = int(args.target_tokens_b * 1e9)
    session_limit = int(args.session_tokens_b * 1e9) if args.session_tokens_b and args.session_tokens_b > 0 else None
    allowed_targets: Dict[int, int] = {}
    for s in wanted_scores:
        remaining = max(0, global_target - buckets[s].total_tokens)
        allowed_targets[s] = buckets[s].total_tokens + (min(session_limit, remaining) if session_limit is not None else remaining)

    # List files from API - use mirror if available, bypassing HfApi to avoid huggingface.co
    print("Fetching file list from repository...")
    
    def fetch_files_via_api(endpoint: str, repo_id: str) -> List[str]:
        """Fetch file list directly via HTTP API, bypassing huggingface_hub library."""
        import urllib.request
        import urllib.error
        
        all_files = []
        cursor = None
        api_url_base = f"{endpoint}/api/datasets/{repo_id}/tree/main"
        
        while True:
            params = "?recursive=true&expand=false&limit=1000"
            if cursor:
                params += f"&cursor={cursor}"
            url = api_url_base + params
            
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.load(response)
                    
                    # Extract file paths
                    for item in data:
                        if isinstance(item, dict) and item.get("type") == "file":
                            path = item.get("path")
                            if path:
                                all_files.append(path)
                    
                    # Check for pagination
                    # The API returns a special marker or we check if we got fewer than limit
                    if len(data) < 1000:
                        break
                    # Try to get next cursor (implementation may vary)
                    cursor = None  # Simplified - API may not support pagination this way
                    break  # For now, just fetch first batch
                    
            except Exception as e:
                print(f"Warning: API call to {endpoint} failed: {e}")
                raise
        
        return all_files
    
    try:
        # Try mirror first if set
        if HF_ENDPOINT and HF_ENDPOINT != "https://huggingface.co":
            print(f"Using mirror: {HF_ENDPOINT}")
            files = fetch_files_via_api(HF_ENDPOINT, repo_id)
        else:
            # Fall back to official endpoint
            files = fetch_files_via_api("https://huggingface.co", repo_id)
        
        parquet_files = [f for f in files if f.endswith(".parquet")]
        parquet_files.sort()
        print(f"Found {len(parquet_files)} parquet files")
        
    except Exception as e:
        print(f"Failed to fetch file list: {e}")
        print("Falling back to HfApi...")
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        parquet_files.sort()
        print(f"Found {len(parquet_files)} parquet files")

    # Optional prefix filter (sample config or explicit repo prefix)
    repo_prefix = args.repo_prefix
    if not repo_prefix and args.config:
        m = re.match(r"^sample-(\d+)BT$", str(args.config).strip())
        if m:
            repo_prefix = f"sample/{m.group(1)}BT/"
        else:
            # Allow passing an actual prefix via --config as a convenience
            repo_prefix = str(args.config).strip()
    if repo_prefix:
        repo_prefix = str(repo_prefix).lstrip("/")
        if repo_prefix and not repo_prefix.endswith("/"):
            repo_prefix += "/"
        before = len(parquet_files)
        parquet_files = [f for f in parquet_files if f.startswith(repo_prefix)]
        parquet_files.sort()
        print(f"Repo prefix filter: {repo_prefix}  -> {len(parquet_files)}/{before} parquet files")
        if not parquet_files:
            raise SystemExit(f"No parquet files match repo_prefix={repo_prefix!r} (repo_id={repo_id})")

    def next_batch() -> List[str]:
        batch: List[str] = []
        skipped_processed = 0
        skipped_train = 0
        for f in parquet_files:
            if f in processed_files:
                skipped_processed += 1
                continue
            base = f.split("/")[-1]
            if args.skip_train_splits and base.startswith("train-"):
                skipped_train += 1
                continue
            batch.append(f)
            if len(batch) >= args.batch_files:
                break
        if not batch and skipped_processed > 0:
            print(f"Info: All {skipped_processed} unprocessed files were already processed")
            print(f"      Skipped {skipped_train} train-* files")
        return batch

    # Print summary before starting
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"{'='*60}")
    print(f"Total parquet files found: {len(parquet_files)}")
    print(f"Already processed files: {len(processed_files)}")
    print(f"Remaining unprocessed: {len([f for f in parquet_files if f not in processed_files])}")
    for s in wanted_scores:
        current = buckets[s].total_tokens / 1e9
        target = allowed_targets[s] / 1e9
        remaining = (allowed_targets[s] - buckets[s].total_tokens) / 1e9
        print(f"Score {s}: {current:.1f}B / {target:.1f}B tokens ({remaining:.1f}B remaining)")
    print(f"{'='*60}\n")
    
    pbars = {s: tqdm(total=allowed_targets[s], initial=buckets[s].total_tokens, unit="tok", unit_scale=True, desc=f"score {s}") for s in wanted_scores}

    # Helper to process a single document given its score and text
    def process_example(score_value: int, text_value: str) -> None:
        if score_value not in wanted_scores:
            return
        if buckets[score_value].total_tokens >= allowed_targets[score_value]:
            return
        if not isinstance(text_value, str) or not text_value:
            return
        tok_ids = enc.encode_ordinary(text_value)
        doc_tokens = len(tok_ids) + 1

        if buckets[score_value].total_tokens + doc_tokens <= allowed_targets[score_value]:
            rec = {"text": text_value, "score": int(score_value)}
            writers[score_value].write(rec)
            buckets[score_value].total_docs += 1
            buckets[score_value].total_tokens += doc_tokens
            pbars[score_value].update(doc_tokens)
        else:
            remaining = allowed_targets[score_value] - buckets[score_value].total_tokens
            allow = remaining - 1
            if allow >= 0:
                trimmed = tok_ids[:allow] if allow > 0 else []
                trimmed_text = enc.decode(trimmed) if trimmed else ""
                rec = {"text": trimmed_text, "score": int(score_value)}
                writers[score_value].write(rec)
                buckets[score_value].total_docs += 1
                buckets[score_value].total_tokens += (len(trimmed) + 1)
                pbars[score_value].update(len(trimmed) + 1)

    while True:
        if all(buckets[s].total_tokens >= allowed_targets[s] for s in wanted_scores):
            break
        to_fetch = next_batch()
        if not to_fetch:
            break
        # Process the current batch
        for repo_path in to_fetch:
            if all(buckets[s].total_tokens >= allowed_targets[s] for s in wanted_scores):
                break

            # Mark file as in-progress to avoid duplication on crash
            in_progress_files.add(repo_path)
            save_manifest(manifest_path, buckets, processed_files, in_progress_files, preserved_buckets)

            # Download to stage_dir if provided, otherwise to local_dir
            dl_dir = str(stage_dir) if stage_dir is not None else str(local_dir)
            # Ensure subdirectories exist to avoid name collisions
            local_file_path = Path(dl_dir) / repo_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            def _safe_stream_download(dst_path: Path) -> str:
                endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                # Resolve path under main; works for public dataset files listed by list_repo_files
                url = f"{endpoint}/datasets/{repo_id}/resolve/main/{repo_path}"
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                headers = {}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(url, headers=headers)
                chunk_size = max(32 * 1024, int(args.download_chunk_kb) * 1024)
                fsync_threshold = max(0, int(args.download_fsync_bytes_mb)) * 1024 * 1024
                sleep_secs = (int(args.download_sleep_ms) / 1000.0) if (hasattr(args, "download_sleep_ms") and args.download_sleep_ms and args.download_sleep_ms > 0) else 0.0
                with urllib.request.urlopen(req, timeout=120) as resp, open(dst_path, "wb") as out_f:
                    bytes_since_fsync = 0
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        bytes_since_fsync += len(chunk)
                        if fsync_threshold and bytes_since_fsync >= fsync_threshold:
                            out_f.flush()
                            try:
                                os.fsync(out_f.fileno())
                            except Exception:
                                pass
                            bytes_since_fsync = 0
                        if sleep_secs:
                            time.sleep(sleep_secs)
                    out_f.flush()
                    try:
                        os.fsync(out_f.fileno())
                    except Exception:
                        pass
                return str(dst_path)

            # hf_hub_download uses HF_ENDPOINT environment variable automatically
            if args.safe_download:
                try:
                    local_file = _safe_stream_download(local_file_path)
                except Exception:
                    # Fallback to hub downloader on failure
                    local_file = hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=repo_path,
                        local_dir=dl_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )
            else:
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=repo_path,
                    local_dir=dl_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

            if args.reader == "datasets":
                ds = load_dataset("parquet", data_files=local_file, split="train", streaming=True)
                for ex in ds:
                    s_value: Optional[int] = None
                    for col in score_cols:
                        if col in ex:
                            s_value = clamp_score(ex[col])
                            if s_value is not None:
                                break
                    if s_value is None:
                        continue
                    if all(buckets[x].total_tokens >= allowed_targets[x] for x in wanted_scores):
                        break
                    process_example(s_value, ex.get(args.text_column))
            else:
                # Low-memory pyarrow reader without pandas conversion
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(local_file)
                cols_to_read = [args.text_column] + [c for c in score_cols if c != args.text_column]
                batch_size = max(1024, int(args.arrow_batch_rows))
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols_to_read):
                    schema_names = list(batch.schema.names)
                    arrays_by_name = {name: batch.column(i) for i, name in enumerate(schema_names)}

                    text_arr = arrays_by_name.get(args.text_column)
                    if text_arr is None:
                        continue

                    for i in range(batch.num_rows):
                        # Find score value from first available score column
                        s_value: Optional[int] = None
                        for col in score_cols:
                            arr = arrays_by_name.get(col)
                            if arr is None:
                                continue
                            try:
                                scalar = arr[i]
                                val = scalar.as_py()
                            except Exception:
                                val = None
                            s_value = clamp_score(val)
                            if s_value is not None:
                                break
                        if s_value is None:
                            continue

                        try:
                            text_scalar = text_arr[i]
                            text_value = text_scalar.as_py()
                            if isinstance(text_value, (bytes, bytearray)):
                                text_value = text_value.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                        if text_value is None:
                            continue

                        if all(buckets[x].total_tokens >= allowed_targets[x] for x in wanted_scores):
                            break
                        process_example(s_value, text_value)
                    if all(buckets[x].total_tokens >= allowed_targets[x] for x in wanted_scores):
                        break

            # Mark file as completed and remove from in-progress
            processed_files.add(repo_path)
            in_progress_files.discard(repo_path)
            save_manifest(manifest_path, buckets, processed_files, in_progress_files, preserved_buckets)
            # Always delete staged file; otherwise honor --delete_after
            if stage_dir is not None or args.delete_after:
                try:
                    os.remove(local_file)
                except Exception:
                    pass
            if args.sync_between_files:
                try:
                    os.sync()
                except Exception:
                    pass

        if not args.continuous:
            break

    for s in wanted_scores:
        writers[s].close()
        pbars[s].close()

    for s in wanted_scores:
        out_dir = out_root / str(s)
        existing = sorted([p for p in out_dir.glob("part-*.jsonl.gz")])
        buckets[s].shards = len(existing)
    save_manifest(manifest_path, buckets, processed_files, in_progress_files, preserved_buckets)


if __name__ == "__main__":
    main()

