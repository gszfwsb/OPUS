"""
Download HuggingFaceFW/fineweb dataset in parquet format with resume support.

Features:
- Direct parquet download (no processing)
- Manifest-based resume for network interruptions
- Uses HF mirror for fast downloads in China
- Target: 200B tokens (configurable)
- Prevents duplicate downloads on interruption
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_ROOT = Path(os.environ.get("HF_CACHE_ROOT", DEFAULT_HF_CACHE_ROOT))
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")

import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Set

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DEFAULT_REPO_ID = "HuggingFaceFW/fineweb"


@dataclass
class DownloadStats:
    total_files: int = 0
    total_bytes: int = 0
    failed_files: int = 0


def load_manifest(path: Path) -> tuple[DownloadStats, Set[str], Set[str]]:
    """Load download progress from manifest."""
    stats = DownloadStats()
    completed_files: Set[str] = set()
    in_progress_files: Set[str] = set()
    
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            stats.total_files = int(data.get("total_files", 0))
            stats.total_bytes = int(data.get("total_bytes", 0))
            stats.failed_files = int(data.get("failed_files", 0))
            completed_files = set(data.get("completed_files", []))
            in_progress_files = set(data.get("in_progress_files", []))
        except Exception as e:
            print(f"Warning: Failed to load manifest: {e}")
    
    return stats, completed_files, in_progress_files


def save_manifest(
    path: Path,
    stats: DownloadStats,
    completed_files: Set[str],
    in_progress_files: Set[str]
) -> None:
    """Save download progress to manifest."""
    out = {
        "total_files": stats.total_files,
        "total_bytes": stats.total_bytes,
        "failed_files": stats.failed_files,
        "completed_files": sorted(list(completed_files)),
        "in_progress_files": sorted(list(in_progress_files)),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def fetch_files_via_api(endpoint: str, repo_id: str, branch: str = "main") -> List[str]:
    """Fetch file list directly via HTTP API with proper pagination."""
    all_files = []
    api_url_base = f"{endpoint}/api/datasets/{repo_id}/tree/{branch}"
    
    try:
        req = urllib.request.Request(
            api_url_base + "?recursive=true&expand=false", 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.load(response)
            for item in data:
                if isinstance(item, dict) and item.get("type") == "file":
                    path = item.get("path")
                    if path:
                        all_files.append(path)
                
    except Exception as e:
        print(f"Warning: API call to {endpoint} failed: {e}")
        raise
    
    return all_files


def safe_stream_download(
    dst_path: Path,
    url: str,
    chunk_kb: int = 512,
    fsync_mb: int = 16
) -> str:
    """Stream download with periodic fsync to limit dirty cache."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, headers=headers)
    chunk_size = max(32 * 1024, chunk_kb * 1024)
    fsync_threshold = max(0, fsync_mb) * 1024 * 1024
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    with urllib.request.urlopen(req, timeout=120) as resp, open(dst_path, "wb") as out_f:
        # Get file size from headers if available
        content_length = resp.headers.get('Content-Length')
        total_size = int(content_length) if content_length else None
        
        bytes_since_fsync = 0
        bytes_downloaded = 0
        
        # Progress bar for this file
        pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=f"  {dst_path.name}",
            leave=False
        )
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            bytes_since_fsync += len(chunk)
            bytes_downloaded += len(chunk)
            pbar.update(len(chunk))
            
            if fsync_threshold and bytes_since_fsync >= fsync_threshold:
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except Exception:
                    pass
                bytes_since_fsync = 0
        
        pbar.close()
        
        out_f.flush()
        try:
            os.fsync(out_f.fileno())
        except Exception:
            pass
    
    return str(dst_path)


def estimate_tokens_from_bytes(bytes_size: int) -> int:
    """
    Rough estimate: FineWeb parquet files contain ~1.5 tokens per byte (compressed).
    This is just for progress tracking, not exact.
    """
    return int(bytes_size * 1.5)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download HuggingFaceFW/fineweb parquet with resume support"
    )
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument(
        "--output_dir",
        type=str,
        default="fineweb_200B_parquet",
        help="Directory to store downloaded parquet files and the manifest.",
    )
    p.add_argument("--batch_files", type=int, default=10,
                   help="Number of files to download per batch")
    p.add_argument("--target_tokens_b", type=float, default=200.0,
                   help="Target tokens in billions (default: 200B, approximate based on file sizes)")
    p.add_argument("--max_files", type=int, default=None,
                   help="Maximum number of files to download (overrides target_tokens_b)")
    p.add_argument("--continuous", action="store_true",
                   help="Keep downloading until target is met")
    p.add_argument("--safe_download", action="store_true", default=True,
                   help="Use streaming HTTP downloader with fsync (default: True)")
    p.add_argument("--download_chunk_kb", type=int, default=512)
    p.add_argument("--download_fsync_mb", type=int, default=16)
    p.add_argument("--subset", type=str, default="default",
                   help="FineWeb subset to download (default: 'default', or 'CC-MAIN-*')")
    p.add_argument("--skip_in_progress", action="store_true",
                   help="Skip files marked as in-progress (useful after crashes)")
    
    args = p.parse_args()

    repo_id = args.repo_id
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_root / "download_manifest.json"
    
    # Load progress
    stats, completed_files, in_progress_files = load_manifest(manifest_path)
    
    # Skip files that were in-progress from crashed run
    if args.skip_in_progress and in_progress_files:
        print(f"Skipping {len(in_progress_files)} in-progress files from previous run")
        completed_files.update(in_progress_files)
        in_progress_files.clear()
        save_manifest(manifest_path, stats, completed_files, in_progress_files)
    
    # Calculate targets
    if args.max_files:
        target_files = args.max_files
        target_description = f"{target_files} files"
    else:
        # Estimate based on target tokens (rough: ~1B tokens per file for FineWeb)
        target_files = int(args.target_tokens_b)
        target_description = f"{args.target_tokens_b:.0f}B tokens (~{target_files} files)"
    
    print("Fetching file list from repository...")
    
    parquet_files = []
    if HF_ENDPOINT and HF_ENDPOINT != "https://huggingface.co":
        print(f"   Using mirror: {HF_ENDPOINT}")
        # Try direct HTTP API to mirror first (most reliable)
        for branch in ["main", "master"]:
            try:
                print(f"   Trying branch: {branch}")
                files = fetch_files_via_api(HF_ENDPOINT, repo_id, branch=branch)
                
                # Filter for parquet files and subset if specified
                parquet_files = [f for f in files if f.endswith(".parquet")]
                
                # Filter by subset if not 'default'
                if args.subset != "default":
                    parquet_files = [f for f in parquet_files if args.subset in f]
                
                if parquet_files:
                    parquet_files.sort()
                    print(f"  Found {len(parquet_files)} parquet files (branch: {branch})")
                    break
                else:
                    print(f"   No parquet files found in branch: {branch}")
            except Exception as e:
                print(f"   Branch {branch} failed: {e}")
        
        if not parquet_files:
            raise RuntimeError("Cannot find parquet files from mirror. Check repo structure.")
    else:
        print("  No mirror configured, trying huggingface.co")
        for branch in ["main", "master"]:
            try:
                print(f"   Trying branch: {branch}")
                files = fetch_files_via_api("https://huggingface.co", repo_id, branch=branch)
                parquet_files = [f for f in files if f.endswith(".parquet")]
                
                if args.subset != "default":
                    parquet_files = [f for f in parquet_files if args.subset in f]
                
                if parquet_files:
                    parquet_files.sort()
                    print(f"  Found {len(parquet_files)} parquet files (branch: {branch})")
                    break
            except Exception as e:
                print(f"   Branch {branch} failed: {e}")
        
        if not parquet_files:
            print("  HTTP API failed, trying HfApi...")
            try:
                api = HfApi()
                files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                parquet_files = [f for f in files if f.endswith(".parquet")]
                
                if args.subset != "default":
                    parquet_files = [f for f in parquet_files if args.subset in f]
                
                parquet_files.sort()
                print(f"  Found {len(parquet_files)} parquet files via HfApi")
            except Exception as e2:
                print(f"  HfApi also failed: {e2}")
                raise RuntimeError("Cannot connect to download server.")
    
    print(f"\n{'='*60}")
    print(f"FineWeb Parquet Download")
    print(f"{'='*60}")
    print(f"Total files:   {len(parquet_files)}")
    print(f"Completed:     {len(completed_files)}")
    print(f"Remaining:     {len([f for f in parquet_files if f not in completed_files])}")
    print(f"Progress:      {stats.total_files} files, {stats.total_bytes/1e9:.2f} GB")
    print(f"Target:        {target_description}")
    print(f"{'='*60}\n")
    
    # Process files
    def next_batch() -> List[str]:
        batch = []
        for f in parquet_files:
            if f in completed_files:
                continue
            batch.append(f)
            if len(batch) >= args.batch_files:
                break
        return batch
    
    try:
        while True:
            if stats.total_files >= target_files:
                print(f"\nTarget reached: {stats.total_files} files")
                break
            
            to_fetch = next_batch()
            if not to_fetch:
                print(f"\nNo more files to download")
                break
            
            print(f"\nDownloading batch of {len(to_fetch)} files...")
            
            for repo_path in to_fetch:
                if stats.total_files >= target_files:
                    break
                
                # Mark as in-progress
                in_progress_files.add(repo_path)
                save_manifest(manifest_path, stats, completed_files, in_progress_files)
                
                # Prepare output path (preserve directory structure)
                local_file_path = out_root / repo_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"\n[{stats.total_files + 1}/{target_files if args.max_files else '?'}] {repo_path}")
                
                # Download file
                try:
                    if args.safe_download:
                        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                        url = f"{endpoint}/datasets/{repo_id}/resolve/main/{repo_path}"
                        local_file = safe_stream_download(
                            local_file_path,
                            url,
                            chunk_kb=args.download_chunk_kb,
                            fsync_mb=args.download_fsync_mb
                        )
                    else:
                        local_file = hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=repo_path,
                            local_dir=str(out_root),
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                    
                    # Get file size
                    file_size = os.path.getsize(local_file)
                    stats.total_bytes += file_size
                    
                    print(f"  Downloaded: {file_size/1e6:.1f} MB")
                    
                    # Mark as completed
                    completed_files.add(repo_path)
                    in_progress_files.discard(repo_path)
                    stats.total_files += 1
                    save_manifest(manifest_path, stats, completed_files, in_progress_files)
                    
                except Exception as e:
                    print(f"  Failed: {e}")
                    stats.failed_files += 1
                    in_progress_files.discard(repo_path)
                    save_manifest(manifest_path, stats, completed_files, in_progress_files)
                    
                    # Continue with next file instead of crashing
                    continue
            
            if not args.continuous:
                break
    
    except KeyboardInterrupt:
        print(f"\nDownload interrupted by user")
    finally:
        # Final manifest save
        save_manifest(manifest_path, stats, completed_files, in_progress_files)
    
    print(f"\n{'='*60}")
    print(f"Download complete: {stats.total_files} files, {stats.total_bytes/1e9:.2f} GB")
    if stats.failed_files:
        print(f"Failed: {stats.failed_files} files")
    print(f"Output: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
