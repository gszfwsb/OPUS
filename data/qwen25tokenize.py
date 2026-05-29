"""
Convert FineWeb Parquet to OPUS-compatible .bin shards using Qwen2.5-7B tokenizer.

Output format matches what `OPUS/train.py` expects in `_load_data_shard`:
- Header: 256 * int32
  - header[0] = 20240520 (magic)
  - header[1] = 1        (version)
  - header[2] = num_tokens
  - header[3] = token_bytes (4 -> int32 payload)
  - header[4] = eod_token_id (tokenizer.eos_token_id)
- Payload: int32 tokens (token_bytes=4)

Important: Some OSS/FUSE mounts don't support "seek back and rewrite header" while writing.
This writer avoids random writes by buffering payload into a temp file and then writing the final
bin shard as "header + payload" in a single forward-only pass.
"""

from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
TOKEN_BYTES_INT32 = 4

DEFAULT_TOKENIZER_DIR = "Qwen/Qwen2.5-7B"
DEFAULT_HF_HOME = None
DEFAULT_TMP_DIR = "/tmp/qwen25tokenize"


def _iter_chunks(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield list(items[i : i + chunk_size])


def _setup_hf_cache_dir(hf_home: Optional[str]) -> None:
    if not hf_home:
        return
    hf_home_path = Path(hf_home).expanduser()
    hf_home_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home_path))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home_path / "hub"))


def _load_tokenizer(tokenizer_dir: str, trust_remote_code: bool, local_files_only: bool):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tok.eos_token_id is None:
        raise ValueError(f"Tokenizer at {tokenizer_dir} has eos_token_id=None; cannot build doc boundaries.")
    # We are building a raw token stream; avoid warnings for very long documents.
    try:
        tok.model_max_length = int(10**18)
    except Exception:
        pass
    return tok


def parquet_worker_qwen25(
    worker_id: int,
    files: List[Path],
    output_queue: Queue,
    tokenizer_dir: str,
    text_column: str,
    max_tokens: Optional[int] = None,
    val_queue: Optional[Queue] = None,
    val_tokens_per_file: int = 0,
    queue_batch_docs: int = 512,
    tokenize_batch_texts: int = 256,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"Token indices sequence length is longer than the specified maximum sequence length.*",
        )
    except Exception:
        pass

    tok = _load_tokenizer(tokenizer_dir, trust_remote_code=trust_remote_code, local_files_only=local_files_only)

    total_train_tokens = 0
    total_val_tokens = 0
    docs_processed = 0

    train_batch: List[tuple[List[int], int]] = []
    val_batch: List[tuple[List[int], int]] = []

    def flush_batches() -> None:
        nonlocal train_batch, val_batch
        if train_batch:
            output_queue.put(("batch", train_batch))
            train_batch = []
        if val_queue is not None and val_batch:
            val_queue.put(("batch", val_batch))
            val_batch = []

    for parquet_file in files:
        try:
            parquet_obj = pq.ParquetFile(str(parquet_file))
            file_val_tokens = 0

            for row_group_idx in range(parquet_obj.num_row_groups):
                table = parquet_obj.read_row_group(row_group_idx, columns=[text_column])
                if text_column not in table.column_names:
                    continue

                col = table.column(text_column)
                try:
                    texts_all = col.to_pylist()
                except Exception:
                    texts_all = [col[i].as_py() for i in range(len(col))]

                texts = [t for t in texts_all if isinstance(t, str) and t]
                if not texts:
                    continue

                for texts_chunk in _iter_chunks(texts, tokenize_batch_texts):
                    enc = tok(
                        texts_chunk,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    ids_list = enc["input_ids"]

                    for ids in ids_list:
                        if not ids:
                            continue

                        token_count = len(ids) + 1  # +EOD
                        docs_processed += 1

                        if val_queue is not None and file_val_tokens < val_tokens_per_file:
                            val_batch.append((ids, token_count))
                            file_val_tokens += token_count
                            total_val_tokens += token_count
                        else:
                            train_batch.append((ids, token_count))
                            total_train_tokens += token_count

                        if len(train_batch) >= queue_batch_docs or (val_queue is not None and len(val_batch) >= queue_batch_docs):
                            flush_batches()

                        if max_tokens is not None and total_train_tokens >= max_tokens:
                            flush_batches()
                            output_queue.put(("done", None))
                            if val_queue is not None:
                                val_queue.put(("done", None))
                            print(
                                f"Worker {worker_id}: Hit max_tokens. docs={docs_processed:,} "
                                f"train={total_train_tokens:,} val={total_val_tokens:,} tokens"
                            )
                            return

        except Exception as e:
            print(f"Worker {worker_id}: Error processing {parquet_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    flush_batches()
    output_queue.put(("done", None))
    if val_queue is not None:
        val_queue.put(("done", None))
    print(
        f"Worker {worker_id}: Finished. docs={docs_processed:,} "
        f"train={total_train_tokens:,} val={total_val_tokens:,} tokens"
    )


class StreamingBinWriterInt32:
    """
    Append-only shard writer compatible with OSS/FUSE.

    - Payload is appended to a temporary file (in tmp_dir).
    - On shard finalize, we write final shard file in out_dir as:
        header(with correct num_tokens) + payload_bytes
      and then rename into place.
    """

    def __init__(
        self,
        out_dir: Path,
        tokens_per_shard: int,
        eod_token_id: int,
        prefix: str,
        tmp_dir: Optional[Path] = None,
        copy_buf_bytes: int = 8 * 1024 * 1024,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_per_shard = int(tokens_per_shard)
        self.eod_token_id = int(eod_token_id)
        self.prefix = str(prefix)

        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else self.out_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.copy_buf_bytes = int(copy_buf_bytes)

        self._shard_idx = 0
        self._payload_f = None
        self._payload_path: Optional[Path] = None
        self._current_tokens = 0
        self._total_tokens = 0
        self._eod_bytes = np.asarray([self.eod_token_id], dtype=np.int32).tobytes()

    def _payload_tmp_path(self, shard_idx: int) -> Path:
        pid = os.getpid()
        return self.tmp_dir / f".{self.prefix}_{shard_idx:06d}.payload.{pid}"

    def _final_tmp_path(self, shard_idx: int) -> Path:
        pid = os.getpid()
        return self.out_dir / f".{self.prefix}_{shard_idx:06d}.bin.tmp.{pid}"

    def _final_path(self, shard_idx: int) -> Path:
        return self.out_dir / f"{self.prefix}_{shard_idx:06d}.bin"

    def _open_new_shard(self) -> None:
        self._current_tokens = 0
        self._payload_path = self._payload_tmp_path(self._shard_idx)
        self._payload_f = open(self._payload_path, "wb", buffering=1024 * 1024)

    def _finalize_shard(self) -> None:
        if self._payload_f is None or self._payload_path is None:
            return

        self._payload_f.flush()
        try:
            os.fsync(self._payload_f.fileno())
        except OSError:
            pass
        self._payload_f.close()
        self._payload_f = None

        payload_size = self._payload_path.stat().st_size
        expected_payload_size = int(self._current_tokens) * TOKEN_BYTES_INT32
        if payload_size != expected_payload_size:
            raise OSError(
                f"Payload size mismatch for shard {self._shard_idx}: "
                f"expected {expected_payload_size} bytes ({self._current_tokens} tokens) "
                f"but got {payload_size} bytes at {self._payload_path}"
            )

        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0] = MAGIC
        header[1] = VERSION
        header[2] = int(self._current_tokens)
        header[3] = TOKEN_BYTES_INT32
        header[4] = self.eod_token_id

        final_tmp = self._final_tmp_path(self._shard_idx)
        final_path = self._final_path(self._shard_idx)

        with open(final_tmp, "wb", buffering=1024 * 1024) as out_f:
            out_f.write(header.tobytes())
            with open(self._payload_path, "rb", buffering=1024 * 1024) as in_f:
                shutil.copyfileobj(in_f, out_f, length=self.copy_buf_bytes)
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass

        try:
            os.replace(final_tmp, final_path)
        except OSError:
            os.rename(final_tmp, final_path)

        try:
            self._payload_path.unlink(missing_ok=True)
        except TypeError:
            if self._payload_path.exists():
                self._payload_path.unlink()
        self._payload_path = None

        self._shard_idx += 1

    def add_tokens(self, tokens: List[int]) -> None:
        token_count = len(tokens) + 1
        if self._payload_f is None:
            self._open_new_shard()
        elif self._current_tokens > 0 and self._current_tokens + token_count > self.tokens_per_shard:
            self._finalize_shard()
            self._open_new_shard()

        assert self._payload_f is not None
        arr = np.asarray(tokens, dtype=np.int32)
        self._payload_f.write(arr.tobytes())
        self._payload_f.write(self._eod_bytes)

        self._current_tokens += token_count
        self._total_tokens += token_count

    def finalize(self) -> dict:
        if self._payload_f is not None and self._current_tokens > 0:
            self._finalize_shard()
        return {"total_shards": int(self._shard_idx), "total_tokens": int(self._total_tokens)}


def discover_parquet_files(parquet_dir: Path, recursive: bool) -> List[Path]:
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if (not files) and recursive:
        files = sorted(parquet_dir.rglob("*.parquet"))
    return files


def convert_parquets_to_bins(
    parquet_dir: Path,
    output_dir: Path,
    tokenizer_dir: str,
    tokens_per_shard: int = 100_000_000,
    num_workers: int = 32,
    max_tokens: Optional[int] = None,
    val_output_dir: Optional[Path] = None,
    val_tokens_total: int = 0,
    text_column: str = "text",
    recursive: bool = True,
    queue_batch_docs: int = 512,
    tokenize_batch_texts: int = 256,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    tmp_dir: Optional[Path] = None,
) -> dict:
    tok = _load_tokenizer(tokenizer_dir, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    eod_token_id = int(tok.eos_token_id)

    tmp_dir_eff = Path(tmp_dir) if tmp_dir is not None else Path(DEFAULT_TMP_DIR)
    tmp_dir_eff.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Parquet -> Qwen2.5 .bin (int32 payload)")
    print(f"{'='*70}")
    print(f"Input:         {parquet_dir}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Text column:   {text_column}")
    print(f"EOD token id:  {eod_token_id}")
    print(f"Train out:     {output_dir}")
    if val_output_dir:
        print(f"Val out:       {val_output_dir}")
        print(f"Val tokens:    {val_tokens_total:,}")
    print(f"Shard size:    {tokens_per_shard:,} tokens")
    print(f"Workers:       {num_workers}")
    print(f"Recursive:     {recursive}")
    print(f"Tmp dir:       {tmp_dir_eff}")
    if max_tokens is not None:
        print(f"Max tokens:    {max_tokens:,}")
    print(f"{'='*70}\n")

    print("Discovering parquet files...")
    parquet_files = discover_parquet_files(parquet_dir, recursive=recursive)
    print(f"  Found {len(parquet_files)} parquet files")
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir} (recursive={recursive})")

    num_workers_eff = max(1, min(int(num_workers), len(parquet_files)))
    if num_workers_eff != num_workers:
        print(f"  Adjusted workers: {num_workers} -> {num_workers_eff} (based on file count)")
    num_workers = num_workers_eff

    val_tokens_per_file = val_tokens_total // len(parquet_files) if val_tokens_total > 0 else 0
    if val_tokens_per_file > 0:
        print(f"Validation: ~{val_tokens_per_file:,} tokens per file (uniform prefix sampling)")

    worker_files: List[List[Path]] = [parquet_files[i::num_workers] for i in range(num_workers)]

    train_queue: Queue = Queue(maxsize=200)
    val_queue: Optional[Queue] = Queue(maxsize=200) if (val_tokens_per_file > 0 and val_output_dir is not None) else None

    worker_max_tokens = (max_tokens // num_workers) if max_tokens is not None else None

    print(f"Starting {num_workers} worker processes...")
    workers: List[Process] = []
    for wid in range(num_workers):
        p = Process(
            target=parquet_worker_qwen25,
            args=(
                wid,
                worker_files[wid],
                train_queue,
                tokenizer_dir,
                text_column,
                worker_max_tokens,
                val_queue,
                val_tokens_per_file,
                queue_batch_docs,
                tokenize_batch_texts,
                trust_remote_code,
                local_files_only,
            ),
        )
        p.start()
        workers.append(p)

    train_writer = StreamingBinWriterInt32(
        Path(output_dir),
        tokens_per_shard,
        eod_token_id=eod_token_id,
        prefix="train",
        tmp_dir=tmp_dir_eff / "train_payload",
    )
    val_writer = (
        StreamingBinWriterInt32(
            Path(val_output_dir),
            tokens_per_shard,
            eod_token_id=eod_token_id,
            prefix="val",
            tmp_dir=tmp_dir_eff / "val_payload",
        )
        if val_output_dir is not None and val_queue is not None
        else None
    )

    total_train_tokens = 0
    total_train_docs = 0
    total_val_tokens = 0
    total_val_docs = 0
    train_done = 0
    val_done = 0
    stop_early = False

    pbar_total = max_tokens if max_tokens is not None else None
    with tqdm(total=pbar_total, desc="Training tokens", unit="tok", unit_scale=True, position=0) as train_pbar, tqdm(
        total=val_tokens_total if val_tokens_total > 0 else None,
        desc="Validation tokens",
        unit="tok",
        unit_scale=True,
        position=1,
        disable=val_queue is None,
    ) as val_pbar:
        while train_done < num_workers or (val_queue is not None and val_done < num_workers):
            if train_done < num_workers:
                try:
                    tag, payload = train_queue.get(timeout=0.1)
                    if tag == "done":
                        train_done += 1
                    elif tag == "batch":
                        batch = payload
                        for tokens, token_count in batch:
                            train_writer.add_tokens(tokens)
                            total_train_tokens += token_count
                            total_train_docs += 1
                            train_pbar.update(token_count)
                        if max_tokens is not None and total_train_tokens >= max_tokens:
                            stop_early = True
                            for wp in workers:
                                wp.terminate()
                            break
                except Empty:
                    pass

            if val_queue is not None and val_done < num_workers:
                try:
                    tag, payload = val_queue.get(timeout=0.1)
                    if tag == "done":
                        val_done += 1
                    elif tag == "batch":
                        batch = payload
                        assert val_writer is not None
                        for tokens, token_count in batch:
                            val_writer.add_tokens(tokens)
                            total_val_tokens += token_count
                            total_val_docs += 1
                            val_pbar.update(token_count)
                except Empty:
                    pass

            if stop_early:
                break

    for wp in workers:
        wp.join(timeout=5)
        if wp.is_alive():
            wp.terminate()

    train_stats = train_writer.finalize()
    val_stats = val_writer.finalize() if val_writer is not None else {"total_tokens": 0, "total_shards": 0}

    print("\nConversion complete:")
    print(f"  Train: {total_train_docs:,} docs, {total_train_tokens:,} tokens, {train_stats['total_shards']} shards")
    if val_writer is not None:
        print(f"  Val:   {total_val_docs:,} docs, {total_val_tokens:,} tokens, {val_stats['total_shards']} shards")

    return {"train": train_stats, "val": val_stats}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert FineWeb Parquet to Qwen2.5 .bin shards (int32 payload)")

    p.add_argument("--parquet_dir", type=str, required=True, help="Directory containing parquet files (supports recursion)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for training .bin shards")
    p.add_argument("--val_output_dir", type=str, default=None, help="Output directory for validation .bin shards")
    p.add_argument("--val_tokens", type=int, default=0, help="Total validation tokens to collect (default: 0)")
    p.add_argument("--tokens_per_shard", type=int, default=100_000_000, help="Tokens per shard (default: 100M)")
    p.add_argument("--num_workers", type=int, default=32, help="Number of worker processes (default: 32)")
    p.add_argument("--max_tokens", type=int, default=None, help="Maximum TRAINING tokens to process (default: None)")

    p.add_argument("--tokenizer_dir", type=str, default=DEFAULT_TOKENIZER_DIR, help="Local tokenizer directory")
    p.add_argument("--hf_home", type=str, default=DEFAULT_HF_HOME, help="HF_HOME cache directory (set if unset)")
    p.add_argument("--tmp_dir", type=str, default=DEFAULT_TMP_DIR, help="Temporary dir for payload buffering (default: /tmp/qwen25tokenize)")

    p.add_argument("--text_column", type=str, default="text", help="Parquet column to tokenize (default: text)")
    p.add_argument("--recursive", action="store_true", help="Recursively discover parquet files")
    p.add_argument("--no_recursive", action="store_true", help="Disable recursive discovery")

    p.add_argument("--queue_batch_docs", type=int, default=512, help="Docs per IPC batch (default: 512)")
    p.add_argument("--tokenize_batch_texts", type=int, default=256, help="Texts per tokenizer batch (default: 256)")

    p.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for tokenizer")
    p.add_argument("--no_local_files_only", action="store_true", help="Allow loading tokenizer from network if missing locally")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _setup_hf_cache_dir(args.hf_home)

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    val_output_dir = Path(args.val_output_dir) if args.val_output_dir else None

    recursive = True
    if args.no_recursive:
        recursive = False
    elif args.recursive:
        recursive = True

    convert_parquets_to_bins(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        tokenizer_dir=args.tokenizer_dir,
        tokens_per_shard=args.tokens_per_shard,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        val_output_dir=val_output_dir,
        val_tokens_total=args.val_tokens,
        text_column=args.text_column,
        recursive=recursive,
        queue_batch_docs=args.queue_batch_docs,
        tokenize_batch_texts=args.tokenize_batch_texts,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=not bool(args.no_local_files_only),
        tmp_dir=Path(args.tmp_dir) if args.tmp_dir else None,
    )


if __name__ == "__main__":
    main()

