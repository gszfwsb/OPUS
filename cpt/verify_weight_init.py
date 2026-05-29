#!/usr/bin/env python3
"""
Verify CPT weight initialization for Llama/Qwen models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist


def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


def _pick_checksums(model: torch.nn.Module, names: List[str], seed: int = 1234) -> torch.Tensor:
    """Return a small float64 tensor of checksums for selected parameter tensors."""
    g = torch.Generator(device=next(model.parameters()).device)
    g.manual_seed(seed)

    sd = model.state_dict()
    vals: List[torch.Tensor] = []
    for n in names:
        if n not in sd:
            # Use a sentinel so missing keys differ
            vals.append(torch.tensor(float("nan"), device=next(model.parameters()).device, dtype=torch.float64))
            continue
        t = sd[n]
        flat = t.reshape(-1)
        # sample up to 1024 elements (or all if smaller)
        k = min(1024, flat.numel())
        if k == 0:
            vals.append(torch.tensor(0.0, device=flat.device, dtype=torch.float64))
            continue
        idx = torch.randint(0, flat.numel(), (k,), generator=g, device=flat.device)
        s = flat.index_select(0, idx).float().sum().to(torch.float64)
        vals.append(s)
    return torch.stack(vals, dim=0)


def _read_weight_map(model_dir: Path) -> Dict[str, str] | None:
    """
    Return HF safetensors weight_map if present.
    `model.safetensors.index.json` contains {"weight_map": {key: filename, ...}}.
    """
    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        return None
    try:
        with idx.open("r") as f:
            j = json.load(f)
        wm = j.get("weight_map", None)
        return wm if isinstance(wm, dict) else None
    except Exception:
        return None


def _load_safetensor_key(model_dir: Path, key: str, weight_map: Dict[str, str] | None):
    """
    Load a single tensor by key from (possibly sharded) safetensors without scanning all shards
    if an index json exists.
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:
        raise RuntimeError("safetensors is required for verify_weight_init spot-check") from e

    # Determine which file contains the key
    candidates: List[Path] = []
    if weight_map is not None and key in weight_map:
        candidates = [model_dir / weight_map[key]]
    else:
        # Single-file common case
        single = model_dir / "model.safetensors"
        if single.exists():
            candidates = [single]
        else:
            # Fallback: scan shards (still fine; usually small number of files)
            candidates = sorted(model_dir.glob("*.safetensors"))

    for fp in candidates:
        if not fp.exists():
            continue
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"Key not found in safetensors: {key}")


def _spotcheck_against_hf_qwen_or_llama(
    model: torch.nn.Module,
    init_model_path: str,
    *,
    layers_to_check: Iterable[int],
    verbose: bool,
) -> Tuple[bool, float, List[Tuple[str, float]]]:
    """
    Rank0-only: Compare a handful of tensors vs HF checkpoint on disk.
    Returns (ok, max_diff, top_bad_diffs).
    """
    model_dir = Path(init_model_path)
    if not model_dir.exists():
        if verbose:
            print(f"[verify_weight_init] init_model_path not found: {init_model_path}")
        return False, float("inf"), [("init_model_path_missing", float("inf"))]

    weight_map = _read_weight_map(model_dir)

    our_sd = model.state_dict()
    n_layers = len(getattr(model, "blocks"))

    # Heuristic: Llama/Qwen HF key prefixes are identical for these weights.
    # We intentionally skip the huge embedding matrix to keep this check lightweight.
    to_check: List[Tuple[str, str]] = [("ln_f.weight", "model.norm.weight")]

    for i in layers_to_check:
        if i < 0 or i >= n_layers:
            continue
        hf_prefix = f"model.layers.{i}"
        our_prefix = f"blocks.{i}"
        to_check.extend(
            [
                (f"{our_prefix}.input_layernorm.weight", f"{hf_prefix}.input_layernorm.weight"),
                (f"{our_prefix}.self_attn.q_proj.weight", f"{hf_prefix}.self_attn.q_proj.weight"),
                (f"{our_prefix}.self_attn.k_proj.weight", f"{hf_prefix}.self_attn.k_proj.weight"),
                (f"{our_prefix}.self_attn.v_proj.weight", f"{hf_prefix}.self_attn.v_proj.weight"),
                (f"{our_prefix}.self_attn.o_proj.weight", f"{hf_prefix}.self_attn.o_proj.weight"),
                (f"{our_prefix}.post_attention_layernorm.weight", f"{hf_prefix}.post_attention_layernorm.weight"),
                (f"{our_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.gate_proj.weight"),
                (f"{our_prefix}.mlp.up_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"),
                (f"{our_prefix}.mlp.down_proj.weight", f"{hf_prefix}.mlp.down_proj.weight"),
            ]
        )
        # Qwen3 may include q_norm/k_norm. If our model has them, check if HF has them.
        if f"{our_prefix}.self_attn.q_norm.weight" in our_sd:
            to_check.append((f"{our_prefix}.self_attn.q_norm.weight", f"{hf_prefix}.self_attn.q_norm.weight"))
            to_check.append((f"{our_prefix}.self_attn.k_norm.weight", f"{hf_prefix}.self_attn.k_norm.weight"))

    bad: List[Tuple[str, float]] = []
    max_diff = 0.0

    for our_k, hf_k in to_check:
        if our_k not in our_sd:
            bad.append((f"missing_our:{our_k}", float("inf")))
            max_diff = float("inf")
            continue
        try:
            hf_t = _load_safetensor_key(model_dir, hf_k, weight_map)
        except Exception as e:
            bad.append((f"missing_hf:{hf_k}", float("inf")))
            max_diff = float("inf")
            if verbose:
                print(f"[verify_weight_init] failed loading HF key {hf_k}: {e}")
            continue

        our_t = our_sd[our_k].detach().to("cpu")
        # Compare in float32; exact match is expected for copied weights.
        d = (our_t.float() - hf_t.float()).abs()
        md = float(d.max().item()) if d.numel() else 0.0
        max_diff = max(max_diff, md)
        if md != 0.0:
            bad.append((f"{our_k} <= {hf_k}", md))

    bad_sorted = sorted(bad, key=lambda kv: kv[1], reverse=True)[:10]
    ok = (max_diff == 0.0)
    return ok, max_diff, bad_sorted


def verify_weights(
    model: torch.nn.Module,
    rank: int,
    init_model_path: str,
    *,
    verbose: bool = True,
) -> bool:
    """
    Returns True if:
    - Parameters are identical across ranks (broadcast sanity check)
    - Rank0 spot-check vs HF checkpoint passes (max diff == 0 for selected tensors)

    The boolean result is broadcast to all ranks to keep behavior consistent.
    """
    if not dist.is_available() or not dist.is_initialized():
        # Single-process / unit tests
        return True

    device = next(model.parameters()).device
    world_size = dist.get_world_size()

    # --- 1) Cross-rank broadcast consistency check ---
    check_names = [
        "ln_f.weight",
        "blocks.0.self_attn.q_proj.weight",
        "blocks.0.self_attn.k_proj.weight",
        "blocks.0.mlp.down_proj.weight",
        f"blocks.{len(getattr(model, 'blocks')) - 1}.self_attn.o_proj.weight",
    ]
    local = _pick_checksums(model, check_names).to(device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    # Decide on rank0, then broadcast verdict
    ok_broadcast = True
    if rank == 0:
        ref = gathered[0]
        for r in range(1, world_size):
            # exact match should hold after broadcast (checksums are deterministic)
            if not torch.equal(ref, gathered[r]):
                ok_broadcast = False
                break
        if verbose:
            print(f"[verify_weight_init] broadcast_consistency={ok_broadcast}")

    ok_broadcast_t = torch.tensor(1 if ok_broadcast else 0, device=device, dtype=torch.int32)
    dist.broadcast(ok_broadcast_t, src=0)
    ok_broadcast = bool(ok_broadcast_t.item() == 1)

    # --- 2) Rank0 spot-check vs HF checkpoint (optional but recommended) ---
    ok_ref = True
    if rank == 0:
        n_layers = len(getattr(model, "blocks"))
        layers = sorted(set([0, n_layers // 2, n_layers - 1]))
        ok_ref, max_diff, bad = _spotcheck_against_hf_qwen_or_llama(
            model, init_model_path, layers_to_check=layers, verbose=verbose
        )
        if verbose:
            print(f"[verify_weight_init] hf_spotcheck_ok={ok_ref} max|diff|={max_diff:.6f} layers={layers}")
            if not ok_ref:
                print("[verify_weight_init] top mismatches:")
                for k, v in bad:
                    print(f"  - {k}: {v}")

    ok_ref_t = torch.tensor(1 if ok_ref else 0, device=device, dtype=torch.int32)
    dist.broadcast(ok_ref_t, src=0)
    ok_ref = bool(ok_ref_t.item() == 1)

    ok = ok_broadcast and ok_ref
    if verbose and rank == 0:
        print(f"[verify_weight_init] verify_weights -> {ok}")

    # final barrier so ranks don't drift if something printed/loaded slowly
    dist.barrier()
    return ok


