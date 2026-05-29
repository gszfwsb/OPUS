from operator import truediv
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
import itertools
import math
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path
import threading
import queue
import random
from typing import List

# Add OPUS path to sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from OPUS.train import data_selection as opus_data_selection
    OPUS_AVAILABLE = True
except ImportError:
    print("Could not import OPUS modules. Make sure OPUS directory is in the right location.")
    opus_data_selection = None
    OPUS_AVAILABLE = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # Increase NCCL timeout to 30 minutes for long evaluations during training
# # Use newer TORCH_NCCL_* environment variables to avoid deprecation warnings
# os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1" 
# os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes in seconds
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

from model import GPT, next_multiple_of_n
import tiktoken
from datasets import load_dataset
from transformers import GPT2LMHeadModel
# Add for OPUS data generator
import torch.utils.data
# Add wandb for logging
import wandb
from contextlib import contextmanager
import argparse
import signal

# Time profiling context manager
@contextmanager
def time_profile(name: str, step: int = None, enabled: bool = True):
    """Context manager for timing code blocks and logging to wandb"""
    if not enabled:
        yield
        return
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Log to wandb if initialized and on master process
        if master_process and args.use_wandb and wandb.run is not None:
            log_dict = {f"timing/{name}_ms": elapsed_ms}
            if step is not None:
                log_dict["step"] = step
            wandb.log(log_dict)
        
        # Save to local timing file
        if master_process and 'experiment_name' in globals() and experiment_name is not None:
            timing_file = str(logs_root_path() / f"{experiment_name}_timing.csv")
            os.makedirs(os.path.dirname(timing_file), exist_ok=True)
            # Create header if file doesn't exist
            if not os.path.exists(timing_file):
                with open(timing_file, "w") as f:
                    f.write("timestamp,step,operation,elapsed_ms\n")
            # Append timing data
            with open(timing_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                step_str = step if step is not None else ""
                f.write(f"{timestamp},{step_str},{name},{elapsed_ms:.2f}\n")
        
        # # Also print for debugging
        # if master_process:
        #     step_str = f" (step {step})" if step is not None else ""
        #     # print(f"⏱️  {name}: {elapsed_ms:.2f}ms{step_str}") 

# Define the HellaSwag evaluation function directly in this file
def get_most_likely_ending(model, tokenizer, context, endings):
    """Calculate the most likely ending based on loss."""
    context_tokens = torch.tensor(tokenizer.encode(context), dtype=torch.long, device='cuda')

    lowest_loss = float('inf')
    best_ending_idx = -1

    for i, ending in enumerate(endings):
        ending_tokens = torch.tensor(tokenizer.encode(' ' + ending), dtype=torch.long, device='cuda')
        full_tokens = torch.cat([context_tokens, ending_tokens]).unsqueeze(0)

        target_tokens = full_tokens.clone()
        # The loss for the context part is irrelevant, so we can mask it.
        # A common practice is to use -100, which is ignored by CrossEntropyLoss.
        target_tokens[:, :len(context_tokens)] = -100

        with torch.no_grad():
            # The model's forward pass now returns loss directly if targets are provided.
            # We don't need sliding_window_num_blocks for eval.
            loss = model(full_tokens, target_seq=target_tokens)

        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            best_ending_idx = i

    return best_ending_idx

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            # Arrange gradients of all parameters in this group first, then pad with world_size zero tensors
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            # Take world_size parameter gradients at a time for reduce_scatter(… AVG …), i.e., first all-reduce average, then scatter each rank's portion back to its output buffer
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    # m ← μ m + (1-μ) g
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    # m ← μ m_new + (1-μ) g
                    grad = grad.lerp_(momentum_buffer, momentum)
                    # g ← U V^T g
                    # Apply "zeroth power/orthogonalization" transformation to grad via Newton-Schulz (quintic) iteration
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    # p += -eff_lr * v
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                # FIX: Check if parameter dimension divides evenly by world_size (8 GPUs)
                # Some GPT-2 XL parameters (like embedding: 50257 dims) don't divide by 8
                if grad.shape[0] % world_size != 0:
                    # FALLBACK: Use all_reduce for non-divisible parameters (less memory efficient but works)
                    grad_slices.append(grad.clone())  # Each GPU gets full gradient
                    reduce_scatter_futures.append(None)  # Mark as "use all_reduce instead"
                else:
                    # NORMAL: Use memory-efficient reduce_scatter for divisible parameters  
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                p = params[base]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                
                # FIX: Handle both reduce_scatter and all_reduce cases
                if reduce_scatter_futures[idx] is None:
                    # FALLBACK CASE: Parameter didn't divide evenly, so we used all_reduce
                    dist.all_reduce(g_slice, op=dist.ReduceOp.AVG)  # Average gradient across all GPUs
                    p_slice = p  # Work with full parameter (uses more memory)
                else:
                    # NORMAL CASE: Parameter divided evenly, so we used reduce_scatter
                    reduce_scatter_futures[idx].wait()  # Wait for our gradient slice
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]  # Work with parameter slice
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    # p ← (1 - lr*wd) p
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                # m ← β1 m + (1-β1) g
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                # v ← β2 v + (1-β2) g∘g
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                # denom ← sqrt(v) + ϵ
                denom = exp_avg_sq.sqrt().add_(eps)
                # update ← m / lr * ((1-β2^t)^0.5 / (1-β1^t))
                step_size = lr * (torch.sqrt(bias2) / bias1)
                # updata <- lr * m / (sqrt(v) + ϵ)
                update = exp_avg.div(denom).mul_(step_size)
                # p ← p - update
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                
                # FIX: Only gather parameter slices back if we used reduce_scatter
                if reduce_scatter_futures[idx-1] is not None:
                    # NORMAL CASE: Gather parameter slices back to full parameter
                    all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
                # FALLBACK CASE: No gathering needed since we worked with full parameter
        
        # Wait for all parameter gathering to complete
        if all_reduce_futures:
            torch.futures.collect_all(all_reduce_futures).wait()

# -----------------------------------------------------------------------------
# HuggingFace GPT-2 weight loading for Continue Pretraining (CPT)

def load_hf_gpt2_weights(model: nn.Module, hf_model_path: str, master_process: bool = True):
    """
    Load weights from a HuggingFace GPT-2 model into our custom GPT architecture.
    
    Our architecture differs from HuggingFace GPT-2:
    - Uses RMSNorm instead of LayerNorm (no learnable params to copy)
    - Uses RoPE instead of learned position embeddings (skip wpe)
    - Uses QK normalization (no equivalent in HF)
    
    We can copy:
    - Token embeddings (wte -> embed)
    - Attention QKV projections (c_attn -> qkv_proj)
    - Attention output projection (c_proj -> c_proj)
    - MLP fc weights (c_fc -> c_fc, c_proj -> c_proj)
    
    Args:
        model: Our custom GPT model instance
        hf_model_path: Path to HuggingFace model or model name (e.g., 'openai-community/gpt2-xl')
        master_process: Whether this is the master process (for logging)
    """
    if master_process:
        print(f"Loading HuggingFace GPT-2 weights from: {hf_model_path}")
    
    # Load HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_path)
    hf_state = hf_model.state_dict()
    
    # Track what we copied
    copied_params = []
    skipped_params = []
    
    # 1. Token embeddings: transformer.wte.weight -> embed.weight
    if 'transformer.wte.weight' in hf_state:
        hf_wte = hf_state['transformer.wte.weight']
        # Our embed might be padded to next_multiple_of_n(vocab_size, 128)
        our_vocab_size = model.embed.weight.shape[0]
        hf_vocab_size = hf_wte.shape[0]
        if our_vocab_size >= hf_vocab_size:
            model.embed.weight.data[:hf_vocab_size] = hf_wte
            copied_params.append(f'embed.weight [{hf_vocab_size}/{our_vocab_size}]')
        else:
            model.embed.weight.data = hf_wte[:our_vocab_size]
            copied_params.append(f'embed.weight (truncated)')
    
    # Skip position embeddings (we use RoPE)
    skipped_params.append('transformer.wpe.weight (using RoPE)')
    
    # 2. Layer weights
    num_layers = len(model.blocks)
    for i in range(num_layers):
        hf_prefix = f'transformer.h.{i}'
        
        # Attention QKV: HF has c_attn.weight [3*dim, dim], c_attn.bias [3*dim]
        # Our qkv_proj has weight [3*hdim, dim] where hdim = num_heads * head_dim
        if f'{hf_prefix}.attn.c_attn.weight' in hf_state:
            hf_qkv = hf_state[f'{hf_prefix}.attn.c_attn.weight']  # [dim, 3*dim] (transposed)
            # HuggingFace uses Conv1D style: weight is [in_features, out_features]
            # Our Linear uses [out_features, in_features]
            model.blocks[i].attn.qkv_proj.weight.data = hf_qkv.T.contiguous()
            copied_params.append(f'blocks.{i}.attn.qkv_proj.weight')
        
        # Attention output projection: c_proj
        if f'{hf_prefix}.attn.c_proj.weight' in hf_state:
            hf_c_proj = hf_state[f'{hf_prefix}.attn.c_proj.weight']  # [dim, dim]
            model.blocks[i].attn.c_proj.weight.data = hf_c_proj.T.contiguous()
            copied_params.append(f'blocks.{i}.attn.c_proj.weight')
        
        # MLP: c_fc (expansion) and c_proj (contraction)
        if f'{hf_prefix}.mlp.c_fc.weight' in hf_state:
            hf_fc = hf_state[f'{hf_prefix}.mlp.c_fc.weight']  # [dim, 4*dim]
            model.blocks[i].mlp.c_fc.weight.data = hf_fc.T.contiguous()
            copied_params.append(f'blocks.{i}.mlp.c_fc.weight')
        
        if f'{hf_prefix}.mlp.c_proj.weight' in hf_state:
            hf_proj = hf_state[f'{hf_prefix}.mlp.c_proj.weight']  # [4*dim, dim]
            model.blocks[i].mlp.c_proj.weight.data = hf_proj.T.contiguous()
            copied_params.append(f'blocks.{i}.mlp.c_proj.weight')
        
        # Skip LayerNorm weights (we use RMSNorm with no learnable params)
        skipped_params.append(f'{hf_prefix}.ln_1 (using RMSNorm)')
        skipped_params.append(f'{hf_prefix}.ln_2 (using RMSNorm)')
        
        # Skip attention biases (our model has no bias)
        if f'{hf_prefix}.attn.c_attn.bias' in hf_state:
            skipped_params.append(f'{hf_prefix}.attn.c_attn.bias (no bias)')
        if f'{hf_prefix}.attn.c_proj.bias' in hf_state:
            skipped_params.append(f'{hf_prefix}.attn.c_proj.bias (no bias)')
        if f'{hf_prefix}.mlp.c_fc.bias' in hf_state:
            skipped_params.append(f'{hf_prefix}.mlp.c_fc.bias (no bias)')
        if f'{hf_prefix}.mlp.c_proj.bias' in hf_state:
            skipped_params.append(f'{hf_prefix}.mlp.c_proj.bias (no bias)')
    
    # Skip final LayerNorm
    skipped_params.append('transformer.ln_f (using RMSNorm)')
    
    # lm_head is tied to embed in our model, so no need to copy separately
    skipped_params.append('lm_head.weight (tied to embed)')
    
    # Clean up HF model to free memory
    del hf_model
    del hf_state
    torch.cuda.empty_cache()
    
    if master_process:
        print(f"✅ Copied {len(copied_params)} parameter groups from HuggingFace GPT-2")
        print(f"⏭️  Skipped {len(skipped_params)} parameter groups (architecture differences)")
        print(f"   Skipped: position embeddings (RoPE), LayerNorms (RMSNorm), biases (no bias)")

# -----------------------------------------------------------------------------
# Distributed data loader

# Default doc boundary token for legacy GPT-2 bins (uint16) is 50256 (<|endoftext|>).
# For HF-tokenized bins (Qwen/Llama), the converter stores boundary token in header[4]
# (usually tokenizer.eos_token_id). `_load_data_shard` updates this value when present.
DATA_EOD_TOKEN_ID = 50256

def discover_all_train_files_flattened(
    root_dir: str,
    seed: int = 42,
    exclude_val: bool = True,
    shuffle_files: bool = True,
) -> List[Path]:
    """
    Discover all .bin training files from a directory.
    Returns a deterministically shuffled list.
    
    Supports two directory structures:
    1. Domain-organized (subdirectories):
        /path/to/data/
            Science and Technology/
                02_software_dev_finewebedu_000000.bin
            Lifestyle/
                01_art_design_finewebedu_000000.bin
    
    2. Flat directory (all files in root):
        /path/to/data/
            s3_01_art_design_finewebedu_000000.bin
            s4_02_software_dev_finewebedu_000000.bin
    
    Args:
        root_dir: Root directory containing .bin files or domain subdirectories
        seed: Random seed for reproducible shuffling (default: 42)
        exclude_val: If True, exclude files with 'val' in the name (default: True)
        shuffle_files: If True, shuffle the final file list with the provided seed
    
    Returns:
        List of Path objects for training files in shuffled order
    """
    root_dir = Path(root_dir)
    all_files = []
    
    # First, check if there are .bin files directly in the root directory (flat structure)
    root_bins = sorted(root_dir.glob("*.bin"))
    if root_bins:
        # Flat directory structure - files are directly in root
        if exclude_val:
            root_bins = [f for f in root_bins if "val" not in f.stem.lower()]
        all_files.extend(root_bins)
    else:
        # Domain-organized structure - files are in subdirectories
        for domain_dir in sorted(root_dir.iterdir()):
            if domain_dir.is_dir():
                domain_bins = sorted(domain_dir.glob("*.bin"))
                if exclude_val:
                    # Exclude validation files
                    domain_bins = [f for f in domain_bins if "val" not in f.stem.lower()]
                all_files.extend(domain_bins)
    
    # Sort for deterministic base ordering
    all_files.sort()
    
    if shuffle_files:
        random.seed(seed)
        random.shuffle(all_files)
    
    return all_files

def interleave_multi_source_files(root_dirs: List[str], seed: int = 42, exclude_val: bool = True) -> List[Path]:
    """
    Interleave files from multiple data sources for balanced data distribution.
    
    This ensures even distribution across sources during training, particularly important
    when sources have different sizes (e.g., Score 3: 200B tokens vs Score 4: 70B tokens).
    
    Strategy:
    1. Shuffle files within each source independently
    2. Interleave sources proportionally (round-robin with proper ratios)
    3. After smallest source exhausted, continue with remaining sources
    
    This maintains consistent data distribution throughout training!
    
    Args:
        root_dirs: List of root directories (e.g., ["/path/to/score3", "/path/to/score4"])
        seed: Random seed for reproducible shuffling
        exclude_val: If True, exclude files with 'val' in the name
        
    Returns:
        List of Path objects interleaved from all sources
    """
    random.seed(seed)
    
    # Collect and shuffle files from each source independently
    all_source_files = []
    for root_dir in root_dirs:
        source_files = discover_all_train_files_flattened(root_dir, seed=seed, exclude_val=exclude_val)
        # Shuffle within source for diversity
        random.shuffle(source_files)
        all_source_files.append(source_files)
    
    # Interleave files proportionally (round-robin)
    interleaved = []
    source_indices = [0] * len(all_source_files)
    
    # Continue until all sources exhausted
    while any(idx < len(files) for idx, files in zip(source_indices, all_source_files)):
        for source_idx, (files, idx) in enumerate(zip(all_source_files, source_indices)):
            if idx < len(files):
                interleaved.append(files[idx])
                source_indices[source_idx] += 1
    
    return interleaved

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        # Default legacy format: uint16 payload (2 bytes per token).
        # Extension for large-vocab tokenizers (Qwen/Llama): header[3]=4 -> int32 payload.
        token_bytes = int(header[3].item()) if header.numel() > 3 else 2
        if token_bytes not in (2, 4):
            token_bytes = 2
        global DATA_EOD_TOKEN_ID
        if header.numel() > 4 and int(header[4].item()) > 0:
            DATA_EOD_TOKEN_ID = int(header[4].item())
        tokens_dtype = torch.uint16 if token_bytes == 2 else torch.int32
        tokens = torch.empty(num_tokens, dtype=tokens_dtype, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == token_bytes * num_tokens, "number of tokens read does not match header"
    return tokens

def load_multi_file_proxy_tokens(proxy_pattern: str, total_tokens: int, seed: int = 42) -> torch.Tensor:
    """
    Load proxy tokens from multiple files by sampling evenly across files.
    Memory-efficient version: only reads the needed portion of each file.
    
    Args:
        proxy_pattern: Glob pattern(s) for proxy files. Can be:
                      - Single pattern: '/path/to/score5/*/*.bin'
                      - Multiple patterns (comma-separated): '/path/to/3/*/*.bin,/path/to/4/*/*.bin'
        total_tokens: Total number of tokens to sample
        seed: Random seed for reproducible sampling
        
    Returns:
        Tensor of sampled proxy tokens (uint16)
    """
    # Support comma-separated multiple patterns
    proxy_files = []
    patterns = [p.strip() for p in proxy_pattern.split(',')]
    for pattern in patterns:
        proxy_files.extend([Path(f) for f in sorted(glob.glob(pattern))])
    
    # Remove duplicates and sort
    proxy_files = sorted(set(proxy_files))
    
    if len(proxy_files) == 0:
        raise ValueError(f"No proxy files found for pattern(s): {proxy_pattern}")
    
    is_master = int(os.environ.get("RANK", 0)) == 0
    if is_master:
        print(f"Loading proxy data from {len(proxy_files)} files: {proxy_pattern}")
        print(f"Target proxy tokens: {total_tokens:,}")
    
    # Calculate tokens per file (sample evenly)
    tokens_per_file = total_tokens // len(proxy_files)
    remaining_tokens = total_tokens % len(proxy_files)
    
    all_proxy_tokens = []
    random.seed(seed)
    
    for i, proxy_file in enumerate(proxy_files):
        # Read only the header to get file size (memory efficient!)
        header = torch.from_file(str(proxy_file), False, 256, dtype=torch.int32)
        assert header[0] == 20240520, f"magic number mismatch in {proxy_file}"
        assert header[1] == 1, "unsupported version"
        file_len = int(header[2])  # number of tokens in file
        token_bytes = int(header[3].item()) if header.numel() > 3 else 2
        if token_bytes not in (2, 4):
            token_bytes = 2
        tokens_dtype = torch.uint16 if token_bytes == 2 else torch.int32
        
        # Calculate how many tokens to sample from this file
        n_tokens = tokens_per_file
        if i < remaining_tokens:  # Distribute remainder across first N files
            n_tokens += 1
        
        # Sample tokens from this file (random starting position)
        if file_len <= n_tokens:
            # Use entire file if it's smaller than requested
            start_idx = 0
            n_tokens = file_len
        else:
            # Random starting position for diversity
            start_idx = random.randint(0, file_len - n_tokens)
        
        # Memory-efficient: only read the needed portion
        sampled_tokens = torch.empty(n_tokens, dtype=tokens_dtype, pin_memory=True)
        with proxy_file.open("rb", buffering=0) as f:
            # Skip header (256 int32 = 1024 bytes) and seek to start position
            offset = 256 * 4 + start_idx * token_bytes  # header + start_idx * sizeof(token)
            f.seek(offset)
            nbytes = f.readinto(sampled_tokens.numpy())
            assert nbytes == token_bytes * n_tokens, f"read {nbytes} bytes but expected {token_bytes * n_tokens}"
        
        all_proxy_tokens.append(sampled_tokens)
        
        if is_master and i < 3:  # Show first 3 files as example
            print(f"  {i+1}. {proxy_file.name}: sampled {len(sampled_tokens):,} tokens from offset {start_idx}")
    
    # Concatenate all sampled tokens
    proxy_tokens = torch.cat(all_proxy_tokens, dim=0)
    
    if is_master:
        print(f"Total proxy tokens loaded: {len(proxy_tokens):,} (memory-efficient loading)")
    
    return proxy_tokens

def load_multi_file_proxy_tokens_and_masks(
    proxy_pattern: str,
    total_tokens: int,
    loss_mask_suffix: str = ".lossmask",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load proxy tokens AND their loss masks from multiple files.
    Memory-efficient version: only reads the needed portion of each file.
    
    Returns:
        Tuple of (proxy_tokens, proxy_mask_tokens) tensors
    """
    proxy_files = []
    patterns = [p.strip() for p in proxy_pattern.split(',')]
    for pattern in patterns:
        proxy_files.extend([Path(f) for f in sorted(glob.glob(pattern))])
    
    proxy_files = sorted(set(proxy_files))
    
    if len(proxy_files) == 0:
        raise ValueError(f"No proxy files found for pattern(s): {proxy_pattern}")
    
    is_master = int(os.environ.get("RANK", 0)) == 0
    if is_master:
        print(f"Loading proxy data + masks from {len(proxy_files)} files: {proxy_pattern}")
        print(f"Target proxy tokens: {total_tokens:,}")
    
    tokens_per_file = total_tokens // len(proxy_files)
    remaining_tokens = total_tokens % len(proxy_files)
    
    all_proxy_tokens = []
    all_proxy_masks = []
    random.seed(seed)
    
    for i, proxy_file in enumerate(proxy_files):
        # Read token file header
        header = torch.from_file(str(proxy_file), False, 256, dtype=torch.int32)
        assert header[0] == 20240520, f"magic number mismatch in {proxy_file}"
        assert header[1] == 1, "unsupported version"
        file_len = int(header[2])
        token_bytes = int(header[3].item()) if header.numel() > 3 else 2
        if token_bytes not in (2, 4):
            token_bytes = 2
        tokens_dtype = torch.uint16 if token_bytes == 2 else torch.int32
        
        # Read mask file header
        mask_file = Path(proxy_file).with_suffix(loss_mask_suffix)
        if not mask_file.is_file():
            raise FileNotFoundError(f"Loss mask sidecar not found for proxy file {proxy_file}: expected {mask_file}")
        mask_header = torch.from_file(str(mask_file), False, 256, dtype=torch.int32)
        assert mask_header[0] == 20240520, f"magic number mismatch in mask {mask_file}"
        mask_len = int(mask_header[2])
        mask_bytes = int(mask_header[3].item()) if mask_header.numel() > 3 else 2
        if mask_bytes not in (2, 4):
            mask_bytes = 2
        mask_dtype = torch.uint16 if mask_bytes == 2 else torch.int32
        
        if file_len != mask_len:
            raise ValueError(f"Token/mask length mismatch for {proxy_file}: tokens={file_len}, mask={mask_len}")
        
        n_tokens = tokens_per_file
        if i < remaining_tokens:
            n_tokens += 1
        
        if file_len <= n_tokens:
            start_idx = 0
            n_tokens = file_len
        else:
            start_idx = random.randint(0, file_len - n_tokens)
        
        # Read tokens
        sampled_tokens = torch.empty(n_tokens, dtype=tokens_dtype, pin_memory=True)
        with proxy_file.open("rb", buffering=0) as f:
            offset = 256 * 4 + start_idx * token_bytes
            f.seek(offset)
            nbytes = f.readinto(sampled_tokens.numpy())
            assert nbytes == token_bytes * n_tokens
        
        # Read masks
        sampled_masks = torch.empty(n_tokens, dtype=mask_dtype, pin_memory=True)
        with mask_file.open("rb", buffering=0) as f:
            offset = 256 * 4 + start_idx * mask_bytes
            f.seek(offset)
            nbytes = f.readinto(sampled_masks.numpy())
            assert nbytes == mask_bytes * n_tokens
        
        all_proxy_tokens.append(sampled_tokens)
        all_proxy_masks.append(sampled_masks)
        
        if is_master and i < 3:
            print(f"  {i+1}. {proxy_file.name}: sampled {len(sampled_tokens):,} tokens+masks from offset {start_idx}")
    
    proxy_tokens = torch.cat(all_proxy_tokens, dim=0)
    proxy_masks = torch.cat(all_proxy_masks, dim=0)
    
    if is_master:
        print(f"Total proxy tokens+masks loaded: {len(proxy_tokens):,}")
    
    return proxy_tokens, proxy_masks

# find world_size starting indicies, such that each begins with EOD token and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == DATA_EOD_TOKEN_ID
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase max_batch_span if necessary

def distributed_data_generator(
    filename_pattern,
    batch_size: int,
    align_to_bos: bool,
    use_loss_mask: bool = False,
    loss_mask_suffix: str = ".lossmask",
):
    """
    Distributed data generator that supports both glob patterns and file lists.
    
    Args:
        filename_pattern: Either a glob pattern string (e.g., "/path/*.bin") 
                         or a list of Path objects
        batch_size: Total batch size across all processes
        align_to_bos: Whether to align batches to beginning-of-sequence tokens
        use_loss_mask: Whether to use loss masking (for CPT)
        loss_mask_suffix: Suffix for loss mask sidecar files
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Support both glob pattern and file list
    if isinstance(filename_pattern, (list, tuple)):
        files = [Path(f) for f in filename_pattern]
    else:
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    
    assert len(files) > 0, f"No files found for pattern/list: {filename_pattern}"
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = itertools.cycle(files)
    cur_file = next(file_iter)
    tokens, pos = _load_data_shard(cur_file), 0
    loss_mask_tokens = None
    if use_loss_mask:
        mask_file = Path(cur_file).with_suffix(loss_mask_suffix)
        if not mask_file.is_file():
            raise FileNotFoundError(f"Loss mask sidecar not found for {cur_file}: expected {mask_file}")
        loss_mask_tokens = _load_data_shard(mask_file)
        if loss_mask_tokens.numel() != tokens.numel():
            raise ValueError(f"Loss mask length mismatch for {cur_file}: tokens={tokens.numel()}, mask={loss_mask_tokens.numel()}")
    max_batch_span = 8 * batch_size if align_to_bos else batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            cur_file = next(file_iter)
            tokens, pos = _load_data_shard(cur_file), 0
            if use_loss_mask:
                mask_file = Path(cur_file).with_suffix(loss_mask_suffix)
                if not mask_file.is_file():
                    raise FileNotFoundError(f"Loss mask sidecar not found for {cur_file}: expected {mask_file}")
                loss_mask_tokens = _load_data_shard(mask_file)
                if loss_mask_tokens.numel() != tokens.numel():
                    raise ValueError(f"Loss mask length mismatch for {cur_file}: tokens={tokens.numel()}, mask={loss_mask_tokens.numel()}")
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        if use_loss_mask:
            assert loss_mask_tokens is not None
            m = loss_mask_tokens[start_idx:][:local_batch_size + 1]
            # Mask applies to the target positions (shifted by 1)
            target_mask = m[1:].to(device="cuda", dtype=torch.bool, non_blocking=True)
            targets = targets.masked_fill(~target_mask, -100)
        pos += batch_span
        yield inputs, targets

def opus_data_generator(model, optimizer, filename_pattern: str, batch_size: int,
                        proxy_files: str, buffer_size: int, selection_ratio: float,
                        proxy_dir: str = None, proxy_tokens_target: int = 1_000_000,
                        score_len: int | None = None, proxy_batch_size: int | None = None,
                        global_selection: bool = False,
                        use_loss_mask: bool = False,
                        loss_mask_suffix: str = ".lossmask"):
    if opus_data_selection is None:
        raise ImportError("OPUS module not loaded. Cannot use opus_data_generator.")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seq_len = batch_size # In this script, batch_size is used as seq_len for the generator

    # === Find trainable layers once ===
    trainable_layers = opus_data_selection.find_GClayers(model)
    assert len(trainable_layers) > 0
    
    # Print gradient dimensions for each trainable layer (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("GRADIENT DIMENSION ANALYSIS")
        print("="*70)
        total_grad_dim = 0
        max_grad_dim = 0
        max_layer_name = ""
        for i, item in enumerate(trainable_layers):
            # Handle both (name, layer) tuples and plain layer objects
            if isinstance(item, tuple) and len(item) == 2:
                name, layer = item
            else:
                layer = item
                name = f"layer_{i}"
            grad_dim = layer.weight.numel()
            total_grad_dim += grad_dim
            if grad_dim > max_grad_dim:
                max_grad_dim = grad_dim
                max_layer_name = name
            print(f"  {name:50s} | shape {str(list(layer.weight.shape)):20s} | grad_dim = {grad_dim:>12,}")
        print("-"*70)
        print(f"  Total layers: {len(trainable_layers)}")
        print(f"  Total gradient dimensions: {total_grad_dim:,}")
        print(f"  Max layer gradient dim: {max_grad_dim:,} ({max_layer_name})")
        print(f"  Recommended projection_dim: >= {max(2048, max_grad_dim // 1000)} (compression ~{max_grad_dim // 8192}x at 8192)")
        print("="*70 + "\n")

    # === 0) Create a validation loader from proxy data (once) ===
    # Support both single-file and multi-file proxy data
    proxy_mask_tokens = None
    if proxy_dir is not None:
        if use_loss_mask:
            proxy_tokens, proxy_mask_tokens = load_multi_file_proxy_tokens_and_masks(
                proxy_dir, proxy_tokens_target, loss_mask_suffix=loss_mask_suffix
            )
        else:
            proxy_tokens = load_multi_file_proxy_tokens(proxy_dir, proxy_tokens_target)
    else:
        # Single-file proxy (legacy behavior)
        proxy_tokens = _load_data_shard(Path(proxy_files))
        if use_loss_mask:
            mask_file = Path(proxy_files).with_suffix(loss_mask_suffix)
            if not mask_file.is_file():
                raise FileNotFoundError(f"Loss mask sidecar not found for proxy file {proxy_files}: expected {mask_file}")
            proxy_mask_tokens = _load_data_shard(mask_file)
    
    proxy_inputs  = proxy_tokens[:-1].to(dtype=torch.int32)   # CPU
    proxy_targets = proxy_tokens[1:].to(dtype=torch.int64)    # CPU
    if use_loss_mask and proxy_mask_tokens is not None:
        proxy_target_mask = proxy_mask_tokens[1:].to(dtype=torch.bool)  # align with targets
        proxy_targets = proxy_targets.masked_fill(~proxy_target_mask, -100)
    p_num = max(1, (proxy_inputs.numel() // seq_len))
    proxy_inputs_b  = proxy_inputs[:p_num * seq_len].view(p_num, seq_len)
    proxy_targets_b = proxy_targets[:p_num * seq_len].view(p_num, seq_len)
    proxy_dataset = torch.utils.data.TensorDataset(proxy_inputs_b, proxy_targets_b)

    def collate_fn(batch):
        inputs, labels = zip(*batch)  # These are still CPU tensors
        inputs = torch.stack(inputs).to(device="cuda", non_blocking=True)
        labels = torch.stack(labels).to(device="cuda", non_blocking=True)
        return {
            'input_ids': inputs,
            'attention_mask': torch.ones_like(inputs),
            'labels': labels
        }
    # This loader will be used by compute_GradProd_GC_per_iter
    # Use larger batch size to better utilize proxy data (default: 8 instead of 2)
    _proxy_bs = min(8, p_num) if (proxy_batch_size is None) else min(int(proxy_batch_size), p_num)
    validation_loader = torch.utils.data.DataLoader(proxy_dataset, batch_size=_proxy_bs, shuffle=True, drop_last=True, collate_fn=collate_fn)
    validation_loader._proxy_mode = getattr(args, "opus_proxy_mode", "refresh")
    validation_loader._proxy_refresh_interval = max(1, int(getattr(args, "opus_proxy_refresh_interval", 1)))
    validation_loader._proxy_call_count = 0
    validation_loader._cached_proxy_batch = None
    validation_loader._fixed_batch = None

    # Support both glob patterns and file lists (from domain_root_dir)
    if isinstance(filename_pattern, (list, tuple)):
        files = [Path(f) for f in filename_pattern]
    else:
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    
    assert len(files) > 0, f"No training files found for pattern/list: {filename_pattern}"
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        # === 1) Rank 0 loads full buffer, others wait for broadcast ===
        with time_profile("data_loading_io", enabled=args.enable_profiling):
            if rank == 0:
                # Master rank loads the full buffer for all ranks
                needed_tokens = buffer_size * world_size + 1
                
                # Pre-allocate tensor for efficiency (avoid list operations)
                candidate_tokens = torch.empty(needed_tokens, dtype=torch.int32)
                filled = 0
                
                while filled < needed_tokens:
                    remaining_need = needed_tokens - filled
                    
                    if pos + remaining_need >= len(tokens):
                        # Copy remaining tokens from current file
                        available = len(tokens) - pos
                        if available > 0:
                            candidate_tokens[filled:filled + available] = tokens[pos:].to(torch.int32)
                            filled += available
                        # Load new file
                        tokens, pos = _load_data_shard(next(file_iter)), 0
                    else:
                        # Copy needed tokens from current file
                        candidate_tokens[filled:filled + remaining_need] = tokens[pos:pos + remaining_need].to(torch.int32)
                        pos += remaining_need
                        filled += remaining_need
                
                # Move to GPU
                candidate_tokens = candidate_tokens.cuda()
            else:
                # Other ranks create empty buffer - will receive via broadcast
                candidate_tokens = torch.empty(buffer_size * world_size + 1, dtype=torch.int32, device='cuda')
        
        with time_profile("data_loading_broadcast", enabled=args.enable_profiling):
            # Broadcast the full buffer from rank 0 to all ranks (now on GPU with int32)
            dist.broadcast(candidate_tokens, src=0)

        # === 2) Each rank extracts its portion ===
        with time_profile("data_preprocessing", enabled=args.enable_profiling):
            start_idx = rank * buffer_size
            local_candidate_tokens = candidate_tokens[start_idx : start_idx + buffer_size + 1]

            # Tokens are already on GPU as int32, convert targets to int64
            local_candidate_inputs  = local_candidate_tokens[:-1]  # Already int32
            local_candidate_targets = local_candidate_tokens[ 1:].to(dtype=torch.int64)

            # === 3) Reshape 1D tokens to [num_seqs, seq_len] ===
            num_seqs = (local_candidate_inputs.numel() // seq_len)
            if num_seqs == 0:
                raise ValueError(f"No sequences to process. Buffer size ({buffer_size}) may be smaller than seq_len ({seq_len}).")

            cut_tokens = num_seqs * seq_len
            local_candidate_inputs  = local_candidate_inputs[:cut_tokens].view(num_seqs, seq_len)
            local_candidate_targets = local_candidate_targets[:cut_tokens].view(num_seqs, seq_len)

        # === 4) Get selected indices from OPUS ===
        with time_profile("opus_selection_total", enabled=args.enable_profiling):
            if global_selection:
                # === Global Selection Mode ===
                # IMPORTANT: Synchronize proxy batch across all ranks so the proxy direction
                # matches the paper's single global selection step.
                with time_profile("sync_proxy_batch", enabled=args.enable_profiling):
                    proxy_mode = getattr(validation_loader, "_proxy_mode", "refresh")
                    refresh_interval = max(1, int(getattr(validation_loader, "_proxy_refresh_interval", 1)))
                    proxy_call_count = int(getattr(validation_loader, "_proxy_call_count", 0))
                    cached_proxy_batch = getattr(validation_loader, "_fixed_batch", None)
                    use_cached_proxy = cached_proxy_batch is not None and (
                        proxy_mode == "fixed" or (proxy_call_count % refresh_interval != 0)
                    )

                    if rank == 0:
                        if use_cached_proxy:
                            proxy_batch = cached_proxy_batch
                        else:
                            if not hasattr(validation_loader, "_it"):
                                validation_loader._it = iter(validation_loader)
                            try:
                                proxy_batch = next(validation_loader._it)
                            except StopIteration:
                                validation_loader._it = iter(validation_loader)
                                proxy_batch = next(validation_loader._it)

                        proxy_input_ids = proxy_batch['input_ids'].contiguous()
                        proxy_labels = proxy_batch['labels'].contiguous()
                    else:
                        proxy_input_ids = torch.empty(_proxy_bs, seq_len, dtype=torch.int32, device='cuda')
                        proxy_labels = torch.empty(_proxy_bs, seq_len, dtype=torch.int64, device='cuda')

                    dist.broadcast(proxy_input_ids, src=0)
                    dist.broadcast(proxy_labels, src=0)

                    synced_proxy_batch = {
                        'input_ids': proxy_input_ids,
                        'attention_mask': torch.ones_like(proxy_input_ids),
                        'labels': proxy_labels
                    }
                    validation_loader._fixed_batch = synced_proxy_batch

                    validation_loader._proxy_call_count = proxy_call_count + 1

                with time_profile("gather_global_candidates", enabled=args.enable_profiling):
                    all_candidate_inputs = [torch.empty_like(local_candidate_inputs) for _ in range(world_size)]
                    all_candidate_targets = [torch.empty_like(local_candidate_targets) for _ in range(world_size)]
                    dist.all_gather(all_candidate_inputs, local_candidate_inputs)
                    dist.all_gather(all_candidate_targets, local_candidate_targets)
                    global_candidate_inputs = torch.cat(all_candidate_inputs, dim=0)
                    global_candidate_targets = torch.cat(all_candidate_targets, dim=0)

                local_num_seqs = local_candidate_inputs.shape[0]
                global_num_seqs = global_candidate_inputs.shape[0]
                global_num_to_select = max(1, int(global_num_seqs * selection_ratio))

                if rank == 0:
                    global_selected_idx = opus_data_selection.get_batch_opus(
                        model=model,
                        buffer_seq=global_candidate_inputs,
                        buffer_labels=global_candidate_targets,
                        proxy_seq=None,
                        proxy_labels=None,
                        optimizer=optimizer,
                        trainable_layers=trainable_layers,
                        validation_loader=validation_loader,
                        selection_ratio=selection_ratio,
                        seq_len=seq_len,
                        selection_strategy=args.selection_strategy,
                        selection_method=args.opus_selection_method,
                        preconditioner=args.opus_preconditioner,
                        temperature=args.opus_temperature,
                        score_len=(score_len if score_len is not None else seq_len),
                        n_windows=args.opus_n_windows,
                    ).to(dtype=torch.long, device='cuda')
                else:
                    global_selected_idx = torch.empty(global_num_to_select, dtype=torch.long, device='cuda')

                dist.broadcast(global_selected_idx, src=0)
                del all_candidate_inputs, all_candidate_targets, global_candidate_inputs, global_candidate_targets

                local_start = rank * local_num_seqs
                local_end = local_start + local_num_seqs

                # 5a: Extract locally selected samples (samples that belong to this rank)
                local_mask = (global_selected_idx >= local_start) & (global_selected_idx < local_end)
                my_local_selected_idx = global_selected_idx[local_mask] - local_start

                if my_local_selected_idx.numel() > 0:
                    my_selected_inputs = local_candidate_inputs[my_local_selected_idx]
                    my_selected_targets = local_candidate_targets[my_local_selected_idx]
                else:
                    my_selected_inputs = torch.empty(0, seq_len, dtype=local_candidate_inputs.dtype, device='cuda')
                    my_selected_targets = torch.empty(0, seq_len, dtype=local_candidate_targets.dtype, device='cuda')

                my_count = my_local_selected_idx.numel()

                # 5b: Communicate counts - how many selected samples each rank has
                all_counts = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
                my_count_tensor = torch.tensor([my_count], dtype=torch.long, device='cuda')
                dist.all_gather(all_counts, my_count_tensor)
                all_counts = torch.cat(all_counts).tolist()  # [count_rank0, count_rank1, ...]

                # 5c: AllGather variable-length tensors using padding
                max_count = max(all_counts) if max(all_counts) > 0 else 1

                padded_inputs = torch.zeros(max_count, seq_len, dtype=local_candidate_inputs.dtype, device='cuda')
                padded_targets = torch.zeros(max_count, seq_len, dtype=local_candidate_targets.dtype, device='cuda')
                if my_count > 0:
                    padded_inputs[:my_count] = my_selected_inputs
                    padded_targets[:my_count] = my_selected_targets

                all_inputs_list = [torch.empty_like(padded_inputs) for _ in range(world_size)]
                all_targets_list = [torch.empty_like(padded_targets) for _ in range(world_size)]
                dist.all_gather(all_inputs_list, padded_inputs)
                dist.all_gather(all_targets_list, padded_targets)

                # 5d: Concatenate actual (non-padded) data from all ranks
                all_selected_inputs = []
                all_selected_targets = []
                for r in range(world_size):
                    count = all_counts[r]
                    if count > 0:
                        all_selected_inputs.append(all_inputs_list[r][:count])
                        all_selected_targets.append(all_targets_list[r][:count])

                if all_selected_inputs:
                    global_selected_inputs = torch.cat(all_selected_inputs, dim=0)  # [global_num_to_select, seq_len]
                    global_selected_targets = torch.cat(all_selected_targets, dim=0)
                else:
                    # Fallback: no samples selected at all (shouldn't happen)
                    global_selected_inputs = local_candidate_inputs[:1]
                    global_selected_targets = local_candidate_targets[:1]

                # 5e: Each rank takes its assigned portion (even distribution)
                total_selected = global_selected_inputs.shape[0]
                samples_per_rank = total_selected // world_size
                remainder = total_selected % world_size

                my_start = rank * samples_per_rank + min(rank, remainder)
                my_count_final = samples_per_rank + (1 if rank < remainder else 0)
                my_end = my_start + my_count_final

                if my_count_final == 0:
                    my_start = 0
                    my_end = 1

                _global_selected_inputs = global_selected_inputs[my_start:my_end]
                _global_selected_targets = global_selected_targets[my_start:my_end]
                _use_global_redistribution = True
                
            else:
                # === Local Selection Mode (original behavior) ===
                selected_indices = opus_data_selection.get_batch_opus(
                    model=model,
                    buffer_seq=local_candidate_inputs,
                    buffer_labels=local_candidate_targets,
                    proxy_seq=None,
                    proxy_labels=None,
                    optimizer=optimizer,
                    trainable_layers=trainable_layers,
                    validation_loader=validation_loader,
                    selection_ratio=selection_ratio,
                    seq_len=seq_len,
                    selection_strategy=args.selection_strategy,
                    selection_method=args.opus_selection_method,
                    preconditioner=args.opus_preconditioner,
                    temperature=args.opus_temperature,
                    score_len=(score_len if score_len is not None else seq_len),
                    n_windows=args.opus_n_windows,
                )
            
            # Log detailed OPUS metrics
            if master_process and args.use_wandb and wandb.run is not None:
                num_candidates = local_candidate_inputs.shape[0] * (world_size if global_selection else 1)
                num_selected = global_num_to_select if global_selection else selected_indices.numel()
                wandb.log({
                    "opus/candidates": num_candidates,
                    "opus/selected": num_selected,
                    "opus/selection_ratio_actual": num_selected / num_candidates,
                    "opus/buffer_size_tokens": local_candidate_inputs.numel(),
                    "opus/global_selection": global_selection,
                })
        
        # === 5) Get the selected batches ===
        if global_selection and '_use_global_redistribution' in dir() and _use_global_redistribution:
            # Global mode: use redistributed data (already extracted above)
            selected_inputs = _global_selected_inputs
            selected_targets = _global_selected_targets
            final_batch_size = selected_inputs.shape[0]
            # Clean up flag
            del _use_global_redistribution
        else:
            # Local mode: use index_select on local candidates
            if selected_indices.dtype != torch.long:
                selected_indices = selected_indices.to(torch.long)
            if selected_indices.device != local_candidate_inputs.device:
                selected_indices = selected_indices.to(local_candidate_inputs.device)
            selected_inputs = local_candidate_inputs.index_select(0, selected_indices)
            selected_targets = local_candidate_targets.index_select(0, selected_indices)
            final_batch_size = selected_indices.numel()

        # === 6) Yield selected sequences one by one ===
        for i in range(final_batch_size):
            yield selected_inputs[i].contiguous(), selected_targets[i].contiguous()



# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    """Training hyperparameters - values are set from command line arguments."""
    # Model and dataset
    model_type: str = "gpt2-xl"
    dataset: str = "fineweb_edu3plus_custom"
    optimizer_type: str = "adamw_unified"
    
    # Sequence lengths
    train_seq_len: int = 6 * 1024
    val_seq_len: int = 8 * 1024
    val_tokens: int = 100_000_000
    
    # Token-based training
    total_tokens_b: float = 30.0
    eval_every_tokens_b: float = 1.0
    checkpoint_every_tokens_b: float = 0.0
    num_iterations: int = 0  # Calculated from tokens
    cooldown_frac: float = 0.45
    
    # Internal training parameters (not CLI-exposed)
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    warmup_frac: float = 0.0
    min_lr_ratio: float = 0.1
    val_loss_every: int = 0  # Deprecated, kept for compatibility
    
    # Data paths (set via CLI)
    train_files: str = None
    val_files: str = None
    data_shuffle_seed: int = 42
    
    # Training settings
    save_checkpoint: bool = True
    grad_clip_norm: float = 1.0
    eval_mode: str = "inline"
    
    # OPUS settings
    use_opus: bool = False
    selection_strategy: str = "opus"
    opus_selection_method: str = "stochastic"
    opus_preconditioner: str = "auto"
    opus_buffer_size: int = 32 * 6 * 1024  # 32 * train_seq_len
    opus_selection_ratio: float = 0.5
    opus_temperature: float = 0.9
    opus_score_len: int = 512
    opus_proxy_batch: int = 16
    opus_proxy_dir: str = None
    opus_proxy_files: str = None  # Internal, same as proxy_dir
    opus_proxy_tokens: int = 30_000_000
    opus_proxy_mode: str = "refresh"
    opus_proxy_refresh_interval: int = 1
    opus_n_windows: int = 1
    opus_global_selection: bool = False
    
    # Random Projection
    use_random_projection: bool = False
    projection_dim: int = 8192
    projection_seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "muon-pretrain"
    wandb_run_name: str = None
    enable_profiling: bool = False
    output_root: str = "."
    
    def get_experiment_name(self) -> str:
        """Generate experiment name based on key hyperparameters"""
        import re
        # Convert sequence lengths to readable format (e.g., 48K, 64K)
        train_seq_k = self.train_seq_len // 1024
        val_seq_k = self.val_seq_len // 1024
        
        # Create base name with model type, dataset, and sequence lengths
        model_name = self.model_type.upper().replace("-", "")  # gpt2-xl -> GPT2XL
        dataset_name = self.dataset.upper()  # fineweb30b -> FINEWEB30B
        optimizer_name = self.optimizer_type.upper().replace("_", "") # muon_hybrid -> MUONHYBRID
        name_parts = ["experiment", model_name, dataset_name, optimizer_name, f"train{train_seq_k}K", f"val{val_seq_k}K"]
        
        # Include total training tokens budget (in billions)
        try:
            tok_str = f"TOK{self.total_tokens_b:g}B"
            name_parts.append(tok_str)
        except Exception:
            pass
        
        # Add score directory information if using domain_root_dir
        if hasattr(self, 'domain_root_dir') and self.domain_root_dir is not None:
            root_dirs = [d.strip() for d in self.domain_root_dir.split(',')]
            score_nums = []
            for root_dir in root_dirs:
                last_component = root_dir.rstrip('/').split('/')[-1]
                if last_component.isdigit():
                    score_nums.append(last_component)
            
            if score_nums:
                score_str = "score" + "-".join(sorted(score_nums))
                name_parts.append(score_str)
        
        # Add OPUS configuration
        if self.use_opus:
            name_parts.append("OPUS")
            name_parts.append(self.selection_strategy.upper())
            name_parts.append(self.opus_selection_method[:4])
            name_parts.append(self.opus_preconditioner.upper())
            
            buffer_multiplier = self.opus_buffer_size // self.train_seq_len
            name_parts.append(f"B{buffer_multiplier}")
            
            ratio_pct = int(self.opus_selection_ratio * 100)
            name_parts.append(f"R{ratio_pct}")
            
            if self.opus_selection_method == "stochastic":
                temp_scaled = int(self.opus_temperature * 10)
                name_parts.append(f"T{temp_scaled:02d}")
            if getattr(self, 'opus_score_len', None):
                name_parts.append(f"S{self.opus_score_len}")
            if getattr(self, 'opus_proxy_batch', None):
                name_parts.append(f"PB{self.opus_proxy_batch}")
        else:
            name_parts.append("BASELINE")
        
        return "_".join(name_parts)

def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="GPT Training with OPUS Data Selection")
    
    # ============ Model & Dataset ============
    parser.add_argument("--model_type", type=str, default="gpt2-xl", 
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-7b",
                                "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"],
                       help="Model architecture (GPT-2 or Qwen)")
    parser.add_argument("--dataset", type=str, default="fineweb_edu3plus_custom",
                       help="Dataset name (for logging)")
    parser.add_argument("--optimizer_type", type=str, default="adamw_unified",
                       choices=["muon_hybrid", "adamw_unified"],
                       help="Optimizer type")
    
    # ============ Data Paths ============
    parser.add_argument("--train_files", type=str, default=None,
                       help="Glob pattern for training files (e.g., '/path/to/*.bin')")
    parser.add_argument("--val_files", type=str, default=None,
                       help="Glob pattern for validation files")
    parser.add_argument("--domain_root_dir", type=str, default=None,
                       help="Comma-separated directories for domain-organized training data (CPT)")
    parser.add_argument("--val_dir", type=str, default=None,
                       help="Comma-separated directories for validation data (CPT)")
    parser.add_argument("--data_shuffle_seed", type=int, default=42,
                       help="Random seed for data shuffling")
    parser.add_argument("--no_train_file_shuffle", action="store_true",
                       help="Disable shuffling of training file order when using --domain_root_dir (keep sorted order).")
    
    # ============ Sequence Length & Attention ============
    parser.add_argument("--train_seq_len", type=int, default=6*1024,
                       help="Training sequence length")
    parser.add_argument("--val_seq_len", type=int, default=8*1024,
                       help="Validation sequence length")
    parser.add_argument("--val_tokens", type=int, default=100_000_000,
                       help="Total validation tokens evaluated per validation run (default: 100M). "
                            "For smoke tests, set small (e.g., 65536 or 262144).")
    parser.add_argument("--attention_mode", type=str, default="flex",
                       choices=["flex", "full", "sdpa", "manual"],
                       help="Attention mode: 'flex' uses FlexAttention sliding-window; "
                            "'full' uses full causal attention (pass None to model, recommended for Qwen3 CPT)")
    
    # ============ Token-based Training ============
    parser.add_argument("--total_tokens_b", type=float, default=30.0,
                       help="Total tokens to train (billions)")
    parser.add_argument("--eval_every_tokens_b", type=float, default=1.0,
                       help="Evaluate every N billion tokens")
    parser.add_argument("--checkpoint_every_tokens_b", type=float, default=0.0,
                       help="Save a checkpoint every N billion tokens (0 disables periodic checkpoint-only saves)")
    parser.add_argument("--eval_mode", type=str, default="inline", choices=["external", "inline"],
                       help="'inline' continues after eval; 'external' exits for separate eval")
    
    # ============ Training Parameters ============
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps (overrides warmup_frac if > 0)")
    parser.add_argument("--warmup_frac", type=float, default=0.0,
                       help="Warmup fraction of total training")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                       choices=["cosine", "linear", "legacy"],
                       help="Learning rate schedule")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                       help="Minimum LR ratio for cosine schedule")
    
    # ============ Optimizer Parameters ============
    parser.add_argument("--adam_lr", type=float, default=None,
                       help="Override Adam learning rate (default: model-specific)")
    parser.add_argument("--muon_lr", type=float, default=None,
                       help="Override Muon learning rate (default: model-specific)")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                       help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0.1,
                       help="Adam weight decay")
    
    # ============ CPT (Continual Pre-Training) ============
    parser.add_argument("--init_model", type=str, default=None,
                       help="Path to pretrained model for CPT (e.g., HuggingFace model path)")
    parser.add_argument("--hf_compatible", action="store_true",
                       help="Use HuggingFace-compatible architecture")
    parser.add_argument("--use_loss_mask", action="store_true",
                       help="Use loss masking for CPT (mask metadata/titles)")
    parser.add_argument("--loss_mask_suffix", type=str, default=".lossmask",
                       help="Suffix for loss mask files")
    
    # ============ OPUS Configuration ============
    parser.add_argument("--use_opus", action="store_true",
                       help="Enable OPUS data selection")
    parser.add_argument("--selection_strategy", type=str, default="opus",
                       choices=["opus", "ppl", "random"],
                       help="Selection strategy: opus/ppl/random")
    parser.add_argument("--opus_selection_method", type=str, default="stochastic",
                       choices=["greedy", "stochastic", "high", "low", "mid"],
                       help="Selection method")
    parser.add_argument("--opus_preconditioner", type=str, default="auto",
                       choices=["auto", "sgd", "adamw", "muon"],
                       help="Preconditioner type")
    parser.add_argument("--opus_buffer_size_multiplier", type=int, default=32,
                       help="Buffer size = multiplier * train_seq_len")
    parser.add_argument("--opus_selection_ratio", type=float, default=0.5,
                       help="Fraction of buffer to select")
    parser.add_argument("--opus_temperature", type=float, default=0.9,
                       help="Temperature for stochastic selection")
    parser.add_argument("--opus_score_len", type=int, default=512,
                       help="Scoring window length (multiple of 128)")
    parser.add_argument("--opus_proxy_batch", type=int, default=16,
                       help="Proxy batch size for scoring")
    parser.add_argument("--opus_n_windows", type=int, default=1,
                       help="Number of random windows per candidate for scoring (mean-aggregated)")
    parser.add_argument("--opus_global_selection", action="store_true",
                       help="Use global selection across all GPUs")
    
    # ============ Proxy Data ============
    parser.add_argument("--opus_proxy_dir", type=str, default=None,
                       help="Glob pattern for proxy files (e.g., '/path/to/proxy/*.bin')")
    parser.add_argument("--opus_proxy_tokens", type=int, default=30_000_000,
                       help="Total proxy tokens to sample")
    parser.add_argument("--opus_proxy_mode", type=str, default="refresh",
                       choices=["fixed", "refresh"],
                       help="Proxy batch mode: 'fixed' reuses same batch, 'refresh' resamples")
    parser.add_argument("--opus_proxy_refresh_interval", type=int, default=1,
                       help="Number of OPUS calls between proxy batch refreshes (when mode=refresh)")
    
    # ============ Random Projection ============
    parser.add_argument("--use_random_projection", action="store_true",
                       help="Enable random projection for gradient embeddings")
    parser.add_argument("--projection_dim", type=int, default=8192,
                       help="Projection dimension")
    parser.add_argument("--projection_seed", type=int, default=42,
                       help="Projection random seed")
    
    # ============ Logging & Checkpoints ============
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Custom experiment name (required for clear identification)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint for resuming training")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="muon-pretrain",
                       help="Wandb project name")
    parser.add_argument("--enable_profiling", action="store_true",
                       help="Enable timing profiling")
    parser.add_argument("--output_root", type=str, default=".",
                       help="Root directory for logs/checkpoints (logs are written under <output_root>/logs)")
    parser.add_argument(
        "--no_torch_compile", "--no_compile",
        action="store_true",
        dest="no_torch_compile",
        help="Disable torch.compile (saves compile time; may affect peak memory)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in transformer blocks (trades compute for memory)",
    )
    
    return parser.parse_args()

# Parse command line arguments and create config
cmd_args = parse_args()
args = Hyperparameters()

# Copy all CLI arguments to args
# Model and dataset
args.model_type = cmd_args.model_type
args.dataset = cmd_args.dataset
args.optimizer_type = cmd_args.optimizer_type

# Data paths
args.train_files = cmd_args.train_files
args.val_files = cmd_args.val_files
args.domain_root_dir = cmd_args.domain_root_dir
args.val_dir = cmd_args.val_dir
args.data_shuffle_seed = cmd_args.data_shuffle_seed
args.no_train_file_shuffle = getattr(cmd_args, "no_train_file_shuffle", False)

# Sequence lengths and attention
args.train_seq_len = cmd_args.train_seq_len
args.val_seq_len = cmd_args.val_seq_len
args.val_tokens = cmd_args.val_tokens
args.attention_mode = cmd_args.attention_mode
use_flex_attention = str(args.attention_mode).lower() == "flex"

# Token-based training
args.total_tokens_b = cmd_args.total_tokens_b
args.eval_every_tokens_b = cmd_args.eval_every_tokens_b
args.checkpoint_every_tokens_b = cmd_args.checkpoint_every_tokens_b
args.eval_mode = cmd_args.eval_mode

# Training parameters
args.grad_accum_steps = cmd_args.grad_accum_steps
args.grad_clip_norm = cmd_args.grad_clip_norm
args.warmup_steps = cmd_args.warmup_steps
args.warmup_frac = cmd_args.warmup_frac
args.lr_schedule = cmd_args.lr_schedule
args.min_lr_ratio = cmd_args.min_lr_ratio

# CPT settings
args.init_model = cmd_args.init_model
args.hf_compatible = cmd_args.hf_compatible
args.use_loss_mask = cmd_args.use_loss_mask
args.loss_mask_suffix = cmd_args.loss_mask_suffix

# OPUS settings
args.use_opus = cmd_args.use_opus
args.selection_strategy = cmd_args.selection_strategy
args.opus_selection_method = cmd_args.opus_selection_method
args.opus_preconditioner = cmd_args.opus_preconditioner
args.opus_buffer_size = cmd_args.opus_buffer_size_multiplier * args.train_seq_len
args.opus_selection_ratio = cmd_args.opus_selection_ratio
args.opus_temperature = cmd_args.opus_temperature
args.opus_score_len = cmd_args.opus_score_len
args.opus_proxy_batch = cmd_args.opus_proxy_batch
args.opus_proxy_dir = cmd_args.opus_proxy_dir
args.opus_proxy_tokens = cmd_args.opus_proxy_tokens
args.opus_n_windows = cmd_args.opus_n_windows
args.opus_proxy_mode = cmd_args.opus_proxy_mode
args.opus_proxy_refresh_interval = cmd_args.opus_proxy_refresh_interval
args.opus_global_selection = cmd_args.opus_global_selection
args.opus_proxy_files = args.opus_proxy_dir  # Alias for compatibility

# Random Projection
args.use_random_projection = cmd_args.use_random_projection
args.projection_dim = cmd_args.projection_dim
args.projection_seed = cmd_args.projection_seed

# Logging
args.use_wandb = cmd_args.use_wandb
args.wandb_project = cmd_args.wandb_project
args.enable_profiling = cmd_args.enable_profiling
args.output_root = cmd_args.output_root

# Checkpoint resumption
resume_checkpoint_path = cmd_args.resume_from_checkpoint

def output_root_path() -> Path:
    return Path(args.output_root).expanduser()

def logs_root_path() -> Path:
    return output_root_path() / "logs"

def experiment_logs_dir(name: str) -> Path:
    return logs_root_path() / name

# Validate data paths - need either train_files or domain_root_dir
if args.train_files is None and args.domain_root_dir is None:
    raise ValueError("Either --train_files or --domain_root_dir must be provided")
if args.val_files is None and args.val_dir is None:
    raise ValueError("Either --val_files or --val_dir must be provided")

# For file lists (used with domain_root_dir)
args.train_files_list = None
args.val_files_list = None

# Handle domain-organized data (overrides train_files with shuffled file list)
if args.domain_root_dir is not None:
    is_master = int(os.environ.get("RANK", 0)) == 0
    
    # Parse comma-separated directories
    root_dirs = [d.strip() for d in args.domain_root_dir.split(',')]
    do_shuffle = not bool(getattr(args, "no_train_file_shuffle", False))
    
    if is_master:
        print(f"Using domain-organized data from {len(root_dirs)} directory(ies):")
        for root_dir in root_dirs:
            print(f"  - {root_dir}")
        if do_shuffle:
            print(f"Shuffle seed: {args.data_shuffle_seed}")
        else:
            print("Shuffle: disabled (sorted file order)")
    
    # Collect files from all sources
    all_train_files = []
    for root_dir in root_dirs:
        train_files = discover_all_train_files_flattened(
            root_dir,
            seed=args.data_shuffle_seed,
            exclude_val=True,
            shuffle_files=do_shuffle,
        )
        all_train_files.extend(train_files)
        if is_master:
            print(f"  Found {len(train_files)} files in {root_dir}")
    
    if do_shuffle:
        random.seed(args.data_shuffle_seed)
        random.shuffle(all_train_files)
    else:
        all_train_files.sort()
    
    if is_master:
        shuffle_str = "shuffled" if do_shuffle else "sorted"
        print(f"Total: {len(all_train_files)} training files ({shuffle_str})")
        if len(all_train_files) > 0:
            print(f"Example files (first 5):")
            for i, f in enumerate(all_train_files[:5]):
                print(f"  {i+1}. {f}")
    
    args.train_files_list = all_train_files

# Handle validation directory
if args.val_dir is not None:
    is_master = int(os.environ.get("RANK", 0)) == 0
    
    # Parse comma-separated directories
    val_dirs = [d.strip() for d in args.val_dir.split(',')]
    
    if is_master:
        print(f"Using validation data from {len(val_dirs)} directory(ies):")
        for val_dir in val_dirs:
            print(f"  - {val_dir}")
    
    # Collect files from all validation directories
    all_val_files = []
    for val_dir in val_dirs:
        val_files = discover_all_train_files_flattened(
            val_dir,
            seed=args.data_shuffle_seed,
            exclude_val=False
        )
        all_val_files.extend(val_files)
        if is_master:
            print(f"  Found {len(val_files)} files in {val_dir}")
    
    if is_master:
        print(f"Total: {len(all_val_files)} validation files")
    
    args.val_files_list = all_val_files


# Calculate num_iterations from total tokens if not explicitly overridden
def calculate_iterations_from_tokens(total_tokens_b: float, world_size: int, train_seq_len: int, grad_accum_steps: int = 1) -> int:
    """Calculate number of iterations needed to process total_tokens_b billion tokens"""
    if total_tokens_b <= 0:
        # Keep >=1 to avoid divide-by-zero in warmup/window schedule.
        return 1
    total_tokens = total_tokens_b * 1e9  # Convert billions to actual tokens
    tokens_per_step = world_size * train_seq_len * grad_accum_steps  # Tokens processed per training step
    iterations = round(total_tokens / tokens_per_step)  # Round to nearest integer for better accuracy
    return max(1, int(iterations))

def calculate_tokens_processed(step: int, world_size: int, train_seq_len: int, grad_accum_steps: int = 1) -> float:
    """Calculate total tokens processed up to current step (in billions)"""
    tokens_per_step = world_size * train_seq_len * grad_accum_steps
    total_tokens = step * tokens_per_step
    return total_tokens / 1e9  # Convert to billions

def run_opencompass_evaluation(checkpoint_path: str, model_type: str, experiment_name: str, step: int, tokens_b: float, is_final: bool) -> tuple[bool, float]:
    """Run OpenCompass evaluation and return success status and average accuracy"""
    import subprocess
    import shutil
    
    try:
        # Run evaluation using run_fast_eval.sh
        eval_cmd = [
            "./run_fast_eval.sh",
            checkpoint_path,
            model_type,
            f"{experiment_name}_eval_step{step:06d}",
            str(world_size),
        ]
        
        print0(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Parse evaluation results to get average accuracy
            avg_accuracy = parse_evaluation_results(f"{experiment_name}_eval_step{step:06d}")
            return True, avg_accuracy
        else:
            print0(f"Evaluation failed with return code {result.returncode}")
            print0(f"Error output: {result.stderr}")
            return False, None
            
    except Exception as e:
        print0(f"Evaluation exception: {str(e)}")
        return False, None

def parse_evaluation_results(eval_experiment_name: str) -> float:
    """Parse OpenCompass evaluation results and calculate average accuracy"""
    import csv
    import glob
    
    # Find the latest evaluation summary CSV
    summary_pattern = f"outputs/nanogpt_eval_{eval_experiment_name}/*/summary/summary_*.csv"
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        print0(f"No summary files found for pattern: {summary_pattern}")
        return None
    
    # Use the most recent summary file
    summary_file = max(summary_files, key=os.path.getmtime)
    print0(f"Parsing results from: {summary_file}")
    
    accuracies = []
    try:
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get the accuracy value (last column should be the model results)
                model_columns = [col for col in row.keys() if col.startswith('nanogpt-')]
                if model_columns:
                    accuracy_str = row[model_columns[0]]
                    if accuracy_str and accuracy_str != '-' and accuracy_str.replace('.', '').isdigit():
                        accuracies.append(float(accuracy_str))
        
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print0(f"Parsed {len(accuracies)} accuracy values, average: {avg_accuracy:.4f}")
            return avg_accuracy
        else:
            print0("No valid accuracy values found")
            return None
            
    except Exception as e:
        print0(f"Error parsing evaluation results: {str(e)}")
        return None

def cleanup_intermediate_files(checkpoint_path: str, experiment_name: str, step: int):
    """Clean up intermediate checkpoint and evaluation outputs (keep only final)"""
    import shutil
    import glob
    
    try:
        # Remove intermediate checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print0(f"Removed intermediate checkpoint: {checkpoint_path}")
        
        # Remove intermediate evaluation outputs
        eval_output_pattern = f"outputs/nanogpt_eval_{experiment_name}_eval_step{step:06d}"
        eval_dirs = glob.glob(eval_output_pattern)
        for eval_dir in eval_dirs:
            if os.path.exists(eval_dir):
                shutil.rmtree(eval_dir)
                print0(f"Removed intermediate evaluation output: {eval_dir}")
                
    except Exception as e:
        print0(f"Error during cleanup: {str(e)}")

def cleanup_previous_checkpoints_and_evals(experiment_name: str, current_step: int, preserve_checkpoint_path: str = None):
    """Clean up all previous checkpoints and evaluation outputs after resuming training"""
    import shutil
    import glob
    
    try:
        # Clean up all previous checkpoints (keep only the one we just loaded from)
        checkpoint_pattern = str(experiment_logs_dir(experiment_name) / "state_step*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        for checkpoint_file in checkpoint_files:
            # Extract step number from filename
            try:
                step_str = checkpoint_file.split('state_step')[1].split('.pt')[0]
                step_num = int(step_str)
                
                # Remove checkpoints from previous steps (but keep current one and the one we resumed from)
                if step_num < current_step and (preserve_checkpoint_path is None or os.path.abspath(checkpoint_file) != os.path.abspath(preserve_checkpoint_path)):
                    os.remove(checkpoint_file)
                    print0(f"Cleaned up previous checkpoint: {checkpoint_file}")
                elif preserve_checkpoint_path and os.path.abspath(checkpoint_file) == os.path.abspath(preserve_checkpoint_path):
                    print0(f"Preserved checkpoint we resumed from: {checkpoint_file}")
                    
            except (IndexError, ValueError):
                # Skip files that don't match expected pattern
                continue
        
        # Clean up all previous evaluation outputs
        eval_output_pattern = f"outputs/nanogpt_eval_{experiment_name}_eval_step*"
        eval_dirs = glob.glob(eval_output_pattern)
        
        for eval_dir in eval_dirs:
            try:
                # Extract step number from directory name
                step_str = eval_dir.split('_eval_step')[1]
                step_num = int(step_str)
                
                # Remove evaluation outputs from previous steps
                if step_num < current_step:
                    if os.path.exists(eval_dir):
                        shutil.rmtree(eval_dir)
                        print0(f"Cleaned up previous evaluation output: {eval_dir}")
                        
            except (IndexError, ValueError):
                # Skip directories that don't match expected pattern
                continue
                
        print0(f"Cleanup completed for experiment {experiment_name} up to step {current_step}")
                
    except Exception as e:
        print0(f"Error during previous files cleanup: {str(e)}")

def cleanup_intermediate_checkpoints_and_evals(experiment_name: str, final_step: int):
    """Clean up all intermediate checkpoints and evaluation outputs, keeping only the final step"""
    import shutil
    import glob
    
    if not master_process:
        return  # Only master process should do cleanup
    
    try:
        print0(f"Starting final cleanup for experiment {experiment_name} (keeping only step {final_step})", console=True)
        
        # Clean up intermediate checkpoints (keep only final step)
        checkpoint_pattern = str(experiment_logs_dir(experiment_name) / "state_step*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        removed_checkpoints = 0
        for checkpoint_file in checkpoint_files:
            # Extract step number from filename
            try:
                step_str = checkpoint_file.split('state_step')[1].split('.pt')[0]
                step_num = int(step_str)
                
                # Remove checkpoints from intermediate steps (keep only final step)
                if step_num < final_step:
                    os.remove(checkpoint_file)
                    print0(f"Removed intermediate checkpoint: {checkpoint_file}")
                    removed_checkpoints += 1
                elif step_num == final_step:
                    print0(f"Preserved final checkpoint: {checkpoint_file}")
                    
            except (IndexError, ValueError):
                # Skip files that don't match expected pattern
                continue
        
        # Clean up intermediate evaluation outputs (keep only final step)
        eval_output_pattern = f"outputs/nanogpt_eval_{experiment_name}_eval_step*"
        eval_dirs = glob.glob(eval_output_pattern)
        
        removed_evaluations = 0
        for eval_dir in eval_dirs:
            try:
                # Extract step number from directory name
                step_str = eval_dir.split('_eval_step')[1]
                step_num = int(step_str)
                
                # Remove evaluation outputs from intermediate steps (keep only final step)
                if step_num < final_step:
                    if os.path.exists(eval_dir):
                        shutil.rmtree(eval_dir)
                        print0(f"Removed intermediate evaluation: {eval_dir}")
                        removed_evaluations += 1
                elif step_num == final_step:
                    print0(f"Preserved final evaluation: {eval_dir}")
                        
            except (IndexError, ValueError):
                # Skip directories that don't match expected pattern
                continue
                
        print0(f"Final cleanup completed for experiment {experiment_name}:", console=True)
        print0(f"  Removed {removed_checkpoints} intermediate checkpoints", console=True)
        print0(f"  Removed {removed_evaluations} intermediate evaluations", console=True)
        print0(f"  Kept final checkpoint and evaluation at step {final_step}", console=True)
                
    except Exception as e:
        print0(f"Error during final cleanup: {str(e)}", console=True)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Calculate iterations from tokens if using token-based training
calculated_iterations = calculate_iterations_from_tokens(args.total_tokens_b, world_size, args.train_seq_len, args.grad_accum_steps)
if args.num_iterations <= 0 or args.num_iterations == 8000:  # Default/sentinel value, use calculated
    args.num_iterations = calculated_iterations
    
# Calculate evaluation intervals in steps
eval_interval_steps = calculate_iterations_from_tokens(args.eval_every_tokens_b, world_size, args.train_seq_len, args.grad_accum_steps)
checkpoint_interval_steps = 0
if args.checkpoint_every_tokens_b > 0:
    checkpoint_interval_steps = calculate_iterations_from_tokens(
        args.checkpoint_every_tokens_b,
        world_size,
        args.train_seq_len,
        args.grad_accum_steps,
    )

# Print configuration
if int(os.environ.get("RANK", 0)) == 0:  # Only print on master process
    print(f"   Model: {args.model_type}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Optimizer: {args.optimizer_type}")
    print(f"   Sequence lengths: train={args.train_seq_len//1024}K, val={args.val_seq_len//1024}K")
    print(f"   Token-based training: {args.total_tokens_b}B tokens, eval every {args.eval_every_tokens_b}B tokens")
    if checkpoint_interval_steps > 0:
        print(f"   Checkpoints: every {args.checkpoint_every_tokens_b}B tokens ({checkpoint_interval_steps} steps)")
    else:
        print(f"   Checkpoints: validation/final only")
    print(f"   Output root: {output_root_path()}")
    print(f"   Calculated iterations: {args.num_iterations} (eval every {eval_interval_steps} steps)")
    tokens_per_step = world_size * args.train_seq_len * args.grad_accum_steps
    print(f"   Tokens per step: {tokens_per_step:,} ({tokens_per_step/1e6:.1f}M)")
    print(f"   Gradient clipping: {'enabled' if args.grad_clip_norm > 0.0 else 'disabled'} (norm={args.grad_clip_norm})")
    # Show warmup configuration (important for CPT)
    if args.warmup_frac > 0:
        warmup_steps_eff = int(args.warmup_frac * calculated_iterations)
        print(f"   Warmup: {args.warmup_frac*100:.1f}% of training ({warmup_steps_eff} steps)")
    elif args.warmup_steps > 0:
        print(f"   Warmup: {args.warmup_steps} steps")
    else:
        print(f"   Warmup: disabled (not recommended for CPT!)")
    print(f"   OPUS: {'enabled' if args.use_opus else 'disabled'}")
    if args.use_opus:
        print(f"   OPUS config: {args.selection_strategy}/{args.opus_selection_method}/{args.opus_preconditioner}")
        print(f"   Buffer: {args.opus_buffer_size//args.train_seq_len}x, Ratio: {args.opus_selection_ratio}")
        print(f"   Score len: {args.opus_score_len}, Proxy batch: {args.opus_proxy_batch}")
    print(f"   Experiment: {args.get_experiment_name()}")
    print()

# assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Initialize random projection for OPUS (if enabled)
if args.use_opus and args.use_random_projection:
    if master_process:
        print(f"\n{'='*80}")
        print("🎲 Random Projection Configuration (DoReMi-style CountSketch)")
        print(f"{'='*80}")
        print(f"  Status: ENABLED")
        print(f"  Projection dimension: {args.projection_dim}")
        print(f"  Random seed: {args.projection_seed}")
        print(f"  Expected memory reduction: ~{100 * (1 - args.projection_dim / 1e6):.1f}% for embeddings")
        print(f"{'='*80}\n")
    
    # Initialize the global projector (all ranks must do this for consistency)
    opus_data_selection.initialize_projector(
        sketch_dim=args.projection_dim,
        seed=args.projection_seed,
        device=device.type,
        enabled=True
    )
elif args.use_opus:
    if master_process:
        print(f"\n{'='*80}")
        print("🔍 Random Projection: DISABLED (using full-dimensional gradients)")
        print(f"{'='*80}\n")
    
    # Initialize disabled projector
    opus_data_selection.initialize_projector(enabled=False)

# Signal handler for graceful shutdown with cleanup
def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT with graceful cleanup"""
    if master_process:
        print0(f"Received signal {signum}, performing cleanup before exit...", console=True)
        
        # IMPORTANT: Do NOT delete older checkpoints on kill by default.
        # Users often want to keep all checkpoints for debugging / analysis / resume.
        # Optional: enable space-saving cleanup with --cleanup_checkpoints.
        if getattr(cmd_args, "cleanup_checkpoints", False):
            # Try to find the experiment name and final step for cleanup
            if 'experiment_name' in globals() and experiment_name:
                try:
                    # Find the latest checkpoint to determine final step
                    import glob
                    checkpoint_pattern = str(experiment_logs_dir(experiment_name) / "state_step*.pt")
                    checkpoint_files = glob.glob(checkpoint_pattern)
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('state_step')[1].split('.pt')[0]))
                        final_step = int(latest_checkpoint.split('state_step')[1].split('.pt')[0])
                        print0(f"[cleanup_checkpoints] Keeping only latest checkpoint at step {final_step}", console=True)
                        cleanup_intermediate_checkpoints_and_evals(experiment_name, final_step)
                    else:
                        print0("[cleanup_checkpoints] No checkpoints found for cleanup", console=True)
                except Exception as e:
                        print0(f"[cleanup_checkpoints] Error during emergency cleanup: {str(e)}", console=True)
            else:
                print0("Preserving all checkpoints (no cleanup on SIGTERM/SIGINT).", console=True)
    
    # Clean shutdown
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print0(f"Process {rank}: Emergency exit complete", console=True)
    sys.exit(1)

# Register signal handlers after distributed initialization
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# begin logging
logfile = None
run_id = None
# Use custom experiment name if provided, otherwise auto-generate
experiment_name = cmd_args.experiment_name if cmd_args.experiment_name else args.get_experiment_name()
if master_process:
    run_id = uuid.uuid4()  # Keep for wandb config only
    os.makedirs(logs_root_path(), exist_ok=True)
    logfile = str(logs_root_path() / f"{experiment_name}.txt")
    print(logfile)
    
    # Initialize wandb with smart naming (if enabled)
    if args.use_wandb:
        wandb_run_name = args.wandb_run_name or experiment_name
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "experiment_name": experiment_name,
                "optimizer_type": args.optimizer_type,
                "model_type": args.model_type,
                "dataset": args.dataset,
                "train_seq_len": args.train_seq_len,
                "val_seq_len": args.val_seq_len,
                "num_iterations": args.num_iterations,
                "val_tokens": args.val_tokens,
                "cooldown_frac": args.cooldown_frac,
                "val_loss_every": args.val_loss_every,
                "grad_accum_steps": args.grad_accum_steps,
                "grad_clip_norm": args.grad_clip_norm,
                "warmup_steps": args.warmup_steps,
                "warmup_frac": args.warmup_frac,
                "checkpoint_every_tokens_b": args.checkpoint_every_tokens_b,
                "use_opus": args.use_opus,
                "selection_strategy": args.selection_strategy,
                "opus_selection_method": args.opus_selection_method,
                "opus_preconditioner": args.opus_preconditioner,
                "opus_buffer_size": args.opus_buffer_size,
                "opus_selection_ratio": args.opus_selection_ratio,
                "opus_temperature": args.opus_temperature,
                "output_root": args.output_root,
                "world_size": world_size,
                "run_id": str(run_id)
            }
        )
        print(f"🚀 Wandb run: {wandb_run_name}")
    else:
        print("📊 Wandb logging disabled")
else:
    run_id = uuid.uuid4()  # Still need this for non-master processes
def format_time_ms(ms):
    """Convert milliseconds to minutes and seconds format"""
    total_seconds = int(ms / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}m{seconds}s"

def parse_time_ms(time_str):
    """Parse time string like '123m45s' back to milliseconds"""
    import re
    match = re.match(r'(\d+)m(\d+)s', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return (minutes * 60 + seconds) * 1000
    return 0

def truncate_logfile_to_step(logfile_path, checkpoint_step):
    """Truncate log file to the last entry of checkpoint_step, and return the training_time_ms from that line"""
    if not os.path.exists(logfile_path):
        return 0
    
    try:
        with open(logfile_path, 'r') as f:
            lines = f.readlines()
        
        # Find the last line that contains the checkpoint step (validation log line)
        # Format: "step:X tokens:... val_loss:... val_ppl:..." or 
        #         "step:X/Y tokens:... train_time:..."
        import re
        last_valid_idx = -1
        training_time_ms = 0
        
        for i, line in enumerate(lines):
            # Match validation step line: "step:123 tokens:..."
            step_match = re.search(r'step:(\d+)(?:/\d+)?\s+tokens:', line)
            if step_match:
                step_num = int(step_match.group(1))
                if step_num <= checkpoint_step:
                    last_valid_idx = i
                    # Try to extract train_time from this line
                    time_match = re.search(r'train_time:(\d+m\d+s)', line)
                    if time_match:
                        training_time_ms = parse_time_ms(time_match.group(1))
        
        if last_valid_idx >= 0:
            # Truncate the file to include only lines up to and including last_valid_idx
            with open(logfile_path, 'w') as f:
                f.writelines(lines[:last_valid_idx + 1])
            print(f"📝 Truncated log file to step {checkpoint_step} (line {last_valid_idx + 1})")
            print(f"   Recovered training_time: {format_time_ms(training_time_ms)}")
            return training_time_ms
        else:
            print(f"⚠️ Could not find step {checkpoint_step} in log file, keeping original")
            return 0
            
    except Exception as e:
        print(f"⚠️ Error truncating log file: {e}")
        return 0

def print0(s, console=False):
    if master_process:
        log_dir = os.path.dirname(logfile)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

def save_training_checkpoint(step: int, tokens_processed_b: float, training_time_ms: float):
    if not (master_process and args.save_checkpoint):
        return None

    print0(f"Saving checkpoint at step {step} (tokens: {tokens_processed_b:.5f}B)...", console=True)
    model_state_dict = model.state_dict()
    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
        clean_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                clean_state_dict[key[len('_orig_mod.'):]] = value
            else:
                clean_state_dict[key] = value
        model_state_dict = clean_state_dict
        print0("Cleaned _orig_mod prefix from checkpoint keys for compatibility")

    log = dict(
        step=step,
        tokens_processed_b=tokens_processed_b,
        code=code,
        model=model_state_dict,
        optimizers=[opt.state_dict() for opt in optimizers],
        training_time_ms=training_time_ms,
    )
    checkpoint_dir = experiment_logs_dir(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"state_step{step:06d}.pt"
    torch.save(log, str(checkpoint_path))
    return str(checkpoint_path)

# begin by printing this file (the Python code)
# print0(code)
# print0("="*100)
# # log information about the hardware/software environment this is running on
# print0(f"Running Python {sys.version}")
# print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
# def nvidia_smi():
#     import subprocess  # avoid top level import
#     return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
# print0(nvidia_smi())
# print0("="*100)

# Create model with optional HuggingFace-compatible architecture
hf_compatible = cmd_args.hf_compatible
model: nn.Module = GPT.from_model_type(
    args.model_type, 
    max_seq_len=max(args.train_seq_len, args.val_seq_len),
    hf_compatible=hf_compatible
).cuda()
model.gradient_checkpointing = bool(cmd_args.gradient_checkpointing)

# Load HuggingFace pretrained weights if specified (for CPT)
# IMPORTANT: For Qwen models, load_qwen_weights may UNTIE embeddings to match HF architecture.
# We need to sync this architecture change across all ranks before broadcasting weights.
embeddings_untied = torch.tensor([0], dtype=torch.int32, device="cuda")

if cmd_args.init_model is not None:
    if model.llama_compatible:
        # Llama/Qwen architecture - use appropriate weight loading
        if args.model_type.startswith('qwen'):
            if master_process:
                # Check if embeddings were tied before loading
                was_tied = model.lm_head.weight is model.embed.weight
                model.load_qwen_weights(cmd_args.init_model, verbose=True)
                # Check if embeddings were untied during loading
                is_tied_now = model.lm_head.weight is model.embed.weight
                if was_tied and not is_tied_now:
                    embeddings_untied[0] = 1
                    if master_process:
                        print(f"[INFO] Embeddings were untied on rank 0 to match HF architecture")
        else:
            if master_process:
                model.load_llama_weights(cmd_args.init_model, verbose=True)
    elif hf_compatible:
        # GPT-2 HF compatible architecture - use direct weight loading
        if master_process:
            model.load_hf_weights(cmd_args.init_model, verbose=True)
    else:
        # Custom GPT-2 architecture - use approximate weight mapping (some weights skipped)
        if master_process:
            load_hf_gpt2_weights(model, cmd_args.init_model, master_process=True)

# Sync architecture change: if rank 0 untied embeddings, all ranks must untie too
dist.broadcast(embeddings_untied, 0)
if embeddings_untied[0] == 1 and not master_process:
    # Untie embeddings on non-master ranks to match rank 0's architecture
    vocab_size, model_dim = model.embed.weight.shape
    model.lm_head = nn.Linear(model_dim, vocab_size, bias=False, device=model.embed.weight.device, dtype=model.embed.weight.dtype)
    if rank == 1:  # Only print from one non-master rank to avoid spam
        print(f"[INFO] Rank {rank}: Untied embeddings to match rank 0")

# Convert model to bfloat16
# For Llama/Qwen models, convert entire model to maintain dtype consistency
if model.llama_compatible:
    model.bfloat16()
else:
    # For GPT-2 models, only convert embeddings
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

# Broadcast model weights from master to all other processes
# CRITICAL: Use param.data, not param.detach() - detach() creates a new tensor
# and broadcasting it won't update the original parameter!
for param in model.parameters():
    dist.broadcast(param.data, 0)

# Verify weights are correctly loaded and broadcast (for CPT debugging)
if cmd_args.init_model is not None and model.llama_compatible:
    from cpt.verify_weight_init import verify_weights
    weights_ok = verify_weights(model, rank, cmd_args.init_model)
    if not weights_ok:
        raise RuntimeError("Weight initialization verification failed! Check broadcast logic.")

# Log model parameters to wandb after model creation
if master_process and args.use_wandb and wandb.run is not None:
    wandb.config.update({"model_params": model.get_num_params()})

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

def unique_params(params):
    """Deduplicate tied/shared parameters while preserving order."""
    seen = set()
    result = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        result.append(p)
    return result

def get_model_learning_rates(model_type: str):
    """Get model-specific learning rates (unified DistAdam LR + Muon LR)"""
    if model_type == "gpt2":
        # GPT2-Small (124M params) - Keep original working config
        return {
            "adam_lr": 0.008,      # Unified DistAdam LR for all non-matrix params
            "muon_lr": 0.05,       # Muon LR for matrix params
        }
    elif model_type == "gpt2-medium":
        # GPT2-Medium (350M params) - Much more conservative rates
        return {
            "adam_lr": 0.006,      # Much lower for stability
            "muon_lr": 0.025,      # Much lower for stability
        }
    elif model_type == "gpt2-large":
        # GPT2-Large (774M params) - More conservative for larger model
        return {
            "adam_lr": 0.004,      # Lower for larger model
            "muon_lr": 0.015,       # Lower for larger model
        }
    elif model_type == "gpt2-xl":
        # GPT2-XL (1.5B params) - Most conservative for largest model
        return {
            "adam_lr": 0.002,      # Most conservative
            "muon_lr": 0.010,      # Most conservative
        }
    elif model_type == "gpt2-7b":
        # GPT2-7B (~7.1B params) - conservative defaults for random-init scratch training
        return {
            "adam_lr": 0.0003,
            "muon_lr": 0.003,
        }
    elif model_type == "qwen3-0.6b":
        # Qwen3-0.6B (600M params) - Conservative for CPT
        return {
            "adam_lr": 0.002,      # Conservative for CPT
            "muon_lr": 0.012,      # Conservative for CPT
        }
    elif model_type == "qwen3-1.7b":
        # Qwen3-1.7B (1.7B params) - Conservative for CPT
        return {
            "adam_lr": 0.001,      # Conservative for CPT
            "muon_lr": 0.008,      # Conservative for CPT
        }
    elif model_type == "qwen3-4b":
        # Qwen3-4B (4B params) - Very conservative
        return {
            "adam_lr": 0.0005,     # Very conservative
            "muon_lr": 0.004,      # Very conservative
        }
    elif model_type == "qwen3-8b":
        # Qwen3-8B (8B params) - Most conservative
        return {
            "adam_lr": 0.0002,     # Most conservative (same as Llama-3.1-8B)
            "muon_lr": 0.002,      # Most conservative (same as Llama-3.1-8B)
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Get model-specific learning rates
lr_config = get_model_learning_rates(args.model_type)

# Override with command line arguments if provided (useful for CPT)
if cmd_args.adam_lr is not None:
    lr_config['adam_lr'] = cmd_args.adam_lr
    if master_process:
        print(f"📌 Using custom Adam LR from command line: {cmd_args.adam_lr}")
if cmd_args.muon_lr is not None:
    lr_config['muon_lr'] = cmd_args.muon_lr
    if master_process:
        print(f"📌 Using custom Muon LR from command line: {cmd_args.muon_lr}")

# init the optimizer(s) based on selected type
if args.optimizer_type == "muon_hybrid":
    # Muon Hybrid: DistAdam for non-matrix params + Muon for matrix params
    if master_process:
        print(f"Using Muon Hybrid optimizer (DistAdam + Muon) with unified learning rates:")
        print(f"  DistAdam LR: {lr_config['adam_lr']} (for head/embed/scalar params)")
        print(f"  Muon LR: {lr_config['muon_lr']} (for matrix params)")
    
    # DistAdam for all non-matrix parameters (head, embed, scalar)
    non_matrix_params = unique_params(scalar_params + head_params + embed_params)
    optimizer1 = DistAdam(
        non_matrix_params,
        lr=lr_config['adam_lr'],
        betas=(cmd_args.adam_beta1, cmd_args.adam_beta2),
        eps=1e-8,
        weight_decay=cmd_args.adam_weight_decay,
    )
    
    # Muon for matrix parameters
    optimizer2 = Muon(hidden_matrix_params, lr=lr_config['muon_lr'], momentum=0.95, weight_decay=0.0)
    optimizers = [optimizer1, optimizer2]
    
elif args.optimizer_type == "adamw_unified":
    # Unified AdamW: DistAdam for all parameters
    if master_process:
        print(f"Using unified DistAdam optimizer with single learning rate: {lr_config['adam_lr']}")
    
    all_params = list(model.parameters())
    optimizer1 = DistAdam(
        all_params,
        lr=lr_config['adam_lr'],
        betas=(cmd_args.adam_beta1, cmd_args.adam_beta2),
        eps=1e-8,
        weight_decay=cmd_args.adam_weight_decay,
    )
    optimizers = [optimizer1]
else:
    raise ValueError(f"Unknown optimizer_type: {args.optimizer_type}")

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# Load checkpoint if resuming (ALL processes must load)
start_step = 0
if resume_checkpoint_path:
    if master_process:
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
    checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
    
    # Handle torch.compile() state_dict key mismatch
    model_state_dict = checkpoint['model']
    
    # Check if we have _orig_mod prefix (compiled model checkpoint)
    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
        # Remove _orig_mod prefix from all keys
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]  # Remove the prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model_state_dict = new_state_dict
        if master_process:
            print("Removed _orig_mod prefix from checkpoint keys for compatibility")
    
    model.load_state_dict(model_state_dict)
    for i, opt in enumerate(optimizers):
        opt.load_state_dict(checkpoint['optimizers'][i])
    start_step = checkpoint['step'] + 1  # Resume from next step
    checkpoint_step = checkpoint['step']
    
    # Get training_time_ms: prefer checkpoint value, fallback to parsing from log file
    resumed_training_time_ms = checkpoint.get('training_time_ms', 0)
    
    if master_process:
        print(f"Resumed from step {checkpoint_step}, starting at step {start_step}")
        
        # Truncate log file to checkpoint step and recover training_time if not in checkpoint
        logfile_training_time_ms = truncate_logfile_to_step(logfile, checkpoint_step)
        
        if resumed_training_time_ms > 0:
            print(f"Previous training time from checkpoint: {format_time_ms(resumed_training_time_ms)}")
        elif logfile_training_time_ms > 0:
            resumed_training_time_ms = logfile_training_time_ms
            print(f"Previous training time from log file: {format_time_ms(resumed_training_time_ms)}")
        else:
            print("⚠️ No previous training time found, starting from 0")
        
        # Clean up previous checkpoints and evaluation outputs after successful resume
        # But preserve the checkpoint we just resumed from
        # DEBUGGING: Temporarily disabled cleanup to debug constant accuracy issue
        # cleanup_previous_checkpoints_and_evals(experiment_name, start_step, resume_checkpoint_path)

# Initialize resumed_training_time_ms for non-resume case
if not resume_checkpoint_path:
    resumed_training_time_ms = 0

# Broadcast start_step and resumed_training_time_ms to all processes (for consistency)
if dist.is_initialized():
    start_step_tensor = torch.tensor(start_step, device=device)
    dist.broadcast(start_step_tensor, src=0)
    start_step = start_step_tensor.item()
    
    # Also broadcast resumed_training_time_ms
    resumed_time_tensor = torch.tensor(resumed_training_time_ms, device=device, dtype=torch.float64)
    dist.broadcast(resumed_time_tensor, src=0)
    resumed_training_time_ms = resumed_time_tensor.item()
    
    # Add a barrier to ensure all processes are synchronized after checkpoint load
    dist.barrier()

# learning rate schedule: warmup -> stable -> decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    
    # Calculate effective warmup steps
    # warmup_frac takes priority if specified
    if args.warmup_frac > 0:
        warmup_steps = int(args.warmup_frac * args.num_iterations)
    else:
        warmup_steps = args.warmup_steps
    
    # Warmup phase: linear increase from 0 to 1
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    
    # Cosine decay schedule (Megatron-like): warmup -> cosine to min_lr_ratio
    if getattr(args, "lr_schedule", "legacy") == "cosine":
        # If warmup_steps == num_iterations, keep LR at 1.0 (degenerate)
        denom = max(1, args.num_iterations - warmup_steps)
        p = (step - warmup_steps) / denom
        # Clamp to [0, 1] for safety
        if p < 0:
            p = 0.0
        elif p > 1:
            p = 1.0
        # cosine from 1.0 -> min_lr_ratio
        return args.min_lr_ratio + 0.5 * (1.0 - args.min_lr_ratio) * (1.0 + math.cos(math.pi * p))
    
    # Legacy schedule: warmup -> stable -> linear cooldown to min_lr_ratio
    # Stable phase: constant LR
    if x < 1 - args.cooldown_frac:
        return 1.0
    # Cooldown phase: linear decay to min_lr_ratio
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * args.min_lr_ratio

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    # For gradient accumulation, we need to use the effective step number
    # which should be step // grad_accum_steps for the main training loop
    effective_step = step // args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else step
    x = effective_step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

if not cmd_args.no_torch_compile:
    model = torch.compile(model, dynamic=False)
elif master_process:
    print0("torch.compile disabled (--no_torch_compile / --no_compile)")

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state

# Use file list if domain_root_dir was provided, otherwise use glob pattern
warmup_train_data = args.train_files_list if args.train_files_list is not None else args.train_files
train_loader = distributed_data_generator(
    warmup_train_data,
    world_size * args.train_seq_len,
    align_to_bos=True,
    use_loss_mask=getattr(args, 'use_loss_mask', False),
    loss_mask_suffix=getattr(args, 'loss_mask_suffix', '.lossmask'),
)
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(1) if use_flex_attention else None).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

# Use file list if domain_root_dir was provided, otherwise use glob pattern
actual_train_data = args.train_files_list if args.train_files_list is not None else args.train_files

# Switch based on selection_strategy:
# - "opus": Use OPUS gradient-based selection
# - "ppl": Use PPL-based selection (high/low/mid difficulty)
# - "random": Use random selection from the same candidate buffer as OPUS
#             (strict control group - same data flow, only selection differs)
# - Others: Uniform streaming (no selection, no buffer)
if args.selection_strategy in ["opus", "ppl", "random"] and args.use_opus:
    # OPUS, PPL, and Random all use the same data generator infrastructure.
    # - OPUS: gradient-based selection via get_batch_opus(..., selection_strategy="opus")
    # - PPL: perplexity-based selection via get_batch_opus(..., selection_strategy="ppl")
    # - Random: uniform random selection via get_batch_opus(..., selection_strategy="random")
    #   This is a strict control group: same candidate buffer, only selection method differs.
    train_data_source = actual_train_data  # Either glob pattern or file list
    train_loader = opus_data_generator(
        model, 
        optimizers, 
        train_data_source,  # Can be glob pattern string or list of files
        args.train_seq_len, 
        args.opus_proxy_files, 
        args.opus_buffer_size, 
        args.opus_selection_ratio,
        proxy_dir=args.opus_proxy_dir,
        proxy_tokens_target=args.opus_proxy_tokens,
        score_len=args.opus_score_len,
        proxy_batch_size=args.opus_proxy_batch,
        global_selection=args.opus_global_selection,
        use_loss_mask=args.use_loss_mask,
        loss_mask_suffix=args.loss_mask_suffix,
    )
else:
    # Default: uniform streaming without selection (no buffer, no filtering)
    train_loader = distributed_data_generator(
        actual_train_data, 
        world_size * args.train_seq_len, 
        align_to_bos=False,
        use_loss_mask=args.use_loss_mask,
        loss_mask_suffix=args.loss_mask_suffix,
    )

# Initialize training time (restore from checkpoint if resuming)
training_time_ms = resumed_training_time_ms
if master_process and training_time_ms > 0:
    print(f"✅ Starting with accumulated training time: {format_time_ms(training_time_ms)}")
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
grad_accum = args.grad_accum_steps
# each update simulates 8-GPU synchronous update by accumulating gradients
for step in range(start_step, train_steps):
    # Check if we've processed enough tokens (more accurate than fixed iteration count)
    tokens_processed_b = calculate_tokens_processed(step + 1, world_size, args.train_seq_len, args.grad_accum_steps)
    last_step = (step == train_steps - 1) or (tokens_processed_b >= args.total_tokens_b)

    # --------------- VALIDATION / CHECKPOINT SECTION -----------------
    should_validate = last_step or (eval_interval_steps > 0 and step > 0 and step % eval_interval_steps == 0)
    periodic_checkpoint = (
        checkpoint_interval_steps > 0 and step > 0 and step % checkpoint_interval_steps == 0
    )
    should_checkpoint = should_validate or periodic_checkpoint

    if should_validate or should_checkpoint:
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        checkpoint_path = None

        if should_validate:
            with time_profile("validation", step=step, enabled=args.enable_profiling):
                model.eval()
            val_batch_size = world_size * args.val_seq_len

            # Make validation robust to batch size mismatches
            if args.val_tokens % val_batch_size != 0:
                # Use the largest number of complete validation steps possible
                val_steps = args.val_tokens // val_batch_size
            else:
                val_steps = args.val_tokens // val_batch_size

            assert val_steps > 0, f"No validation steps possible with val_tokens={args.val_tokens}, val_batch_size={val_batch_size}"

            # Use file list if val_dir was provided, otherwise use glob pattern
            actual_val_data = args.val_files_list if args.val_files_list is not None else args.val_files
            val_loader = distributed_data_generator(actual_val_data, val_batch_size, align_to_bos=False)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step) if use_flex_attention else None)
                val_loss /= val_steps
                del val_loader
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            # Calculate perplexity from validation loss
            val_ppl = torch.exp(val_loss)

            # Log validation loss and perplexity for plotting
            print0(f"step:{step} tokens:{tokens_processed_b:.5f}B val_loss:{val_loss.item():.4f} val_ppl:{val_ppl.item():.2f}")
            print0(f"step:{step}/{train_steps} tokens:{tokens_processed_b:.5f}B/{args.total_tokens_b}B val_loss:{val_loss:.4f} val_ppl:{val_ppl:.2f} train_time:{format_time_ms(training_time_ms)} step_avg:{format_time_ms(training_time_ms/max(step, 1))}", console=True)

            # Log to wandb
            if master_process and args.use_wandb and wandb.run is not None:
                wandb.log({
                    "val_loss": val_loss.item(),
                    "val_perplexity": val_ppl.item(),
                    "step": step,
                    "tokens_processed_b": tokens_processed_b,
                    "epoch": step / args.num_iterations,
                    "token_progress": tokens_processed_b / args.total_tokens_b,
                    "training_time_ms": training_time_ms,
                    "step_avg_ms": training_time_ms / max(step, 1),
                    "memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
                })
        elif master_process:
            print0(
                f"step:{step}/{train_steps} tokens:{tokens_processed_b:.5f}B/{args.total_tokens_b}B "
                f"checkpoint_only train_time:{format_time_ms(training_time_ms)} "
                f"step_avg:{format_time_ms(training_time_ms/max(step, 1))}",
                console=True,
            )

        if should_checkpoint:
            checkpoint_path = save_training_checkpoint(step, tokens_processed_b, training_time_ms)
        
        # Handle evaluation marker creation and coordinated exit for all processes
        if should_validate and not last_step and args.eval_mode == "external":
            # Only master process creates the marker file
            if master_process:
                eval_marker_file = str(experiment_logs_dir(experiment_name) / f"eval_needed_step{step:06d}.marker")
                with open(eval_marker_file, "w") as f:
                    f.write(f"checkpoint_path={checkpoint_path}\n")
                    f.write(f"model_type={args.model_type}\n")
                    f.write(f"experiment_name={experiment_name}\n")
                    f.write(f"step={step}\n")
                    f.write(f"tokens_b={tokens_processed_b}\n")
                
                print0(f"Created evaluation marker: {eval_marker_file}", console=True)
                print0(f"Training will exit for external evaluation. Use run_experiments.sh to resume.", console=True)
            
            # ALL processes must coordinate the exit to avoid distributed deadlock
            if dist.is_initialized():
                print0(f"Process {rank}: Synchronizing all processes before exit...", console=True)
                dist.barrier()  # Ensure all processes reach this point together
                torch.cuda.synchronize()  # Ensure all CUDA operations complete
            
            # Clean shutdown for master process only
            if master_process and args.use_wandb and wandb.run is not None:
                wandb.finish()
            
            # Destroy process group on ALL processes
            if dist.is_initialized():
                dist.destroy_process_group()
            
            print0(f"Process {rank}: Exiting cleanly for evaluation.", console=True)
            sys.exit(0)
        
        # Final step: create marker file for evaluation (same as intermediate steps)
        elif should_validate and last_step and args.eval_mode == "external":
            # Only master process creates the marker file
            if master_process:
                eval_marker_file = str(experiment_logs_dir(experiment_name) / f"eval_needed_step{step:06d}.marker")
                with open(eval_marker_file, "w") as f:
                    f.write(f"checkpoint_path={checkpoint_path}\n")
                    f.write(f"model_type={args.model_type}\n")
                    f.write(f"experiment_name={experiment_name}\n")
                    f.write(f"step={step}\n")
                    f.write(f"tokens_b={tokens_processed_b}\n")
                    f.write(f"final_step=true\n")  # Mark as final step
                
                print0(f"Created final evaluation marker: {eval_marker_file}", console=True)
                print0(f"Training completed. Final evaluation will be handled by run_experiments.sh", console=True)
            
            # ALL processes must coordinate the exit to avoid distributed deadlock
            if dist.is_initialized():
                print0(f"Process {rank}: Synchronizing all processes before final exit...", console=True)
                dist.barrier()  # Ensure all processes reach this point together
                torch.cuda.synchronize()  # Ensure all CUDA operations complete
            
            # Clean shutdown for master process only
            if master_process and args.use_wandb and wandb.run is not None:
                wandb.finish()
            
            # Optional: cleanup intermediate checkpoints/evals to save disk space
            if master_process and getattr(cmd_args, "cleanup_checkpoints", False):
                print0(f"[cleanup_checkpoints] Performing final cleanup for {experiment_name} (keeping only step {step})...", console=True)
                cleanup_intermediate_checkpoints_and_evals(experiment_name, step)
            elif master_process:
                print0("Preserving all checkpoints (no final cleanup).", console=True)
            
            # Destroy process group on ALL processes
            if dist.is_initialized():
                dist.destroy_process_group()
            
            print0(f"Process {rank}: Exiting cleanly after final step.", console=True)
            sys.exit(0)

        if should_validate:
            model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        # The last checkpoint is already saved in the validation block above
        break

    # --------------- TRAINING SECTION (with gradient accumulation) -----------------
    acc_loss = 0.0
    
    # Time data loading separately
    with time_profile("data_loading_step", step=step, enabled=args.enable_profiling):
        batch_data = []
        for accum in range(grad_accum):
            inputs, targets = next(train_loader)
            batch_data.append((inputs, targets))
    
    # Time forward/backward passes
    with time_profile("forward_backward", step=step, enabled=args.enable_profiling):
        for accum in range(grad_accum):
            inputs, targets = batch_data[accum]
            l = model(inputs, targets, get_window_size_blocks(step) if use_flex_attention else None)
            (l / grad_accum).backward()
            acc_loss += l.item()
    # Time optimizer steps
    with time_profile("optimizer_step", step=step, enabled=args.enable_profiling):
        # Gradient clipping before optimizer step
        if args.grad_clip_norm > 0.0:
            # Clip gradients by global norm
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
            # Log gradient norm for monitoring
            if master_process and args.use_wandb and wandb.run is not None:
                wandb.log({
                    "grad_norm": total_norm.item(),
                    "grad_clipped": total_norm.item() > args.grad_clip_norm,
                    "step": step + 1
                })
        
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        if args.optimizer_type == "muon_hybrid":
            # optimizer2 is Muon in this case
            for group in optimizer2.param_groups:
                frac = min(step / 300, 1) # momentum warmup for muon
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
    # logging averaged loss
    avg_loss = acc_loss / grad_accum

    loss_avg_t = torch.tensor(avg_loss, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(loss_avg_t, op=torch.distributed.ReduceOp.AVG)
    loss_avg = loss_avg_t.item()

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    current_tokens_b = calculate_tokens_processed(step + 1, world_size, args.train_seq_len, args.grad_accum_steps)
    print0(
        f"step:{step+1}/{train_steps} tokens:{current_tokens_b:.5f}B/{args.total_tokens_b}B loss:{loss_avg:.4f} "
        f"train_time:{format_time_ms(approx_training_time_ms)} "
        f"step_avg:{format_time_ms(approx_training_time_ms/(step + 1))}",
        console=True,
    )
    
    # Log training metrics to wandb
    if master_process and args.use_wandb and wandb.run is not None:
        current_lr1 = optimizers[0].param_groups[0]["lr"]
        
        log_dict = {
            "train_loss": loss_avg,
            "step": step + 1,
            "tokens_processed_b": current_tokens_b,
            "token_progress": current_tokens_b / args.total_tokens_b,
            "lr_adam": current_lr1,
            "lr_schedule_factor": get_lr(step),  # Track warmup/cooldown progress
            "window_size": (get_window_size_blocks(step).item() * 128) if use_flex_attention else int(args.train_seq_len),
        }
        
        if args.optimizer_type == "muon_hybrid":
            current_lr2 = optimizers[1].param_groups[0]["lr"]
            current_momentum = optimizers[1].param_groups[0]["momentum"]
            log_dict["lr_muon"] = current_lr2
            log_dict["momentum_muon"] = current_momentum
        
        wandb.log(log_dict)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {format_time_ms(training_time_ms)}", console=True)

# Final wandb logging
if master_process and args.use_wandb and wandb.run is not None:
    wandb.log({
        "final_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "final_memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        "total_training_time_ms": training_time_ms,
    })
    wandb.finish()

dist.destroy_process_group()
