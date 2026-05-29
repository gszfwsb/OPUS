import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import BlockMask, flex_attention
import torch.distributed as dist
from dataclasses import dataclass
from OPUS.layers.linear import GCLinear


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model
# Supports two modes:
#   1. Custom architecture (default): RMSNorm, RoPE, QK Norm, no bias - optimized for training
#   2. HF-compatible mode: LayerNorm, Learned Position Embeddings, with bias - 100% GPT-2 compatible

def norm(x: Tensor):
    """RMSNorm - faster than LayerNorm"""
    return F.rms_norm(x, (x.size(-1),))


# =============================================================================
# HuggingFace-Compatible Components (for CPT with official GPT-2 weights)
# =============================================================================

class HFCastedLinear(GCLinear):
    """Linear layer with bias for HuggingFace GPT-2 compatibility"""
    def __init__(self, in_features: int, out_features: int):
        # Initialize parent without calling its __init__ to avoid bias=False
        nn.Linear.__init__(self, in_features, out_features, bias=True)
        # Initialize GCLinear tracking
        self.layer_input = None
        self.pre_activation = None

    def reset_parameters(self) -> None:
        std = 0.02  # GPT-2 uses 0.02 std
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: Tensor):
        self.layer_input = x
        out = F.linear(x, self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)
        self.pre_activation = out
        return out


class HFCausalSelfAttention(nn.Module):
    """
    HuggingFace-compatible Self Attention:
    - Uses learned position embeddings (passed from parent)
    - No QK normalization
    - Has bias in all projections
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # QKV projection with bias (HuggingFace style)
        self.qkv_proj = HFCastedLinear(dim, 3 * dim)
        # Output projection with bias
        self.c_proj = HFCastedLinear(dim, dim)
        
        # Standard attention scale
        self.attn_scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x: Tensor, block_mask: BlockMask | None):
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).unbind(dim=2)
        
        # No QK norm, no RoPE - standard GPT-2 attention
        # q, k, v are [B, T, num_heads, head_dim]
        
        # Use standard scaled dot product attention
        if block_mask is None or B > 1:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                is_causal=True, scale=self.attn_scale
            ).transpose(1, 2)
        else:
            assert B == 1, "flex_attention with block_mask requires batch size 1"
            y = flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                block_mask=block_mask, scale=self.attn_scale
            ).transpose(1, 2)

        y = y.contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class HFMLP(nn.Module):
    """HuggingFace-compatible MLP with bias and GELU activation"""
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = HFCastedLinear(dim, hdim)
        self.c_proj = HFCastedLinear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # GPT-2 uses tanh approximation
        x = self.c_proj(x)
        return x


class HFBlock(nn.Module):
    """HuggingFace-compatible transformer block with LayerNorm"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = HFCausalSelfAttention(dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = HFMLP(dim)

    def forward(self, x: Tensor, block_mask: BlockMask | None):
        x = x + self.attn(self.ln_1(x), block_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# Llama-Compatible Components (for CPT with Llama 3.2 weights)
# =============================================================================

class LlamaRotary(nn.Module):
    """
    Llama3-style RoPE with configurable scaling.
    Supports both standard RoPE and Llama3's frequency scaling.
    """
    def __init__(self, dim: int, max_seq_len: int, rope_theta: float = 500000.0,
                 rope_scaling: dict = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        # Apply Llama3 scaling if provided
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            factor = rope_scaling.get("factor", 32.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)
            
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            
            wavelen = 2 * torch.pi / inv_freq
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen,
                inv_freq / factor,
                inv_freq
            )
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        
        # Precompute cos/sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
    
    def forward(self, x: Tensor, position_ids: Tensor = None):
        # x: [B, T, num_heads, head_dim] or [B, num_heads, T, head_dim]
        if x.dim() == 4:
            if x.size(1) == x.size(2):  # Ambiguous, assume [B, T, H, D]
                T = x.size(1)
            elif x.size(2) > x.size(1):  # [B, H, T, D]
                T = x.size(2)
                x = x.transpose(1, 2)  # -> [B, T, H, D]
            else:
                T = x.size(1)
        else:
            T = x.size(-2)
        
        cos = self.cos[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
        sin = self.sin[:T].unsqueeze(0).unsqueeze(2)
        
        # Apply rotary embedding
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat([
            x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x2.size(-1)],
            x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x2.size(-1)]
        ], dim=-1)
        return x_rotated.type_as(x)


class LlamaAttention(nn.Module):
    """
    Llama/Qwen-style attention with Grouped Query Attention (GQA) support.
    Optionally supports QK Normalization (used in Qwen3).
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, max_seq_len: int, rope_theta: float = 500000.0,
                 rope_scaling: dict = None, use_qk_norm: bool = False,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.use_qk_norm = use_qk_norm
        
        # Q, K, V projections (no bias)
        self.q_proj = CastedLinear(dim, num_heads * head_dim)
        self.k_proj = CastedLinear(dim, num_kv_heads * head_dim)
        self.v_proj = CastedLinear(dim, num_kv_heads * head_dim)
        self.o_proj = CastedLinear(num_heads * head_dim, dim)
        
        # Optional QK Normalization (for Qwen3)
        if use_qk_norm:
            self.q_norm = LlamaRMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = LlamaRMSNorm(head_dim, eps=rms_norm_eps)
        
        # Llama3/Qwen3 RoPE
        self.rotary = LlamaRotary(head_dim, max_seq_len, rope_theta, rope_scaling)
        
        self.attn_scale = 1.0 / (head_dim ** 0.5)
    
    def forward(self, x: Tensor, block_mask: BlockMask | None = None):
        B, T, _ = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Optional QK Norm (Qwen3 style)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply RoPE to Q and K
        q = self.rotary(q)
        k = self.rotary(k)
        
        # Repeat K/V heads to match Q heads (for GQA)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        
        # Transpose for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot product attention
        if block_mask is None or B > 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        else:
            assert B == 1, "flex_attention with block_mask requires batch size 1"
            y = flex_attention(q, k, v, block_mask=block_mask, scale=self.attn_scale)
        
        # Reshape back: [B, T, num_heads * head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    """
    Llama-style MLP with SwiGLU activation.
    Uses gate_proj, up_proj, down_proj structure.
    """
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = CastedLinear(dim, intermediate_size)
        self.up_proj = CastedLinear(dim, intermediate_size)
        self.down_proj = CastedLinear(intermediate_size, dim)
        # Zero init for output projection (stability)
        self.down_proj.weight.detach().zero_()
    
    def forward(self, x: Tensor):
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaRMSNorm(nn.Module):
    """Llama-style RMSNorm with learnable weight"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: Tensor):
        # RMSNorm - compute in float32 for numerical stability, then cast back
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # Cast weight and result back to input dtype
        return (x * self.weight.to(torch.float32)).to(input_dtype)


class LlamaBlock(nn.Module):
    """Llama/Qwen transformer block with RMSNorm and GQA"""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 intermediate_size: int, max_seq_len: int, layer_idx: int,
                 rope_theta: float = 500000.0, rope_scaling: dict = None,
                 rms_norm_eps: float = 1e-5, use_qk_norm: bool = False):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(dim, eps=rms_norm_eps)
        self.self_attn = LlamaAttention(
            dim, num_heads, num_kv_heads, head_dim, max_seq_len, rope_theta, rope_scaling,
            use_qk_norm=use_qk_norm, rms_norm_eps=rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=rms_norm_eps)
        self.mlp = LlamaMLP(dim, intermediate_size)
    
    def forward(self, x: Tensor, block_mask: BlockMask | None = None):
        # Pre-norm architecture
        x = x + self.self_attn(self.input_layernorm(x), block_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# =============================================================================
# Custom Architecture Components (default, optimized for training)
# =============================================================================

class CastedLinear(GCLinear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        self.layer_input = x
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            out = out.reshape(*x.shape[:-1], -1)
        else:
            out = F.linear(x, self.weight.type_as(x))
        self.pre_activation = out
        return self.pre_activation

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkv_proj = CastedLinear(dim, 3 * hdim)
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        self.attn_scale = 0.12

    def forward(self, x: Tensor, block_mask: BlockMask | None):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        qkv = self.qkv_proj(x)
        q, k, v = qkv.view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        
        # REMOVED: Value embedding mixing - use standard values only
        # v = lambdas[0] * v + lambdas[1] * ve.view_as(v)  # REMOVED
        
        # Use flex_attention if a block_mask is provided and batch size is 1, otherwise fall back.
        if block_mask is None or B > 1:
             y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=self.attn_scale).transpose(1, 2)
        else:
             assert B == 1, "flex_attention with block_mask requires batch size 1"
             y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.gelu(x) # Use GELU for better numerical stability
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # REMOVED: Skip layer 7 attention (was for U-Net pattern)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, block_mask: BlockMask | None):
        # REMOVED: U-Net skip connections and lambda mixing
        # x = lambdas[0] * x + lambdas[1] * x0  # REMOVED
        
        # Standard residual connections
        x = x + self.attn(norm(x), block_mask)
        x = x + self.mlp(norm(x))
        return x

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    num_layers: int = 48  # GPT-2 XL
    num_heads: int = 25   # GPT-2 XL
    model_dim: int = 1600 # GPT-2 XL
    max_seq_len: int = 4*64*1024
    hf_compatible: bool = False  # Use HuggingFace-compatible architecture
    llama_compatible: bool = False  # Use Llama/Qwen-compatible architecture
    # Llama/Qwen-specific configs
    num_kv_heads: int = None  # For GQA, defaults to num_heads
    intermediate_size: int = None  # MLP hidden size, defaults to 4*model_dim
    head_dim: int = 64
    rope_theta: float = 500000.0
    rope_scaling: dict = None
    rms_norm_eps: float = 1e-5
    use_qk_norm: bool = False  # QK Normalization (used in Qwen3)

class GPT(nn.Module):
    """
    GPT Model with three architecture modes:
    
    1. Custom mode (default):
       - RMSNorm for faster normalization
       - RoPE for position encoding
       - QK normalization for stability
       - No bias in linear layers
       - Optimized for training from scratch
    
    2. HuggingFace GPT-2 compatible mode (hf_compatible=True):
       - LayerNorm (standard GPT-2)
       - Learned position embeddings
       - No QK normalization
       - Bias in all linear layers
       - 100% compatible with official GPT-2 weights for CPT
    
    3. Llama-compatible mode (llama_compatible=True):
       - RMSNorm with learnable weight
       - Llama3-style RoPE with scaling
       - Grouped Query Attention (GQA)
       - SwiGLU MLP
       - 100% compatible with official Llama weights for CPT
    """
    def __init__(self,
                 vocab_size: int = None,
                 num_layers: int = None,
                 num_heads: int = None,
                 model_dim: int = None,
                 max_seq_len: int = 4*64*1024,
                 model_type: str = None,
                 hf_compatible: bool = False,
                 llama_compatible: bool = False,
                 # Llama-specific params
                 num_kv_heads: int = None,
                 intermediate_size: int = None,
                 head_dim: int = 64,
                 rope_theta: float = 500000.0,
                 rope_scaling: dict = None,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        
        self.hf_compatible = hf_compatible
        self.llama_compatible = llama_compatible
        # Runtime eval toggles (kept for compatibility with OpenCompass wrapper patterns):
        # - If either is True, we will skip softcapping and return raw logits.
        self.disable_logit_squash = False
        self.return_raw_logits = False
        self.gradient_checkpointing = False
        
        # Handle model_type configuration
        if model_type is not None:
            # Check if it's a GPT-2 or Llama model
            gpt2_models = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt2-7b'}
            llama_models = {'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b'}
            qwen_models = {'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'}
            
            if model_type in gpt2_models:
                # GPT-2 configuration
                config_args = {
                    'gpt2':         dict(num_layers=12, num_heads=12, model_dim=768),   # 124M params
                    'gpt2-medium':  dict(num_layers=24, num_heads=16, model_dim=1024),  # 350M params  
                    'gpt2-large':   dict(num_layers=36, num_heads=20, model_dim=1280),  # 774M params
                    'gpt2-xl':      dict(num_layers=48, num_heads=25, model_dim=1600),  # 1558M params
                    'gpt2-7b':      dict(num_layers=52, num_heads=52, model_dim=3328),  # ~7.1B params
                }[model_type]
                
                num_layers = num_layers if num_layers is not None else config_args['num_layers']
                num_heads = num_heads if num_heads is not None else config_args['num_heads'] 
                model_dim = model_dim if model_dim is not None else config_args['model_dim']
                vocab_size = vocab_size if vocab_size is not None else 50257
                
                arch_mode = "HuggingFace-compatible" if hf_compatible else "Custom (RMSNorm+RoPE)"
                print(f"Initializing {model_type} ({arch_mode}): {num_layers} layers, {num_heads} heads, {model_dim} dim")
                
            elif model_type in llama_models:
                # Llama configuration
                llama_compatible = True
                self.llama_compatible = True
                
                llama_configs = {
                    'llama-3.2-1b': dict(
                        vocab_size=128256, num_layers=16, num_heads=32, num_kv_heads=8,
                        model_dim=2048, intermediate_size=8192, head_dim=64,
                        rope_theta=500000.0, rms_norm_eps=1e-5,
                        rope_scaling=dict(
                            factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0,
                            original_max_position_embeddings=8192, rope_type="llama3"
                        )
                    ),
                    'llama-3.2-3b': dict(
                        vocab_size=128256, num_layers=28, num_heads=24, num_kv_heads=8,
                        model_dim=3072, intermediate_size=8192, head_dim=128,
                        rope_theta=500000.0, rms_norm_eps=1e-5,
                        rope_scaling=dict(
                            factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0,
                            original_max_position_embeddings=8192, rope_type="llama3"
                        )
                    ),
                    'llama-3.1-8b': dict(
                        vocab_size=128256, num_layers=32, num_heads=32, num_kv_heads=8,
                        model_dim=4096, intermediate_size=14336, head_dim=128,
                        rope_theta=500000.0, rms_norm_eps=1e-5,
                        rope_scaling=dict(
                            factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
                            original_max_position_embeddings=8192, rope_type="llama3"
                        )
                    ),
                }[model_type]
                
                vocab_size = vocab_size if vocab_size is not None else llama_configs['vocab_size']
                num_layers = num_layers if num_layers is not None else llama_configs['num_layers']
                num_heads = num_heads if num_heads is not None else llama_configs['num_heads']
                num_kv_heads = num_kv_heads if num_kv_heads is not None else llama_configs['num_kv_heads']
                model_dim = model_dim if model_dim is not None else llama_configs['model_dim']
                intermediate_size = intermediate_size if intermediate_size is not None else llama_configs['intermediate_size']
                head_dim = llama_configs['head_dim']
                rope_theta = llama_configs['rope_theta']
                rope_scaling = llama_configs['rope_scaling']
                rms_norm_eps = llama_configs['rms_norm_eps']
                
                print(f"Initializing {model_type} (Llama-compatible): {num_layers} layers, {num_heads} heads, {num_kv_heads} kv_heads, {model_dim} dim")
            
            elif model_type in qwen_models:
                # Qwen3 configuration (same architecture as Llama: RMSNorm + GQA + SwiGLU)
                llama_compatible = True
                self.llama_compatible = True
                
                qwen_configs = {
                    'qwen3-0.6b': dict(
                        vocab_size=151936, num_layers=28, num_heads=16, num_kv_heads=8,
                        model_dim=1024, intermediate_size=3072, head_dim=128,
                        rope_theta=1000000.0, rms_norm_eps=1e-6,
                        rope_scaling=None, use_qk_norm=True  # Qwen3 uses QK Norm
                    ),
                    'qwen3-1.7b': dict(
                        vocab_size=151936, num_layers=28, num_heads=16, num_kv_heads=8,
                        model_dim=2048, intermediate_size=6144, head_dim=128,
                        rope_theta=1000000.0, rms_norm_eps=1e-6,
                        rope_scaling=None, use_qk_norm=True
                    ),
                    'qwen3-4b': dict(
                        vocab_size=151936, num_layers=36, num_heads=32, num_kv_heads=8,
                        model_dim=2560, intermediate_size=9216, head_dim=128,
                        rope_theta=1000000.0, rms_norm_eps=1e-6,
                        rope_scaling=None, use_qk_norm=True
                    ),
                }[model_type]
                
                vocab_size = vocab_size if vocab_size is not None else qwen_configs['vocab_size']
                num_layers = num_layers if num_layers is not None else qwen_configs['num_layers']
                num_heads = num_heads if num_heads is not None else qwen_configs['num_heads']
                num_kv_heads = num_kv_heads if num_kv_heads is not None else qwen_configs['num_kv_heads']
                model_dim = model_dim if model_dim is not None else qwen_configs['model_dim']
                intermediate_size = intermediate_size if intermediate_size is not None else qwen_configs['intermediate_size']
                head_dim = qwen_configs['head_dim']
                rope_theta = qwen_configs['rope_theta']
                rope_scaling = qwen_configs['rope_scaling']
                rms_norm_eps = qwen_configs['rms_norm_eps']
                use_qk_norm = qwen_configs.get('use_qk_norm', False)
                
                print(f"Initializing {model_type} (Qwen-compatible): {num_layers} layers, {num_heads} heads, {num_kv_heads} kv_heads, {model_dim} dim, qk_norm={use_qk_norm}")
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Supported: {gpt2_models | llama_models | qwen_models}")
        else:
            # Use provided parameters or defaults
            vocab_size = vocab_size if vocab_size is not None else 50257
            num_layers = num_layers if num_layers is not None else 48
            num_heads = num_heads if num_heads is not None else 25
            model_dim = model_dim if model_dim is not None else 1600
        
        # Set defaults for optional params
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        intermediate_size = intermediate_size if intermediate_size is not None else 4 * model_dim
        
        # Set use_qk_norm default if not defined (Llama doesn't use it, Qwen does)
        try:
            use_qk_norm
        except NameError:
            use_qk_norm = False
        
        self.config = GPTConfig(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            model_dim=model_dim,
            max_seq_len=max_seq_len,
            hf_compatible=hf_compatible,
            llama_compatible=llama_compatible,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=use_qk_norm
        )
        
        # Token embeddings and model architecture based on mode
        if llama_compatible:
            # Llama mode: exact vocab size, RMSNorm, GQA, SwiGLU
            self.embed = nn.Embedding(vocab_size, model_dim)
            
            # No learned position embeddings (using RoPE)
            self.wpe = None
            
            # Llama/Qwen transformer blocks
            self.blocks = nn.ModuleList([
                LlamaBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    max_seq_len=max_seq_len,
                    layer_idx=i,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    rms_norm_eps=rms_norm_eps,
                    use_qk_norm=use_qk_norm
                )
                for i in range(num_layers)
            ])
            
            # Final RMSNorm (Llama style with learnable weight)
            self.ln_f = LlamaRMSNorm(model_dim, eps=rms_norm_eps)
            
            # LM head (tied to embeddings)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight  # Weight tying
            
        elif hf_compatible:
            # HuggingFace GPT-2 mode: use exact vocab size (50257)
            self.embed = nn.Embedding(vocab_size, model_dim)
            
            # Learned position embeddings (GPT-2 style)
            self.wpe = nn.Embedding(max_seq_len, model_dim)
            
            # HuggingFace-compatible transformer blocks
            self.blocks = nn.ModuleList([
                HFBlock(model_dim, num_heads, max_seq_len, i) 
                for i in range(num_layers)
            ])
            
            # Final LayerNorm
            self.ln_f = nn.LayerNorm(model_dim)
            
            # LM head (tied to embeddings)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight  # Weight tying
            
        else:
            # Custom mode: pad vocab to multiple of 128 for efficiency
            vocab_size_padded = next_multiple_of_n(vocab_size, n=128)
            self.embed = nn.Embedding(vocab_size_padded, model_dim)
            
            # No position embeddings (using RoPE in attention)
            self.wpe = None
            
            # Custom transformer blocks (RMSNorm, RoPE, QK norm)
            self.blocks = nn.ModuleList([
                Block(model_dim, num_heads, max_seq_len, i) 
                for i in range(num_layers)
            ])
            
            # No final LayerNorm module (using functional RMSNorm)
            self.ln_f = None
            
            # Tied embeddings with padded vocab
            self.lm_head = nn.Linear(vocab_size_padded, model_dim, bias=False)
            self.lm_head.weight = self.embed.weight  # Weight tying
            
            # Keep embedding learning rate multiplier for custom mode
            for param in self.embed.parameters():
                param.lr_mul = 75.
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
    
    def load_hf_weights(self, hf_model_path: str, verbose: bool = True):
        """
        Load weights from a HuggingFace GPT-2 model.
        
        This method only works when hf_compatible=True, as it requires
        100% architecture compatibility for direct weight loading.
        
        Args:
            hf_model_path: Path to HuggingFace model or model name 
                          (e.g., 'openai-community/gpt2-xl' or local path)
            verbose: If True, print loading progress
        
        Raises:
            AssertionError: If model is not in hf_compatible mode
        """
        from transformers import GPT2LMHeadModel
        
        assert self.hf_compatible, "load_hf_weights() only works with hf_compatible=True"
        
        if verbose:
            print(f"Loading HuggingFace GPT-2 weights from: {hf_model_path}")
        
        # Load HuggingFace model
        hf_model = GPT2LMHeadModel.from_pretrained(hf_model_path)
        hf_state = hf_model.state_dict()
        
        # Direct weight mapping (100% compatible architecture)
        new_state = {}
        
        # 1. Token embeddings: transformer.wte.weight -> embed.weight
        new_state['embed.weight'] = hf_state['transformer.wte.weight']
        
        # 2. Position embeddings: transformer.wpe.weight -> wpe.weight
        wpe_weight = hf_state['transformer.wpe.weight']
        # Truncate or pad if necessary
        if wpe_weight.size(0) > self.wpe.weight.size(0):
            wpe_weight = wpe_weight[:self.wpe.weight.size(0)]
        elif wpe_weight.size(0) < self.wpe.weight.size(0):
            # Pad with zeros (for longer sequence lengths)
            padding = torch.zeros(
                self.wpe.weight.size(0) - wpe_weight.size(0), 
                wpe_weight.size(1),
                dtype=wpe_weight.dtype
            )
            wpe_weight = torch.cat([wpe_weight, padding], dim=0)
        new_state['wpe.weight'] = wpe_weight
        
        # 3. Layer weights
        num_layers = len(self.blocks)
        for i in range(num_layers):
            hf_prefix = f'transformer.h.{i}'
            our_prefix = f'blocks.{i}'
            
            # LayerNorm 1 (before attention)
            new_state[f'{our_prefix}.ln_1.weight'] = hf_state[f'{hf_prefix}.ln_1.weight']
            new_state[f'{our_prefix}.ln_1.bias'] = hf_state[f'{hf_prefix}.ln_1.bias']
            
            # Attention QKV: HF uses Conv1D style [in_features, out_features]
            # Our Linear uses [out_features, in_features]
            new_state[f'{our_prefix}.attn.qkv_proj.weight'] = hf_state[f'{hf_prefix}.attn.c_attn.weight'].T.contiguous()
            new_state[f'{our_prefix}.attn.qkv_proj.bias'] = hf_state[f'{hf_prefix}.attn.c_attn.bias']
            
            # Attention output projection
            new_state[f'{our_prefix}.attn.c_proj.weight'] = hf_state[f'{hf_prefix}.attn.c_proj.weight'].T.contiguous()
            new_state[f'{our_prefix}.attn.c_proj.bias'] = hf_state[f'{hf_prefix}.attn.c_proj.bias']
            
            # LayerNorm 2 (before MLP)
            new_state[f'{our_prefix}.ln_2.weight'] = hf_state[f'{hf_prefix}.ln_2.weight']
            new_state[f'{our_prefix}.ln_2.bias'] = hf_state[f'{hf_prefix}.ln_2.bias']
            
            # MLP
            new_state[f'{our_prefix}.mlp.c_fc.weight'] = hf_state[f'{hf_prefix}.mlp.c_fc.weight'].T.contiguous()
            new_state[f'{our_prefix}.mlp.c_fc.bias'] = hf_state[f'{hf_prefix}.mlp.c_fc.bias']
            new_state[f'{our_prefix}.mlp.c_proj.weight'] = hf_state[f'{hf_prefix}.mlp.c_proj.weight'].T.contiguous()
            new_state[f'{our_prefix}.mlp.c_proj.bias'] = hf_state[f'{hf_prefix}.mlp.c_proj.bias']
        
        # 4. Final LayerNorm
        new_state['ln_f.weight'] = hf_state['transformer.ln_f.weight']
        new_state['ln_f.bias'] = hf_state['transformer.ln_f.bias']
        
        # 5. LM head is tied to embeddings, already handled
        
        # Load the state dict
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        
        if verbose:
            print(f"✅ Successfully loaded {len(new_state)} parameter groups")
            if missing:
                print(f"⚠️  Missing keys (expected, tied weights): {missing}")
            if unexpected:
                print(f"⚠️  Unexpected keys: {unexpected}")
        
        # Clean up
        del hf_model, hf_state
        torch.cuda.empty_cache()

    def load_llama_weights(self, llama_model_path: str, verbose: bool = True):
        """
        Load weights from a HuggingFace Llama model.
        
        This method only works when llama_compatible=True.
        
        Args:
            llama_model_path: Path to HuggingFace model or model name 
                             (e.g., 'meta-llama/Llama-3.2-1B' or local path)
            verbose: If True, print loading progress
        """
        from transformers import LlamaForCausalLM
        
        assert self.llama_compatible, "load_llama_weights() only works with llama_compatible=True"
        
        if verbose:
            print(f"Loading Llama weights from: {llama_model_path}")
        
        # Load HuggingFace model
        hf_model = LlamaForCausalLM.from_pretrained(llama_model_path, dtype=torch.bfloat16)
        hf_state = hf_model.state_dict()
        
        new_state = {}
        
        # 1. Token embeddings: model.embed_tokens.weight -> embed.weight
        new_state['embed.weight'] = hf_state['model.embed_tokens.weight']
        
        # 2. Layer weights
        num_layers = len(self.blocks)
        for i in range(num_layers):
            hf_prefix = f'model.layers.{i}'
            our_prefix = f'blocks.{i}'
            
            # Input LayerNorm (RMSNorm)
            new_state[f'{our_prefix}.input_layernorm.weight'] = hf_state[f'{hf_prefix}.input_layernorm.weight']
            
            # Attention Q, K, V projections (separate in Llama)
            new_state[f'{our_prefix}.self_attn.q_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
            new_state[f'{our_prefix}.self_attn.k_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
            new_state[f'{our_prefix}.self_attn.v_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
            new_state[f'{our_prefix}.self_attn.o_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.o_proj.weight']
            
            # Post-attention LayerNorm (RMSNorm)
            new_state[f'{our_prefix}.post_attention_layernorm.weight'] = hf_state[f'{hf_prefix}.post_attention_layernorm.weight']
            
            # MLP: gate_proj, up_proj, down_proj
            new_state[f'{our_prefix}.mlp.gate_proj.weight'] = hf_state[f'{hf_prefix}.mlp.gate_proj.weight']
            new_state[f'{our_prefix}.mlp.up_proj.weight'] = hf_state[f'{hf_prefix}.mlp.up_proj.weight']
            new_state[f'{our_prefix}.mlp.down_proj.weight'] = hf_state[f'{hf_prefix}.mlp.down_proj.weight']
        
        # 3. Final RMSNorm
        new_state['ln_f.weight'] = hf_state['model.norm.weight']
        
        # 4. LM head is tied to embeddings (handled by weight tying in __init__)
        
        # Load the state dict
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        
        if verbose:
            print(f"✅ Successfully loaded {len(new_state)} parameter groups")
            if missing:
                print(f"⚠️  Missing keys (expected, tied weights): {missing}")
            if unexpected:
                print(f"⚠️  Unexpected keys: {unexpected}")
        
        # Clean up
        del hf_model, hf_state
        torch.cuda.empty_cache()

    def load_qwen_weights(self, qwen_model_path: str, verbose: bool = True):
        """
        Load weights from a HuggingFace Qwen3 model.
        
        Qwen3 shares the same architecture as Llama (RMSNorm, GQA, SwiGLU, RoPE)
        but additionally uses QK Normalization.
        
        Args:
            qwen_model_path: Path to HuggingFace Qwen model or model name 
                             (e.g., 'Qwen/Qwen3-0.6B' or local path)
            verbose: If True, print loading progress
        """
        from transformers import AutoModelForCausalLM
        
        assert self.llama_compatible, "load_qwen_weights() only works with llama_compatible=True (Qwen uses same arch)"
        
        if verbose:
            print(f"Loading Qwen3 weights from: {qwen_model_path}")
        
        # Load HuggingFace model (use AutoModelForCausalLM to auto-detect Qwen3)
        hf_model = AutoModelForCausalLM.from_pretrained(qwen_model_path, dtype=torch.bfloat16, trust_remote_code=True)
        hf_state = hf_model.state_dict()
        
        new_state = {}
        
        # 1. Token embeddings: model.embed_tokens.weight -> embed.weight
        new_state['embed.weight'] = hf_state['model.embed_tokens.weight']
        
        # 2. Layer weights
        num_layers = len(self.blocks)
        for i in range(num_layers):
            hf_prefix = f'model.layers.{i}'
            our_prefix = f'blocks.{i}'
            
            # Input LayerNorm (RMSNorm)
            new_state[f'{our_prefix}.input_layernorm.weight'] = hf_state[f'{hf_prefix}.input_layernorm.weight']
            
            # Attention Q, K, V projections
            new_state[f'{our_prefix}.self_attn.q_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
            new_state[f'{our_prefix}.self_attn.k_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
            new_state[f'{our_prefix}.self_attn.v_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
            new_state[f'{our_prefix}.self_attn.o_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.o_proj.weight']
            
            # QK Normalization (Qwen3 specific)
            if f'{hf_prefix}.self_attn.q_norm.weight' in hf_state:
                new_state[f'{our_prefix}.self_attn.q_norm.weight'] = hf_state[f'{hf_prefix}.self_attn.q_norm.weight']
                new_state[f'{our_prefix}.self_attn.k_norm.weight'] = hf_state[f'{hf_prefix}.self_attn.k_norm.weight']
            
            # Post-attention LayerNorm (RMSNorm)
            new_state[f'{our_prefix}.post_attention_layernorm.weight'] = hf_state[f'{hf_prefix}.post_attention_layernorm.weight']
            
            # MLP: gate_proj, up_proj, down_proj (SwiGLU)
            new_state[f'{our_prefix}.mlp.gate_proj.weight'] = hf_state[f'{hf_prefix}.mlp.gate_proj.weight']
            new_state[f'{our_prefix}.mlp.up_proj.weight'] = hf_state[f'{hf_prefix}.mlp.up_proj.weight']
            new_state[f'{our_prefix}.mlp.down_proj.weight'] = hf_state[f'{hf_prefix}.mlp.down_proj.weight']
        
        # 3. Final RMSNorm
        new_state['ln_f.weight'] = hf_state['model.norm.weight']
        
        # 4. LM head - Qwen ties embeddings
        # If lm_head.weight exists and is different from embed_tokens, load it
        if 'lm_head.weight' in hf_state and not torch.equal(hf_state['lm_head.weight'], hf_state['model.embed_tokens.weight']):
            new_state['lm_head.weight'] = hf_state['lm_head.weight']
        
        # Load the state dict
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        
        if verbose:
            print(f"✅ Successfully loaded {len(new_state)} parameter groups")
            if missing:
                print(f"⚠️  Missing keys (expected, tied weights): {missing}")
            if unexpected:
                print(f"⚠️  Unexpected keys: {unexpected}")
        
        # Clean up
        del hf_model, hf_state
        torch.cuda.empty_cache()

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        """Keep FlexAttention - it's memory efficient and fast"""
        BLOCK_SIZE = 128
        # Assuming input_seq is 1D for this helper
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    @classmethod
    def from_model_type(cls, model_type: str, max_seq_len: int = 4*64*1024, hf_compatible: bool = False, 
                        llama_compatible: bool = False, **kwargs):
        """
        Create a GPT/Llama/Qwen model from a model type string.
        
        Args:
            model_type: One of:
                - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt2-7b'
                - Llama: 'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b'
                - Qwen: 'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'
            max_seq_len: Maximum sequence length for the model
            hf_compatible: If True, use HuggingFace GPT-2 compatible architecture
            llama_compatible: If True, use Llama/Qwen-compatible architecture (auto-set for llama/qwen models)
            **kwargs: Additional arguments to override defaults
        
        Returns:
            GPT model instance
        
        Example:
            # GPT-2 custom architecture (for training from scratch)
            model = GPT.from_model_type('gpt2-xl', max_seq_len=128*1024)
            
            # GPT-2 HuggingFace-compatible (for CPT)
            model = GPT.from_model_type('gpt2-xl', max_seq_len=8192, hf_compatible=True)
            
            # Llama-3.2-1B (for CPT with Llama weights)
            model = GPT.from_model_type('llama-3.2-1b', max_seq_len=8192)
            
            # Qwen3-0.6B (for CPT with Qwen weights)
            model = GPT.from_model_type('qwen3-0.6b', max_seq_len=8192)
        """
        # Auto-detect llama/qwen models (they share the same architecture)
        if model_type.startswith('llama') or model_type.startswith('qwen'):
            llama_compatible = True
        return cls(model_type=model_type, max_seq_len=max_seq_len, hf_compatible=hf_compatible, 
                   llama_compatible=llama_compatible, **kwargs)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        Since we use RoPE (no learned position embeddings), this mainly affects
        whether to count token embeddings (which are tied to lm_head in our case).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'embed'):
            # In tied embedding case, don't double count the shared weights
            if hasattr(self.lm_head, 'weight') and self.lm_head.weight is self.embed.weight:
                n_params -= self.embed.weight.numel()
        return n_params

    def forward(self, input_seq: Tensor, target_seq: Tensor | None = None, sliding_window_num_blocks: Tensor | None = None):
        assert input_seq.ndim == 1 or input_seq.ndim == 2
        if input_seq.ndim == 1:
            input_seq = input_seq[None]  # shape becomes [1, T]

        B, T = input_seq.size()

        if sliding_window_num_blocks is not None:
            # This logic requires batch size 1.
            assert B == 1, f"flex_attention requires batch size 1 to create block masks, but got batch size {B}"
            # Remove batch dimension for mask creation
            seq = input_seq.squeeze(0)
            long_bm, short_bm = self.create_blockmasks(seq, sliding_window_num_blocks)
            # Create pattern for 12 layers, repeat enough times to cover all layers
            pattern = [long_bm, short_bm, short_bm, short_bm, long_bm,
                       short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            num_repeats = (len(self.blocks) + 11) // 12  # ceil division
            block_masks = (pattern * num_repeats)[:len(self.blocks)]  # Trim to exact length
        else:
            # When no sliding window is provided, use no mask.
            block_masks = [None] * len(self.blocks)

        if self.llama_compatible:
            # Llama-compatible forward pass
            # Token embeddings only (RoPE is applied in attention)
            x = self.embed(input_seq)  # [B, T, D]
            
            # Llama transformer blocks (RMSNorm + GQA + SwiGLU)
            for i, block in enumerate(self.blocks):
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    x = checkpoint(block, x, block_masks[i], use_reentrant=False)
                else:
                    x = block(x, block_masks[i])
            
            # Final RMSNorm
            x = self.ln_f(x)
            
            # LM head (no softcapping for Llama mode)
            logits = self.lm_head(x)
            
        elif self.hf_compatible:
            # HuggingFace GPT-2 compatible forward pass
            # Token embeddings + learned position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=input_seq.device)
            tok_emb = self.embed(input_seq)  # [B, T, D]
            pos_emb = self.wpe(pos)  # [T, D]
            x = tok_emb + pos_emb
            
            # Standard transformer blocks (with LayerNorm inside)
            for i, block in enumerate(self.blocks):
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    x = checkpoint(block, x, block_masks[i], use_reentrant=False)
                else:
                    x = block(x, block_masks[i])
            
            # Final LayerNorm
            x = self.ln_f(x)
            
            # LM head (no softcapping for HF-compatible mode)
            logits = self.lm_head(x)
        else:
            # Custom architecture forward pass
            x = norm(self.embed(input_seq))  # RMSNorm after embedding
            
            # Standard transformer processing
            for i, block in enumerate(self.blocks):
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    x = checkpoint(block, x, block_masks[i], use_reentrant=False)
                else:
                    x = block(x, block_masks[i])

            x = norm(x)  # Final RMSNorm
            logits = self.lm_head(x)
            
            # Softcapping for custom mode - prevents extreme logits.
            # IMPORTANT: Keep this ON by default for backward compatibility with existing training runs,
            # but allow disabling for evaluation / logprob parity vs raw-logit GPT-2:
            # - set `self.disable_logit_squash = True` OR `self.return_raw_logits = True`
            if not (getattr(self, "disable_logit_squash", False) or getattr(self, "return_raw_logits", False)):
                logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        
        if target_seq is not None:
            # Use float32 for loss calculation to improve numerical stability
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_seq.view(-1), reduction="sum" if self.training else "mean")
            return loss
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # check if we should stop
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

        return idx