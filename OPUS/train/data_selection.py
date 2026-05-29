"""
OPUS Data Selection: Gradient-based data selection using ghost inner-product.

Core algorithm for computing gradient similarity scores between training candidates
and proxy data, then selecting the most beneficial samples for training.

Key features:
- Ghost inner-product: Efficient gradient similarity without materializing full gradients
- Preconditioner support: AdamW, Muon, or automatic detection
- Random projection: Optional CountSketch-based gradient compression
- Multiple selection strategies: greedy, stochastic greedy, PPL-based
"""

import torch
import numpy as np
import time
import math

# Random Projection support
try:
    from .random_projection import GradientProjector
    _RANDOM_PROJECTION_AVAILABLE = True
except ImportError:
    _RANDOM_PROJECTION_AVAILABLE = False
    GradientProjector = None

# Global projector instance
_GLOBAL_PROJECTOR = None


def initialize_projector(sketch_dim: int = 16384, seed: int = 42, device: str = "cuda", enabled: bool = True):
    """
    Initialize global gradient projector for random projection.
    
    GPT-2 XL gradient dimension analysis:
    - Max layer (c_fc/c_proj2): 10,240,000 dims
    - Recommend sketch_dim >= 8192 (compression ~1250x)
    - Default 16384 (compression ~625x, better ranking preservation)
    
    Args:
        sketch_dim: Projection dimension. Recommended values:
            - 8192: Minimum usable (compression ~1250x)
            - 16384: Default recommended (compression ~625x)
            - 32768: High precision (compression ~312x)
        seed: Random seed
        device: Compute device
        enabled: Whether to enable projection
    """
    global _GLOBAL_PROJECTOR
    if not _RANDOM_PROJECTION_AVAILABLE:
        print("[Random Projection] Module not available, projection disabled")
        _GLOBAL_PROJECTOR = None
        return
    
    if enabled and sketch_dim > 0:
        _GLOBAL_PROJECTOR = GradientProjector(
            sketch_dim=sketch_dim,
            seed=seed,
            device=device,
            enabled=True
        )
        # Estimate compression ratio (based on GPT-2 XL max layer)
        max_grad_dim = 10_240_000  # c_fc / c_proj2
        compression_ratio = max_grad_dim / sketch_dim
        print(f"[Random Projection] Initialized: dim={sketch_dim}, compression≈{compression_ratio:.0f}x, seed={seed}")
    else:
        _GLOBAL_PROJECTOR = None
        print("[Random Projection] Disabled (using full-dimensional gradients)")


def get_projector():
    """Get the global gradient projector instance."""
    return _GLOBAL_PROJECTOR


_GBG_CALL_ID = 0


def make_window_generator(step: int, rank: int, base_seed: int = 12345, device: str = "cuda"):
    """Create a deterministic random generator for window sampling."""
    g = torch.Generator(device=device)
    g.manual_seed(base_seed + step * 100003 + rank * 1009)
    return g


def _as_opt_list(optimizer_or_list):
    """Convert optimizer to list if needed."""
    if isinstance(optimizer_or_list, (list, tuple)):
        return list(optimizer_or_list)
    return [optimizer_or_list]


def _build_param2opt_map(optimizers):
    """Build mapping from parameters to their optimizers."""
    m = {}
    for opt in optimizers:
        for g in opt.param_groups:
            for p in g['params']:
                m[p] = opt
    return m


def _group_kind(group):
    """Detect optimizer type from param group."""
    if 'betas' in group:
        return 'adamw'
    if 'momentum' in group and 'betas' not in group:
        return 'muon'
    return 'unknown'


def _get_rank_safe():
    """Get distributed rank safely."""
    return (torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0)


def _copy_batch_dict(batch):
    """Return a shallow copy so downstream slicing doesn't mutate cached proxy batches."""
    return {k: v for k, v in batch.items()}


def _sample_proxy_batch_for_step(validation_loader):
    """Sample or reuse the proxy batch exactly once per OPUS selection step."""
    fixed_batch = getattr(validation_loader, "_fixed_batch", None)
    if fixed_batch is not None:
        return _copy_batch_dict(fixed_batch)

    if not hasattr(validation_loader, "_it"):
        validation_loader._it = iter(validation_loader)

    proxy_mode = getattr(validation_loader, "_proxy_mode", "refresh")
    refresh_interval = max(1, int(getattr(validation_loader, "_proxy_refresh_interval", 1)))
    cached_batch = getattr(validation_loader, "_cached_proxy_batch", None)
    call_count = int(getattr(validation_loader, "_proxy_call_count", 0))

    if proxy_mode == "fixed":
        if cached_batch is None:
            try:
                cached_batch = next(validation_loader._it)
            except StopIteration:
                validation_loader._it = iter(validation_loader)
                cached_batch = next(validation_loader._it)
            validation_loader._cached_proxy_batch = cached_batch
        return _copy_batch_dict(cached_batch)

    if cached_batch is None or call_count % refresh_interval == 0:
        try:
            cached_batch = next(validation_loader._it)
        except StopIteration:
            validation_loader._it = iter(validation_loader)
            cached_batch = next(validation_loader._it)
        validation_loader._cached_proxy_batch = cached_batch

    validation_loader._proxy_call_count = call_count + 1
    return _copy_batch_dict(cached_batch)


def _get_step_from_optimizer(optimizer):
    """Return the current step of the optimizer."""
    t = 1
    for g in optimizer.param_groups:
        for p in g['params']:
            st = optimizer.state.get(p, None)
            if st is not None and 'step' in st:
                t = max(t, int(st['step']))
    return t


def _find_group_for_param(optimizer, p):
    """Find the param group containing parameter p."""
    for g in optimizer.param_groups:
        # IMPORTANT: don't use `p in g['params']` for torch Tensors/Parameters.
        # Python membership uses `==`, which triggers elementwise tensor compares and can error.
        for gp in g.get('params', ()):
            if gp is p:
                return g
    return optimizer.param_groups[0]


def global_C_t(optimizer, p):
    """
    AdamW scalar factor:
        C_t = alpha_t * (1 - beta1) * sqrt(1 - beta2^t) / (1 - beta1^t)
    where alpha_t is the *current* LR in the optimizer param group.
    """
    g = _find_group_for_param(optimizer, p)
    beta1, beta2 = g['betas']
    lr = g['lr']
    t = _get_step_from_optimizer(optimizer)
    num = max(1e-16, 1.0 - (beta2 ** t)) ** 0.5
    den = max(1e-16, 1.0 - (beta1 ** t))
    device = p.device
    return torch.tensor(lr * (1.0 - beta1) * num / den, dtype=torch.float32, device=device)


def _inv_rms_scalar_for_param(p, group, state):
    """Backward-compatible scalar approximation helper."""
    v = state.get('exp_avg_sq', None)
    eps = group.get('eps', 1e-8)
    if v is None:
        return torch.tensor(1.0, dtype=torch.float32, device=p.device)
    inv = (v.sqrt() + eps).reciprocal().to(dtype=torch.float32)
    return inv.mean()


def _inv_rms_tensor_for_param(group, state):
    """Return the full AdamW diagonal preconditioner 1 / (sqrt(v_t) + eps)."""
    v = state.get('exp_avg_sq', None)
    if v is None:
        return None
    eps = group.get('eps', 1e-8)
    return (v.sqrt() + eps).reciprocal().to(dtype=torch.float32)


def global_C_t_muon(optimizer, p):
    """
    Muon scalar factor:
        C_t = eff_lr * (1 - momentum^2)
    """
    g = _find_group_for_param(optimizer, p)
    lr = g['lr'] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
    momentum = g.get('momentum', 0.0)
    return torch.tensor(lr * (1.0 - momentum**2), dtype=torch.float32, device=p.device)


def _get_muon_S_matrix(p, optimizer):
    """
    Constructs the S_t matrix for Muon preconditioning for a given parameter p.
    S_t = aI + bA_t + cA_t^2, where A_t = m_tilde @ m_tilde.T
    """
    state = optimizer.state.get(p, None)
    out_dim = p.shape[0]
    device = p.device
    if state is None or 'momentum_buffer' not in state:
        return torch.eye(out_dim, dtype=torch.bfloat16, device=device)

    m = state['momentum_buffer'].to(dtype=torch.bfloat16)
    m_norm = torch.norm(m)
    if m_norm < 1e-7:
        return torch.eye(out_dim, dtype=torch.bfloat16, device=device)
        
    m_tilde = m / m_norm
    
    # A_t should be [out_dim, out_dim]
    A = m_tilde @ m_tilde.T
    
    a, b, c = 3.4445, -4.7750, 2.0315
    I = torch.eye(out_dim, device=device, dtype=torch.bfloat16)
    S = a * I + b * A + c * (A @ A)
    return S


def weighted_grad_dotprod(A1, B1, A2, B2, diagonal_weight: torch.Tensor, weight_rhs: bool = False,
                          chunk_rows: int = 32) -> torch.Tensor:
    """
    Compute <D o G1, G2> or <D o G1, D o G2> without materializing the full gradient.

    `diagonal_weight` is the AdamW diagonal preconditioner with shape [D_out, D_in].
    """
    if diagonal_weight.ndim != 2:
        raise ValueError(f"Expected diagonal_weight with shape [D_out, D_in], got {tuple(diagonal_weight.shape)}")

    device = A1.device
    dtype = torch.float32
    result = torch.zeros(A1.shape[0], A2.shape[0], device=device, dtype=dtype)
    diagonal_weight = diagonal_weight.to(device=device, dtype=dtype)
    out_dim = diagonal_weight.size(0)
    lhs_inputs = B1.to(dtype)
    rhs_inputs = B2.to(dtype)

    for start in range(0, out_dim, chunk_rows):
        end = min(start + chunk_rows, out_dim)
        weight_chunk = diagonal_weight[start:end]

        if A1.dim() == 2 and B1.dim() == 2:
            lhs_chunk = torch.einsum('bo,bi->boi', A1[:, start:end].to(dtype), lhs_inputs)
            rhs_chunk = torch.einsum('bo,bi->boi', A2[:, start:end].to(dtype), rhs_inputs)
        elif A1.dim() == 3 and B1.dim() == 3:
            lhs_chunk = torch.bmm(A1[:, :, start:end].transpose(1, 2).to(dtype), lhs_inputs)
            rhs_chunk = torch.bmm(A2[:, :, start:end].transpose(1, 2).to(dtype), rhs_inputs)
        else:
            raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")

        lhs_chunk = lhs_chunk * weight_chunk.unsqueeze(0)
        if weight_rhs:
            rhs_chunk = rhs_chunk * weight_chunk.unsqueeze(0)

        result.add_(torch.matmul(lhs_chunk.flatten(start_dim=1), rhs_chunk.flatten(start_dim=1).T))

    return result


def find_GClayers(model):
    """Find all GCLinear and GCLoRALinear layers in the model."""
    from OPUS.layers.linear import GCLinear
    from OPUS.layers.lora_layers import GCLoRALinear
    return [m for m in model.modules() if isinstance(m, (GCLinear, GCLoRALinear))]


def _random_windows(x, win_len, align_to=128, g=None):
    """Extract random windows from sequences."""
    N, T = x.size()
    if T <= win_len:
        idx = torch.arange(min(T, win_len), device=x.device).expand(N, -1)
        return x[:, :win_len], idx
    max_start = T - win_len
    starts = torch.randint(0, max_start + 1, (N,), device=x.device, generator=g)
    if align_to is not None and align_to > 1:
        starts = (starts // align_to) * align_to
        starts = torch.clamp(starts, 0, max_start)
    ar = torch.arange(win_len, device=x.device)
    idx = starts[:, None] + ar[None, :]
    return x.gather(1, idx), idx


def compute_GradProd_GC_per_iter(model, device, batch_train, validation_loader, optimizer, trainable_layers,
                                 preconditioner='adamw', score_len=512,
                                 per_val=False, return_tracin_and_similarity=True):
    """
    OPUS step-wise scoring (window aligned with training, dropout disabled, proxy batch refreshed each step).
    
    Args:
        model: The model to compute gradients for
        device: Compute device
        batch_train: Training batch dict with input_ids, attention_mask, labels
        validation_loader: DataLoader for proxy/validation data
        optimizer: Optimizer or list of optimizers
        trainable_layers: List of GCLinear/GCLoRALinear layers
        preconditioner: 'adamw', 'muon', 'sgd', or 'auto'
        score_len: Sequence length for scoring (shorter = faster)
        per_val: If True, return per-validation-sample scores
        return_tracin_and_similarity: If True, also return similarity matrix
    
    Returns:
        scores: Gradient similarity scores
        similarity_matrix: Self-similarity matrix (if return_tracin_and_similarity=True)
    """
    # --- Optimizer plumbing ---
    opt_list = _as_opt_list(optimizer)
    param2opt = _build_param2opt_map(opt_list)
    auto_mode = (preconditioner == 'auto')

    def _sample_proxy_batch():
        return _sample_proxy_batch_for_step(validation_loader)

    # --- Get validation batch (refreshed each step) ---
    batch_val = _sample_proxy_batch()

    val_bs = batch_val['input_ids'].shape[0]
    train_bs = batch_train['input_ids'].shape[0]

    # --- Truncate to score_len and align with proxy ---
    L = min(batch_train['input_ids'].shape[1], batch_val['input_ids'].shape[1])
    if score_len is not None:
        L = min(L, score_len)

    for B in (batch_train, batch_val):
        for k in ('input_ids', 'attention_mask', 'labels'):
            B[k] = B[k][:, :L]

    # --- Combined forward pass (window aligned with training) ---
    combined_labels = torch.cat([batch_train['labels'], batch_val['labels']], dim=0)
    combined_inputs = {
        k: torch.cat([batch_train[k], batch_val[k]], dim=0)
        for k in batch_train.keys() if k != 'labels'
    }
    del batch_train, batch_val
    
    was_training = model.training
    model.eval()
    logits = model(
        input_seq=combined_inputs['input_ids'],
        target_seq=None,
        sliding_window_num_blocks=None
    )
    if was_training:
        model.train()

    # --- Per-sample loss ---
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    per_position_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    loss = (per_position_loss * mask).sum(dim=-1) / mask.sum(dim=-1)

    # --- Get pre-activations and compute z-grad ---
    pre_acts_raw = [layer.pre_activation for layer in trainable_layers]
    keep_idx = [
        i for i, z in enumerate(pre_acts_raw)
        if isinstance(z, torch.Tensor) and z.requires_grad and z.grad_fn is not None
    ]
    if not keep_idx:
        raise RuntimeError("No usable pre_activations captured. Check GC hooks / forward path.")

    pre_acts = [pre_acts_raw[i] for i in keep_idx]
    layers_kept = [trainable_layers[i] for i in keep_idx]

    Z_grad = torch.autograd.grad(
        loss.mean(), pre_acts, retain_graph=False, allow_unused=True
    )

    dLdZ_a_train_lst, dLdZ_a_val_lst = [], []
    param_list = []

    # --- Decompose (dLdZ, a) and split train/val ---
    for layer, zgrad in zip(layers_kept, Z_grad):
        if zgrad is None:
            continue
        decompose_results = layer.pe_grad_gradcomp(zgrad, per_sample=True)
        # Normalize to a list of (dLdZ, a) pairs.
        # - GCLinear returns a single pair as a 2-tensor tuple: (dLdZ, H)
        # - LoRA GC layers return a list of pairs: [(dLdZ_B, a), (dLdO, a_A)]
        if isinstance(decompose_results, tuple):
            if len(decompose_results) == 2 and all(isinstance(x, torch.Tensor) for x in decompose_results):
                decompose_results = [decompose_results]
            else:
                decompose_results = list(decompose_results)

        layer_params = []
        if hasattr(layer, 'lora_A'): layer_params.append(layer.lora_A)
        if hasattr(layer, 'lora_B'): layer_params.append(layer.lora_B)

        train_results = [None] * len(decompose_results)
        val_results = [None] * len(decompose_results)

        for i, (dLdZ, a) in enumerate(decompose_results):
            p_ref = (layer_params[i] if i < len(layer_params) else getattr(layer, "weight", None))
            param_list.append(p_ref)

            dLdZ_train, dLdZ_val = dLdZ[:train_bs], dLdZ[train_bs:]
            a_train, a_val = a[:train_bs], a[train_bs:]

            # Scoring doesn't require autograd; detach to avoid building graphs
            # and to allow safe conversion to NumPy later.
            train_results[i] = (dLdZ_train.detach(), a_train.detach())
            val_results[i] = (dLdZ_val.detach(), a_val.detach())

        dLdZ_a_train_lst.extend(train_results)
        dLdZ_a_val_lst.extend(val_results)

    # --- Preconditioner cache ---
    S_map, scalar_muon, adamw_preconditioners = {}, {}, {}
    unique_params = [p for p in set(param_list) if p is not None]
    for p in unique_params:
        opt_p = param2opt.get(p, None)
        if opt_p is None:
            continue
        group = _find_group_for_param(opt_p, p)
        kind = _group_kind(group) if auto_mode else preconditioner

        if kind == 'muon':
            if p in opt_p.state:
                S_map[p] = _get_muon_S_matrix(p, opt_p)
            scalar_muon[p] = global_C_t_muon(opt_p, p).to(dtype=torch.float32)
        elif kind == 'adamw':
            state = opt_p.state.get(p, {})
            Ct_p = global_C_t(opt_p, p)
            adamw_preconditioners[p] = {
                "scale": Ct_p.to(dtype=torch.float32),
                "diag": _inv_rms_tensor_for_param(group, state),
            }

    # --- Scoring ---
    if per_val:
        score_t = torch.zeros((train_bs, val_bs), device=device, dtype=torch.float32)
    else:
        score_t = torch.zeros(train_bs, device=device, dtype=torch.float32)

    projector = get_projector()
    use_projection = projector is not None and projector.enabled

    for idx_pair, ((dLdZ_t, a_t), (dLdZ_v, a_v)) in enumerate(zip(dLdZ_a_train_lst, dLdZ_a_val_lst)):
        p = param_list[idx_pair]

        dLdZ_t_transformed = dLdZ_t
        dLdZ_v_transformed = dLdZ_v
        s_for_score = torch.tensor(1.0, device=device)
        layer_scale = 1.0
        preconditioner_S = None
        diagonal_preconditioner = None
        effective_preconditioner = 'sgd'

        if p is not None:
            opt_p = param2opt.get(p, None)
            if opt_p is not None:
                group = _find_group_for_param(opt_p, p)
                effective_preconditioner = _group_kind(group) if auto_mode else preconditioner
                
                if effective_preconditioner == 'muon' and p in S_map:
                    S = S_map[p].to(device)
                    preconditioner_S = S
                    dLdZ_t_transformed = torch.einsum('...o,po->...p',
                                                      dLdZ_t.float(), S.float().T).to(dLdZ_t.dtype)
                    s_for_score = scalar_muon[p]
                elif effective_preconditioner == 'adamw' and p in adamw_preconditioners:
                    s_for_score = adamw_preconditioners[p]["scale"]
                    diagonal_preconditioner = adamw_preconditioners[p]["diag"]

                if effective_preconditioner != 'sgd' and p.ndim >= 2:
                    layer_scale = 1.0 / float((p.shape[-1] * p.shape[-2]) ** 0.5)

        if use_projection:
            grad_t_proj = projector.project_full_gradient(
                dLdZ_t_transformed,
                a_t,
                param_id=idx_pair,
                preconditioner_S=None,
                diagonal_preconditioner=diagonal_preconditioner,
            )
            grad_v_proj = projector.project_full_gradient(
                dLdZ_v_transformed,
                a_v,
                param_id=idx_pair,
                preconditioner_S=None,
                diagonal_preconditioner=None,
            )
            dot_val = torch.matmul(grad_t_proj, grad_v_proj.T)
        elif diagonal_preconditioner is not None:
            dot_val = weighted_grad_dotprod(
                dLdZ_t_transformed, a_t, dLdZ_v_transformed, a_v, diagonal_preconditioner, weight_rhs=False
            )
        else:
            dot_val = grad_dotprod(dLdZ_t_transformed, a_t, dLdZ_v_transformed, a_v)
        
        scaled = (s_for_score.to(dot_val.device) * layer_scale) * dot_val

        if per_val:
            score_t.add_(scaled.float())
        else:
            score_t.add_(scaled.mean(dim=1).float())

    grad_dotproduct_score = score_t.detach().cpu().numpy()

    # --- Similarity matrix ---
    if return_tracin_and_similarity:
        sim_t = torch.zeros((train_bs, train_bs), device=device, dtype=torch.float32)

        for idx_pair, (dLdZ_i, a_i) in enumerate(dLdZ_a_train_lst):
            p = param_list[idx_pair]

            dLdZ_i_transformed = dLdZ_i
            s_for_sim = torch.tensor(1.0, device=device)
            layer_scale = 1.0
            diagonal_preconditioner = None
            effective_preconditioner = 'sgd'

            if p is not None:
                opt_p = param2opt.get(p, None)
                if opt_p is not None:
                    group = _find_group_for_param(opt_p, p)
                    effective_preconditioner = _group_kind(group) if auto_mode else preconditioner

                    if effective_preconditioner == 'muon' and p in S_map:
                        S = S_map[p].to(device)
                        dLdZ_i_transformed = torch.einsum('...o,po->...p',
                                                          dLdZ_i.float(), S.float().T).to(dLdZ_i.dtype)
                        s_for_sim = scalar_muon[p]
                    elif effective_preconditioner == 'adamw' and p in adamw_preconditioners:
                        s_for_sim = adamw_preconditioners[p]["scale"]
                        diagonal_preconditioner = adamw_preconditioners[p]["diag"]
                
                if effective_preconditioner != 'sgd' and p.ndim >= 2:
                    layer_scale = 1.0 / float((p.shape[-1] * p.shape[-2]) ** 0.5)

            if use_projection:
                grad_i_proj = projector.project_full_gradient(
                    dLdZ_i_transformed,
                    a_i,
                    param_id=idx_pair,
                    preconditioner_S=None,
                    diagonal_preconditioner=diagonal_preconditioner,
                )
                dot_mat = torch.matmul(grad_i_proj, grad_i_proj.T)
            elif diagonal_preconditioner is not None:
                dot_mat = weighted_grad_dotprod(
                    dLdZ_i_transformed, a_i, dLdZ_i_transformed, a_i, diagonal_preconditioner, weight_rhs=True
                )
            else:
                dot_mat = grad_dotprod(dLdZ_i_transformed, a_i, dLdZ_i_transformed, a_i)
            
            similarity_contribution = ((s_for_sim.to(dot_mat.device) ** 2) * (layer_scale ** 2) * dot_mat).float()
            sim_t.add_(similarity_contribution)

        similarity_local_score = sim_t.detach().cpu().numpy()
        return grad_dotproduct_score, similarity_local_score

    else:
        return grad_dotproduct_score, None


def grad_dotprod(A1, B1, A2, B2) -> torch.Tensor:
    """Compute gradient dot product for linear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        return grad_dotprod_non_sequential(A1, B1, A2, B2)
    elif A1.dim() == 3 and B1.dim() == 3:
        return grad_dotprod_sequential(A1, B1, A2, B2)
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")


def grad_dotprod_non_sequential(A1, B1, A2, B2):
    """Gradient dot product for non-sequential (2D) inputs."""
    dot_prod_1 = torch.matmul(A1, A2.T)
    dot_prod_2 = torch.matmul(B1, B2.T)
    return dot_prod_1 * dot_prod_2


def grad_dotprod_sequential(A1, B1, A2, B2, chunk_size=1024):
    """Gradient dot product for sequential (3D) inputs."""
    (b, t, p), (_, _, d) = A1.size(), B1.size()
    A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1)
    B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1)
    return torch.matmul(A, B.T)


def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on highest scores with interaction penalty.
    
    After selecting a point, subtract its interactions with remaining points
    to diversify selection.
    """
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)
        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -np.inf

    return selected_indices


def stochastic_greedy_selection(scores, interaction_matrix, K, temperature=1.0):
    """
    Select K data points using stochastic greedy approach.
    
    Uses softmax over scores to sample points, introducing randomness
    while favoring higher-scoring points.
    """
    scores = scores.copy()
    selected_indices = []
    num_points = len(scores)
    available_mask = np.ones(num_points, dtype=bool)
    
    for _ in range(K):
        current_scores = scores[available_mask]
        
        scaled_scores = current_scores / temperature
        if np.max(scaled_scores) > 700:
            scaled_scores = scaled_scores - np.max(scaled_scores)
        probabilities = np.exp(scaled_scores)
        prob_sum = probabilities.sum()
        
        if prob_sum == 0 or not np.isfinite(prob_sum):
            probabilities = np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities /= prob_sum

        available_indices = np.where(available_mask)[0]
        sampled_local_idx = np.random.choice(len(available_indices), p=probabilities)
        selected_idx = available_indices[sampled_local_idx]
        
        selected_indices.append(selected_idx)
        scores -= interaction_matrix[selected_idx, :]
        available_mask[selected_idx] = False
        
    return selected_indices


def get_batch_opus(model, buffer_seq, buffer_labels, proxy_seq, proxy_labels,
                   optimizer, trainable_layers, validation_loader,
                   selection_ratio, seq_len,   
                   selection_strategy="opus", selection_method="greedy",
                   temperature=1.5, score_len=512, preconditioner="adamw",
                   return_scores_only: bool = False, n_windows: int = 1):
    """
    OPUS batch selection with multiple strategies.
    
    Args:
        model: The model
        buffer_seq: Candidate sequences [N, L] or [T]
        buffer_labels: Candidate labels [N, L] or [T]
        proxy_seq: Proxy sequences (unused for OPUS strategy)
        proxy_labels: Proxy labels (unused for OPUS strategy)
        optimizer: Optimizer(s)
        trainable_layers: GC layers
        validation_loader: Proxy data loader
        selection_ratio: Fraction to select
        seq_len: Full training sequence length
        selection_strategy: 'opus', 'ppl', 'random'
        selection_method: 'greedy' or 'stochastic'
        temperature: Temperature for stochastic selection
        score_len: Shorter length for scoring (faster)
        preconditioner: 'adamw', 'muon', 'sgd', or 'auto'
        return_scores_only: If True, return (scores, interaction_matrix)
        n_windows: Number of random windows to score per candidate, averaged together
    
    Returns:
        Selected indices tensor, or (scores, interaction_matrix) if return_scores_only
    """
    device = buffer_seq.device
    score_len = min(score_len, seq_len)
    assert score_len % 128 == 0, "score_len must be multiple of 128 (FlexAttention BLOCK_SIZE)"

    def to_batches(x, L):
        if x.dim() == 1:
            n = (x.size(0) // L)
            if n == 0:
                raise ValueError(f"Length insufficient: {x.size(0)} < {L}")
            return x[: n * L].view(n, L)
        elif x.dim() == 2 and x.size(1) == L:
            return x
        else:
            raise ValueError(f"Expected [N,{L}] or [T], got {tuple(x.shape)}")

    buffer_seq = to_batches(buffer_seq, seq_len)
    buffer_labels = to_batches(buffer_labels, seq_len)
    if proxy_seq is not None:
        proxy_seq = to_batches(proxy_seq, seq_len)
        proxy_labels = to_batches(proxy_labels, seq_len)

    num_buf = buffer_seq.size(0)
    num_to_select = max(1, int(num_buf * selection_ratio))

    def model_loss_one(seq_1d, lab_1d, use_score_len=False):
        L = score_len if use_score_len else seq_len
        s = seq_1d[:L]
        y = lab_1d[:L]
        win_blocks = torch.tensor(L // 128, dtype=torch.int32, device=s.device)
        return model(s, y, win_blocks)

    if selection_strategy == "random":
        return torch.randperm(num_buf, device=device)[:num_to_select]

    elif selection_strategy == "ppl":
        mode = (selection_method or "high").lower()
        losses = []
        with torch.no_grad():
            for i in range(num_buf):
                l = model_loss_one(buffer_seq[i], buffer_labels[i], use_score_len=True)
                losses.append(l.detach())
        losses = torch.stack(losses)

        if mode == "low":
            _, idx = torch.topk(-losses, k=num_to_select, largest=True)
            return idx
        elif mode == "mid":
            q_low, q_high = 0.2, 0.8
            ql = torch.quantile(losses, torch.tensor(q_low, device=losses.device))
            qh = torch.quantile(losses, torch.tensor(q_high, device=losses.device))
            mask = (losses >= ql) & (losses <= qh)
            mid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if mid_idx.numel() >= num_to_select:
                median = torch.quantile(losses, torch.tensor(0.5, device=losses.device))
                dist = torch.abs(losses[mid_idx] - median)
                _, order = torch.topk(-dist, k=num_to_select, largest=True)
                return mid_idx[order]
            else:
                remaining = num_to_select - mid_idx.numel()
                fill_mask = ~mask
                dist_all = torch.full_like(losses, float('inf'))
                below = losses < ql
                above = losses > qh
                dist_all[below] = ql - losses[below]
                dist_all[above] = losses[above] - qh
                dist_all[~fill_mask] = float('inf')
                fill_k = min(remaining, int(fill_mask.sum().item()))
                _, fill_order = torch.topk(-dist_all, k=fill_k, largest=True)
                return torch.cat([mid_idx, fill_order.to(mid_idx.device)], dim=0)
        else:
            _, idx = torch.topk(losses, k=num_to_select, largest=True)
            return idx
        
    elif selection_strategy == "opus":
        scores = None
        interaction_matrix = None
        n_windows = max(1, int(n_windows))
        created_step_fixed_proxy = False

        if getattr(validation_loader, "_fixed_batch", None) is None:
            validation_loader._fixed_batch = _sample_proxy_batch_for_step(validation_loader)
            created_step_fixed_proxy = True

        try:
            for _ in range(n_windows):
                global _GBG_CALL_ID
                gen = make_window_generator(step=_GBG_CALL_ID, rank=_get_rank_safe(), device=buffer_seq.device.type)
                _GBG_CALL_ID += 1
                win_inputs, idx = _random_windows(buffer_seq, score_len, align_to=128, g=gen)
                win_labels = buffer_labels.gather(1, idx)

                batch_train = {
                    'input_ids': win_inputs,
                    'attention_mask': torch.ones_like(win_inputs),
                    'labels': win_labels
                }

                win_scores, win_interaction = compute_GradProd_GC_per_iter(
                    model, buffer_seq.device, batch_train,
                    validation_loader, optimizer, trainable_layers,
                    preconditioner=preconditioner, score_len=score_len,
                    per_val=False, return_tracin_and_similarity=True
                )
                if scores is None:
                    scores = win_scores
                    interaction_matrix = win_interaction
                else:
                    scores += win_scores
                    interaction_matrix += win_interaction
        finally:
            if created_step_fixed_proxy:
                validation_loader._fixed_batch = None

        scores /= n_windows
        interaction_matrix /= n_windows
        
        if return_scores_only:
            return torch.tensor(scores, device=buffer_seq.device, dtype=torch.float32), \
                   torch.tensor(interaction_matrix, device=buffer_seq.device, dtype=torch.float32)
        
        if selection_method == "greedy":
            selected_idx = greedy_selection(scores, interaction_matrix, num_to_select)
        elif selection_method == "stochastic":
            selected_idx = stochastic_greedy_selection(scores, interaction_matrix, num_to_select, temperature)
        return torch.tensor(selected_idx, device=buffer_seq.device)

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
