"""
Random Projection for OPUS Algorithm

Implements paper equation (9): φ(u_i) = Π(P_t ∇ℓ(u_i; θ_t))

Uses CountSketch for random projection, mapping high-dimensional gradient vectors
to low-dimensional space while approximately preserving inner products
(Johnson-Lindenstrauss lemma).

GPT-2 XL gradient dimension analysis:
- c_attn (Q,K,V): 1600 x 4800 = 7,680,000 dims
- c_proj: 1600 x 1600 = 2,560,000 dims  
- c_fc (MLP): 1600 x 6400 = 10,240,000 dims
- c_proj2 (MLP): 6400 x 1600 = 10,240,000 dims

"""
import torch
import math


class GradientProjector:
    """
    Gradient Projector: Uses CountSketch to project high-dimensional gradients to low-dimensional space.
    
    CountSketch algorithm:
    - h: [d] -> [m]  hash function, maps each element to one of m buckets
    - s: [d] -> {-1, +1}  sign function
    - sketch[h(i)] += s(i) * vector[i]
    
    Mathematical guarantee: E[<Π(u), Π(v)>] = <u, v> (unbiased expectation)
    
    Variance analysis:
    - Var[<Π(u), Π(v)>] ≈ (||u||² ||v||² + <u,v>²) / m
    - For GPT-2 XL max layer (10M dims), recommend m >= 8192
    
    Args:
        sketch_dim: Projection dimension. For GPT-2 XL recommend >= 8192, default 16384
        seed: Random seed
        device: Compute device
        enabled: Whether to enable projection
    """
    
    def __init__(self, sketch_dim: int = 16384, seed: int = 42, device: str = "cuda", enabled: bool = True):
        self.sketch_dim = sketch_dim
        self.seed = seed
        self.device = device
        self.enabled = enabled
        self.hash_functions = {}
        self.sign_functions = {}

    def _get_hash_and_sign(self, param_numel: int, param_id: int):
        """Get or create hash and sign functions for given dimension and param ID."""
        cache_key = (param_numel, param_id)
        if cache_key not in self.hash_functions:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + param_id * 1000003 + param_numel)
            
            # Generate hash function h: [d] -> [m]
            self.hash_functions[cache_key] = torch.randint(
                0, self.sketch_dim, (param_numel,), 
                generator=generator, device=self.device, dtype=torch.long
            )
            
            # Generate sign function s: [d] -> {-1, +1}
            self.sign_functions[cache_key] = torch.randint(
                0, 2, (param_numel,), 
                generator=generator, device=self.device, dtype=torch.float32
            ) * 2 - 1  # Map 0->-1, 1->+1
        
        return self.hash_functions[cache_key], self.sign_functions[cache_key]

    def project_vector(self, vector: torch.Tensor, param_id: int) -> torch.Tensor:
        """
        Project a single vector using CountSketch.
        
        CountSketch guarantee: E[<sketch(u), sketch(v)>] = <u, v>
        
        Args:
            vector: Vector to project [d] or [*, d]
            param_id: Parameter identifier (for deterministic random projection)
        
        Returns:
            Projected vector [m]
        """
        if not self.enabled:
            return vector

        flat_vector = vector.flatten()
        param_numel = flat_vector.numel()
        h, s = self._get_hash_and_sign(param_numel, param_id)
        
        # Initialize sketch vector
        sketch = torch.zeros(self.sketch_dim, dtype=torch.float32, device=self.device)
        
        # Apply CountSketch: sketch[h(p)] += s(p) * vector[p]
        # No scaling needed! E[<sketch(u), sketch(v)>] = <u, v> is already unbiased
        sketch.index_add_(0, h, s * flat_vector.float())
        
        return sketch

    def project_batch(self, vectors: torch.Tensor, param_id: int) -> torch.Tensor:
        """
        Batch project vectors using CountSketch.
        
        CountSketch guarantee: E[<sketch(u), sketch(v)>] = <u, v>
        
        Args:
            vectors: [B, d] batch of vectors
            param_id: Parameter identifier
        
        Returns:
            [B, m] projected vectors
        """
        if not self.enabled:
            return vectors
        
        B, d = vectors.shape
        h, s = self._get_hash_and_sign(d, param_id)
        
        # Batch CountSketch
        # vectors: [B, d], s: [d] -> signed_vectors: [B, d]
        signed_vectors = vectors.float() * s.unsqueeze(0)
        
        # Use scatter_add for batch projection
        sketches = torch.zeros(B, self.sketch_dim, dtype=torch.float32, device=self.device)
        
        # h: [d] -> expand to [B, d]
        h_expanded = h.unsqueeze(0).expand(B, -1)
        sketches.scatter_add_(1, h_expanded, signed_vectors)
        
        # No scaling needed! CountSketch is already unbiased
        return sketches

    def project_full_gradient(self, dLdZ: torch.Tensor, a: torch.Tensor,
                               param_id: int, preconditioner_S: torch.Tensor = None,
                               diagonal_preconditioner: torch.Tensor = None) -> torch.Tensor:
        """
        Project full gradient per paper equation (9): φ(u_i) = Π(P_t ∇ℓ(u_i; θ_t))
        
        For linear layers, gradient is outer product: G = dLdZ @ a^T
        
        Args:
            dLdZ: Loss gradient w.r.t. layer output [B, D_out] or [B, T, D_out]
            a: Layer input (activation) [B, D_in] or [B, T, D_in]
            param_id: Parameter identifier
            preconditioner_S: Preconditioner matrix (optional)
            diagonal_preconditioner: AdamW-style diagonal preconditioner [D_out, D_in]
        
        Returns:
            Projected gradient [B, sketch_dim]
        """
        if not self.enabled:
            raise ValueError("Projector is disabled but project_full_gradient was called")

        # Apply dense left preconditioner before sketching. This is required for Muon-like
        # row mixing and keeps the subsequent CountSketch pass streamed over row chunks.
        if preconditioner_S is not None:
            dLdZ = torch.einsum('...o,po->...p', dLdZ.float(), preconditioner_S.float().T).to(dLdZ.dtype)

        if dLdZ.dim() == 2:
            B, d_out = dLdZ.shape
            d_in = a.shape[1]
        elif dLdZ.dim() == 3:
            B, _, d_out = dLdZ.shape
            d_in = a.shape[2]
        else:
            raise ValueError(f"Unexpected dLdZ shape: {dLdZ.shape}")

        h, s = self._get_hash_and_sign(d_out * d_in, param_id)
        h = h.view(d_out, d_in)
        s = s.view(d_out, d_in)
        sketches = torch.zeros(B, self.sketch_dim, dtype=torch.float32, device=self.device)
        chunk_rows = 32 if d_out > 32 else d_out
        a_float = a.float()
        dLdZ_float = dLdZ.float()
        diag = None if diagonal_preconditioner is None else diagonal_preconditioner.to(device=self.device, dtype=torch.float32)

        for start in range(0, d_out, chunk_rows):
            end = min(start + chunk_rows, d_out)

            if dLdZ.dim() == 3:
                grad_chunk = torch.bmm(dLdZ_float[:, :, start:end].transpose(1, 2), a_float)
            else:
                grad_chunk = torch.einsum('bo,bi->boi', dLdZ_float[:, start:end], a_float)

            if diag is not None:
                grad_chunk = grad_chunk * diag[start:end].unsqueeze(0)

            chunk_h = h[start:end].reshape(1, -1).expand(B, -1)
            chunk_signed = grad_chunk.reshape(B, -1) * s[start:end].reshape(1, -1)
            sketches.scatter_add_(1, chunk_h, chunk_signed)

        return sketches

    def clear_cache(self):
        """Clear cached hash and sign functions."""
        self.hash_functions.clear()
        self.sign_functions.clear()
