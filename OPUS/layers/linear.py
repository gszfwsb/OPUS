"""
GCLinear: Gradient-Computation Linear Layer for OPUS.

Extends nn.Linear to capture layer inputs and pre-activations,
enabling efficient per-example gradient computation without
materializing full gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCLinear(nn.Linear):
    """
    Gradient-Computation Linear layer that captures activations for ghost inner-product.
    
    Stores layer_input and pre_activation for per-example gradient computation:
    - Gradient w.r.t. weight: dLdZ^T @ layer_input
    - Gradient w.r.t. bias: dLdZ
    
    This avoids materializing full per-example gradients by computing
    gradient inner-products directly from activations.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCLinear, self).__init__(in_features, out_features, bias)
        
        # Stored for per-example gradient computation
        self.pre_activation = None
        self.layer_input = None
        self.name = 'linear'
        self.has_bias = bias

    def forward(self, input):        
        self.layer_input = input
        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out            
        return self.pre_activation
    
    def per_example_gradient(self, deriv_pre_activ):
        """
        Compute per-example gradients w.r.t. weights and bias.

        Parameters:
            deriv_pre_activ: Derivative of loss w.r.t. pre-activation [B, D_out] or [B, T, D_out]
        
        Returns:
            pe_grad_weight: Per-example weight gradients [B, D_out, D_in]
            pe_grad_bias: Per-example bias gradients [B, D_out]
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input
        
        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1),
                                       H.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)    
            pe_grad_weight = torch.bmm(dLdZ, H.transpose(0, 1))
            pe_grad_bias = dLdZ.sum(dim=-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Compute squared norm of per-example gradients efficiently.
        
        For 2D inputs, uses the identity:
        ||dLdZ @ H^T||^2 = ||dLdZ||^2 * ||H||^2
        
        Parameters:
            deriv_pre_activ: Derivative of loss w.r.t. pre-activation
        
        Returns:
            Squared norms [B]
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            zsum = dLdZ.pow(2).sum(1)
            hsum = H.pow(2).sum(1)
            s = zsum * hsum
            
            return s + zsum
        else:
            pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
            batch_size = pe_grad_weight.size(0)
            sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)

            return sq_norm_weight + sq_norm_bias

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        """
        Return decomposition for efficient gradient inner-product computation.
        
        For ghost inner-product: <G_i, G_j> = <dLdZ_i, dLdZ_j> * <H_i, H_j>
        
        Parameters:
            deriv_pre_activ: Derivative of loss w.r.t. pre-activation
            per_sample: Whether to return per-sample decomposition
        
        Returns:
            (dLdZ, H) tuple for gradient inner-product computation
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input
        batch_size = deriv_pre_activ.shape[0]
        dLdZ = deriv_pre_activ * batch_size

        if is_2d and self.has_bias:
            # Append ones for bias gradient
            ones_column = torch.ones(H.size(0), 1, device=H.device)
            H = torch.cat((H, ones_column), dim=1)
            
        return dLdZ, H
