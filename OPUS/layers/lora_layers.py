"""
LoRA Layers with Gradient Computation support for OPUS.

Provides LoRA (Low-Rank Adaptation) layers with activation capture
for per-example gradient computation.

Based on Microsoft LoRA implementation.
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALayer:
    """Base class for LoRA layers."""
    
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    """LoRA implemented in a dense layer."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class GCLoRALinear(LoRALinear):
    """
    Gradient-Computation LoRA Linear layer for OPUS.
    
    Captures layer inputs and pre-activations for per-example gradient computation
    on LoRA parameters (lora_A and lora_B).
    """
    
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0., 
                 fan_in_fan_out=False, merge_weights=True, **kwargs):
        super(GCLoRALinear, self).__init__(in_features, out_features, r=r, lora_alpha=lora_alpha,
                                           lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out,
                                           merge_weights=merge_weights, **kwargs)

        self.layer_type = 'GC_Linear_LoRA'
        self.register_forward_hook(self.capture_hook)

    def capture_hook(self, module, input, output):
        """Forward hook to capture layer input and output."""
        self.layer_input = input[0]
        self.pre_activation = output

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        """
        Return decomposition for efficient LoRA gradient inner-product computation.
        
        For LoRA, we need gradients w.r.t. both lora_A and lora_B:
        - dL/d(lora_A) = lora_B^T @ dL/dO @ a^T
        - dL/d(lora_B) = dL/dO @ a @ lora_A^T
        
        Returns decomposed pairs for both parameters.
        """
        a = self.layer_input.to(self.lora_A.dtype)
        dLdO = (deriv_pre_activ * deriv_pre_activ.shape[0]).to(self.lora_B.dtype)
        dLdO_B = torch.matmul(dLdO, self.lora_B)
        a_A = torch.matmul(a, self.lora_A.T)

        return [(dLdO_B, a), (dLdO, a_A)]
