"""
OPUS Layers: Custom layers for gradient computation.

Provides GCLinear and GCLoRALinear layers that capture activations
for per-example gradient computation (ghost inner-product).
"""

from .linear import GCLinear
from .lora_layers import GCLoRALinear, LoRALinear

__all__ = ["GCLinear", "GCLoRALinear", "LoRALinear"]
