"""
OPUS Train: Training utilities for gradient-based data selection.

Provides:
- data_selection: Core OPUS algorithm with ghost inner-product and random projection
- random_projection: CountSketch-based gradient compression
"""

from .data_selection import (
    initialize_projector,
    get_batch_opus,
    compute_GradProd_GC_per_iter,
    find_GClayers,
    greedy_selection,
    stochastic_greedy_selection,
)
from .random_projection import GradientProjector

__all__ = [
    "initialize_projector",
    "get_batch_opus", 
    "compute_GradProd_GC_per_iter",
    "find_GClayers",
    "greedy_selection",
    "stochastic_greedy_selection",
    "GradientProjector",
]
