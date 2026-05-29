"""
OPUS: Optimizer-induced Projected Utility Selection

A gradient-based data selection algorithm for efficient language model pre-training.
It defines sample utility in the optimizer-induced update space and uses ghost
inner-product with random projection for scalable gradient similarity computation.
"""

__version__ = "1.0.0"
