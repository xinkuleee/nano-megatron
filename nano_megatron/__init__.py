"""Megatron's TP, SP, PP, and DP training core, small enough to read."""

from .model import GPT, GPTConfig
from .parallel import ParallelContext, init_parallel
from .pipeline import forward_backward_1f1b

__all__ = [
    "GPT",
    "GPTConfig",
    "ParallelContext",
    "init_parallel",
    "forward_backward_1f1b",
]
