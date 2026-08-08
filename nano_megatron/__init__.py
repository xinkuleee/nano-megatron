"""nano-megatron: tensor and pipeline parallelism, small enough to read."""

from .model import GPT, GPTConfig
from .schedules import get_forward_backward_func

__all__ = ["GPT", "GPTConfig", "get_forward_backward_func"]
