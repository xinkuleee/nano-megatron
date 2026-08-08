"""Tensor-parallel layers.

``ColumnParallelLinear`` shards the weight along its output dim, so its own
output arrives already sharded and any elementwise op that follows needs no
communication. ``RowParallelLinear`` shards along the input dim and pays one
all-reduce to sum the partial products. Pairing them -- column then row, with
the nonlinearity in between -- is what buys a whole MLP for a single all-reduce.

Weights are stored transposed relative to ``nn.Linear`` (``[in, out_local]``)
because that keeps the sharded dimension the trailing one and makes the
split/merge helpers in ``tests`` obvious to read.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import parallel_state as ps
from .mappings import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
)


def divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """y = x @ A, with A sharded column-wise: A = [A_1, ..., A_p].

    The output is left sharded unless ``gather_output`` is set.
    """

    _tensor_parallel_sharded_parameters = frozenset({"weight", "bias"})

    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        world_size = ps.get_tensor_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_local = divide(out_features, world_size)
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(in_features, self.out_features_local))
        self.bias = nn.Parameter(torch.zeros(self.out_features_local)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # f: no-op forward, all-reduce the input gradient on the way back.
        x = copy_to_tensor_parallel_region(x)
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return gather_from_tensor_parallel_region(y) if self.gather_output else y


class RowParallelLinear(nn.Module):
    """y = x @ A, with A sharded row-wise and x already sharded to match."""

    _tensor_parallel_sharded_parameters = frozenset({"weight"})

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        world_size = ps.get_tensor_parallel_world_size()
        self.in_features = in_features
        self.in_features_local = divide(in_features, world_size)
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.in_features_local, out_features))
        # The bias is replicated, so add it after the all-reduce -- adding it
        # before would scale it by the TP world size.
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        y = x @ self.weight
        # g: all-reduce the partial sums; gradient flows straight through.
        y = reduce_from_tensor_parallel_region(y)
        return y + self.bias if self.bias is not None else y


class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary sharded across the TP group.

    Tokens outside this rank's slice contribute a zero row, and the all-reduce
    fills them in from whichever rank owns them.
    """

    _tensor_parallel_sharded_parameters = frozenset({"weight"})

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        world_size = ps.get_tensor_parallel_world_size()
        rank = ps.get_tensor_parallel_rank()
        per_partition = divide(num_embeddings, world_size)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_start = rank * per_partition
        self.vocab_end = self.vocab_start + per_partition

        self.weight = nn.Parameter(torch.empty(per_partition, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids):
        if ps.get_tensor_parallel_world_size() > 1:
            mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
            local_ids = (input_ids - self.vocab_start).clone()
            local_ids[mask] = 0
        else:
            mask = None
            local_ids = input_ids

        y = F.embedding(local_ids, self.weight)
        if mask is not None:
            y = y.clone()
            y[mask, :] = 0.0
            y = reduce_from_tensor_parallel_region(y)
        return y


def set_tensor_parallel_attributes(model: nn.Module) -> None:
    """Tag parameters so the gradient sync knows which ones are replicated.

    LayerNorm/RMSNorm weights and row-parallel biases exist identically on every
    TP rank. Their gradients happen to be identical too (the inputs are), so no
    extra all-reduce is needed here -- but the tag is what a sequence-parallel
    or optimizer-sharding extension would key off.
    """
    # Reset first, then mark in a second pass. A parameter can be shared by two
    # modules (the tied embedding / LM head is the important case), so a module
    # that sees it later must never overwrite an earlier ``True`` with ``False``.
    for param in model.parameters():
        param.tensor_parallel_sharded = False

    for module in model.modules():
        sharded_names = getattr(module, "_tensor_parallel_sharded_parameters", ())
        for name, param in module.named_parameters(recurse=False):
            if name in sharded_names:
                param.tensor_parallel_sharded = True


def all_reduce_replicated_gradients(model: nn.Module) -> None:
    """Sync gradients of TP-replicated parameters.

    A no-op mathematically in this implementation (identical inputs give
    identical gradients), but it removes any drift from non-determinism and
    documents where sequence parallelism would need real communication.
    """
    if ps.get_tensor_parallel_world_size() == 1:
        return
    group = ps.get_tensor_parallel_group()
    for param in model.parameters():
        if param.grad is not None and not getattr(param, "tensor_parallel_sharded", False):
            dist.all_reduce(param.grad, group=group)
            param.grad /= ps.get_tensor_parallel_world_size()
