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
import torch.nn as nn
import torch.nn.functional as F

from .mappings import (
    copy_to_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    reduce_from_tensor_parallel_region,
)
from .parallel import ParallelContext


def divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """y = x @ A, with A sharded column-wise: A = [A_1, ..., A_p].

    The output remains hidden-sharded. ``reduce_input_grad=False`` is used when
    an outer sequence all-gather already owns the input-gradient reduction.
    """

    def __init__(
        self, in_features: int, out_features: int, ctx: ParallelContext, bias: bool = True
    ):
        super().__init__()
        self.ctx = ctx
        self.out_features_local = divide(out_features, ctx.tp_size)
        self.weight = nn.Parameter(torch.empty(in_features, self.out_features_local))
        self.bias = nn.Parameter(torch.zeros(self.out_features_local)) if bias else None
        self.weight.tensor_parallel_sharded = True
        if self.bias is not None:
            self.bias.tensor_parallel_sharded = True
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, *, reduce_input_grad=True):
        # f: no-op forward, all-reduce the input gradient on the way back.
        if reduce_input_grad:
            x = copy_to_tensor_parallel_region(x, self.ctx)
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class RowParallelLinear(nn.Module):
    """y = x @ A, with A sharded row-wise and x already sharded to match."""

    def __init__(
        self, in_features: int, out_features: int, ctx: ParallelContext, bias: bool = True
    ):
        super().__init__()
        self.ctx = ctx
        self.in_features_local = divide(in_features, ctx.tp_size)

        self.weight = nn.Parameter(torch.empty(self.in_features_local, out_features))
        # The bias is replicated, so add it after the all-reduce -- adding it
        # before would scale it by the TP world size.
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight.tensor_parallel_sharded = True
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, *, sequence_parallel=False):
        y = x @ self.weight
        if sequence_parallel:
            y = reduce_scatter_to_sequence_parallel_region(y, self.ctx)
        else:
            # g: all-reduce the partial sums; gradient flows straight through.
            y = reduce_from_tensor_parallel_region(y, self.ctx)
        return y + self.bias if self.bias is not None else y


class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary sharded across the TP group.

    Tokens outside this rank's slice contribute a zero row, and the all-reduce
    fills them in from whichever rank owns them.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, ctx: ParallelContext
    ):
        super().__init__()
        self.ctx = ctx
        per_partition = divide(num_embeddings, ctx.tp_size)

        self.vocab_start = ctx.tp_rank * per_partition
        self.vocab_end = self.vocab_start + per_partition

        self.weight = nn.Parameter(torch.empty(per_partition, embedding_dim))
        self.weight.tensor_parallel_sharded = True
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids, *, sequence_parallel=False):
        vocab_size = self.ctx.tp_size * (self.vocab_end - self.vocab_start)
        if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
            raise ValueError("input token id is outside the model vocabulary")
        if self.ctx.tp_size > 1:
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
            if sequence_parallel:
                y = reduce_scatter_to_sequence_parallel_region(y, self.ctx)
            else:
                y = reduce_from_tensor_parallel_region(y, self.ctx)
        return y
