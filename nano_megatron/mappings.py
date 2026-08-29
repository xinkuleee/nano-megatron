"""Autograd-aware tensor- and sequence-parallel collectives.

Everything TP does rests on two conjugate autograd functions:

    f  (``copy_to_tensor_parallel_region``)    forward: identity
                                               backward: all-reduce
    g  (``reduce_from_tensor_parallel_region``) forward: all-reduce
                                               backward: identity

A column-parallel linear is ``f -> local matmul``; a row-parallel linear is
``local matmul -> g``. Chaining them means the elementwise nonlinearity in
between needs no communication at all, and autograd derives every backward
collective for free. That is the whole trick.

Sequence parallelism keeps the residual stream sharded as ``[B, S/tp, H]``.
Before a column-parallel projection the sequence is all-gathered; after the
matching row-parallel projection a SUM reduce-scatter restores the shard.  The
backward methods are the mathematical adjoints of those forward collectives.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel import ParallelContext


def _all_reduce(
    tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, group=group)
    return tensor


def _check_sequence_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim < 2:
        raise ValueError(f"sequence-parallel tensor needs at least 2 dims, got {tensor.shape}")


def _all_gather_sequence(
    tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int
) -> torch.Tensor:
    """All-gather batch-first sequence shards in TP-rank order."""
    if world_size == 1:
        return tensor
    _check_sequence_tensor(tensor)
    sequence_first = tensor.transpose(0, 1).contiguous()
    output = torch.empty(
        (sequence_first.size(0) * world_size, *sequence_first.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    gather = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    gather(output, sequence_first, group=group)
    return output.transpose(0, 1).contiguous()


def _reduce_scatter_sequence(
    tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int
) -> torch.Tensor:
    """SUM partial full-sequence tensors and scatter dim 1 across TP ranks."""
    if world_size == 1:
        return tensor
    _check_sequence_tensor(tensor)
    if tensor.size(1) % world_size:
        raise ValueError(
            f"sequence length {tensor.size(1)} is not divisible by TP size {world_size}"
        )
    sequence_first = tensor.transpose(0, 1).contiguous()
    output = torch.empty(
        (sequence_first.size(0) // world_size, *sequence_first.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    reduce_scatter = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    reduce_scatter(output, sequence_first, op=dist.ReduceOp.SUM, group=group)
    return output.transpose(0, 1).contiguous()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """f: identity forward, all-reduce backward."""

    @staticmethod
    def forward(ctx, x, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        return x

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad, ctx.group, ctx.world_size), None, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """g: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x, group, world_size):
        return _all_reduce(x, group, world_size)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-gather sequence forward; SUM reduce-scatter backward."""

    @staticmethod
    def forward(ctx, x, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        return _all_gather_sequence(x, group, world_size)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter_sequence(grad, ctx.group, ctx.world_size), None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """SUM reduce-scatter sequence forward; all-gather backward."""

    @staticmethod
    def forward(ctx, x, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        return _reduce_scatter_sequence(x, group, world_size)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather_sequence(grad, ctx.group, ctx.world_size), None, None


def copy_to_tensor_parallel_region(x: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(x, ctx.tp_group, ctx.tp_size)


def reduce_from_tensor_parallel_region(x: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(x, ctx.tp_group, ctx.tp_size)


def gather_from_sequence_parallel_region(x: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(x, ctx.tp_group, ctx.tp_size)


def reduce_scatter_to_sequence_parallel_region(
    x: torch.Tensor, ctx: ParallelContext
) -> torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(x, ctx.tp_group, ctx.tp_size)
