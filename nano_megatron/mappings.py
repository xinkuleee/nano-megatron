"""The communication primitives of tensor parallelism.

Everything TP does rests on two conjugate autograd functions:

    f  (``copy_to_tensor_parallel_region``)    forward: identity
                                               backward: all-reduce
    g  (``reduce_from_tensor_parallel_region``) forward: all-reduce
                                               backward: identity

A column-parallel linear is ``f -> local matmul``; a row-parallel linear is
``local matmul -> g``. Chaining them means the elementwise nonlinearity in
between needs no communication at all, and autograd derives every backward
collective for free. That is the whole trick.

The scatter/gather pair is the sequence-parallel variant, kept here because it
makes the identity ``all_reduce == reduce_scatter + all_gather`` explicit.
"""

import torch
import torch.distributed as dist

from . import parallel_state as ps


def _all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if ps.get_tensor_parallel_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, group=ps.get_tensor_parallel_group())
    return tensor


def _split_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    world_size = ps.get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor
    chunks = torch.chunk(tensor, world_size, dim=-1)
    return chunks[ps.get_tensor_parallel_rank()].contiguous()


def _gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    world_size = ps.get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor
    tensor = tensor.contiguous()
    buffers = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(buffers, tensor, group=ps.get_tensor_parallel_group())
    return torch.cat(buffers, dim=-1)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """f: identity forward, all-reduce backward."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """g: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x):
        return _all_reduce(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Keep this rank's shard of the last dim; all-gather on the way back."""

    @staticmethod
    def forward(ctx, x):
        return _split_last_dim(x)

    @staticmethod
    def backward(ctx, grad):
        return _gather_last_dim(grad)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Rebuild the full last dim; drop back to this rank's shard on the way back."""

    @staticmethod
    def forward(ctx, x):
        return _gather_last_dim(x)

    @staticmethod
    def backward(ctx, grad):
        return _split_last_dim(grad)


def copy_to_tensor_parallel_region(x):
    return _CopyToTensorParallelRegion.apply(x)


def reduce_from_tensor_parallel_region(x):
    return _ReduceFromTensorParallelRegion.apply(x)


def scatter_to_tensor_parallel_region(x):
    return _ScatterToTensorParallelRegion.apply(x)


def gather_from_tensor_parallel_region(x):
    return _GatherFromTensorParallelRegion.apply(x)
