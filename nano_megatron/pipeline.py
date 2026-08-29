"""Point-to-point communication and a non-interleaved 1F1B schedule.

Every rank executes the same single-threaded loop.  Earlier pipeline stages
warm up with more forwards, all stages then alternate one forward and one
backward, and the remaining warmup activations are drained in cooldown.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from .parallel import ParallelContext


Batch = tuple[torch.Tensor, torch.Tensor]


def _exchange(
    ctx: ParallelContext,
    *,
    send_prev: torch.Tensor | None = None,
    send_next: torch.Tensor | None = None,
    recv_prev: bool = False,
    recv_next: bool = False,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Exchange activations/gradients with adjacent pipeline stages."""
    received_prev = (
        torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
        if recv_prev
        else None
    )
    received_next = (
        torch.empty(shape, dtype=dtype, device=device) if recv_next else None
    )
    operations = []
    if received_prev is not None:
        operations.append(
            dist.P2POp(dist.irecv, received_prev, ctx.previous_pipeline_rank)
        )
    if send_next is not None:
        operations.append(
            dist.P2POp(dist.isend, send_next.contiguous(), ctx.next_pipeline_rank)
        )
    if received_next is not None:
        operations.append(
            dist.P2POp(dist.irecv, received_next, ctx.next_pipeline_rank)
        )
    if send_prev is not None:
        operations.append(
            dist.P2POp(dist.isend, send_prev.contiguous(), ctx.previous_pipeline_rank)
        )

    for request in dist.batch_isend_irecv(operations) if operations else ():
        request.wait()
    return received_prev, received_next


def _recv_forward(ctx, shape, dtype, device):
    if ctx.is_first_stage:
        return None
    activation, _ = _exchange(
        ctx, recv_prev=True, shape=shape, dtype=dtype, device=device
    )
    return activation


def _send_forward(ctx, tensor, shape, dtype, device):
    if not ctx.is_last_stage:
        _exchange(
            ctx, send_next=tensor, shape=shape, dtype=dtype, device=device
        )


def _recv_backward(ctx, shape, dtype, device):
    if ctx.is_last_stage:
        return None
    _, gradient = _exchange(
        ctx, recv_next=True, shape=shape, dtype=dtype, device=device
    )
    return gradient


def _send_backward(ctx, tensor, shape, dtype, device):
    if not ctx.is_first_stage:
        _exchange(
            ctx, send_prev=tensor, shape=shape, dtype=dtype, device=device
        )


def _send_forward_recv_backward(ctx, tensor, shape, dtype, device):
    if ctx.is_last_stage:
        return None
    _, gradient = _exchange(
        ctx,
        send_next=tensor,
        recv_next=True,
        shape=shape,
        dtype=dtype,
        device=device,
    )
    return gradient


def _forward_step(
    model: nn.Module,
    batch: Batch,
    hidden: torch.Tensor | None,
    losses: list[torch.Tensor],
    num_microbatches: int,
    ctx: ParallelContext,
) -> torch.Tensor:
    inputs, labels = batch
    stage_input = inputs if ctx.is_first_stage else hidden
    if stage_input is None:
        raise RuntimeError("a non-first pipeline stage requires an activation")

    output = model(stage_input, labels if ctx.is_last_stage else None)
    if ctx.is_last_stage:
        # Accumulating these scaled losses matches one logical large batch.
        output = output / num_microbatches
        losses.append(output.detach())
    return output


def _backward_step(
    input_tensor: torch.Tensor | None,
    output_tensor: torch.Tensor,
    output_gradient: torch.Tensor | None,
    ctx: ParallelContext,
) -> torch.Tensor | None:
    if ctx.is_last_stage:
        torch.autograd.backward(output_tensor)
    else:
        if output_gradient is None:
            raise RuntimeError("a non-last pipeline stage requires an output gradient")
        torch.autograd.backward(output_tensor, grad_tensors=output_gradient)

    if input_tensor is None:
        return None
    if input_tensor.grad is None:
        raise RuntimeError("pipeline input did not receive a gradient")
    return input_tensor.grad


def forward_backward_1f1b(
    model: nn.Module,
    batches: Sequence[Batch],
    config,
    device: torch.device | str,
    ctx: ParallelContext,
) -> list[torch.Tensor]:
    """Run non-interleaved 1F1B and return detached losses on the last stage."""
    num_microbatches = len(batches)
    if num_microbatches == 0:
        raise ValueError("1F1B requires at least one microbatch")

    micro_batch_size = batches[0][0].size(0)
    expected_batch_shape = (micro_batch_size, config.seq_len)
    if any(
        tuple(inputs.shape) != expected_batch_shape
        or tuple(labels.shape) != expected_batch_shape
        for inputs, labels in batches
    ):
        raise ValueError(
            f"every microbatch input and label must have shape {expected_batch_shape}"
        )
    sequence_length = config.seq_len // (ctx.tp_size if config.sequence_parallel else 1)
    shape = (micro_batch_size, sequence_length, config.n_embd)
    dtype = config.activation_dtype

    # Earlier stages run further ahead.  With pp_size == 1 this is zero, so the
    # same steady-state loop becomes ordinary gradient accumulation.
    num_warmup = min(ctx.pp_size - ctx.pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    inputs: list[torch.Tensor | None] = []
    outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    batch_iterator = iter(batches)

    for _ in range(num_warmup):
        hidden = _recv_forward(ctx, shape, dtype, device)
        output = _forward_step(
            model, next(batch_iterator), hidden, losses, num_microbatches, ctx
        )
        _send_forward(ctx, output, shape, dtype, device)
        inputs.append(hidden)
        outputs.append(output)

    for _ in range(num_steady):
        hidden = _recv_forward(ctx, shape, dtype, device)
        output = _forward_step(
            model, next(batch_iterator), hidden, losses, num_microbatches, ctx
        )
        output_gradient = _send_forward_recv_backward(
            ctx, output, shape, dtype, device
        )

        inputs.append(hidden)
        outputs.append(output)
        input_tensor = inputs.pop(0)
        output_tensor = outputs.pop(0)
        input_gradient = _backward_step(
            input_tensor, output_tensor, output_gradient, ctx
        )
        _send_backward(ctx, input_gradient, shape, dtype, device)

    for _ in range(num_warmup):
        input_tensor = inputs.pop(0)
        output_tensor = outputs.pop(0)
        output_gradient = _recv_backward(ctx, shape, dtype, device)
        input_gradient = _backward_step(
            input_tensor, output_tensor, output_gradient, ctx
        )
        _send_backward(ctx, input_gradient, shape, dtype, device)

    return losses
