"""Pipeline schedules.

The scheduler *is* the pipeline parallelism -- there is no separate runtime.
Each rank runs a plain single-threaded loop; the pipelining effect emerges from
different ranks sitting at different points in that loop, with the blocking
receive acting as the only synchronisation needed. No queues, no threads.

GPipe and 1F1B differ purely in the order of the same two step functions.
GPipe runs every forward then every backward, so it holds ``M`` activations.
1F1B interleaves them once the pipe is full, so a stage holds at most
``pp_size - pp_rank``. The bubble ratio is identical, ``(P-1)/M`` -- the win is
entirely in activation memory.
"""

from __future__ import annotations

from typing import Callable

import torch

from . import p2p
from . import parallel_state as ps


def _forward_step(model, batch, hidden, loss_accumulator, num_microbatches):
    """Run one microbatch through this stage.

    The loss is divided by the microbatch count so that accumulating gradients
    over all of them reproduces one big batch exactly -- which is what the
    correctness tests check.
    """
    inputs, labels = batch
    stage_input = inputs if ps.is_pipeline_first_stage() else hidden
    output = model(stage_input, labels if ps.is_pipeline_last_stage() else None)

    if ps.is_pipeline_last_stage():
        output = output / num_microbatches
        loss_accumulator.append(output.detach())
    return output


def _backward_step(input_tensor, output_tensor, grad_output):
    """Backprop one microbatch and return the gradient to send upstream."""
    if input_tensor is not None:
        input_tensor.retain_grad()

    if ps.is_pipeline_last_stage():
        # ``output_tensor`` is the loss; seed the backward pass with 1.
        torch.autograd.backward(output_tensor)
    else:
        torch.autograd.backward(output_tensor, grad_tensors=grad_output)

    return input_tensor.grad if input_tensor is not None else None


def forward_backward_no_pipelining(model, batches, config, device):
    """pp_size == 1: just a loop over microbatches."""
    losses = []
    for batch in batches:
        output = _forward_step(model, batch, None, losses, len(batches))
        torch.autograd.backward(output)
    return losses


def forward_backward_gpipe(model, batches, config, device):
    """All forwards, then all backwards. Holds M activations at once."""
    num_microbatches = len(batches)
    micro_bs = batches[0][0].size(0)
    shape = (micro_bs, config.seq_len, config.n_embd)
    dtype = torch.float32

    input_tensors, output_tensors, losses = [], [], []

    for batch in batches:
        hidden = p2p.recv_forward(shape, dtype, device)
        output = _forward_step(model, batch, hidden, losses, num_microbatches)
        p2p.send_forward(output)
        input_tensors.append(hidden)
        output_tensors.append(output)

    for _ in range(num_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        grad_output = p2p.recv_backward(shape, dtype, device)
        grad_input = _backward_step(input_tensor, output_tensor, grad_output)
        p2p.send_backward(grad_input)

    return losses


def forward_backward_pipelining_1f1b(model, batches, config, device):
    """Warmup, then strict alternation, then cooldown."""
    num_microbatches = len(batches)
    pp_size = ps.get_pipeline_parallel_world_size()
    pp_rank = ps.get_pipeline_parallel_rank()

    micro_bs = batches[0][0].size(0)
    shape = (micro_bs, config.seq_len, config.n_embd)
    dtype = torch.float32

    # Earlier stages must run further ahead to keep the pipe full.
    num_warmup = min(pp_size - pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    input_tensors, output_tensors, losses = [], [], []
    batch_iter = iter(batches)

    # --- warmup: forward only ---
    for _ in range(num_warmup):
        hidden = p2p.recv_forward(shape, dtype, device)
        output = _forward_step(model, next(batch_iter), hidden, losses, num_microbatches)
        p2p.send_forward(output)
        input_tensors.append(hidden)
        output_tensors.append(output)

    # --- steady state: one forward, one backward ---
    for _ in range(num_steady):
        hidden = p2p.recv_forward(shape, dtype, device)
        output = _forward_step(model, next(batch_iter), hidden, losses, num_microbatches)

        # Fuse "send my activation down" with "receive my gradient back".
        grad_output = p2p.send_forward_recv_backward(output, shape, dtype, device)

        input_tensors.append(hidden)
        output_tensors.append(output)

        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        grad_input = _backward_step(input_tensor, output_tensor, grad_output)
        p2p.send_backward(grad_input)

    # --- cooldown: backward only, draining what warmup queued ---
    for _ in range(num_warmup):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        grad_output = p2p.recv_backward(shape, dtype, device)
        grad_input = _backward_step(input_tensor, output_tensor, grad_output)
        p2p.send_backward(grad_input)

    return losses


SCHEDULES: dict[str, Callable] = {
    "gpipe": forward_backward_gpipe,
    "1f1b": forward_backward_pipelining_1f1b,
}


def get_forward_backward_func(schedule: str = "1f1b"):
    if ps.get_pipeline_parallel_world_size() == 1:
        return forward_backward_no_pipelining
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; choose from {sorted(SCHEDULES)}")
    return SCHEDULES[schedule]
