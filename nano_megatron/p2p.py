"""Point-to-point communication between adjacent pipeline stages.

Two rules matter here and both are about deadlock. Sends and receives are
issued together through ``batch_isend_irecv`` so the ordering between stages
can never bite, and the receive buffer shape is fixed by config rather than
negotiated at runtime -- a shape handshake is one more chance to hang.

The traffic itself is tiny: one ``[b, s, h]`` activation per microbatch per
stage boundary, which is why PP tolerates slow inter-node links while TP does
not.
"""

import torch
import torch.distributed as dist

from . import parallel_state as ps


def _exchange(send_prev=None, send_next=None, recv_prev=False, recv_next=False, shape=None,
              dtype=torch.float32, device="cpu"):
    ops = []
    tensor_recv_prev = tensor_recv_next = None

    if recv_prev:
        tensor_recv_prev = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
        ops.append(dist.P2POp(dist.irecv, tensor_recv_prev, ps.get_pipeline_prev_rank()))
    if send_next is not None:
        ops.append(dist.P2POp(dist.isend, send_next.contiguous(), ps.get_pipeline_next_rank()))
    if recv_next:
        tensor_recv_next = torch.empty(shape, dtype=dtype, device=device)
        ops.append(dist.P2POp(dist.irecv, tensor_recv_next, ps.get_pipeline_next_rank()))
    if send_prev is not None:
        ops.append(dist.P2POp(dist.isend, send_prev.contiguous(), ps.get_pipeline_prev_rank()))

    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return tensor_recv_prev, tensor_recv_next


def recv_forward(shape, dtype, device):
    """Receive the activation from the previous stage."""
    if ps.is_pipeline_first_stage():
        return None
    tensor, _ = _exchange(recv_prev=True, shape=shape, dtype=dtype, device=device)
    return tensor


def send_forward(tensor):
    """Send this stage's activation downstream."""
    if ps.is_pipeline_last_stage():
        return
    _exchange(send_next=tensor)


def recv_backward(shape, dtype, device):
    """Receive the activation gradient from the next stage."""
    if ps.is_pipeline_last_stage():
        return None
    _, tensor = _exchange(recv_next=True, shape=shape, dtype=dtype, device=device)
    return tensor


def send_backward(tensor):
    """Send the input gradient upstream."""
    if ps.is_pipeline_first_stage():
        return
    _exchange(send_prev=tensor)


def send_forward_recv_backward(tensor, shape, dtype, device):
    """One exchange doing both -- the steady-state move in 1F1B."""
    if ps.is_pipeline_last_stage():
        return None
    _, grad = _exchange(send_next=tensor, recv_next=True, shape=shape, dtype=dtype, device=device)
    return grad


def send_backward_recv_forward(tensor, shape, dtype, device):
    if ps.is_pipeline_first_stage():
        return None
    activation, _ = _exchange(send_prev=tensor, recv_prev=True, shape=shape, dtype=dtype,
                              device=device)
    return activation
