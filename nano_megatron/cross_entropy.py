"""Cross entropy over a vocabulary-sharded logit tensor.

The point of this file is what it *avoids*: never materialise the full
``[batch, seq, vocab]`` logits. An all-gather of that tensor would dwarf every
other collective in the model. Instead the two reductions softmax actually
needs -- a max and a sum -- are computed across the TP group on tensors of
shape ``[batch * seq]``, dropping the traffic from O(b*s*V) to O(b*s).
"""

import torch
import torch.distributed as dist

from .parallel import ParallelContext


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, vocab_start, vocab_end, group, world_size):
        # Subtract the global max for numerical stability. One all-reduce over
        # a [n] tensor, not [n, V].
        logits_max = logits.max(dim=-1)[0]
        if world_size > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits = logits - logits_max.unsqueeze(-1)

        # Pick out the logit of the target token, zeroing the ranks that do not
        # own it, then sum across the group so every rank ends up with it.
        target_mask = (target < vocab_start) | (target >= vocab_end)
        local_target = (target - vocab_start).clone()
        local_target[target_mask] = 0

        flat_logits = logits.view(-1, logits.size(-1))
        rows = torch.arange(flat_logits.size(0), device=flat_logits.device)
        predicted = flat_logits[rows, local_target.view(-1)]
        predicted = predicted.view_as(target).clone()
        predicted[target_mask] = 0.0
        if world_size > 1:
            dist.all_reduce(predicted, group=group)

        exp_logits = logits.exp()
        sum_exp = exp_logits.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(sum_exp, group=group)

        loss = sum_exp.log() - predicted

        # softmax, reused in backward
        exp_logits.div_(sum_exp.unsqueeze(-1))
        ctx.save_for_backward(exp_logits, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, local_target = ctx.saved_tensors

        # d/dlogit = softmax - onehot(target), with the one-hot only subtracted
        # on the rank that owns the target token.
        grad_input = softmax
        flat = grad_input.view(-1, grad_input.size(-1))
        rows = torch.arange(flat.size(0), device=flat.device)
        keep = (~target_mask).view(-1).to(flat.dtype)
        flat[rows, local_target.view(-1)] -= keep

        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None, None, None, None


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    ctx: ParallelContext,
) -> torch.Tensor:
    """Mean cross-entropy loss from vocab-sharded ``logits``.

    ``logits`` is ``[b, s, V/tp]``; ``target`` is ``[b, s]`` of global token ids.
    """
    vocab_size = ctx.tp_size * (vocab_end - vocab_start)
    if torch.any(target < 0) or torch.any(target >= vocab_size):
        raise ValueError("target token id is outside the model vocabulary")
    loss = _VocabParallelCrossEntropy.apply(
        logits.float(), target, vocab_start, vocab_end, ctx.tp_group, ctx.tp_size
    )
    return loss.mean()
