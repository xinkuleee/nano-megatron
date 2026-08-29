"""Gradient synchronization after pipeline backward completes.

Two things need attention after the backward pass:

  * data-parallel replicas must average their gradients;
  * when word embeddings are tied across a split pipeline, the first and last
    stage each hold a copy whose gradients would otherwise silently diverge.
"""

import torch
import torch.distributed as dist

from .parallel import ParallelContext


def all_reduce_replicated_gradients(
    model, ctx: ParallelContext, *, sequence_parallel: bool = False
) -> None:
    """Synchronize parameters that are replicated across TP ranks.

    Without SP every TP rank sees all tokens, so gradients are duplicates and
    are averaged. With SP they cover disjoint tokens and must remain a SUM.
    """
    if ctx.tp_size == 1:
        return
    for param in model.parameters():
        if param.grad is not None and not getattr(
            param, "tensor_parallel_sharded", False
        ):
            dist.all_reduce(param.grad, group=ctx.tp_group)
            if not sequence_parallel:
                param.grad /= ctx.tp_size


def all_reduce_data_parallel_gradients(model, ctx: ParallelContext) -> None:
    if ctx.dp_size == 1:
        return
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for param in model.parameters():
        if param.grad is not None:
            buckets.setdefault((param.grad.device, param.grad.dtype), []).append(param.grad)

    # One collective per dtype/device instead of one per parameter. The flat
    # buffer is intentionally rebuilt each step: compact, deterministic, and
    # sufficient before introducing overlap or persistent gradient buckets.
    for gradients in buckets.values():
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat, group=ctx.dp_group)
        flat /= ctx.dp_size
        offset = 0
        for gradient in gradients:
            size = gradient.numel()
            gradient.copy_(flat[offset : offset + size].view_as(gradient))
            offset += size


def all_reduce_embedding_gradients(model, ctx: ParallelContext) -> None:
    """Keep tied embedding / lm_head weights in step across the pipeline.

    Only meaningful once weight tying is enabled across stages; with pp_size ==
    1 the two modules are the same object and there is nothing to do.
    """
    if ctx.pp_size == 1:
        return
    if not (ctx.is_first_stage or ctx.is_last_stage):
        return

    if ctx.tied_group is None:
        return

    if ctx.is_first_stage:
        weight = getattr(model, "embedding", None)
    else:
        weight = getattr(model, "lm_head", None)
    if weight is None or weight.weight.grad is None:
        return

    dist.all_reduce(weight.weight.grad, group=ctx.tied_group)


def synchronize_tied_embeddings(model, ctx: ParallelContext) -> None:
    """Give the first-stage embedding and last-stage LM head equal weights.

    With pipeline parallelism they are separate ``Parameter`` objects. Their
    initial values must match before summing their gradients can emulate one
    genuinely shared parameter.
    """
    if ctx.pp_size == 1:
        return
    if not (ctx.is_first_stage or ctx.is_last_stage):
        return

    module = getattr(model, "embedding", None) if ctx.is_first_stage else getattr(
        model, "lm_head", None
    )
    if module is None:
        return
    dist.broadcast(
        module.weight.data,
        src=ctx.pp_ranks[0],
        group=ctx.tied_group,
    )


def synchronize_model_parameters(
    model, ctx: ParallelContext, *, tie_embeddings: bool = False
) -> None:
    """Make replicated parameters agree after rank-aware initialization."""
    if ctx.tp_size > 1:
        for param in model.parameters():
            if not getattr(param, "tensor_parallel_sharded", False):
                dist.broadcast(
                    param.data,
                    src=ctx.tp_ranks[0],
                    group=ctx.tp_group,
                )

    if ctx.dp_size > 1:
        for param in model.parameters():
            dist.broadcast(
                param.data,
                src=ctx.dp_ranks[0],
                group=ctx.dp_group,
            )

    if tie_embeddings:
        synchronize_tied_embeddings(model, ctx)


def finalize_gradients(
    model,
    ctx: ParallelContext,
    *,
    tie_embeddings: bool = False,
    sequence_parallel: bool = False,
) -> None:
    all_reduce_replicated_gradients(model, ctx, sequence_parallel=sequence_parallel)
    if tie_embeddings:
        all_reduce_embedding_gradients(model, ctx)
    all_reduce_data_parallel_gradients(model, ctx)


def clip_grad_norm(
    model,
    ctx: ParallelContext,
    max_norm: float,
    *,
    tie_embeddings: bool = False,
) -> torch.Tensor:
    """Clip using the L2 norm of the logical, unsharded model.

    Tensor-parallel shards are unique and must be summed; TP-replicated
    parameters are divided by the TP size before that reduction. Pipeline
    stages are then summed. Data-parallel replicas already hold identical,
    averaged gradients, so reducing over DP would incorrectly multiply the
    norm. A tied last-stage LM head is the same logical parameter as the
    first-stage embedding and is counted only once.
    """
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")

    parameters = list(model.named_parameters())
    device = next((param.device for _, param in parameters), torch.device("cpu"))

    tp_size = ctx.tp_size
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for name, param in parameters:
        if param.grad is None:
            continue
        duplicate_tied_weight = (
            tie_embeddings
            and ctx.pp_size > 1
            and ctx.is_last_stage
            and name == "lm_head.weight"
        )
        if duplicate_tied_weight:
            continue

        contribution = param.grad.detach().double().pow(2).sum()
        if not getattr(param, "tensor_parallel_sharded", False):
            contribution /= tp_size
        total_sq += contribution

    if tp_size > 1:
        dist.all_reduce(total_sq, group=ctx.tp_group)
    if ctx.pp_size > 1:
        dist.all_reduce(total_sq, group=ctx.pp_group)

    total_norm = total_sq.sqrt()
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for _, param in parameters:
        if param.grad is not None:
            param.grad.mul_(coefficient.to(dtype=param.grad.dtype))
    return total_norm
