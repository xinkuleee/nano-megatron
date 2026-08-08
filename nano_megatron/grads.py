"""Gradient synchronisation that the schedules do not cover.

Two things need attention after the backward pass:

  * data-parallel replicas must average their gradients;
  * when word embeddings are tied across a split pipeline, the first and last
    stage each hold a copy whose gradients would otherwise silently diverge.
"""

import torch
import torch.distributed as dist

from . import parallel_state as ps


def all_reduce_data_parallel_gradients(model) -> None:
    dp_size = ps.get_data_parallel_world_size()
    if dp_size == 1:
        return
    group = ps.get_data_parallel_group()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=group)
            param.grad /= dp_size


def all_reduce_embedding_gradients(model) -> None:
    """Keep tied embedding / lm_head weights in step across the pipeline.

    Only meaningful once weight tying is enabled across stages; with pp_size ==
    1 the two modules are the same object and there is nothing to do.
    """
    if ps.get_pipeline_parallel_world_size() == 1:
        return
    if not (ps.is_pipeline_first_stage() or ps.is_pipeline_last_stage()):
        return

    group = ps.get_embedding_group()
    if group is None:
        return

    if ps.is_pipeline_first_stage():
        weight = getattr(model, "embedding", None)
    else:
        weight = getattr(model, "lm_head", None)
    if weight is None or weight.weight.grad is None:
        return

    dist.all_reduce(weight.weight.grad, group=group)


def synchronize_tied_embeddings(model) -> None:
    """Give the first-stage embedding and last-stage LM head equal weights.

    With pipeline parallelism they are separate ``Parameter`` objects. Their
    initial values must match before summing their gradients can emulate one
    genuinely shared parameter.
    """
    if ps.get_pipeline_parallel_world_size() == 1:
        return
    if not (ps.is_pipeline_first_stage() or ps.is_pipeline_last_stage()):
        return

    module = getattr(model, "embedding", None) if ps.is_pipeline_first_stage() else getattr(
        model, "lm_head", None
    )
    if module is None:
        return
    dist.broadcast(
        module.weight.data,
        src=ps.get_pipeline_first_rank(),
        group=ps.get_embedding_group(),
    )


def synchronize_model_parameters(model, tie_embeddings: bool = False) -> None:
    """Make replicated parameters agree after rank-aware initialization."""
    if ps.get_tensor_parallel_world_size() > 1:
        for param in model.parameters():
            if not getattr(param, "tensor_parallel_sharded", False):
                dist.broadcast(
                    param.data,
                    src=ps.get_tensor_parallel_src_rank(),
                    group=ps.get_tensor_parallel_group(),
                )

    if ps.get_data_parallel_world_size() > 1:
        for param in model.parameters():
            dist.broadcast(
                param.data,
                src=ps.get_data_parallel_src_rank(),
                group=ps.get_data_parallel_group(),
            )

    if tie_embeddings:
        synchronize_tied_embeddings(model)


def finalize_gradients(model, tie_embeddings: bool = False) -> None:
    from .tp_layers import all_reduce_replicated_gradients

    all_reduce_replicated_gradients(model)
    if tie_embeddings:
        all_reduce_embedding_gradients(model)
    all_reduce_data_parallel_gradients(model)


def clip_grad_norm(model, max_norm: float, tie_embeddings: bool = False) -> torch.Tensor:
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

    tp_size = ps.get_tensor_parallel_world_size()
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for name, param in parameters:
        if param.grad is None:
            continue
        duplicate_tied_weight = (
            tie_embeddings
            and ps.get_pipeline_parallel_world_size() > 1
            and ps.is_pipeline_last_stage()
            and name == "lm_head.weight"
        )
        if duplicate_tied_weight:
            continue

        contribution = param.grad.detach().double().pow(2).sum()
        if not getattr(param, "tensor_parallel_sharded", False):
            contribution /= tp_size
        total_sq += contribution

    if tp_size > 1:
        dist.all_reduce(total_sq, group=ps.get_tensor_parallel_group())
    if ps.get_pipeline_parallel_world_size() > 1:
        dist.all_reduce(total_sq, group=ps.get_pipeline_parallel_group())

    total_norm = total_sq.sqrt()
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for _, param in parameters:
        if param.grad is not None:
            param.grad.mul_(coefficient.to(dtype=param.grad.dtype))
    return total_norm
