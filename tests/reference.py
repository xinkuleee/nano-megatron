"""Slice a single-process reference model into per-rank shards.

This is only used by the tests, but it is where the sharding rules are written
down in one place: which dim each parameter is split along, and which
parameters are replicated. If the tests pass, this file and the layers agree.
"""

from __future__ import annotations

import torch

from nano_megatron.model import GPT, GPTConfig, layers_for_stage
from nano_megatron.parallel import ParallelContext

# parameter-name suffix -> dim of the *reference* (unsharded) tensor to split.
# Anything not listed is replicated across the TP group.
_SHARD_DIM = {
    "attn.q.weight": 1,
    "attn.k.weight": 1,
    "attn.v.weight": 1,
    "attn.proj.weight": 0,
    "mlp.gate.weight": 1,
    "mlp.up.weight": 1,
    "mlp.down.weight": 0,
    "embedding.weight": 0,
    "lm_head.weight": 0,
}


def _shard_dim_for(name: str) -> int | None:
    for suffix, dim in _SHARD_DIM.items():
        if name.endswith(suffix):
            return dim
    return None


def build_reference_model(config: GPTConfig, seed: int = 0) -> GPT:
    """A complete, unsharded model -- the ground truth to compare against."""
    torch.manual_seed(seed)
    return GPT(config, ParallelContext.single())


def shard_state_dict(
    reference: GPT, config: GPTConfig, ctx: ParallelContext
) -> dict[str, torch.Tensor]:
    """Extract this rank's slice of the reference weights."""
    tp_size, tp_rank = ctx.tp_size, ctx.tp_rank
    pp_size, pp_rank = ctx.pp_size, ctx.pp_rank

    start, end = layers_for_stage(config.n_layer, pp_size, pp_rank)
    reference_state = reference.state_dict()
    sharded: dict[str, torch.Tensor] = {}

    for name, tensor in reference_state.items():
        if name.startswith("layers."):
            # Renumber global layer index -> local index on this stage.
            _, index, rest = name.split(".", 2)
            index = int(index)
            if not (start <= index < end):
                continue
            local_name = f"layers.{index - start}.{rest}"
        elif name.startswith("embedding."):
            if not ctx.is_first_stage:
                continue
            local_name = name
        elif name.startswith(("lm_head.", "final_norm.")):
            if not ctx.is_last_stage:
                continue
            local_name = name
        else:
            continue

        dim = _shard_dim_for(name)
        if dim is None or tp_size == 1:
            sharded[local_name] = tensor.clone()
        else:
            sharded[local_name] = torch.chunk(tensor, tp_size, dim=dim)[tp_rank].contiguous()

    return sharded


def build_parallel_model(
    config: GPTConfig, reference: GPT, ctx: ParallelContext
) -> GPT:
    """Construct this rank's stage and load its slice of the reference."""
    model = GPT(config, ctx)
    missing, unexpected = model.load_state_dict(
        shard_state_dict(reference, config, ctx), strict=False
    )
    unexpected = [k for k in unexpected if not k.startswith("rope_")]
    if unexpected:
        raise RuntimeError(f"unexpected keys when sharding: {unexpected}")
    real_missing = [k for k in missing if not k.startswith("rope_")]
    if real_missing:
        raise RuntimeError(f"missing keys when sharding: {real_missing}")
    return model


def gather_full_gradient(name: str, model: GPT, config: GPTConfig) -> torch.Tensor | None:
    """Reassemble a full gradient from its TP shards, for comparison."""
    param = dict(model.named_parameters()).get(name)
    if param is None or param.grad is None:
        return None
    return _gather_tensor(name, param.grad, model.ctx)


def gather_full_parameter(name: str, model: GPT) -> torch.Tensor | None:
    """Reassemble a full parameter from its TP shards, for comparison."""
    param = dict(model.named_parameters()).get(name)
    if param is None:
        return None
    return _gather_tensor(name, param.data, model.ctx)


def _gather_tensor(name: str, tensor: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    import torch.distributed as dist

    dim = _shard_dim_for(name)
    tp_size = ctx.tp_size
    if dim is None or tp_size == 1:
        return tensor.clone()

    shards = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(shards, tensor.contiguous(), group=ctx.tp_group)
    return torch.cat(shards, dim=dim)
