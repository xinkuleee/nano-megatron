"""Construction of the orthogonal TP / PP / DP communication groups.

This is the skeleton everything else stands on. The global set of ranks is
viewed as a 3-D grid of shape ``(pp, dp, tp)`` with ``tp`` varying fastest, so
that a TP group is always a contiguous block of ranks -- on real hardware that
puts the chatty all-reduces on intra-node NVLink and leaves PP, whose traffic is
small point-to-point messages, on the slow outer axis.

    rank = tp_rank + tp_size * (dp_rank + dp_size * pp_rank)
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist

_TP_GROUP = None
_PP_GROUP = None
_DP_GROUP = None
# First and last pipeline stage, used to keep tied word embeddings in sync.
_EMBEDDING_GROUP = None

_TP_GLOBAL_RANKS: list[int] = []
_PP_GLOBAL_RANKS: list[int] = []
_DP_GLOBAL_RANKS: list[int] = []

# Set by ``single_rank_context`` so that the same model code can be built
# without any distributed context at all (used to construct the single-process
# reference model the parallel one is checked against).
_OVERRIDE: dict | None = None


def initialize_model_parallel(tp_size: int = 1, pp_size: int = 1) -> None:
    world_size = dist.get_world_size()
    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(
            f"world size {world_size} not divisible by tp_size * pp_size = {tp_size * pp_size}"
        )
    dp_size = world_size // (tp_size * pp_size)
    rank = dist.get_rank()

    grid = torch.arange(world_size).reshape(pp_size, dp_size, tp_size)

    global _TP_GROUP, _PP_GROUP, _DP_GROUP, _EMBEDDING_GROUP
    global _TP_GLOBAL_RANKS, _PP_GLOBAL_RANKS, _DP_GLOBAL_RANKS

    # Tensor parallel: vary tp, hold (pp, dp).
    for pp in range(pp_size):
        for dp in range(dp_size):
            ranks = grid[pp, dp, :].tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                _TP_GROUP, _TP_GLOBAL_RANKS = group, ranks

    # Data parallel: vary dp, hold (pp, tp).
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = grid[pp, :, tp].tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                _DP_GROUP, _DP_GLOBAL_RANKS = group, ranks

    # Pipeline parallel: vary pp, hold (dp, tp).
    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = grid[:, dp, tp].tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                _PP_GROUP, _PP_GLOBAL_RANKS = group, ranks

            # Every group must be created by every rank, in the same order,
            # even when this rank is not a member of it.
            embedding_ranks = [ranks[0], ranks[-1]] if pp_size > 1 else [ranks[0]]
            embedding_group = dist.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = embedding_group


def destroy_model_parallel() -> None:
    global _TP_GROUP, _PP_GROUP, _DP_GROUP, _EMBEDDING_GROUP
    global _TP_GLOBAL_RANKS, _PP_GLOBAL_RANKS, _DP_GLOBAL_RANKS
    _TP_GROUP = _PP_GROUP = _DP_GROUP = _EMBEDDING_GROUP = None
    _TP_GLOBAL_RANKS = _PP_GLOBAL_RANKS = _DP_GLOBAL_RANKS = []


@contextmanager
def single_rank_context():
    """Pretend tp = pp = dp = 1 so model code builds without any groups.

    Used to construct the reference model that parallel runs are validated
    against; all collectives degenerate to no-ops.
    """
    global _OVERRIDE
    previous, _OVERRIDE = _OVERRIDE, {"tp_size": 1, "tp_rank": 0, "pp_size": 1, "pp_rank": 0}
    try:
        yield
    finally:
        _OVERRIDE = previous


def _initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_tensor_parallel_group():
    return None if _OVERRIDE is not None else _TP_GROUP


def get_pipeline_parallel_group():
    return None if _OVERRIDE is not None else _PP_GROUP


def get_data_parallel_group():
    return None if _OVERRIDE is not None else _DP_GROUP


def get_embedding_group():
    return None if _OVERRIDE is not None else _EMBEDDING_GROUP


def get_tensor_parallel_world_size() -> int:
    if _OVERRIDE is not None:
        return _OVERRIDE["tp_size"]
    return dist.get_world_size(_TP_GROUP) if _TP_GROUP is not None else 1


def get_tensor_parallel_rank() -> int:
    if _OVERRIDE is not None:
        return _OVERRIDE["tp_rank"]
    return dist.get_rank(_TP_GROUP) if _TP_GROUP is not None else 0


def get_pipeline_parallel_world_size() -> int:
    if _OVERRIDE is not None:
        return _OVERRIDE["pp_size"]
    return dist.get_world_size(_PP_GROUP) if _PP_GROUP is not None else 1


def get_pipeline_parallel_rank() -> int:
    if _OVERRIDE is not None:
        return _OVERRIDE["pp_rank"]
    return dist.get_rank(_PP_GROUP) if _PP_GROUP is not None else 0


def get_data_parallel_world_size() -> int:
    if _OVERRIDE is not None:
        return 1
    return dist.get_world_size(_DP_GROUP) if _DP_GROUP is not None else 1


def get_data_parallel_rank() -> int:
    if _OVERRIDE is not None:
        return 0
    return dist.get_rank(_DP_GROUP) if _DP_GROUP is not None else 0


def is_pipeline_first_stage() -> bool:
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    return get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1


def get_pipeline_prev_rank() -> int:
    """Global rank of the previous pipeline stage."""
    pp_rank = get_pipeline_parallel_rank()
    return _PP_GLOBAL_RANKS[pp_rank - 1]


def get_pipeline_next_rank() -> int:
    """Global rank of the next pipeline stage."""
    pp_rank = get_pipeline_parallel_rank()
    return _PP_GLOBAL_RANKS[pp_rank + 1]


def get_pipeline_first_rank() -> int:
    """Global rank of the first stage in this rank's pipeline group."""
    return _PP_GLOBAL_RANKS[0]


def get_tensor_parallel_src_rank() -> int:
    """Global source rank used to synchronize TP-replicated parameters."""
    return _TP_GLOBAL_RANKS[0]


def get_data_parallel_src_rank() -> int:
    """Global source rank used to synchronize data-parallel replicas."""
    return _DP_GLOBAL_RANKS[0]


def describe() -> str:
    return (
        f"rank {dist.get_rank():2d}/{dist.get_world_size()} | "
        f"tp {get_tensor_parallel_rank()}/{get_tensor_parallel_world_size()} "
        f"pp {get_pipeline_parallel_rank()}/{get_pipeline_parallel_world_size()} "
        f"dp {get_data_parallel_rank()}/{get_data_parallel_world_size()} | "
        f"tp_ranks={_TP_GLOBAL_RANKS} pp_ranks={_PP_GLOBAL_RANKS} dp_ranks={_DP_GLOBAL_RANKS}"
    )
