"""Orthogonal tensor, pipeline, and data-parallel process groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ParallelContext:
    """All topology state needed by one rank, with no hidden globals."""

    rank: int
    world_size: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    tp_group: dist.ProcessGroup | None = None
    pp_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None
    tied_group: dist.ProcessGroup | None = None
    tp_ranks: tuple[int, ...] = (0,)
    pp_ranks: tuple[int, ...] = (0,)
    dp_ranks: tuple[int, ...] = (0,)

    @classmethod
    def single(cls) -> "ParallelContext":
        return cls(
            rank=0,
            world_size=1,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
        )

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def previous_pipeline_rank(self) -> int:
        if self.is_first_stage:
            raise RuntimeError("the first pipeline stage has no predecessor")
        return self.pp_ranks[self.pp_rank - 1]

    @property
    def next_pipeline_rank(self) -> int:
        if self.is_last_stage:
            raise RuntimeError("the last pipeline stage has no successor")
        return self.pp_ranks[self.pp_rank + 1]

    def describe(self) -> str:
        return (
            f"rank {self.rank:2d}/{self.world_size} | "
            f"tp {self.tp_rank}/{self.tp_size} "
            f"pp {self.pp_rank}/{self.pp_size} "
            f"dp {self.dp_rank}/{self.dp_size} | "
            f"tp_ranks={list(self.tp_ranks)} pp_ranks={list(self.pp_ranks)} "
            f"dp_ranks={list(self.dp_ranks)}"
        )


def init_parallel(tp_size: int = 1, pp_size: int = 1) -> ParallelContext:
    """Create every subgroup in deterministic order and return this rank's view."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before init_parallel")
    if tp_size <= 0 or pp_size <= 0:
        raise ValueError("tp_size and pp_size must be positive")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % (tp_size * pp_size):
        raise ValueError(
            f"world size {world_size} is not divisible by tp_size * pp_size "
            f"= {tp_size * pp_size}"
        )
    dp_size = world_size // (tp_size * pp_size)
    grid = torch.arange(world_size).reshape(pp_size, dp_size, tp_size)

    tp_group = pp_group = dp_group = tied_group = None
    tp_ranks: tuple[int, ...] = ()
    pp_ranks: tuple[int, ...] = ()
    dp_ranks: tuple[int, ...] = ()

    for pp in range(pp_size):
        for dp in range(dp_size):
            ranks = tuple(grid[pp, dp, :].tolist())
            group = dist.new_group(list(ranks))
            if rank in ranks:
                tp_group, tp_ranks = group, ranks

    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = tuple(grid[pp, :, tp].tolist())
            group = dist.new_group(list(ranks))
            if rank in ranks:
                dp_group, dp_ranks = group, ranks

    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = tuple(grid[:, dp, tp].tolist())
            group = dist.new_group(list(ranks))
            if rank in ranks:
                pp_group, pp_ranks = group, ranks

            if pp_size > 1:
                endpoints = (ranks[0], ranks[-1])
                group = dist.new_group(list(endpoints))
                if rank in endpoints:
                    tied_group = group

    tp_rank = tp_ranks.index(rank)
    pp_rank = pp_ranks.index(rank)
    dp_rank = dp_ranks.index(rank)
    return ParallelContext(
        rank=rank,
        world_size=world_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        tp_group=tp_group,
        pp_group=pp_group,
        dp_group=dp_group,
        tied_group=tied_group,
        tp_ranks=tp_ranks,
        pp_ranks=pp_ranks,
        dp_ranks=dp_ranks,
    )
