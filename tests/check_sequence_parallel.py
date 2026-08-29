"""Real-process checks for sequence-parallel collectives and model wiring."""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from nano_megatron.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from nano_megatron.parallel import init_parallel

pytestmark = pytest.mark.distributed


def _worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    ctx = init_parallel(tp_size=world_size)

    full = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    shard = torch.chunk(full, world_size, dim=1)[rank].clone().requires_grad_(True)
    gathered = gather_from_sequence_parallel_region(shard, ctx)
    torch.testing.assert_close(gathered, full)
    gathered.square().sum().backward()
    expected_grad = torch.chunk(2 * full, world_size, dim=1)[rank] * world_size
    torch.testing.assert_close(shard.grad, expected_grad)

    partial = (full / world_size).requires_grad_(True)
    reduced = reduce_scatter_to_sequence_parallel_region(partial, ctx)
    torch.testing.assert_close(reduced, torch.chunk(full, world_size, dim=1)[rank])
    reduced.sum().backward()
    torch.testing.assert_close(partial.grad, torch.ones_like(full))

    queue.put(rank)
    dist.destroy_process_group()


def _run_world_size(world_size):
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "rendezvous")
        processes = [
            context.Process(target=_worker, args=(rank, world_size, init_file, queue))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert sorted(results) == list(range(world_size))
        assert all(process.exitcode == 0 for process in processes)


def test_sequence_collectives():
    for world_size in (2, 4):
        _run_world_size(world_size)


def main():
    for world_size in (2, 4):
        _run_world_size(world_size)
        print(f"PASS  sequence collectives tp={world_size}")


if __name__ == "__main__":
    main()
