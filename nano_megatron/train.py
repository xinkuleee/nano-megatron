"""Training entry point.

Deliberately thin. Everything interesting happens in the schedule function and
in the layers; this file just builds a model, feeds it synthetic data, and
steps the optimizer, so that the loop reads the same whether you are running on
one process or sixteen.

    python -m nano_megatron.train --tp 2 --pp 2      # spawns its own processes
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import parallel_state as ps
from .grads import clip_grad_norm, finalize_gradients, synchronize_model_parameters
from .model import GPT, GPTConfig
from .schedules import get_forward_backward_func
from .tp_layers import set_tensor_parallel_attributes


def synthetic_batches(config, micro_bs, num_microbatches, step, data_parallel_rank=0):
    # Each DP replica consumes a different deterministic batch. Pipeline stages
    # in the same DP lane share ``data_parallel_rank`` and therefore agree.
    generator = torch.Generator().manual_seed(1234 + 100_003 * step + data_parallel_rank)
    return [
        (
            torch.randint(0, config.vocab_size, (micro_bs, config.seq_len), generator=generator),
            torch.randint(0, config.vocab_size, (micro_bs, config.seq_len), generator=generator),
        )
        for _ in range(num_microbatches)
    ]


def train_worker(rank, world_size, init_file, args):
    dist.init_process_group(
        backend=args.backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    ps.initialize_model_parallel(tp_size=args.tp, pp_size=args.pp)
    device = torch.device(args.device)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        tie_word_embeddings=True,
    )

    # Different PP stages and TP shards should not receive identical random
    # tensors; DP replicas of the same shard should. Replicated parameters are
    # explicitly broadcast below.
    model_seed = args.seed + args.tp * ps.get_pipeline_parallel_rank() + ps.get_tensor_parallel_rank()
    torch.manual_seed(model_seed)
    model = GPT(config).to(device)
    set_tensor_parallel_attributes(model)
    synchronize_model_parameters(model, tie_embeddings=config.tie_word_embeddings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    forward_backward = get_forward_backward_func(args.schedule)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"{ps.describe()}\nlocal params: {total/1e6:.2f}M  schedule: {args.schedule}\n")

    for step in range(args.steps):
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        batches = synthetic_batches(
            config,
            args.micro_batch_size,
            args.num_microbatches,
            step,
            ps.get_data_parallel_rank(),
        )
        batches = [(x.to(device), y.to(device)) for x, y in batches]

        losses = forward_backward(model, batches, config, device)
        finalize_gradients(model, tie_embeddings=config.tie_word_embeddings)

        clip_grad_norm(model, args.clip, tie_embeddings=config.tie_word_embeddings)
        optimizer.step()
        elapsed = time.perf_counter() - start_time

        # Only the last stage computes a loss.
        if ps.is_pipeline_last_stage() and ps.get_tensor_parallel_rank() == 0:
            loss = torch.stack(losses).sum()
            if ps.get_data_parallel_world_size() > 1:
                dist.all_reduce(loss, group=ps.get_data_parallel_group())
                loss /= ps.get_data_parallel_world_size()
            tokens = args.micro_batch_size * args.num_microbatches * config.seq_len
            if ps.get_data_parallel_rank() == 0:
                print(
                    f"step {step:3d} | loss {loss.item():.4f} | {elapsed*1e3:6.1f} ms | "
                    f"{tokens * ps.get_data_parallel_world_size()/elapsed:8.0f} tok/s",
                    flush=True,
                )

    dist.barrier()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description="Train a small GPT with TP + PP.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--schedule", choices=["gpipe", "1f1b"], default="1f1b")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    return parser


def main():
    args = build_parser().parse_args()
    world_size = args.tp * args.pp * args.dp

    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "rendezvous")
        processes = [
            context.Process(target=train_worker, args=(rank, world_size, init_file, args))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise SystemExit(f"{len(failed)} worker(s) failed: {failed}")


if __name__ == "__main__":
    main()
