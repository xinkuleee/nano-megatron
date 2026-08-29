"""Training entry point.

Deliberately thin. Everything interesting happens in the pipeline and layers;
this file wires a torchrun process to a token stream, one model
stage, an optimizer, and rank-local checkpoints.

    torchrun --nproc-per-node=4 -m nano_megatron.train --tp 2 --pp 2
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import load_checkpoint, save_checkpoint
from .data import TokenStream
from .grads import clip_grad_norm, finalize_gradients, synchronize_model_parameters
from .model import GPT, GPTConfig
from .parallel import init_parallel
from .pipeline import forward_backward_1f1b


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


def train(args):
    if (args.backend == "nccl") != (args.device == "cuda"):
        raise ValueError("use gloo with cpu or nccl with cuda")
    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = args.tp * args.pp * args.dp
    if world_size != expected_world_size:
        raise ValueError(
            f"torchrun world size {world_size} != tp * pp * dp = {expected_world_size}"
        )
    ctx = init_parallel(tp_size=args.tp, pp_size=args.pp)
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        tie_word_embeddings=True,
        sequence_parallel=args.sequence_parallel,
        recompute=args.recompute,
        compute_dtype=args.dtype,
    )

    # Different PP stages and TP shards should not receive identical random
    # tensors; DP replicas of the same shard should. Replicated parameters are
    # explicitly broadcast below.
    model_seed = args.seed + args.tp * ctx.pp_rank + ctx.tp_rank
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    model = GPT(config, ctx).to(device)
    synchronize_model_parameters(
        model, ctx, tie_embeddings=config.tie_word_embeddings
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    stream = (
        TokenStream(
            args.data,
            dtype=args.data_dtype,
            seq_len=config.seq_len,
            dp_rank=ctx.dp_rank,
            dp_size=ctx.dp_size,
        )
        if args.data
        else None
    )
    training_state = {
        "lr": args.lr,
        "clip": args.clip,
        "micro_batch_size": args.micro_batch_size,
        "num_microbatches": args.num_microbatches,
    }
    loaded = (
        load_checkpoint(args.load, model, optimizer, ctx, data=stream, map_location=device)
        if args.load
        else (0, {})
    )
    start_step, loaded_extra = loaded
    if loaded_extra and loaded_extra != training_state:
        raise RuntimeError(
            f"checkpoint training configuration mismatch: saved={loaded_extra}, "
            f"current={training_state}"
        )

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"{ctx.describe()}\nlocal params: {total/1e6:.2f}M  schedule: 1f1b\n")

    for step in range(start_step, args.steps):
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        if stream is None:
            batches = synthetic_batches(
                config,
                args.micro_batch_size,
                args.num_microbatches,
                step,
                ctx.dp_rank,
            )
        else:
            batches = [
                stream.next_batch(args.micro_batch_size)
                for _ in range(args.num_microbatches)
            ]
        batches = [(x.to(device), y.to(device)) for x, y in batches]

        losses = forward_backward_1f1b(model, batches, config, device, ctx)
        finalize_gradients(
            model,
            ctx,
            tie_embeddings=config.tie_word_embeddings,
            sequence_parallel=config.sequence_parallel,
        )

        clip_grad_norm(
            model, ctx, args.clip, tie_embeddings=config.tie_word_embeddings
        )
        optimizer.step()
        elapsed = time.perf_counter() - start_time

        # Only the last stage computes a loss.
        if ctx.is_last_stage and ctx.tp_rank == 0:
            loss = torch.stack(losses).sum()
            if ctx.dp_size > 1:
                dist.all_reduce(loss, group=ctx.dp_group)
                loss /= ctx.dp_size
            tokens = args.micro_batch_size * args.num_microbatches * config.seq_len
            if ctx.dp_rank == 0:
                print(
                    f"step {step:3d} | loss {loss.item():.4f} | {elapsed*1e3:6.1f} ms | "
                    f"{tokens * ctx.dp_size/elapsed:8.0f} tok/s",
                    flush=True,
                )

        next_step = step + 1
        should_save = args.save and (
            next_step == args.steps
            or (args.save_every > 0 and next_step % args.save_every == 0)
        )
        if should_save:
            checkpoint_dir = Path(args.save) / f"step_{next_step:08d}"
            save_checkpoint(
                checkpoint_dir, model, optimizer, ctx, step=next_step, data=stream,
                extra=training_state,
            )

    dist.barrier()
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description="Train a small GPT with TP + PP.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--data", help="raw pre-tokenized uint16/uint32 file")
    parser.add_argument("--data-dtype", choices=["uint16", "uint32"], default="uint16")
    parser.add_argument("--save", help="directory for rank-local checkpoint steps")
    parser.add_argument("--save-every", type=int, default=0, help="0 saves only the final step")
    parser.add_argument("--load", help="exact checkpoint step directory to resume")
    return parser


def main():
    args = build_parser().parse_args()
    if args.steps < 0 or args.save_every < 0:
        raise ValueError("steps and save-every must be non-negative")
    if args.num_microbatches <= 0 or args.micro_batch_size <= 0:
        raise ValueError("num-microbatches and micro-batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
