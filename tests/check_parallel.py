"""Correctness harness: every parallel config must match a single-process run.

Processes are spawned directly with a file-backed rendezvous rather than going
through torchrun, so this runs anywhere -- including a laptop with no GPU and no
open ports.

    python -m tests.check_parallel --tp 2 --pp 2
    python -m tests.check_parallel --matrix
"""

from __future__ import annotations

import argparse
import os
import queue
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from nano_megatron.grads import (
    clip_grad_norm,
    finalize_gradients,
    synchronize_model_parameters,
)
from nano_megatron.model import GPT, GPTConfig
from nano_megatron.parallel import init_parallel
from nano_megatron.pipeline import forward_backward_1f1b
from tests.reference import (
    build_parallel_model,
    build_reference_model,
    gather_full_gradient,
    gather_full_parameter,
)

TOLERANCE = dict(rtol=1e-4, atol=1e-5)
pytestmark = pytest.mark.distributed


def make_batches(config, micro_bs, num_microbatches, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randint(0, config.vocab_size, (micro_bs, config.seq_len), generator=generator),
            torch.randint(0, config.vocab_size, (micro_bs, config.seq_len), generator=generator),
        )
        for _ in range(num_microbatches)
    ]


def run_reference(config, batches, loss_scale=1.0):
    """Single-process forward/backward over the same microbatches."""
    reference = build_reference_model(config)
    total = 0.0
    for inputs, labels in batches:
        loss = reference(inputs, labels) * loss_scale / len(batches)
        loss.backward()
        total = total + loss.detach()
    return reference, total


def reference_name_for(name, layer_offset, tied_across_pipeline):
    if name.startswith("layers."):
        _, index, rest = name.split(".", 2)
        return f"layers.{int(index) + layer_offset}.{rest}"
    if tied_across_pipeline and name == "lm_head.weight":
        return "embedding.weight"
    return name


def tensors_equal_across_group(tensor, group) -> bool:
    copies = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(copies, tensor.contiguous(), group=group)
    return all(torch.equal(copies[0], other) for other in copies[1:])


def check_parameter_initialization(config, rank, failures, ctx):
    """Exercise the rank-aware initialization path used by ``train``."""
    torch.manual_seed(10_000 + rank)
    initialized = GPT(config, ctx)
    synchronize_model_parameters(
        initialized, ctx, tie_embeddings=config.tie_word_embeddings
    )

    distinct_tp_shard = False
    for name, param in initialized.named_parameters():
        if ctx.dp_size > 1 and not tensors_equal_across_group(
            param.data, ctx.dp_group
        ):
            failures.append(f"init {name}: DP replicas differ after synchronization")

        if ctx.tp_size > 1:
            equal = tensors_equal_across_group(param.data, ctx.tp_group)
            if getattr(param, "tensor_parallel_sharded", False):
                distinct_tp_shard = distinct_tp_shard or not equal
            elif not equal:
                failures.append(f"init {name}: TP-replicated values differ")

    if ctx.tp_size > 1 and not distinct_tp_shard:
        failures.append("init: every TP shard is identical")

    if ctx.pp_size > 1 and (ctx.is_first_stage or ctx.is_last_stage):
        tied = initialized.embedding.weight if ctx.is_first_stage else initialized.lm_head.weight
        if not tensors_equal_across_group(tied.data, ctx.tied_group):
            failures.append("init: tied embedding copies differ across pipeline stages")


def check_one_rank(rank, world_size, init_file, args, result_queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    ctx = init_parallel(tp_size=args.tp, pp_size=args.pp)
    device = torch.device("cpu")

    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        # Exercise the real training configuration: split pipelines keep a
        # copy on their first and last stages and explicitly sum its gradient.
        tie_word_embeddings=True,
        sequence_parallel=args.sequence_parallel,
        recompute=args.recompute,
    )

    # Every model-parallel rank in one DP replica sees the same microbatches,
    # while different DP replicas see deliberately different data.  If the DP
    # all-reduce is missing, grouped incorrectly, or not averaged, the gradient
    # comparison below fails instead of accidentally passing on identical data.
    dp_rank = ctx.dp_rank
    batches = make_batches(
        config, args.micro_batch_size, args.num_microbatches, seed=1234 + dp_rank
    )
    reference_batches = [
        batch
        for replica in range(args.dp)
        for batch in make_batches(
            config, args.micro_batch_size, args.num_microbatches, seed=1234 + replica
        )
    ]
    reference, reference_loss = run_reference(config, reference_batches)

    model = build_parallel_model(config, reference, ctx)

    activation_shapes = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: activation_shapes.append(tuple(output.shape))
        )
        for layer in model.layers
    ]
    losses = forward_backward_1f1b(model, batches, config, device, ctx)
    for handle in handles:
        handle.remove()
    finalize_gradients(
        model,
        ctx,
        tie_embeddings=config.tie_word_embeddings,
        sequence_parallel=config.sequence_parallel,
    )

    failures = []
    expected_sequence = config.seq_len // (args.tp if config.sequence_parallel else 1)
    if any(shape[1] != expected_sequence for shape in activation_shapes):
        failures.append(
            f"activation sequence shapes {[shape[1] for shape in activation_shapes]} "
            f"!= expected {expected_sequence}"
        )

    # --- loss, checked on the stage that computes it ---
    if ctx.is_last_stage:
        parallel_loss = torch.stack(losses).sum()
        # Loss is local to a DP replica. Gradients, unlike the diagnostic loss,
        # are averaged across replicas by ``finalize_gradients``.
        _, local_reference_loss = run_reference(config, batches)
        if not torch.allclose(parallel_loss, local_reference_loss, **TOLERANCE):
            failures.append(
                f"loss: parallel={parallel_loss.item():.6f} "
                f"reference={local_reference_loss.item():.6f}"
            )

    # --- gradients, reassembled from their shards ---
    reference_parameters = dict(reference.named_parameters())
    reference_grads = {name: p.grad for name, p in reference_parameters.items()}
    layer_offset = ctx.pp_rank * (config.n_layer // args.pp)

    for name, _ in model.named_parameters():
        full = gather_full_gradient(name, model, config)
        if full is None:
            failures.append(f"no gradient produced for {name}")
            continue

        reference_name = reference_name_for(name, layer_offset, args.pp > 1)

        expected = reference_grads.get(reference_name)
        if expected is None:
            failures.append(f"no reference gradient for {reference_name}")
        elif not torch.allclose(full, expected, **TOLERANCE):
            error = (full - expected).abs().max().item()
            failures.append(f"grad {reference_name}: max_abs_err={error:.3e}")

    # Compare the norm and clipped gradients against the logical full model.
    reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), args.clip)
    parallel_norm = clip_grad_norm(model, ctx, args.clip, tie_embeddings=True)
    if not torch.allclose(parallel_norm.float(), reference_norm.float(), **TOLERANCE):
        failures.append(
            f"grad norm: parallel={parallel_norm.item():.6f} "
            f"reference={reference_norm.item():.6f}"
        )

    for name, _ in model.named_parameters():
        full = gather_full_gradient(name, model, config)
        reference_name = reference_name_for(name, layer_offset, args.pp > 1)
        expected = reference_grads.get(reference_name)
        if full is not None and expected is not None and not torch.allclose(full, expected, **TOLERANCE):
            error = (full - expected).abs().max().item()
            failures.append(f"clipped grad {reference_name}: max_abs_err={error:.3e}")

    # A real step catches synchronized-initial-value and tied-weight defects.
    # Use a linear update here. Adam's first step is close to ``sign(grad)``
    # for small gradients, so harmless collective round-off near zero can turn
    # into an O(lr) parameter difference and obscure the synchronization defect
    # this assertion is meant to catch.
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    parallel_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    reference_optimizer.step()
    parallel_optimizer.step()

    for name, _ in model.named_parameters():
        full = gather_full_parameter(name, model)
        reference_name = reference_name_for(name, layer_offset, args.pp > 1)
        expected = reference_parameters.get(reference_name)
        if full is not None and expected is not None and not torch.allclose(
            full, expected.data, **TOLERANCE
        ):
            error = (full - expected.data).abs().max().item()
            failures.append(f"updated param {reference_name}: max_abs_err={error:.3e}")

    check_parameter_initialization(config, rank, failures, ctx)

    result_queue.put((rank, ctx.describe(), failures))

    dist.destroy_process_group()


def run_config(args, verbose=True) -> bool:
    world_size = args.tp * args.pp * args.dp
    context = mp.get_context("spawn")
    result_queue = context.Queue()

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "rendezvous")
        processes = []
        for rank in range(world_size):
            process = context.Process(
                target=check_one_rank,
                args=(rank, world_size, init_file, args, result_queue),
            )
            process.start()
            processes.append(process)

        results = []
        deadline = time.monotonic() + args.timeout
        while len(results) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                results.append(result_queue.get(timeout=min(0.2, remaining)))
            except queue.Empty:
                # A worker that exits before writing a result used to leave this
                # harness blocked in Queue.get forever.
                if all(not process.is_alive() for process in processes) or any(
                    process.exitcode not in (None, 0) for process in processes
                ):
                    break

        for process in processes:
            process.join(timeout=1)
        timed_out = [process for process in processes if process.is_alive()]
        for process in timed_out:
            process.terminate()
            process.join()

        crashed = [p.exitcode for p in processes if p.exitcode != 0]
        missing_results = world_size - len(results)

    results.sort()
    total_failures = sum(len(failures) for _, _, failures in results)

    if verbose:
        for _, description, failures in results:
            print(f"  {description}")
            for failure in failures:
                print(f"      FAIL {failure}")

    label = (
        f"tp={args.tp} pp={args.pp} dp={args.dp} world={world_size}"
    )
    ok = total_failures == 0 and not crashed and missing_results == 0
    detail = (
        ""
        if ok
        else f"  ({total_failures} mismatches, {len(crashed)} crashes, "
        f"{missing_results} missing results)"
    )
    print(f"{'PASS' if ok else 'FAIL'}  {label}{detail}", flush=True)
    return ok


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--matrix", action="store_true", help="sweep all configs")
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="seconds allowed per configuration"
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not args.matrix:
        raise SystemExit(0 if run_config(args) else 1)

    # (tp, pp, dp, microbatches, sequence_parallel, recompute)
    configs = [
        (1, 1, 1, 4, False, False),
        (2, 1, 1, 4, False, False),
        (4, 1, 1, 4, False, False),
        (1, 2, 1, 4, False, False),
        (2, 2, 1, 4, False, False),
        (1, 4, 1, 4, False, False),
        (1, 4, 1, 1, False, False),
        (1, 1, 2, 4, False, False),
        (2, 1, 2, 4, False, False),
        (1, 2, 2, 4, False, False),
        (2, 2, 2, 2, False, False),
        (2, 1, 1, 4, True, False),
        (4, 1, 1, 4, True, False),
        (2, 2, 1, 4, True, False),
        (2, 2, 2, 2, True, False),
        (2, 2, 1, 2, True, True),
    ]
    results = []
    for tp, pp, dp, microbatches, sequence_parallel, recompute in configs:
        args.tp, args.pp, args.dp = tp, pp, dp
        args.num_microbatches = microbatches
        args.sequence_parallel = sequence_parallel
        args.recompute = recompute
        results.append(run_config(args, verbose=False))

    print(f"\n{sum(results)}/{len(results)} configurations passed")
    raise SystemExit(0 if all(results) else 1)


def test_distributed_matrix():
    args = build_parser().parse_args(["--matrix", "--n-embd=32", "--vocab-size=64", "--seq-len=8", "--micro-batch-size=1"] )
    configs = [
        (2, 1, 1, False, False),
        (2, 2, 1, True, False),
        (2, 2, 2, True, True),
    ]
    for tp, pp, dp, sequence_parallel, recompute in configs:
        args.tp, args.pp, args.dp = tp, pp, dp
        args.num_microbatches = 2
        args.sequence_parallel = sequence_parallel
        args.recompute = recompute
        assert run_config(args, verbose=False)


if __name__ == "__main__":
    main()
