"""Socket-free checks for logical-model gradient clipping.

The real implementation reduces squared norms over TP and PP groups.  These
checks replace the collectives with deterministic peer contributions, so they
exercise the accounting and clipping math without requiring gloo or a GPU.

    python -m tests.check_grad_clip
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from nano_megatron import grads
from nano_megatron.parallel import ParallelContext

TOLERANCE = dict(rtol=1e-6, atol=1e-6)


class _Weight(nn.Module):
    def __init__(self, gradient, *, sharded: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros_like(torch.as_tensor(gradient, dtype=torch.float32)))
        self.weight.grad = torch.as_tensor(gradient, dtype=torch.float32).clone()
        self.weight.tensor_parallel_sharded = sharded


class _Stage(nn.Module):
    def __init__(self, pp_rank: int, tp_rank: int):
        super().__init__()
        shard_gradients = {(0, 0): [3.0], (0, 1): [4.0], (1, 0): [5.0], (1, 1): [12.0]}
        replicated_gradients = {0: [2.0], 1: [6.0]}
        embedding_gradients = {0: [1.0, 2.0], 1: [2.0, 1.0]}

        self.shard = _Weight(shard_gradients[pp_rank, tp_rank], sharded=True)
        self.replicated = _Weight(replicated_gradients[pp_rank], sharded=False)
        if pp_rank == 0:
            self.embedding = _Weight(embedding_gradients[tp_rank], sharded=True)
        else:
            # After tied-gradient synchronization this is the same logical
            # gradient as the first-stage embedding, so it must be scaled but
            # not counted a second time in the norm.
            self.lm_head = _Weight(embedding_gradients[tp_rank], sharded=True)


@contextlib.contextmanager
def _parallel_context(
    *,
    tp_size: int,
    pp_size: int,
    pp_rank: int,
    tp_peer_contribution: float = 0.0,
    pp_peer_contribution: float = 0.0,
):
    """Substitute peer sums for TP/PP collectives on one simulated rank."""
    tp_group, pp_group, dp_group = object(), object(), object()
    calls = []
    ctx = ParallelContext(
        rank=pp_rank * tp_size,
        world_size=tp_size * pp_size * 2,
        tp_size=tp_size,
        tp_rank=0,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=2,
        dp_rank=0,
        tp_group=tp_group,
        pp_group=pp_group,
        dp_group=dp_group,
        tp_ranks=tuple(range(tp_size)),
        pp_ranks=tuple(i * tp_size for i in range(pp_size)),
        dp_ranks=(0, tp_size),
    )
    original_all_reduce = grads.dist.all_reduce

    def fake_all_reduce(tensor, group=None):
        if group is tp_group:
            calls.append("tp")
            tensor.add_(tp_peer_contribution)
        elif group is pp_group:
            calls.append("pp")
            tensor.add_(pp_peer_contribution)
        elif group is dp_group:
            raise AssertionError("gradient norm must not be reduced over DP replicas")
        else:
            raise AssertionError("unexpected process group")

    grads.dist.all_reduce = fake_all_reduce
    try:
        yield calls, ctx
    finally:
        grads.dist.all_reduce = original_all_reduce


def check_tp_pp_dp_tied_norm_and_scaling():
    """TP shards sum, TP/DP replicas count once, PP stages sum, tie counts once."""
    # Intended per-rank contributions after dividing TP-replicated parameters
    # by two.  The last-stage tied LM-head copy is intentionally absent.
    local_squared = {(0, 0): 16.0, (0, 1): 23.0, (1, 0): 43.0, (1, 1): 162.0}
    stage_squared = {0: 39.0, 1: 205.0}
    expected_norm = torch.tensor(244.0, dtype=torch.float64).sqrt()
    max_norm = expected_norm.item() / 2.0
    expected_scale = max_norm / (expected_norm.item() + 1e-6)

    for pp_rank in (0, 1):
        for tp_rank in (0, 1):
            model = _Stage(pp_rank, tp_rank)
            original = {name: param.grad.clone() for name, param in model.named_parameters()}
            with _parallel_context(
                tp_size=2,
                pp_size=2,
                pp_rank=pp_rank,
                tp_peer_contribution=local_squared[pp_rank, 1 - tp_rank],
                pp_peer_contribution=stage_squared[1 - pp_rank],
            ) as (calls, ctx):
                norm = grads.clip_grad_norm(
                    model, ctx, max_norm, tie_embeddings=True
                )

            assert calls == ["tp", "pp"], f"rank {(pp_rank, tp_rank)} collectives: {calls}"
            assert torch.allclose(norm, expected_norm, **TOLERANCE), (
                f"rank {(pp_rank, tp_rank)} norm={norm.item():.8f}, "
                f"expected={expected_norm.item():.8f}"
            )
            for name, param in model.named_parameters():
                expected = original[name] * expected_scale
                assert torch.allclose(param.grad, expected, **TOLERANCE), (
                    f"rank {(pp_rank, tp_rank)} did not scale {name} uniformly"
                )

    return f"logical TP=2, PP=2, DP=2 norm={expected_norm.item():.6f}"


def check_tied_lm_head_counting_is_conditional():
    """An LM head is skipped only when it really represents a tied PP copy."""
    model = nn.Module()
    model.lm_head = _Weight([3.0], sharded=True)

    with _parallel_context(
        tp_size=1, pp_size=2, pp_rank=1, pp_peer_contribution=16.0
    ) as (_, ctx):
        untied_norm = grads.clip_grad_norm(
            model, ctx, 100.0, tie_embeddings=False
        )
    assert torch.allclose(untied_norm, torch.tensor(5.0, dtype=torch.float64), **TOLERANCE)

    model.lm_head.weight.grad.fill_(3.0)
    with _parallel_context(
        tp_size=1, pp_size=2, pp_rank=1, pp_peer_contribution=16.0
    ) as (_, ctx):
        tied_norm = grads.clip_grad_norm(model, ctx, 100.0, tie_embeddings=True)
    assert torch.allclose(tied_norm, torch.tensor(4.0, dtype=torch.float64), **TOLERANCE)
    return "untied LM head counted; tied last-stage copy deduplicated"


def check_limits_and_empty_gradients():
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2))
    model.weight.grad = torch.tensor([3.0, 4.0])
    model.weight.tensor_parallel_sharded = False

    with _parallel_context(tp_size=1, pp_size=1, pp_rank=0) as (_, ctx):
        norm = grads.clip_grad_norm(model, ctx, 100.0)
    assert norm.item() == 5.0
    assert torch.equal(model.weight.grad, torch.tensor([3.0, 4.0]))

    model.weight.grad = torch.tensor([3.0, 4.0])
    with _parallel_context(tp_size=1, pp_size=1, pp_rank=0) as (_, ctx):
        norm = grads.clip_grad_norm(model, ctx, 0.0)
    assert norm.item() == 5.0
    assert torch.equal(model.weight.grad, torch.zeros(2))

    model.weight.grad = None
    with _parallel_context(tp_size=1, pp_size=1, pp_rank=0) as (_, ctx):
        assert grads.clip_grad_norm(model, ctx, 1.0).item() == 0.0

    try:
        grads.clip_grad_norm(model, ParallelContext.single(), -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative max_norm must raise ValueError")
    return "large/zero/negative limits and empty gradients handled"


CHECKS = [
    check_tp_pp_dp_tied_norm_and_scaling,
    check_tied_lm_head_counting_is_conditional,
    check_limits_and_empty_gradients,
]


def test_gradient_clipping_checks():
    for check in CHECKS:
        check()


def main():
    failures = 0
    for check in CHECKS:
        try:
            print(f"  ok    {check.__name__}: {check()}")
        except (AssertionError, RuntimeError, ValueError) as error:
            failures += 1
            print(f"  FAIL  {check.__name__}: {error}")

    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} gradient clipping checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
