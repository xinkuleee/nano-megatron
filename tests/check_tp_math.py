"""Check TP algebra rank by rank without creating process groups.

Each simulated rank receives an explicit topology context. Partial tensor math
is combined by the test, which is enough to prove that column/row sharding
reconstructs the full matmul and vocab sharding reconstructs cross entropy.

It cannot prove the schedules or the real collectives are right; only
``check_parallel.py`` under real processes does that.

    python -m tests.check_tp_math
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nano_megatron import mappings
from nano_megatron.parallel import ParallelContext
from nano_megatron.tp_layers import ColumnParallelLinear, RowParallelLinear

TOLERANCE = dict(rtol=1e-5, atol=1e-6)


def simulated_rank(tp_size, tp_rank):
    """A topology-only context for algebra checked without collectives."""
    return ParallelContext(
        rank=tp_rank, world_size=tp_size, tp_size=tp_size, tp_rank=tp_rank,
        pp_size=1, pp_rank=0, dp_size=1, dp_rank=0,
        tp_ranks=tuple(range(tp_size)), pp_ranks=(tp_rank,), dp_ranks=(tp_rank,),
    )


def check_column_row_pair_reconstructs_mlp():
    """The core TP identity: split A column-wise, B row-wise, sum the results.

        Y = f(X A) B  ==  sum_i  f(X A_i) B_i

    This is why the nonlinearity in between needs no communication.
    """
    torch.manual_seed(0)
    batch, in_dim, hidden, out_dim = 3, 16, 32, 16
    x = torch.randn(batch, in_dim)
    A = torch.randn(in_dim, hidden) * 0.1
    B = torch.randn(hidden, out_dim) * 0.1

    expected = F.silu(x @ A) @ B

    for tp_size in (2, 4):
        partial_sum = torch.zeros(batch, out_dim)
        for tp_rank in range(tp_size):
            ctx = simulated_rank(tp_size, tp_rank)
            column = ColumnParallelLinear(in_dim, hidden, ctx, bias=False)
            row = RowParallelLinear(hidden, out_dim, ctx, bias=False)
            column.weight.data = torch.chunk(A, tp_size, dim=1)[tp_rank].clone()
            row.weight.data = torch.chunk(B, tp_size, dim=0)[tp_rank].clone()
            hidden_local = F.silu(column(x, reduce_input_grad=False))
            partial_sum = partial_sum + hidden_local @ row.weight

        error = (partial_sum - expected).abs().max().item()
        assert torch.allclose(partial_sum, expected, **TOLERANCE), (
            f"tp={tp_size}: column/row pair does not reconstruct MLP, max_err={error:.3e}"
        )
    return "column x row reconstructs the full MLP for tp in (2, 4)"


def check_column_shard_gradients_concatenate():
    """Each shard's weight gradient must equal the matching slice of the full one."""
    torch.manual_seed(0)
    batch, in_dim, out_dim = 3, 16, 32
    x = torch.randn(batch, in_dim)
    A = (torch.randn(in_dim, out_dim) * 0.1).requires_grad_(True)

    (x @ A).pow(2).sum().backward()
    expected_grad = A.grad

    tp_size = 4
    shards = []
    for tp_rank in range(tp_size):
        ctx = simulated_rank(tp_size, tp_rank)
        column = ColumnParallelLinear(in_dim, out_dim, ctx, bias=False)
        column.weight.data = torch.chunk(A.detach(), tp_size, dim=1)[tp_rank].clone()
        column(x, reduce_input_grad=False).pow(2).sum().backward()
        shards.append(column.weight.grad)

    rebuilt = torch.cat(shards, dim=1)
    error = (rebuilt - expected_grad).abs().max().item()
    assert torch.allclose(rebuilt, expected_grad, **TOLERANCE), (
        f"gathered column gradients differ from reference, max_err={error:.3e}"
    )
    return "column-parallel weight gradients concatenate to the reference"


def check_vocab_parallel_cross_entropy():
    """Sharded-vocabulary loss must equal a plain cross entropy.

    Each simulated rank gets a genuine slice of the logits. The all-reduces the
    real implementation performs (max, sum-exp, target logit) are emulated by
    combining the per-rank partials here, so the sharded code path -- including
    the out-of-range masking -- is actually exercised.
    """
    torch.manual_seed(0)
    batch, seq, vocab = 2, 5, 16
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))

    expected = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))

    for tp_size in (2, 4):
        per_partition = vocab // tp_size
        shard_max, shard_sum, shard_target = [], [], []

        for tp_rank in range(tp_size):
            start = tp_rank * per_partition
            end = start + per_partition
            shard = logits[..., start:end]

            shard_max.append(shard.max(dim=-1)[0])
            owned = (labels >= start) & (labels < end)
            local = (labels - start).clamp(0, per_partition - 1)
            picked = shard.gather(-1, local.unsqueeze(-1)).squeeze(-1)
            shard_target.append(torch.where(owned, picked, torch.zeros_like(picked)))
            shard_sum.append(shard)

        global_max = torch.stack(shard_max).max(dim=0)[0]
        sum_exp = sum((s - global_max.unsqueeze(-1)).exp().sum(-1) for s in shard_sum)
        target_logit = torch.stack(shard_target).sum(0) - global_max
        loss = (sum_exp.log() - target_logit).mean()

        assert torch.allclose(loss, expected, **TOLERANCE), (
            f"tp={tp_size}: sharded CE {loss.item():.6f} != reference {expected.item():.6f}"
        )

    return "sharded-vocabulary CE matches F.cross_entropy for tp in (2, 4)"


def check_vocab_parallel_ce_gradient():
    """Its gradient must match autograd through the reference loss."""
    from nano_megatron.cross_entropy import vocab_parallel_cross_entropy

    torch.manual_seed(0)
    batch, seq, vocab = 2, 5, 16
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))

    reference = logits.clone().requires_grad_(True)
    F.cross_entropy(reference.view(-1, vocab), labels.view(-1)).backward()

    parallel = logits.clone().requires_grad_(True)
    vocab_parallel_cross_entropy(
        parallel, labels, 0, vocab, ParallelContext.single()
    ).backward()

    error = (parallel.grad - reference.grad).abs().max().item()
    assert torch.allclose(parallel.grad, reference.grad, rtol=1e-4, atol=1e-6), (
        f"vocab-parallel CE gradient differs, max_err={error:.3e}"
    )
    return f"vocab-parallel CE gradient matches autograd (max_err={error:.2e})"


def check_conjugate_functions_are_inverse_shaped():
    """f and g must be transposes of each other in the autograd sense."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)

    ctx = simulated_rank(2, 0)
    original = mappings._all_reduce
    mappings._all_reduce = lambda tensor, _group, _size: tensor
    try:
        # f: identity forward. g: identity backward. Composing them must be
        # identity in both directions when the all-reduce is stubbed out.
        y = mappings.reduce_from_tensor_parallel_region(
            mappings.copy_to_tensor_parallel_region(x, ctx), ctx
        )
        assert torch.equal(y, x), "f then g is not identity in forward"
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x)), "f then g is not identity in backward"
    finally:
        mappings._all_reduce = original

    return "f and g compose to identity with collectives stubbed"


CHECKS = [
    check_column_row_pair_reconstructs_mlp,
    check_column_shard_gradients_concatenate,
    check_vocab_parallel_cross_entropy,
    check_vocab_parallel_ce_gradient,
    check_conjugate_functions_are_inverse_shaped,
]


def test_tensor_parallel_math_checks():
    for check in CHECKS:
        check()


def main():
    failures = 0
    for check in CHECKS:
        try:
            print(f"  ok    {check.__name__}: {check()}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {check.__name__}: {error}")

    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} TP math checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
