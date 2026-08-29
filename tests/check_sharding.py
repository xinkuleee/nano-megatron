"""Checks that need no sockets: sharding shapes, coverage, and reference math.

The full correctness proof lives in ``check_parallel.py`` and needs real
processes. This file catches the errors that are cheap to catch -- a parameter
nobody knows how to shard, a stage that builds the wrong layers, a chunk/concat
pair that does not round-trip -- and runs anywhere.

    python -m tests.check_sharding
"""

from __future__ import annotations

import torch

from nano_megatron.model import GPTConfig, layers_for_stage
from tests.reference import _SHARD_DIM, _shard_dim_for, build_reference_model

CONFIG = GPTConfig(vocab_size=128, n_layer=4, n_head=4, n_embd=64, seq_len=16)


def check_reference_runs():
    """The single-process model must produce a finite loss and full gradients."""
    model = build_reference_model(CONFIG)
    inputs = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.seq_len))
    labels = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.seq_len))

    loss = model(inputs, labels)
    loss.backward()

    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
    return f"reference loss={loss.item():.4f}, {len(list(model.parameters()))} params all with grads"


def check_every_parameter_has_a_rule():
    """No parameter may be silently treated as replicated by accident."""
    model = build_reference_model(CONFIG)
    replicated_ok = ("norm.weight", "_norm.weight", ".bias")

    unclassified = []
    for name, _ in model.named_parameters():
        if _shard_dim_for(name) is None and not name.endswith(replicated_ok):
            unclassified.append(name)

    assert not unclassified, (
        f"these parameters match no sharding rule and are not obviously "
        f"replicated: {unclassified}"
    )
    return f"all parameters classified ({len(_SHARD_DIM)} shard rules)"


def check_shard_concat_roundtrip():
    """Splitting then concatenating along the recorded dim must be identity."""
    model = build_reference_model(CONFIG)
    checked = 0
    for name, param in model.named_parameters():
        dim = _shard_dim_for(name)
        if dim is None:
            continue
        for tp_size in (2, 4):
            if param.shape[dim] % tp_size:
                continue
            shards = torch.chunk(param.data, tp_size, dim=dim)
            assert len(shards) == tp_size, f"{name}: got {len(shards)} shards"
            rebuilt = torch.cat(shards, dim=dim)
            assert torch.equal(rebuilt, param.data), f"{name} tp={tp_size} round-trip failed"
            checked += 1
    return f"{checked} shard/concat round-trips exact"


def check_head_divisibility():
    """Attention shards over heads, so heads must divide by the TP size."""
    for tp_size in (1, 2, 4):
        assert CONFIG.n_head % tp_size == 0, f"n_head={CONFIG.n_head} not divisible by tp={tp_size}"
        assert CONFIG.n_embd % tp_size == 0
        assert CONFIG.vocab_size % tp_size == 0
    return "head/embd/vocab divisible for tp in (1, 2, 4)"


def check_layer_partition():
    """Stages must cover every layer exactly once, with no gaps or overlap."""
    for pp_size in (1, 2, 4):
        covered = []
        for pp_rank in range(pp_size):
            start, end = layers_for_stage(CONFIG.n_layer, pp_size, pp_rank)
            covered.extend(range(start, end))
        assert covered == list(range(CONFIG.n_layer)), (
            f"pp={pp_size} covers {covered}, expected 0..{CONFIG.n_layer - 1}"
        )
    return "layer partition exact for pp in (1, 2, 4)"


def check_tied_embedding_is_shared():
    """With one stage the tie must be the same tensor, not a copy."""
    config = GPTConfig(**{**CONFIG.__dict__, "tie_word_embeddings": True})
    model = build_reference_model(config)
    assert model.lm_head.weight is model.embedding.weight, "tie did not share storage"
    return "tied embedding shares storage at pp=1"


def check_gradient_sync_tags_match_sharding_rules():
    """Never classify a TP shard as replicated during gradient finalization."""
    model = build_reference_model(CONFIG)

    mismatches = []
    for name, parameter in model.named_parameters():
        expected = _shard_dim_for(name) is not None
        actual = getattr(parameter, "tensor_parallel_sharded", False)
        if actual != expected:
            mismatches.append(f"{name}: expected sharded={expected}, got {actual}")

    assert not mismatches, "gradient-sync tags disagree with sharding rules: " + "; ".join(
        mismatches
    )
    return "gradient-sync tags match all parameter sharding rules"


CHECKS = [
    check_reference_runs,
    check_every_parameter_has_a_rule,
    check_shard_concat_roundtrip,
    check_head_divisibility,
    check_layer_partition,
    check_tied_embedding_is_shared,
    check_gradient_sync_tags_match_sharding_rules,
]


def test_sharding_checks():
    for check in CHECKS:
        check()


def main():
    failures = 0
    for check in CHECKS:
        try:
            detail = check()
            print(f"  ok    {check.__name__}: {detail}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {check.__name__}: {error}")

    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} sharding checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
