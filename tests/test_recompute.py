from dataclasses import replace

import torch

from nano_megatron.model import GPT, GPTConfig


def test_block_recompute_matches_normal_forward_and_backward():
    config = GPTConfig(vocab_size=64, n_layer=2, n_head=4, n_embd=32, seq_len=8)
    inputs = torch.randint(0, config.vocab_size, (2, config.seq_len))
    labels = torch.randint(0, config.vocab_size, (2, config.seq_len))

    torch.manual_seed(0)
    normal = GPT(config)
    recomputed = GPT(replace(config, recompute=True))
    recomputed.load_state_dict(normal.state_dict())

    normal_logits = normal(inputs)
    recomputed_logits = recomputed(inputs)
    torch.testing.assert_close(recomputed_logits, normal_logits)

    normal_loss = normal(inputs, labels)
    recomputed_loss = recomputed(inputs, labels)
    normal_loss.backward()
    recomputed_loss.backward()

    torch.testing.assert_close(recomputed_loss, normal_loss)
    normal_grads = dict(normal.named_parameters())
    recomputed_grads = dict(recomputed.named_parameters())
    assert normal_grads.keys() == recomputed_grads.keys()
    for name, parameter in normal_grads.items():
        torch.testing.assert_close(recomputed_grads[name].grad, parameter.grad)


def test_recompute_is_disabled_during_evaluation(monkeypatch):
    config = GPTConfig(
        vocab_size=64, n_layer=2, n_head=4, n_embd=32, seq_len=8, recompute=True
    )
    calls = 0

    def counting_checkpoint(function, *args, **kwargs):
        nonlocal calls
        calls += 1
        return function(*args)

    monkeypatch.setattr("nano_megatron.model.checkpoint", counting_checkpoint)
    inputs = torch.randint(0, config.vocab_size, (2, config.seq_len))

    model = GPT(config)
    model(inputs)
    assert calls == config.n_layer

    model.eval()
    model(inputs)
    assert calls == config.n_layer


def test_bfloat16_compute_keeps_loss_and_gradients_finite():
    config = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        seq_len=8,
        compute_dtype="bfloat16",
    )
    model = GPT(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.seq_len))
    labels = torch.randint(0, config.vocab_size, (2, config.seq_len))

    logits = model(inputs)
    assert logits.dtype == torch.bfloat16
    loss = model(inputs, labels)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
