from __future__ import annotations

import copy
import random
import struct
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nano_megatron import checkpoint
from nano_megatron.data import TokenStream
from nano_megatron.parallel import ParallelContext


CTX = ParallelContext.single()


def _stream(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(struct.pack("=32H", *range(32)))
    return TokenStream(path, dtype="uint16", seq_len=2)


def _make_training_state(tmp_path):
    torch.manual_seed(11)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    data = _stream(tmp_path)

    inputs, labels = data.next_batch(2)
    loss = model(inputs.float()).sub(labels.float()).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, data


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_nested_equal(a, e)
    else:
        assert actual == expected


def test_restores_model_optimizer_step_rng_and_data_cursor(tmp_path):
    model, optimizer, data = _make_training_state(tmp_path)
    random.seed(23)
    checkpoint.save_checkpoint(tmp_path / "ckpt", model, optimizer, CTX, step=7, data=data)
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())

    expected_torch_random = torch.rand(4)
    expected_python_random = random.random()
    expected_batch = data.next_batch(2)

    for parameter in model.parameters():
        parameter.data.add_(100)
    optimizer.param_groups[0]["lr"] = 9.0
    torch.manual_seed(999)
    random.seed(999)

    step, extra = checkpoint.load_checkpoint(
        tmp_path / "ckpt", model, optimizer, CTX, data=data
    )
    assert step == 7
    assert extra == {}
    _assert_nested_equal(model.state_dict(), saved_model)
    _assert_nested_equal(optimizer.state_dict(), saved_optimizer)
    assert torch.equal(torch.rand(4), expected_torch_random)
    assert random.random() == expected_python_random
    actual_batch = data.next_batch(2)
    assert torch.equal(actual_batch[0], expected_batch[0])
    assert torch.equal(actual_batch[1], expected_batch[1])


def test_atomic_failure_preserves_previous_rank_shard(tmp_path):
    model, optimizer, data = _make_training_state(tmp_path)
    directory = tmp_path / "ckpt"
    path = checkpoint.save_checkpoint(directory, model, optimizer, CTX, step=3, data=data)
    original = path.read_bytes()

    def fail_after_partial_write(_state, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("simulated write failure")

    with mock.patch.object(checkpoint.torch, "save", side_effect=fail_after_partial_write):
        with pytest.raises(OSError, match="simulated"):
            checkpoint.save_checkpoint(directory, model, optimizer, CTX, step=4, data=data)

    assert path.name == "rank_00000.pt"
    assert path.read_bytes() == original
    assert list(directory.glob("*.tmp")) == []


def test_rejects_checkpoint_from_different_topology_before_loading(tmp_path):
    model, optimizer, data = _make_training_state(tmp_path)
    directory = tmp_path / "ckpt"
    path = checkpoint.save_checkpoint(directory, model, optimizer, CTX, step=3, data=data)
    state = torch.load(path, weights_only=False)
    state["topology"]["tp_size"] = 2
    torch.save(state, path)
    original = copy.deepcopy(model.state_dict())

    with pytest.raises(RuntimeError, match="topology mismatch"):
        checkpoint.load_checkpoint(directory, model, optimizer, CTX, data=data)
    _assert_nested_equal(model.state_dict(), original)


def test_rejects_incomplete_checkpoint(tmp_path):
    model, optimizer, data = _make_training_state(tmp_path)
    directory = tmp_path / "ckpt"
    checkpoint.save_checkpoint(directory, model, optimizer, CTX, step=3, data=data)
    (directory / "manifest.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint.load_checkpoint(directory, model, optimizer, CTX, data=data)


def test_rejects_negative_step(tmp_path):
    model, optimizer, data = _make_training_state(tmp_path)
    with pytest.raises(ValueError, match="step"):
        checkpoint.save_checkpoint(
            tmp_path / "ckpt", model, optimizer, CTX, step=-1, data=data
        )
