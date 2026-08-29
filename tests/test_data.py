from __future__ import annotations

import struct

import pytest
import torch

from nano_megatron.data import TokenStream


def _write_tokens(path, values, dtype="uint16"):
    code = {"uint16": "H", "uint32": "I"}[dtype]
    path.write_bytes(struct.pack(f"={len(values)}{code}", *values))
    return path


def test_dp_ranks_read_disjoint_deterministic_windows(tmp_path):
    path = _write_tokens(tmp_path / "tokens.bin", range(40))
    rank0 = TokenStream(path, dtype="uint16", seq_len=3, dp_rank=0, dp_size=2)
    rank1 = TokenStream(path, dtype="uint16", seq_len=3, dp_rank=1, dp_size=2)

    x0, y0 = rank0.next_batch(2)
    x1, y1 = rank1.next_batch(2)
    assert x0.flatten().tolist() == list(range(6))
    assert y0.flatten().tolist() == list(range(1, 7))
    assert x1.flatten().tolist() == list(range(6, 12))
    assert y1.flatten().tolist() == list(range(7, 13))
    assert rank0.cursor == rank1.cursor == 12

    x0, _ = rank0.next_batch(2)
    x1, _ = rank1.next_batch(2)
    assert x0.flatten().tolist() == list(range(12, 18))
    assert x1.flatten().tolist() == list(range(18, 24))
    assert rank0.cursor == rank1.cursor == 24


def test_cursor_round_trip_reproduces_next_batch_and_wraps(tmp_path):
    path = _write_tokens(tmp_path / "tokens.bin", range(10))
    stream = TokenStream(path, dtype="uint16", seq_len=3, dp_rank=1, dp_size=2)
    stream.next_batch(1)
    saved = stream.state_dict()
    expected = stream.next_batch(1)

    resumed = TokenStream(path, dtype="uint16", seq_len=3, dp_rank=1, dp_size=2)
    resumed.load_state_dict(saved)
    actual = resumed.next_batch(1)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[0].flatten().tolist() == [9, 0, 1]
    assert actual[1].flatten().tolist() == [0, 1, 2]


def test_uint32_tokens_are_read_without_truncation(tmp_path):
    values = [0, 70_000, 80_000, 90_000]
    path = _write_tokens(tmp_path / "tokens.bin", values, dtype="uint32")
    stream = TokenStream(path, dtype="uint32", seq_len=2)
    inputs, labels = stream.next_batch(1)
    assert inputs.dtype == labels.dtype == torch.int64
    assert inputs.tolist() == [[0, 70_000]]
    assert labels.tolist() == [[70_000, 80_000]]


def test_rejects_invalid_format_parallelism_and_state(tmp_path):
    path = _write_tokens(tmp_path / "tokens.bin", range(4))
    with pytest.raises(ValueError, match="dtype"):
        TokenStream(path, dtype="int64", seq_len=2)
    with pytest.raises(ValueError, match="dp_rank"):
        TokenStream(path, dtype="uint16", seq_len=2, dp_rank=2, dp_size=2)

    malformed = tmp_path / "malformed.bin"
    malformed.write_bytes(b"abc")
    with pytest.raises(ValueError, match="not divisible"):
        TokenStream(malformed, dtype="uint16", seq_len=2)

    stream = TokenStream(path, dtype="uint16", seq_len=2)
    state = stream.state_dict()
    state["cursor"] = -1
    with pytest.raises(ValueError, match="cursor"):
        stream.load_state_dict(state)
    mismatched = stream.state_dict()
    mismatched["seq_len"] = 4
    with pytest.raises(ValueError, match="configuration mismatch"):
        stream.load_state_dict(mismatched)
