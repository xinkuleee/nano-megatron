"""End-to-end torchrun checkpoint equivalence on a tiny CPU model."""

from __future__ import annotations

import socket
import struct
import subprocess
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.distributed


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _run(output: Path, tokens: Path, steps: int, load: Path | None = None) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--master-addr=127.0.0.1",
        f"--master-port={_free_port()}",
        "-m",
        "nano_megatron.train",
        "--tp=2",
        "--pp=2",
        f"--steps={steps}",
        "--num-microbatches=2",
        "--micro-batch-size=1",
        "--n-layer=2",
        "--n-head=4",
        "--n-embd=32",
        "--vocab-size=64",
        "--seq-len=8",
        "--sequence-parallel",
        "--recompute",
        "--dtype=bfloat16",
        f"--data={tokens}",
        f"--save={output}",
    ]
    if load is not None:
        command.append(f"--load={load}")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_torchrun_interrupted_training_matches_uninterrupted(tmp_path):
    tokens = tmp_path / "tokens.bin"
    tokens.write_bytes(struct.pack("=4096H", *(index % 64 for index in range(4096))))

    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    resumed = tmp_path / "resumed"
    _run(continuous, tokens, steps=2)
    _run(split, tokens, steps=1)
    _run(resumed, tokens, steps=2, load=split / "step_00000001")

    continuous_step = continuous / "step_00000002"
    resumed_step = resumed / "step_00000002"
    for checkpoint_dir in (continuous_step, resumed_step):
        assert (checkpoint_dir / "manifest.pt").is_file()
        assert len(list(checkpoint_dir.glob("rank_*.pt"))) == 4

    for rank in range(4):
        filename = f"rank_{rank:05d}.pt"
        uninterrupted = torch.load(continuous_step / filename, weights_only=False)
        after_resume = torch.load(resumed_step / filename, weights_only=False)
        _assert_equal(after_resume, uninterrupted)
