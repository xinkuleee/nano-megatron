"""A minimal reader for raw, pre-tokenized training data.

The file format is deliberately just a flat native-endian stream of uint16 or
uint32 token ids.  Tokenization and dataset building stay outside the training
runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


_DTYPES = {
    "uint16": (torch.uint16, 2),
    "uint32": (torch.uint32, 4),
}


class TokenStream:
    """Deterministic next-token batches from a memory-mapped token file.

    Every DP rank reads a disjoint window from the same global step.  The
    cursor advances by the whole DP batch, so equal cursors identify the same
    training position on every TP/PP rank::

        base = cursor + dp_rank * batch_size * seq_len
        cursor += dp_size * batch_size * seq_len

    The stream wraps at EOF.  Only the monotonically increasing cursor is
    checkpointed; modulo is applied when tokens are read.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        dtype: str,
        seq_len: int,
        dp_rank: int = 0,
        dp_size: int = 1,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"dp_rank must be in [0, {dp_size}), got {dp_rank}")

        path = Path(path)
        torch_dtype, item_size = _DTYPES[dtype]
        file_size = path.stat().st_size
        if file_size % item_size:
            raise ValueError(
                f"{path} has {file_size} bytes, not divisible by {dtype} item size {item_size}"
            )
        num_tokens = file_size // item_size
        if num_tokens < 2:
            raise ValueError(f"token stream needs at least 2 tokens, got {num_tokens}")

        self.path = path
        self.dtype = dtype
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.cursor = 0
        stat = path.stat()
        self._identity = {
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        self._tokens = torch.from_file(
            str(path), shared=False, size=num_tokens, dtype=torch_dtype
        )

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, labels)`` with shape ``[batch, seq_len]``."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        local_tokens = batch_size * self.seq_len
        start = self.cursor + self.dp_rank * local_tokens
        indices = torch.arange(local_tokens + 1, dtype=torch.int64)
        indices.add_(start).remainder_(self._tokens.numel())
        window = self._tokens[indices].to(torch.int64)

        self.cursor += self.dp_size * local_tokens
        inputs = window[:-1].view(batch_size, self.seq_len)
        labels = window[1:].view(batch_size, self.seq_len)
        return inputs, labels

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "dtype": self.dtype,
            "seq_len": self.seq_len,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "identity": self._identity,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "dtype": self.dtype,
            "seq_len": self.seq_len,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "identity": self._identity,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"token-stream configuration mismatch: saved={actual}, current={expected}"
            )
        cursor = state["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError(f"cursor must be a non-negative integer, got {cursor!r}")
        self.cursor = cursor
