"""Rank-local training checkpoints for a fixed parallel topology."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .parallel import ParallelContext


_VERSION = 1
_MANIFEST = "manifest.pt"


def _topology(ctx: ParallelContext) -> dict[str, int]:
    return {
        "world_size": ctx.world_size,
        "global_rank": ctx.rank,
        "tp_size": ctx.tp_size,
        "tp_rank": ctx.tp_rank,
        "pp_size": ctx.pp_size,
        "pp_rank": ctx.pp_rank,
        "dp_size": ctx.dp_size,
        "dp_rank": ctx.dp_rank,
    }


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def _rank_path(directory: str | os.PathLike, ctx: ParallelContext) -> Path:
    return Path(directory) / f"rank_{ctx.rank:05d}.pt"


def _model_config(model):
    config = getattr(model, "config", None)
    return asdict(config) if config is not None and is_dataclass(config) else None


def _atomic_save(state, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _uses_distributed_checkpoint(ctx: ParallelContext) -> bool:
    return ctx.world_size > 1 and torch.distributed.is_initialized()


def save_checkpoint(
    directory: str | os.PathLike,
    model,
    optimizer,
    ctx: ParallelContext,
    *,
    step: int,
    data=None,
    extra: dict | None = None,
) -> Path:
    """Atomically save this rank's model, optimizer, RNG, and data cursor."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _rank_path(directory, ctx)
    state = {
        "version": _VERSION,
        "topology": _topology(ctx),
        "model": model.state_dict(),
        "model_config": _model_config(model),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": _rng_state(),
        "data": data.state_dict() if data is not None else None,
        "extra": extra or {},
    }

    distributed = _uses_distributed_checkpoint(ctx)
    manifest_path = directory / _MANIFEST
    if distributed:
        if ctx.rank == 0:
            manifest_path.unlink(missing_ok=True)
        torch.distributed.barrier()

    _atomic_save(state, path)

    if distributed:
        torch.distributed.barrier()
    if ctx.rank == 0:
        manifest = {
            "version": _VERSION,
            "step": step,
            "world_size": ctx.world_size,
            "parallel_sizes": {
                "tp": ctx.tp_size, "pp": ctx.pp_size, "dp": ctx.dp_size
            },
            "shards": [f"rank_{rank:05d}.pt" for rank in range(ctx.world_size)],
        }
        if all((directory / shard).is_file() for shard in manifest["shards"]):
            _atomic_save(manifest, manifest_path)
        else:
            raise RuntimeError("checkpoint is missing one or more rank shards")
    if distributed:
        torch.distributed.barrier()
    return path


def load_checkpoint(
    directory: str | os.PathLike,
    model,
    optimizer,
    ctx: ParallelContext,
    *,
    data=None,
    map_location="cpu",
) -> tuple[int, dict]:
    """Restore this rank and return the next step plus extra state.

    A shard is intentionally valid only for the exact world size, parallel
    sizes, and rank coordinates that wrote it.
    """
    directory = Path(directory)
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"checkpoint is incomplete: missing {manifest_path}")
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    expected_manifest = {
        "version": _VERSION,
        "world_size": ctx.world_size,
        "parallel_sizes": {"tp": ctx.tp_size, "pp": ctx.pp_size, "dp": ctx.dp_size},
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"checkpoint manifest mismatch for {key}: "
                f"saved={manifest.get(key)!r}, current={expected!r}"
            )
    if not all((directory / shard).is_file() for shard in manifest.get("shards", ())):
        raise RuntimeError("checkpoint is incomplete: a rank shard is missing")

    path = _rank_path(directory, ctx)
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("version") != _VERSION:
        raise RuntimeError(
            f"unsupported checkpoint version {state.get('version')!r}; expected {_VERSION}"
        )

    saved_topology = state.get("topology")
    current_topology = _topology(ctx)
    if saved_topology != current_topology:
        raise RuntimeError(
            f"checkpoint topology mismatch: saved={saved_topology}, current={current_topology}"
        )
    saved_config = state.get("model_config")
    current_config = _model_config(model)
    if saved_config != current_config:
        raise RuntimeError(
            f"checkpoint model configuration mismatch: "
            f"saved={saved_config}, current={current_config}"
        )

    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(f"invalid checkpoint step {step!r}")
    if manifest.get("step") != step:
        raise RuntimeError("checkpoint manifest and rank shard have different steps")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if state["data"] is not None:
        if data is None:
            raise RuntimeError("checkpoint contains data state but no data stream was provided")
        data.load_state_dict(state["data"])
    _restore_rng_state(state["rng"])
    return step, state.get("extra", {})
