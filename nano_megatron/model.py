"""A small GPT, built so that each pipeline stage constructs only its own layers.

Nothing here knows how the pipeline is scheduled. A stage that is not first
takes a hidden state instead of token ids; a stage that is not last returns a
hidden state instead of a loss. That contract keeps pipeline scheduling outside
the model implementation.

Q/K/V and the SwiGLU gate/up are kept as separate projections rather than one
fused GEMM. Fusing them would shard wrong: chunking ``[q|k|v]`` along the output
dim hands rank 0 all of ``q`` and half of ``k`` instead of a slice of each head.
Megatron does fuse, but only by storing the weight in a per-shard interleaved
layout -- correctness first here, and the sharding rules stay obvious.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cross_entropy import vocab_parallel_cross_entropy
from .mappings import copy_to_tensor_parallel_region, gather_from_sequence_parallel_region
from .parallel import ParallelContext
from .tp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    divide,
)


@dataclass
class GPTConfig:
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 32
    # Dropout is left out entirely: doing it correctly under TP needs two RNG
    # states (different seeds inside the parallel region, identical outside).
    tie_word_embeddings: bool = True
    sequence_parallel: bool = False
    recompute: bool = False
    compute_dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        return divide(self.n_embd, self.n_head)

    @property
    def activation_dtype(self) -> torch.dtype:
        try:
            return {"float32": torch.float32, "bfloat16": torch.bfloat16}[
                self.compute_dtype
            ]
        except KeyError as error:
            raise ValueError(
                f"compute_dtype must be 'float32' or 'bfloat16', got {self.compute_dtype!r}"
            ) from error


class RMSNorm(nn.Module):
    """Replicated across the TP group -- every rank sees the same input."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight.to(x.dtype)


def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: [b, n_head_local, s, head_dim]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ParallelAttention(nn.Module):
    """Attention sharded over heads.

    Heads are independent, so once Q/K/V are column-sharded every head's
    attention is computed locally with no communication at all. The single
    all-reduce lives in the row-parallel output projection.
    """

    def __init__(self, config: GPTConfig, ctx: ParallelContext):
        super().__init__()
        self.ctx = ctx
        self.n_head_local = divide(config.n_head, ctx.tp_size)
        self.head_dim = config.head_dim
        self.sequence_parallel = config.sequence_parallel

        self.q = ColumnParallelLinear(config.n_embd, config.n_embd, ctx, bias=False)
        self.k = ColumnParallelLinear(config.n_embd, config.n_embd, ctx, bias=False)
        self.v = ColumnParallelLinear(config.n_embd, config.n_embd, ctx, bias=False)
        self.proj = RowParallelLinear(config.n_embd, config.n_embd, ctx, bias=False)

    def forward(self, x, cos, sin):
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, self.ctx)
        b, s, _ = x.shape

        def split_heads(t):
            return t.view(b, s, self.n_head_local, self.head_dim).transpose(1, 2)

        reduce_input_grad = not self.sequence_parallel
        q = split_heads(self.q(x, reduce_input_grad=reduce_input_grad))
        k = split_heads(self.k(x, reduce_input_grad=reduce_input_grad))
        v = split_heads(self.v(x, reduce_input_grad=reduce_input_grad))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, s, -1)
        return self.proj(y, sequence_parallel=self.sequence_parallel)


class ParallelMLP(nn.Module):
    """SwiGLU: column-shard the two up-projections, row-shard the down.

    Both column shards land on the same rank, so the elementwise product
    between them needs no communication.
    """

    def __init__(self, config: GPTConfig, ctx: ParallelContext):
        super().__init__()
        self.ctx = ctx
        hidden = 4 * config.n_embd
        self.sequence_parallel = config.sequence_parallel
        self.gate = ColumnParallelLinear(config.n_embd, hidden, ctx, bias=False)
        self.up = ColumnParallelLinear(config.n_embd, hidden, ctx, bias=False)
        self.down = RowParallelLinear(hidden, config.n_embd, ctx, bias=False)

    def forward(self, x):
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, self.ctx)
        reduce_input_grad = not self.sequence_parallel
        hidden = F.silu(self.gate(x, reduce_input_grad=reduce_input_grad))
        hidden = hidden * self.up(x, reduce_input_grad=reduce_input_grad)
        return self.down(hidden, sequence_parallel=self.sequence_parallel)


class TransformerLayer(nn.Module):
    def __init__(self, config: GPTConfig, ctx: ParallelContext):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = ParallelAttention(config, ctx)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = ParallelMLP(config, ctx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class ParallelLMHead(nn.Module):
    """Output projection, vocab-sharded.

    Stored as ``[V/tp, n_embd]`` -- the same layout as the embedding -- so the
    two can literally share one tensor when weights are tied.
    """

    def __init__(self, config: GPTConfig, ctx: ParallelContext):
        super().__init__()
        self.ctx = ctx
        per_partition = divide(config.vocab_size, ctx.tp_size)
        self.weight = nn.Parameter(torch.empty(per_partition, config.n_embd))
        self.weight.tensor_parallel_sharded = True
        self.sequence_parallel = config.sequence_parallel
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # Same f as ColumnParallelLinear: identity out, all-reduce the input
        # gradient on the way back.
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, self.ctx)
        else:
            x = copy_to_tensor_parallel_region(x, self.ctx)
        return x @ self.weight.t()


def layers_for_stage(n_layer: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    per_stage = divide(n_layer, pp_size)
    return pp_rank * per_stage, (pp_rank + 1) * per_stage


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, ctx: ParallelContext | None = None):
        super().__init__()
        ctx = ctx or ParallelContext.single()
        self.config = config
        self.ctx = ctx
        pp_size = ctx.pp_size
        pp_rank = ctx.pp_rank
        self.first_stage = ctx.is_first_stage
        self.last_stage = ctx.is_last_stage
        config.activation_dtype  # validate before allocating parameters
        for name in ("vocab_size", "n_layer", "n_head", "n_embd", "seq_len"):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive")
        divide(config.n_layer, ctx.pp_size)
        divide(config.n_embd, ctx.tp_size)
        divide(config.n_head, ctx.tp_size)
        divide(config.vocab_size, ctx.tp_size)
        if config.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if config.sequence_parallel:
            divide(config.seq_len, ctx.tp_size)

        start, end = layers_for_stage(config.n_layer, pp_size, pp_rank)
        self.layers = nn.ModuleList([TransformerLayer(config, ctx) for _ in range(start, end)])

        if self.first_stage:
            self.embedding = VocabParallelEmbedding(config.vocab_size, config.n_embd, ctx)
        if self.last_stage:
            self.final_norm = RMSNorm(config.n_embd)
            self.lm_head = ParallelLMHead(config, ctx)

        # With one stage the tie is a genuine shared tensor. Split across
        # stages the two copies are distinct parameters, and keeping them equal
        # is the job of the embedding-group all-reduce in ``grads.py``.
        if config.tie_word_embeddings and self.first_stage and self.last_stage:
            self.lm_head.weight = self.embedding.weight

        cos, sin = build_rope_cache(config.seq_len, config.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def vocab_range(self) -> tuple[int, int]:
        per = divide(self.config.vocab_size, self.ctx.tp_size)
        rank = self.ctx.tp_rank
        return rank * per, (rank + 1) * per

    def forward(self, x, labels=None):
        """``x`` is token ids on the first stage, a hidden state elsewhere.

        Returns the loss on the last stage when ``labels`` is given, otherwise a
        hidden state to hand to the next stage.
        """
        device_type = x.device.type
        use_bfloat16 = self.config.compute_dtype == "bfloat16"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            h = (
                self.embedding(x, sequence_parallel=self.config.sequence_parallel)
                if self.first_stage
                else x
            )
            if use_bfloat16:
                h = h.to(torch.bfloat16)

            for layer in self.layers:
                if self.config.recompute and self.training:
                    h = checkpoint(
                        layer, h, self.rope_cos, self.rope_sin, use_reentrant=False
                    )
                else:
                    h = layer(h, self.rope_cos, self.rope_sin)

            if not self.last_stage:
                return h

            logits = self.lm_head(self.final_norm(h))
            if labels is None:
                return logits
            start, end = self.vocab_range()
            return vocab_parallel_cross_entropy(logits, labels, start, end, self.ctx)
