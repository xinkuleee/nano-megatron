"""A small GPT, built so that each pipeline stage constructs only its own layers.

Nothing here knows how the pipeline is scheduled. A stage that is not first
takes a hidden state instead of token ids; a stage that is not last returns a
hidden state instead of a loss. That is the entire contract, and it is why the
same model runs unchanged under GPipe, 1F1B, or no pipeline at all.

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

from . import parallel_state as ps
from .cross_entropy import vocab_parallel_cross_entropy
from .mappings import copy_to_tensor_parallel_region
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

    @property
    def head_dim(self) -> int:
        return divide(self.n_embd, self.n_head)


class RMSNorm(nn.Module):
    """Replicated across the TP group -- every rank sees the same input."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


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

    def __init__(self, config: GPTConfig):
        super().__init__()
        tp = ps.get_tensor_parallel_world_size()
        self.n_head_local = divide(config.n_head, tp)
        self.head_dim = config.head_dim

        self.q = ColumnParallelLinear(config.n_embd, config.n_embd, bias=False)
        self.k = ColumnParallelLinear(config.n_embd, config.n_embd, bias=False)
        self.v = ColumnParallelLinear(config.n_embd, config.n_embd, bias=False)
        self.proj = RowParallelLinear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape

        def split_heads(t):
            return t.view(b, s, self.n_head_local, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, s, -1)
        return self.proj(y)


class ParallelMLP(nn.Module):
    """SwiGLU: column-shard the two up-projections, row-shard the down.

    Both column shards land on the same rank, so the elementwise product
    between them needs no communication.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.gate = ColumnParallelLinear(config.n_embd, hidden, bias=False)
        self.up = ColumnParallelLinear(config.n_embd, hidden, bias=False)
        self.down = RowParallelLinear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerLayer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = ParallelAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = ParallelMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class ParallelLMHead(nn.Module):
    """Output projection, vocab-sharded.

    Stored as ``[V/tp, n_embd]`` -- the same layout as the embedding -- so the
    two can literally share one tensor when weights are tied.
    """

    _tensor_parallel_sharded_parameters = frozenset({"weight"})

    def __init__(self, config: GPTConfig):
        super().__init__()
        per_partition = divide(config.vocab_size, ps.get_tensor_parallel_world_size())
        self.weight = nn.Parameter(torch.empty(per_partition, config.n_embd))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # Same f as ColumnParallelLinear: identity out, all-reduce the input
        # gradient on the way back.
        x = copy_to_tensor_parallel_region(x)
        return x @ self.weight.t()


def layers_for_stage(n_layer: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    per_stage = divide(n_layer, pp_size)
    return pp_rank * per_stage, (pp_rank + 1) * per_stage


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        pp_size = ps.get_pipeline_parallel_world_size()
        pp_rank = ps.get_pipeline_parallel_rank()
        self.first_stage = ps.is_pipeline_first_stage()
        self.last_stage = ps.is_pipeline_last_stage()

        start, end = layers_for_stage(config.n_layer, pp_size, pp_rank)
        self.layer_offset = start
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(start, end)])

        if self.first_stage:
            self.embedding = VocabParallelEmbedding(config.vocab_size, config.n_embd)
        if self.last_stage:
            self.final_norm = RMSNorm(config.n_embd)
            self.lm_head = ParallelLMHead(config)

        # With one stage the tie is a genuine shared tensor. Split across
        # stages the two copies are distinct parameters, and keeping them equal
        # is the job of the embedding-group all-reduce in ``grads.py``.
        if config.tie_word_embeddings and self.first_stage and self.last_stage:
            self.lm_head.weight = self.embedding.weight

        cos, sin = build_rope_cache(config.seq_len, config.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def vocab_range(self) -> tuple[int, int]:
        per = divide(self.config.vocab_size, ps.get_tensor_parallel_world_size())
        rank = ps.get_tensor_parallel_rank()
        return rank * per, (rank + 1) * per

    def forward(self, x, labels=None):
        """``x`` is token ids on the first stage, a hidden state elsewhere.

        Returns the loss on the last stage when ``labels`` is given, otherwise a
        hidden state to hand to the next stage.
        """
        h = self.embedding(x) if self.first_stage else x

        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)

        if not self.last_stage:
            return h

        logits = self.lm_head(self.final_norm(h))
        if labels is None:
            return logits
        start, end = self.vocab_range()
        return vocab_parallel_cross_entropy(logits, labels, start, end)
