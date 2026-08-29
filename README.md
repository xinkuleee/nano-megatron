# nano-megatron

A compact implementation of Megatron's defining training path: tensor and
sequence parallel Transformer layers, non-interleaved 1F1B pipeline parallelism,
and data-parallel gradient reduction. The implementation is pure PyTorch and is
verified numerically against an unsharded single-process model on CPU/Gloo.

This is a teaching system, not a smaller copy of Megatron-Core. It keeps the
communication and scheduling algorithms and deliberately leaves out production
feature breadth, compatibility layers, and custom GPU kernels.

## Implemented

- Orthogonal TP × PP × DP process groups in an explicit `ParallelContext`
- Column/row-parallel linear layers and attention-head/MLP sharding
- Vocab-parallel embedding, LM head, and cross entropy without full-logit gather
- Sequence Parallel residual stream using sequence AllGather/ReduceScatter
- Non-interleaved 1F1B with explicit activation/gradient P2P
- Flat data-parallel gradient reduction, tied embeddings, and logical grad norm
- Whole-block activation recomputation
- FP32 and BF16 compute paths (BF16 smoke-tested on CPU; NCCL remains unverified)
- Deterministic uint16/uint32 pre-tokenized token streams
- Atomic rank-local checkpoint/resume for an unchanged topology
- Standard `torchrun` entry point for CPU/Gloo and GPU/NCCL

## Quick start

```bash
uv sync --group dev
./run_checks.sh
```

Run a four-process synthetic training example:

```bash
torchrun --nproc-per-node=4 -m nano_megatron.train \
  --tp 2 --pp 2 --sequence-parallel --recompute --steps 20
```

On macOS, if `--standalone` resolves an invalid local IPv6 hostname, use an
explicit loopback rendezvous:

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m nano_megatron.train --tp 2 --pp 2 --sequence-parallel
```

For a pre-tokenized stream, pass a flat native-endian uint16 or uint32 file:

```bash
torchrun --nproc-per-node=4 -m nano_megatron.train \
  --tp 2 --pp 2 --sequence-parallel \
  --data tokens.bin --data-dtype uint16 \
  --save checkpoints --save-every 100 --steps 1000
```

Resume from an exact step directory using the same TP/PP/DP topology:

```bash
torchrun --nproc-per-node=4 -m nano_megatron.train \
  --tp 2 --pp 2 --sequence-parallel \
  --data tokens.bin --load checkpoints/step_00000100 --steps 1000
```

For NVIDIA GPUs add `--backend nccl --device cuda --dtype bfloat16`. The CUDA
device is bound from `LOCAL_RANK` before any NCCL process group is created.

## Architecture

| File | Responsibility |
|---|---|
| `parallel.py` | Rank coordinates and TP/PP/DP/tied process groups |
| `mappings.py` | Autograd-aware AllReduce, sequence AllGather, ReduceScatter |
| `tp_layers.py` | Column/row linear and vocab-parallel embedding |
| `cross_entropy.py` | Cross entropy over vocabulary-sharded logits |
| `model.py` | One decoder-only GPT stage: RMSNorm, RoPE, MHA, SwiGLU |
| `pipeline.py` | P2P communication and the single non-interleaved 1F1B schedule |
| `grads.py` | TP/SP/tied/DP gradient finalization and global clipping |
| `data.py` | Minimal pre-tokenized token stream |
| `checkpoint.py` | Atomic same-topology rank-local checkpointing |
| `train.py` | Thin torchrun CLI and training loop |

Each rank owns coordinates `(pp_rank, dp_rank, tp_rank)` with:

$$W = PP \times DP \times TP$$

Tensor-parallel ranks split layer weights. Sequence Parallel reuses the TP group
and keeps the residual stream as `[B, S/TP, H]`: each sublayer AllGathers the
sequence before its column-parallel projections and ReduceScatters the row-
parallel output. PP ranks own consecutive Transformer blocks and exchange only
activation shards and their gradients. DP replicas consume different token
windows and average flat gradient buffers after microbatch accumulation.

The vocabulary-parallel loss communicates only the per-token global maximum,
exponential sum, and target logit, reducing communication from $O(BSV)$ to
$O(BS)$.

## Correctness contract

The tests do not treat "the program ran" as correctness. They build an
unsharded reference model, slice the exact same weights across ranks, and compare:

- loss and every reassembled gradient
- TP/SP activation shapes
- logical full-model gradient norm and clipped gradients
- parameters after an optimizer step
- heterogeneous DP data and TP×PP×DP composition
- tied embedding initialization and gradients
- recomputation on/off results
- checkpoint model, optimizer, RNG, step, and data-cursor restoration

Run `./run_checks.sh` for the socket-free algebra checks, real SP collectives,
and the CPU/Gloo distributed matrix. The script fails if distributed checks
cannot run rather than reporting a skipped matrix as success.

The end-to-end CLI test also compares uninterrupted training with a four-rank
`torchrun` save/restart/resume run under TP=2, PP=2, SP, BF16, and recomputation.

## Deliberate non-goals

- MoE/Expert Parallel and Context Parallel
- ZeRO/FSDP or a distributed optimizer
- virtual/interleaved pipeline stages or automatic graph partitioning
- communication/computation overlap and persistent gradient buckets
- FP8/FP4, Transformer Engine, Triton, or custom CUDA kernels
- model registries, multiple architectures, tokenizer training, or data cleaning
- inference serving, KV caches, RL, elastic recovery, async checkpoints
- checkpoint resharding across a changed topology

These are important production features, but each creates a separate systems
problem. Keeping them out lets every collective in this repository correspond
to a visible mathematical reason and keeps the full training step readable.
