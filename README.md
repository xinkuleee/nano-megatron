# nano-megatron

Tensor, pipeline, and data parallelism, implemented from scratch in a compact
Python codebase and verified against a single-process reference.

Pure Python. No CUDA, no C++, no NCCL source. The parallelism lives entirely in
*orchestration* — which ranks form a group, which slice of a weight each holds,
what order microbatches execute in — while the actual work stays in PyTorch's
existing GEMM and collective kernels. That split is the point: it is what makes
Megatron's core ideas fit in a file you can read in one sitting.

## Quick start

```bash
uv sync
```

```bash
./run_checks.sh
```

```bash
uv run python -m nano_megatron.train --tp 2 --pp 2 --steps 20
```

Everything runs on CPU with the `gloo` backend, so a laptop is enough to develop
and validate against. Pass `--backend nccl --device cuda` on a real GPU box.

## What's here

| File | Lines | What it does |
|---|---|---|
| [`parallel_state.py`](nano_megatron/parallel_state.py) | 201 | Carves the world into orthogonal TP/PP/DP groups |
| [`mappings.py`](nano_megatron/mappings.py) | 111 | The conjugate `f`/`g` autograd pair — the heart of TP |
| [`tp_layers.py`](nano_megatron/tp_layers.py) | 163 | Column/Row parallel linear, vocab-parallel embedding |
| [`cross_entropy.py`](nano_megatron/cross_entropy.py) | 76 | Loss over sharded logits without ever gathering them |
| [`model.py`](nano_megatron/model.py) | 220 | GPT where each stage builds only its own layers |
| [`p2p.py`](nano_megatron/p2p.py) | 84 | Stage-to-stage send/recv |
| [`schedules.py`](nano_megatron/schedules.py) | 154 | GPipe and 1F1B |
| | **1,168** | parallelism and model core (comments included) |
| [`sharding.py`](nano_megatron/sharding.py) | 122 | Sharding rules in one place, used by the tests |
| [`train.py`](nano_megatron/train.py) | 159 | Training loop |
| [`grads.py`](nano_megatron/grads.py) | 153 | DP/tied gradients, parameter sync, global grad norm |

## Tensor parallelism

Two conjugate autograd functions do all the work:

```
f  (copy_to_tensor_parallel_region)     forward: identity     backward: all-reduce
g  (reduce_from_tensor_parallel_region) forward: all-reduce   backward: identity
```

A column-parallel linear is `f → local matmul`. A row-parallel linear is
`local matmul → g`. Chain them with a nonlinearity in between and the
nonlinearity needs no communication at all, because each rank's slice of the
hidden dimension is self-contained:

$$\text{silu}(XA)B = \sum_i \text{silu}(XA_i)B_i$$

One all-reduce per MLP, one per attention block. Autograd derives every backward
collective for free — that's why `f` and `g` are the only communication code in
the whole TP path.

Attention shards over heads, which are independent by construction, so
everything between the QKV projections and the output projection is local.

**Q/K/V are three separate projections here, not one fused GEMM.** Fusing them
shards wrong: chunking `[q|k|v]` along the output dim gives rank 0 all of `q`
plus half of `k`, not a slice of each head. Megatron does fuse, but pays for it
with a per-shard interleaved weight layout. Correctness first.

**The loss never materialises full logits.** A `[b, s, V]` all-gather would
dwarf every other collective in the model. Instead the two reductions softmax
actually needs — a max and a sum — happen on `[b, s]` tensors, which is
`O(b·s)` traffic instead of `O(b·s·V)`.

## Pipeline parallelism

The scheduler *is* the pipeline parallelism. There is no runtime, no queue, no
thread pool. Each rank runs an ordinary single-threaded loop:

```python
for _ in range(num_steady):
    hidden = recv_forward(...)              # blocks until upstream sends
    output = forward_step(...)
    grad = send_forward_recv_backward(...)  # one exchange, two jobs
    backward_step(...)
    send_backward(...)
```

The pipelining effect emerges from different processes sitting at different
points in that loop. The blocking receive is the only synchronisation needed.

GPipe and 1F1B run the *same two step functions in a different order*:

|  | Bubble | Activations held |
|---|---|---|
| GPipe | `(P-1)/M` | `M` |
| 1F1B | `(P-1)/M` | `P - rank` |

Identical bubble. The entire win is activation memory — and it costs nothing but
reordering. Model code is untouched between the two.

## Verification

Correctness is the deliverable, so it is checked four ways.

**Sharding invariants** ([`tests/check_sharding.py`](tests/check_sharding.py)) —
every parameter has an explicit shard rule, chunk/concat round-trips exactly,
stages cover every layer exactly once.

**TP algebra** ([`tests/check_tp_math.py`](tests/check_tp_math.py)) — simulates
a TP group inside one process with collectives stubbed out, then proves
column×row reconstructs the full MLP, that sharded gradients concatenate to the
reference, and that the vocab-parallel loss and its gradient match
`F.cross_entropy`.

**Logical gradient norm** ([`tests/check_grad_clip.py`](tests/check_grad_clip.py)) —
checks TP-shard summation, replicated-parameter deduplication, PP-stage
summation, tied-weight deduplication, and verifies that DP replicas are not
counted twice.

**End-to-end** ([`tests/check_parallel.py`](tests/check_parallel.py)) — the real
one. Builds a full single-process model, shards it, runs both schedules under
real processes, and asserts the loss, every reassembled gradient, the global
gradient norm, clipped gradients, and parameters after one optimizer step all match
elementwise. DP replicas deliberately consume different batches so their
all-reduce cannot pass by averaging already-identical gradients:

```bash
python -m tests.check_parallel --matrix
```

sweeping TP, PP, DP, combined 3-D configurations, both schedules, and the
`num_microbatches < pp` boundary case. Split pipelines also exercise tied input
and output embeddings.

Loss is divided by the microbatch count so gradient accumulation reproduces one
large batch exactly — that equivalence is what makes the comparison meaningful.

### Current status

Sharding, TP-math, and logical-gradient checks pass (7/7, 5/5, and 3/3), and
the 17-configuration CPU/Gloo distributed matrix passes. `run_checks.sh` fails rather than reporting success when socket
binding is unavailable, because skipping the distributed matrix is not a
correctness pass. GPU/NCCL, mixed precision, and performance remain separate
validation work.

## Deliberate omissions

Not oversights — each is a performance optimisation orthogonal to the parallelism
itself, and leaving them out is why the core stays readable:

- **Interleaved 1F1B** — divides the bubble by `v`; needs virtual stages
- **Sequence parallelism** — the natural next step; splits each all-reduce into
  reduce-scatter + all-gather for the same traffic and `1/tp` the activation memory
- **Communication/computation overlap** — bucketed gradient all-reduce; the
  single biggest real-world throughput win
- **ZeRO optimizer sharding**, activation recomputation, FP8, MoE
- **Dropout** — correct TP dropout needs two RNG states (different seeds inside
  the parallel region, identical outside). Omitted rather than done wrong.

## Notes

Dividing `world_size` as `(pp, dp, tp)` with `tp` varying fastest keeps each TP
group contiguous. On real hardware that lands the chatty all-reduces on
intra-node NVLink and leaves PP — small point-to-point messages — on the slow
outer axis. It is the reason `tp_size` should not exceed one node.
