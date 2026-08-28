# Sparse attention on the Hexagon HTP: vocabulary, constants, and the deployment table

Entry point for the block-sparse attention work. Everything here is measured on SM8750 /
Hexagon v79 (6 HVX threads, 1 HMX unit, 8 MB VTCM) across four physical units, whose dense
reference rows agree to within 1.4%. The detailed studies are indexed at the end.

## 1. Vocabulary

Six things are called a "block size" or a "tile" in this work and they are independent. Most
confusion in this project traced to conflating two of them.

| symbol | what it is | set by | typical |
|:--|:--|:--|:--|
| `bs` (= `Bl`, `Bk`) | **KV block** the selection *names* | `op_params[4]` | 64 |
| `bq` | **query block** one selection list *serves* | `op_params[5]`; `0` = chunk-wide | 0, 64…512 |
| `k` | query blocks merged under one list; `bq = k·bs` | the scorer | 1, 2, 4, 8 |
| `u` | **size of the selection list**, `sel->ne[0]` | the scorer | 3…32 |
| `Br` | kernel's **query tile** height | chunk-size search; pinned to `bq` when `bq≠0` | 64…1152 |
| `Bc` | kernel's **KV chunk** width, `= m·bs` | chunk-size search | 64…512 |

Two derived quantities do all the work:

**`kv_eff = u · bs`** — the KV rows the kernel actually reads, [ggml-hexagon.cpp:2095](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2095):
```c
const uint32_t kv_effective = sel ? (uint32_t)(sel->ne[0] * bs) : nek1;
```
Every KV-derived decision uses `kv_eff`, not `kv`: the chunk count, whether the pipeline can
run, the VTCM budget, the `Bc` search. `kv_eff / kv` *is* the density.

**`n_kv_blocks` ("chunks") — the trip count of the inner loop over KV.** The kernel is query-tile
outer, KV-chunk inner:
```
for each query tile (Br rows):
    for each of n_kv_blocks chunks:
        DMA K,V -> QK^T on HMX -> softmax -> PV -> rescale the O accumulator
```
```
n_kv_blocks = kv_eff / Bc = u / m
m = largest divisor of u with m <= FA_SPARSE_MAX_M (8) and m*bs <= Bc_cap
```
`m` must **divide** `u` or the last chunk is ragged and the staging buffers carry stale rows;
`Bc_cap` is derived so `n_kv_blocks >= FA_MIN_KV_BLOCKS` (3) and the pipeline can run
([flash-attn-ops.h:436-446](../../../ggml/src/ggml-hexagon/htp/flash-attn-ops.h#L436)).

## 2. Measured constants

| constant | value | what it is |
|:--|--:|:--|
| per-`graph_compute` | **~620 µs** | additive, flat over a 17× range of work. **Not** a 1.605 slope — see `xattn-block-selection.md` |
| per KV chunk | **~161 µs** | fixed, independent of `Bc`. Matches the independent ~190 µs at `flash-attn-ops.h:401` |
| per KV row | **0.554 µs** | the actual streaming and arithmetic |
| Q-side floor | **~740 µs** at `nb=512` | ~1.4 µs per query token; irreducible, sparsity never touches it |
| query-block tax | **1.10 / 1.51 / 1.89×** | at `bq` = 256 / 128 / 64 against chunk-wide. Multiplicative, independent of `u` |

Cost model, validated to ±7% on held-out real unions:
```
cost(bq, u) = tax(bq) * [ 121 + 161*n_kv_blocks(u) + 0.554*kv_eff(u) ]      µs, nb=512
```
Below three chunks it does not apply: `u=2` gives one chunk, one thread, no pipeline, and
costs **3.85×** the prediction.

> The **composition** of the 161 µs is not established. An earlier claim that it was the
> fork/join and HMX-queue handoff does not survive its own evidence — idle time is only 7.6%
> of wall in the shared-selection configuration these were fitted on. The likelier dominant
> term is the online-softmax O-accumulator rescale, which runs once per (query tile, chunk)
> over `Br·G·DV` elements regardless of `Bc`. Untested.

## 3. The deployment table

Attention kernel only, `bs=64`, 25% fine density, `u` rounded **up** to hold the measured
union and then to a divisor-friendly size. `k` adjacent query blocks merged, so `Br = k·64`.

| kv | dense | k=1 | k=2 | k=4 | k=8 | best |
|--:|--:|--:|--:|--:|--:|:--|
| 512 | 933 | 0.71× (u=3) | 0.87× (u=3) | 1.13× (u=3) | **1.19× (u=6)** | **k=8** |
| 1024 | 1244 | 0.85× (u=6) | 1.07× (u=6) | **1.45× (u=6)** | 1.27× (u=8) | **k=4** |
| 2048 | 2077 | 1.11× (u=8) | 1.25× (u=12) | **1.75× (u=12)** | 1.54× (u=16) | **k=4** |
| 4096 | 4286 | 1.64× (u=16) | 1.69× (u=24) | **2.35× (u=24)** | 2.08× (u=28) | **k=4** |

Four rules follow, all measured:

1. **Merge `k=4` query blocks** (`k=8` below kv=1024). Interior optimum: along a fixed-`u` row
   a larger `k` is always cheaper, but the union grows with `k` and overtakes the saving.
2. **Round `u` to a divisor-friendly size, never to the nearest integer.** At `k=4`, `u=11`
   costs 2604 µs and `u=12` costs 1188 — **2.2× for one fewer block**. At kv=4096, `u=23` vs
   `u=24` is **2.9×**. Good sizes have a divisor near `u/3`: 3, 6, 8, 12, 16, 24, 32. Avoid
   5, 7, 11, 13, 19, 23.
3. **Never select fewer than 3 blocks.** `u=2` is one chunk, one thread, no pipeline: at
   kv=512 it costs 1364 µs against 747 for `u=3` — **1.83× slower for 33% fewer blocks**. A
   nominal 25% budget at kv=512 lands exactly there and must be overridden.
4. **A faithful per-query-block selection (`k=1`) never beats dense below kv=2048.**

## 4. Achieved rate

The harness counts *useful* FLOPs (selected blocks only), so these are real rates.

| kv | kv_eff | attn µs | GFLOP | TF/s | dense @ full kv | dense @ `kv_eff` | **% of ceiling** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 128 | 1633 | 0.54 | 0.33 | 2.32 | 0.40 | 83% |
| 1024 | 256 | 2058 | 2.15 | 1.04 | 3.52 | 1.24 | 84% |
| 2048 | 512 | 3955 | 8.59 | 2.17 | 4.01 | 2.42 | 90% |
| 4096 | 1024 | 5276 | 17.18 | **3.26** | 4.31 | 3.43 | **95%** |

"% of dense at full `kv`" is the wrong denominator — a sparse kernel does the work of a
*shorter* KV, and short KV is intrinsically harder to run fast here. Against **dense at the
same `kv_eff`** the kernel reaches **83–95%**, consistent with the 91–102%-of-ceiling result
obtained by a different route in `xattn-scoring.md`. **Block-sparse indexing costs essentially
nothing; what costs is having less work.**

The query block shows up directly as a rate collapse, identical arithmetic throughout:

| selection | µs | TF/s |
|:--|--:|--:|
| chunk-wide | 5181 | **3.32** |
| `bq=128` | 8856 | 1.94 |
| `bq=64` | 10696 | 1.61 |

For scale: dense FA peaks at 4.31 TF/s here, and mllm's GEMM study on the same die puts
practical HMX fp16 peak near 7 TF/s for a large square, falling to 1.6–2.1 TF/s at `K=128` —
which *is* attention's QK shape. Both FA kernels sit above that because the fused kernel keeps
the score tile in VTCM instead of round-tripping it.

## 5. Why the query block costs so much here, and not on a GPU

`Br` is the factor over which every KV byte and every synchronisation event is amortised.
Per-query-block selection pins `Br` to `bq` ([:436](../../../ggml/src/ggml-hexagon/htp/flash-attn-ops.h#L436)),
collapsing it from 512–1152 to 64. The MAC count does not change; the schedule does.

**On a CUDA GPU the same constraint is vacuous, because `Br <= 64 <= bq` already.** From
llama.cpp's own CUDA backend, the query tile per CTA is capped at 64
([fattn.cu:33](../../../ggml/src/ggml-cuda/fattn.cu#L33)):
```c
ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);   // largest case
```
And the dependency is *inverted*: query tiles **are** the parallelism
([fattn-common.cuh:1086-1128](../../../ggml/src/ggml-cuda/fattn-common.cuh#L1086)):
```c
ntiles_x     = ceil(Q->ne[1] / ncols1);
blocks_num.x = ntiles_dst;                 // one CTA per output tile
```
More query tiles means more CTAs means more SMs busy. The CUDA code's explicit worry is having
too *few*: when `tiles_efficiency_percent < 75` it switches to **stream-K**, splitting the KV
axis to manufacture parallelism — the exact opposite of what this kernel spends VTCM on.

| | HTP | CUDA |
|:--|:--|:--|
| query tiles are… | a **serial** loop | the **grid** |
| parallelism from | splitting rows across 6 HVX threads | one CTA per tile across ~100+ SMs |
| more tiles means | more serial iterations | more occupancy |
| tile size | 512–1152 (wants big) | ≤ 64 (capped) |

Independently corroborated: mllm measures the same effect on the same silicon through a
different framework — `BQ=64` vs `BQ=256` is 1.93×, against our 1.84–2.09× — but note their
production graph *inverts* it, because there the `nqb = Sq/BQ` batch items are what fills the
6 threads. **"Keep `Br` large" is a property of this kernel's structure, not of the silicon.**

## 6. What is not measured

Every number here is latency. **No accuracy result exists for any configuration in this
document.** The configurations that win are also the coarsest: chunk-wide selection means one
KV set for up to 2048 query tokens, and the best end-to-end result additionally replaces
QUOKA's query sub-selection with an evenly spaced sample, discarding its central claim. Read
every speedup as an upper bound conditional on a quality result nobody has produced.

## 7. Index

| doc | what it settles |
|:--|:--|
| `flash-attn-htp-anatomy.md` | the kernel: tiling, phases, the per-query-block attribution, the barrier work |
| `xattn-scoring.md` | XAttention scoring, the context-length curves, op-by-op shapes, the CPU/GPU split |
| `xattn-block-selection.md` | the selection pipeline, the per-call constant, the OpenCL comparison |
| `block-selection-overlap.md` | union/intersection on real activations, the `(k, u)` surface, the cost predictor |
| `quoka-on-htp.md` | QUOKA (arXiv:2602.08722) implemented and measured; the block-selection hybrid |
| `sparse-attn-inkernel-vs-gather.md` | in-kernel indexing vs a materialised gather |
| `opencl-gather-vs-mllm.md`, `gemm-latency-qwen3-1.7b.md` | mllm cross-checks |
| `rpcmem-cpu-access.md` | CPU/DSP shared-buffer costs |
