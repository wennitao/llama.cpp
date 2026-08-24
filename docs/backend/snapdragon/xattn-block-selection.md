# Block selection on HTP: what scoring actually costs

Companion to [sparse-attn-inkernel-vs-gather.md](sparse-attn-inkernel-vs-gather.md),
which measured block-sparse *attention* against a materialized gather but supplied
the block selection as a constant leaf. This doc measures the missing half: what it
costs to **compute** the selection with real attention-score-based ranking
(XAttention, Xu et al. ICML 2025, arXiv:2503.16428), and where that cost lives.

Headline: the score matmul was running at ~7% of the HMX's demonstrated throughput,
but fixing that does not reduce latency, because the matmul is only ~10% of the
marginal cost. The scoring cost is the **unfused op chain around it**.

## Setup

`test_flash_attn_ext_xattn_sel` in `tests/test-backend-ops.cpp`. Qwen3-1.7B-ish
shape: `hs=128`, `nh=8` KV heads, `nr=2` (16 Q heads), `nb=512` query tokens,
selection block `bs=64`, 25% density (`n_sel = nblk/4`). Device `b4bd0901`
(SM8750, Hexagon V79), `build-sparse`.

The graph is 17 ops:

```
q -> view/cont      -> q_reps  [hs, R*nr, nh]     R query representatives per head
mul_mat(k, q_reps)  -> scores  [kv, R*nr, nh]     the score matmul
soft_max_ext        -> sm      [kv, R*nr, nh]     temperature 1/sqrt(hs)
reshape+sum_rows    ->         [nblk, R*nr, nh]   collapse bs keys into one block
cont(permute)+sum_rows -> block_scores [nblk,1,nh]  collapse the query axis
add(bias) + argsort -> ranked                     descending
view(n_sel)         -> sel                        top-k, a strided view
flash_attn_ext(src[5]=sel)                        block-sparse FA on a COMPUTED sel
```

`R` is the knob under test: `R*nr` is the number of query rows entering the score
matmul. `R=8` gives 16 rows, which only half-fills one 32-row HMX tile.

### Stages, and why they are measured differently

| stage | measures | timing mode |
|---|---|---|
| 1 | the score matmul alone | replicated — the matmul **is** the output node, so per-call overhead amortizes |
| 2 | matmul + softmax + both reductions | whole-graph, `n_runs=1` |
| 0 | everything, including the sparse FA | whole-graph, `n_runs=1` |
| 4 | identical FA with `sel` supplied as a **leaf** | whole-graph, `n_runs=1` |

Stage 4 exists purely so that `stage0 - stage4` cancels the whole-graph per-call
overhead, which is large here (~590 µs, measured as stage-4 minus the replicated
`test_flash_attn_ext_sparse` figure). **Never compare a whole-graph absolute
against a replicated one** — doing exactly that produced a wrong "dense beats
sparse everywhere" conclusion earlier in this work.

## Result 1 — the score matmul was starved

Stage 1, µs/run and GFLOP/s from `2·nh·(R·nr)·kv·hs`:

| kv | R=8 | R=16 | R=32 | R=64 | R=128 |
|--:|--:|--:|--:|--:|--:|
| 512 | 80.3 · 209 | 90.5 · 371 | 116.7 · 575 | 172.9 · 776 | 191.1 · **1410** |
| 1024 | 125.0 · 269 | 139.1 · 482 | 182.6 · 735 | 259.2 · 1040 | 380.3 · **1410** |
| 2048 | 216.9 · 309 | 243.0 · 552 | 307.3 · 874 | 449.1 · 1200 | 691.6 · **1550** |

16× the work costs 2.4–3.2× the time; utilization improves **5.0–6.8×**. For
scale, the flash-attention kernel on the same device sustains ~3.15 TFLOPS, so
`R=8` was running the matrix unit at roughly **7% of demonstrated throughput**
and `R=128` gets within 2×.

This corroborates against mllm, which scores the same way on QNN HTP. At L=2048
mllm's antidiagonal scorer does `Hq·Lr²·(S·d)` = **1.07 GMAC** in 763 µs; ours at
`R=8` does `nh·(R·nr)·kv·hs` = **33.5 MMAC** in 654 µs. Thirty-two times less work
for the same wall time — the shape, not the kernel, was the problem.

### Which dimension is small, and what that actually costs

Not tile under-fill. `ggml_mul_mat(k, q_reps)` is, per KV head, an
`(kv x 128) x (128 x R*nr)` product: `M = kv` (512-2048), `K = hs = 128`,
`N = R*nr = 16` at `R=8`. `N=16` fills half of a 32x32 HMX tile — but half-filled
tiles alone would cap efficiency at 50%, and we measure 7%.

The real shape problem is that this is a **tall-skinny, GEMV-like** matmul. The
entire K tensor must be streamed regardless of `R`, and at `N=16` there are only
16 output columns to amortize that streaming against. Fitting `t = a + b·R` by
least squares over the five `R` points:

| kv | fixed `a` | per-`R` `b` | K bytes | implied K bandwidth | `a` as % of t(R=8) |
|--:|--:|--:|--:|--:|--:|
| 512 | 83.2 µs | 0.95 µs | 1.0 MB | 12.6 GB/s | ~100% |
| 1024 | 110.7 | 2.15 | 2.1 MB | 18.9 GB/s | 89% |
| 2048 | 183.2 | 4.00 | 4.2 MB | 22.9 GB/s | 84% |

So **84-100% of the score matmul at `R=8` is a fixed cost that buys no output**:
streaming `nh · kv · hs` fp16 of K through the machine at 13-23 GB/s. The fit's
`a` tracks K's size almost exactly, which is the signature of a K-streaming
bound rather than a compute bound.

Two consequences:

1. **Extra query representatives are nearly free in the matmul.** At kv=2048,
   going `R=8 -> R=64` adds 8x the useful work for +232 µs on top of a 183 µs
   floor you were paying anyway. The reason `R` is expensive today is entirely
   the unfused chain, not the matmul.
2. **~183 µs at kv=2048 is an irreducible floor for scoring**, because ranking
   blocks requires looking at every key at least once. XAttention's antidiagonal
   reshape does not help here — it repacks the same `L·d` elements, so it moves
   the same bytes. This is what the fusion projection below is anchored to.

## Result 2 — but ~90% of the marginal cost is not the matmul

Cost of going `R=8 → R=64`. Both operands of each difference are whole-graph, so
the per-call overhead cancels and no overhead estimate is needed:

| kv | matmul Δ (stage 1) | matmul+softmax+reduce Δ (stage 2) | matmul's share |
|--:|--:|--:|--:|
| 512 | +92.5 µs | +804.4 µs | **11.5%** |
| 1024 | +134.2 | +1353.3 | **9.9%** |
| 2048 | +232.1 | +2368.6 | **9.8%** |

The post-matmul chain dominates, and its traffic is proportional to `R`. So
raising `R` buys much better HMX utilization on a term that is not the
bottleneck, while inflating the term that is.

Total scoring cost, `stage0 - stage4`:

| kv | R=8 | R=64 |
|--:|--:|--:|
| 512 | +164 µs | +654 |
| 1024 | +267 | +1699 |
| 2048 | +688 | +2812 |

Stage 4 is `R`-independent by construction, so its two rows are a free noise
check: they differ by 1.6% / 8.5% / 2.8%. Read these deltas as **±100 µs**. That
also revises an earlier pass which reported +323 / +434 / +654; the kv=2048 figure
reproduces, the shorter-context ones were inside the noise band.

**`R` is a quality knob, not a speed knob.**

## Result 3 — it is not a hidden CPU fallback

Worth ruling out, because it would imply a completely different fix. `perf` mode
in `test-backend-ops` support-checks **only the output node**
(`tests/test-backend-ops.cpp:1535`), so a silent graph split was plausible.

It is not happening. `ggml_backend_hexagon_graph_compute`
(`ggml/src/ggml-hexagon/ggml-hexagon.cpp:3760`) walks every compute node,
remaps it with `op_remap_to_htp`, and enqueues it. There is no per-node fallback
path. All 17 ops are NPU-resident, and every op type in the chain — `MUL_MAT`,
`SOFT_MAX`, `SUM_ROWS`, `CONT`, `ADD`, `ARGSORT` — has an HTP opcode.

So the cost is genuine per-op dispatch and DRAM round-trip, not a host detour.

## Where the traffic goes

With `S = [kv, R*nr, nh]` fp32 as the big intermediate, the chain touches it 4×:

| op | traffic |
|---|---|
| `mul_mat` | writes `S` |
| `soft_max` | reads `S`, writes `S` |
| `sum_rows` | reads `S`, writes `S/bs` |

At `kv=2048, R=64` that is 8.4 MB × 4 ≈ 34 MB per call. Everything after the
first `sum_rows` operates on a tensor `bs=64`× smaller and is negligible.

A fused kernel that keeps `S` in VTCM collapses 4× to 1× — only the small output
reaches DRAM.

## Why fusion is a small job here

The fused scorer is **flash attention with the PV matmul replaced by a block-wise
sum**: same tiled QKᵀ, same online softmax, same DMA descriptor chain, same HMX
queue. The online softmax is required anyway — at `kv=2048, R=64` the score
tensor is 8.4 MB against 8 MB of VTCM, so it cannot be materialized whole
regardless.

`ggml-hexagon` already has the hook: `try_fuse_node`
(`ggml/src/ggml-hexagon/ggml-hexagon.cpp:3670`) pattern-matches consecutive nodes
and emits one `htp_opnode` with a new opcode, which is how `RMS_NORM_MUL`,
`MUL_MAT_QKV` and `MUL_MAT_FFN` are done. No ggml-core change and no CPU
reference implementation are needed — the CPU path keeps running the unfused ops.

### Projected payoff

This assumes the HMX-staged design (attempt 2 below), not the HVX one that was
actually built first. If fusion brings scoring to near its matmul floor, at `kv=2048`:

| | today | fused (projected) |
|---|--:|--:|
| scoring, R=8 | 688 µs | ~220 µs |
| scoring, R=64 | 2812 µs | ~450 µs |
| end-to-end vs dense (2105 µs), R=8 | 1643 µs — **1.28×** | ~1210 µs — **~1.7×** |

The R=64 line is the more interesting one: fusion is what makes good selection
affordable, and good selection is what makes block-sparse attention correct
enough to deploy.

## Attempt 1 at fusion: correct (now actually tested), and 1.4-3.8x slower

Implemented as `HTP_OP_XATTN_SCORE`
(`ggml/src/ggml-hexagon/htp/xattn-score-ops.c`), matched on the host by
`try_fuse_xattn_score` over `MUL_MAT -> SOFT_MAX -> RESHAPE -> SUM_ROWS`. The
`[kv, nq, nh]` intermediate stays in VTCM exactly as intended, so DRAM traffic
does drop to K plus the small output.

> **Correctness: established, after two false starts.** This section first claimed
> the fused kernel passed 2/2 backends, then retracted that saying `test` mode
> "structurally cannot" run a fused op. Both were wrong, for different reasons.
>
> The real cause was mundane: `test_flash_attn_ext_xattn_sel` was registered only in
> `make_test_cases_perf()`, never in `make_test_cases_eval()`. `test -o
> XATTN_BLOCK_SCORES` therefore reported **`0/0 tests passed`** and printed OK -- zero
> failures out of zero cases. Every correctness claim about this kernel was vacuous.
>
> The retraction was also wrong. `ggml_backend_compare_graph_backend` does support
> whole-graph comparison: with a non-empty `test_nodes` list it calls
> `ggml_backend_graph_compute` ONCE over the whole graph and compares only the listed
> outputs (`ggml/src/ggml-backend.cpp:2240-2243`), and `run_whole_graph()` already
> feeds it that list. Node-by-node is only the `num_test_nodes == 0` branch.
>
> With the cases registered in the eval list, `XATTN_BLOCK_SCORES` now reports
> **3/3 passed with the fusion trace firing 3 times**, and 3/3 unfused. The fused
> scoring kernel is genuinely correct. Only its speed is the problem.
>
> **Stage 0 stays perf-only.** It runs argsort -> selection -> sparse FA, and fails
> 0/3 against CPU *with fusion both on and off*. HTP and CPU block scores differ in
> the last fp bits, argsort flips a rank, a different set of blocks is chosen, and the
> outputs diverge entirely. Checking stage 0 needs the ranking pinned via a dominant
> `bias` leaf; the graph has the slot, the harness fills it randomly.

At kv=2048/R=64 the fused kernel sustains 42.8 GFLOP/s against the standalone
HMX matmul's 1.55 TFLOP/s -- **36x worse**. Saving ~470 µs of chain traffic
bought ~1800 µs of extra matmul.

**Root cause: `hs = 128` is only two HVX f16 vectors per dot product.**
`hvx_dot_f16_f16_aa_rx4` issues 2 multiply-accumulates per row and then a full
horizontal reduction -- 4 vector adds plus a multi-step shuffle-reduce -- and
`hvx_dot_f16_f16_aa_rx32` adds a `vsetq`/`vmux` per group of 4 on top. The
reduction costs more than the arithmetic it reduces. HMX never pays this: its
32x32 tile accumulates in place, so a short K dimension costs it nothing.

The earlier reasoning in this doc -- that the pass is K-streaming bound, so the
matmul engine hardly matters -- was wrong in a specific way. The K-streaming
term bounds *HMX*, which is fast enough to be waiting on memory. HVX is far
enough below that bound to be compute-bound instead, and at `hs=128` its
per-dot reduction overhead dominates.

**The fusion is therefore gated off by default** (`opt_xattn_fusion`, enable with
`GGML_HEXAGON_XATTN_FUSION=1`). With it off the numbers reproduce the unfused
baseline within noise, so nothing regresses.

### What attempt 2 needs

The structure is right and worth keeping: VTCM-resident intermediate, one pass,
block sums emitted directly. Only the matmul engine has to change -- stage the
score tile through HMX into VTCM, then let HVX do the softmax and block-sum over
it in place. That keeps the 4x-to-1x traffic win *and* the 1.55 TFLOP/s matmul,
and it is the combination the projection below assumes. The FA kernel already
does exactly this staging for QK^T, so the machinery exists in
`hmx-fa-kernels.h`.

## The whole-graph per-call constant, measured

Stage 5 was added to settle this: byte-identical graph to stage 1, but timed
whole-graph, so `t(stage5) - t(stage1)` **is** the per-`ggml_backend_graph_compute`
cost that every `n_runs=1` figure in this doc carries.

| kv | R | replicated | whole-graph | constant |
|--:|--:|--:|--:|--:|
| 512 | 8 | 80.0 | 459.7 | **379.7** |
| 512 | 64 | 173.3 | 599.2 | 426.0 |
| 1024 | 8 | 124.7 | 500.0 | **375.4** |
| 1024 | 64 | 257.8 | 712.2 | 454.3 |
| 2048 | 8 | 213.9 | 612.6 | **398.7** |
| 2048 | 64 | 448.1 | 1041.6 | 593.5 |

So **~375-400 µs** for a small graph, drifting to ~590 µs as the work grows -- it is
not a pure constant, because a replicated run also overlaps successive iterations
that a single run cannot. Two earlier estimates were both too high: ~590 µs (from
stage4 minus replicated FA) and ~690 µs (derived from stage2-stage1 flatness).

What this does and does not invalidate:

- **Scoring deltas are unaffected.** `stage0 - stage4` cancels the constant, since
  both sides are whole-graph. Those numbers stand.
- **Absolute whole-graph figures are inflated** by ~400-600 µs. Read every stage-2
  and stage-4 number in this doc with that in mind.
- **Cross-backend whole-graph comparisons are the real casualty.** The NPU pays a
  FastRPC/dspqueue round trip and the GPU pays an OpenCL queue flush; there is no
  reason those constants are equal, and neither has been measured against the other.
  Any NPU-vs-GPU table built from `n_runs=1` numbers needs each backend's own
  constant subtracted first.

## Not measured / not verified

- **HTP == CPU is verified; the algorithm's quality is not.**
  `test-backend-ops test -b HTP0 -o FLASH_ATTN_EXT_XATTN` passes, so the whole
  chain -- score matmul, softmax, both reductions, argsort, and the sparse FA
  consuming the computed `sel` -- reproduces the CPU reference within NMSE
  tolerance, and the ranking is stable enough that both backends select the same
  blocks. What is *not* tested is whether XAttention picks *good* blocks: there is
  no retrieval or perplexity arm. Every number here is a latency, and the accuracy
  claim is only "HTP agrees with CPU", not "the selector works".
- ctx=4096 end-to-end.
- `R` between 128 and `nb`, where the matmul should saturate.
- OpenCL scoring. mllm already measured eager-op OpenCL scoring at 48.4 ms for
  L=2048 versus 0.76 ms on QNN HTP — 60× worse, with no `sum` or `topk` op on
  that backend to build on. See [opencl-gather-vs-mllm.md](opencl-gather-vs-mllm.md).
  Not worth pursuing without a custom fused kernel.

## Reproduce

```sh
cmake --build build-sparse --target test-backend-ops -j 24
adb push build-sparse/bin/test-backend-ops /data/local/tmp/llama.cpp/bin/

# ADSP_LIBRARY_PATH is required or session 0 fails to open with 0x80000406
D=/data/local/tmp/llama.cpp
adb shell "cd $D && ADSP_LIBRARY_PATH=$D/lib LD_LIBRARY_PATH=./lib \
  ./bin/test-backend-ops perf -b HTP0 -o XATTN_SCORE_MM"     # stage 1, R sweep
adb shell "cd $D && ADSP_LIBRARY_PATH=$D/lib LD_LIBRARY_PATH=./lib \
  ./bin/test-backend-ops perf -b HTP0 -o XATTN_BLOCK_SCORES" # stage 2
adb shell "cd $D && ADSP_LIBRARY_PATH=$D/lib LD_LIBRARY_PATH=./lib \
  ./bin/test-backend-ops perf -b HTP0 -o FLASH_ATTN_EXT_XATTN" # stage 0
adb shell "cd $D && ADSP_LIBRARY_PATH=$D/lib LD_LIBRARY_PATH=./lib \
  ./bin/test-backend-ops perf -b HTP0 -o XATTN_FA_ONLY"      # stage 4 control
```

If `adb devices` is empty, check `ss -lntp | grep 5037` — anything other than
`ssh` listening there is a local adb server shadowing the tunnel; kill it.
