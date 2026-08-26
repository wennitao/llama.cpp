# XAttention block-selection scoring in ggml

The faithful implementation of the block-score estimator from **XAttention: Block Sparse
Attention with Antidiagonal Scoring** (Xu et al., ICML 2025, arXiv:2503.16428), reference
implementation [mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention).
This is the only file in this directory that cites that paper.

Its predecessor, [xattn-block-selection.md](xattn-block-selection.md), measured
`test_flash_attn_ext_qsubsample_topk` — query-subsampled top-k selection — while calling it
XAttention. That doc now opens with a correction. Everything here is a different arm, in a
different test case, with a different name.

## The algorithm

Per head, with `Q, K ∈ R^{L×d}`, block size `Bl`, antidiagonal stride `S` (`S | Bl`):

```
reshaped_key   = cat([K[k::S, :]     for k in range(S)], dim=-1)   # [L/S, S·d]
reshaped_query = cat([Q[S-1-q::S, :] for q in range(S)], dim=-1)   # [L/S, S·d]
attn     = softmax(reshaped_query @ reshaped_keyᵀ / (√d · S) + causal_mask, dim=-1)
attn_sum = attn.view(NBq, Bl/S, NBk, Bl/S).sum(dim=(1,3))          # [NBq, NBk]
```

Entry `(rq, rk)` is `Σ_q Q[(S-1-q) + rq·S] · K[q + rk·S]`. The query offset `S-1-q` always
pairs with key offset `q`, so the two sum to `S-1` — the antidiagonal of the `S×S` sub-tile.
QK MACs drop from `L²d` to `L²d/S`, and unlike subsampling, **every input element is still
read exactly once**: the reduction is a re-association of the same products, not a
discard.

Three properties of the reference that are easy to get wrong, each read out of
`/mnt/raid0_ssd/wentao/x-attention` @ `e379887`:

1. **The scale is `1/(√d · S)` and it is applied to the logits**, before the mask and
   before the softmax (`xattn/src/Xattention.py:187-189`, which divides by
   `math.sqrt(head_dim) / stride / norm`; `norm` defaults to 1 and every call site passes
   1). The extra `1/S` makes the reduced logit the arithmetic *mean* of `S` ordinary
   attention logits, so the softmax temperature matches ordinary attention. The paper text
   is not in the repo, so that rationale is a reading of the code, not a quotation.
2. **The reduced causal mask is permissive**: cell `(rq, rk)` is allowed iff `rk ≤ rq`
   (`Xattention.py:204-214`, `torch.triu(ones * -inf, diagonal=1)` on the diagonal chunk).
   On the diagonal the antidiagonal pairs are (query `S-1-q`, key `q`), which is causal
   only when `q ≤ S-1-q`, so `⌊S/2⌋` of the `S` products are non-causal and the reference
   includes them all. It has no choice: the antidiagonal sum is one scalar and cannot be
   partially masked. **This is the algorithm, not a bug — do not "fix" it.**
3. **Chunked prefill (`Lk > Lq`) shows up as a column offset.** The reference's
   `current_index = k_block_num - q_block_num + chunk_idx·num_blocks_per_chunk`
   (`Xattention.py:255-256`); on the reduced grid that is `rk ≤ rq + (Nk - Nq)`.

## The ggml graph

`test_xattn_score` in `tests/test-backend-ops.cpp`. Twelve nodes and two constant leaves;
six do real work (`GET_ROWS`, `MUL_MAT`, `SOFT_MAX`, `SUM_ROWS`, `CONT`, `SUM_ROWS`).

```
leaf  idx    I32 [Lq, Hq]        idx[l] = (l/S)·S + (S-1 - l%S)        the reversal
leaf  cmask  F32 [Nk, Nq]        0 where rk ≤ rq + (Nk-Nq), else -1e30
      get_rows(q, idx)      -> qperm    F32 [d,   Lq, Hq]   THE only data movement
      reshape               -> rq       F32 [S·d, Nq, Hq]   free view
      reshape(k)            -> rk       F16 [S·d, Nk, Hkv]  free view
      mul_mat(rk, rq)       -> mm       F32 [Nk,  Nq, Hq]   the reduced matmul
      soft_max_ext(mm, cmask, 1/(√d·S), 0)                  scale+mask+softmax, ONE node
      reshape + sum_rows    ->          [1, NBk, Nq,  Hq]   pool the KEY sub-block axis
      reshape + permute + cont + sum_rows -> [1, NBk, NBq, Hq]  pool the QUERY sub-block axis
      reshape               -> attn_sum F32 [NBk, NBq, Hq]
```

Notes that are load-bearing:

- **The K reshape is genuinely free.** `[d, Lk, Hkv] → [S·d, Nk, Hkv]` is a plain
  contiguous reshape: element `(k·d+e, rk)` sits at `rk·S·d + k·d + e = K[e, rk·S+k]`,
  which is exactly `cat([K[k::S, :]])`.
  *Integration hazard:* `ggml_reshape_3d` asserts `ggml_is_contiguous`, and a KV-cache K
  view with `n_kv < kv_size` is **not** contiguous. In `test-backend-ops` K is a fresh
  contiguous tensor; in llama.cpp proper it is not. This is the single biggest obstacle to
  reusing this graph as written.
- **The scale lives inside `soft_max_ext`**, not in a separate `ggml_scale`. A `SCALE` node
  would be a full extra pass over the `Nk·Nq·Hq` intermediate.
- **The mask is F32 and uses `-1e30f`, not `-INFINITY`.** F32 because hexagon's vectorized
  mask path is only reached for an F32 mask (an F16 mask drops to a scalar loop). `-1e30f`
  because the fast softmax prep multiplies the mask by the slope in the non-IEEE `qf32`
  format, whose Inf behaviour is undocumented in-tree — this is defensive, not a
  confirmed bug.
- **Q is reversed, not K.** The two are mathematically identical: relabel the contraction
  index `c = S-1-q` and `(Q reversed, K plain)` becomes `(Q plain, K reversed)`, term for
  term. The choice is forced by dtypes. `ggml_get_rows` returns F32 unconditionally
  (`ggml/src/ggml.c:3901`), so reversing K would upcast it to F32, and
  `ggml_hexagon_matmul_is_hmx_eligible` rejects a *batched* matmul whose `src0` is not F16
  (`ggml/src/ggml-hexagon/ggml-hexagon.cpp:2331-2333`) — an F32 K loses HMX outright. Q is
  already F32, so reversing Q is dtype-preserving. The byte-count argument points the other
  way (K is f16 and `Hkv ≤ Hq`); it is the smaller effect and is written down here so
  nobody optimizes into the cliff.
- **Reducing the KEY axis first is what makes the one copy affordable.** The key sub-index
  is the fast axis of `rk`, so that pool is a free reshape; it shrinks the tensor by
  `P = Bl/S` *before* the permute+cont that the query pool needs.

### Shape constraints, all asserted

```c
GGML_ASSERT(Lq % S == 0 && Lk % S == 0);
GGML_ASSERT(Bl % S == 0);            // reference floors Bl // S and gets the grid wrong SILENTLY
GGML_ASSERT(Lq % Bl == 0 && Lk % Bl == 0);
GGML_ASSERT(S <= Bl);                // S == Bl is legal (P == 1); S > Bl divides by zero
GGML_ASSERT(Hq % Hkv == 0);
GGML_ASSERT(Lk >= Lq);
```

The reference asserts **none** of these. `S ∤ Bl` in particular fails silently there:
`reshaped_block_size = block_size // stride` floors, and the whole block grid comes out the
wrong size with no error.

Backend-specific, to keep the graph off CPU fallbacks:

| constraint | why |
|---|---|
| `Nk ≤ 32` or `Nk % 32 == 0` | `ggml_hexagon_supported_softmax`, `ggml-hexagon.cpp:3204`. `Nk` is the softmax `ne0`; violating it sends `SOFT_MAX` to CPU and splits the graph. |
| `S·d % 32 == 0`, `Nq > 4` | HMX eligibility for the reduced matmul. |
| `Nk % 32 == 0` | keeps the F32 mask's `nb[1] = 4·Nk` a multiple of 128, which the fast softmax path's *aligned* mask load needs and never checks. |

`op_flops()` reports `2 · Hq · Nq · Nk · (S·d) = 2 · Hq · Lq · Lk · d / S`. That is the
**dense-equivalent** count: the graph computes the masked upper triangle too, while the
reference's triton kernel skips future blocks, so the realized saving over a
causality-aware baseline is roughly `S/2`-fold, not `S`-fold.

## Correctness

### What is checked in-tree

| stage | output | mode | what it proves |
|--:|---|---|---|
| 0 | `attn_sum` `[NBk, NBq, Hq]` | eval, NMSE 5e-4 | whole graph vs the CPU reference |
| 3 | `sum_rows(attn_sum)` `[1, NBq, Hq]` | eval, absolute 1e-2 | the analytic invariant, below |
| 1 | `mm` `[Nk, Nq, Hq]` | perf, replicated | the reduced matmul alone |
| 2 | `qperm` `[d, Lq, Hq]` | perf, replicated | `GET_ROWS` — the reversal alone |
| 4 | `mm`, whole-graph | perf | `t(4) − t(1)` is the per-call constant |

Stage 0 is a real correctness arm, unlike the stand-in's stage 0: it ends at `attn_sum`,
which is a smooth, well-conditioned function of the inputs (softmax then two sums, no
discontinuity anywhere). There is no argsort to flip a rank on a last-bit difference, and
no `src[5]` that the CPU backend cannot see.

**The invariant (stage 3):** every reduced softmax row sums to 1 and the pool adds
`P = Bl/S` of them, so **every entry of `sum_rows(attn_sum)` is exactly `P`**. Stage 3
overrides `err()` to check *both* backends against that constant rather than against each
other, which catches the one class of bug an NMSE-vs-CPU check cannot see: a pooling shape
that both backends misunderstand identically. Its tolerance is loose (1e-2 absolute)
because hexagon's softmax normalizes with `hvx_vec_inverse_f32`, an approximate reciprocal
(`ggml/src/ggml-hexagon/htp/softmax-ops.c:198`); a structural pooling error is off by a
whole factor of `P` or `NBk`, so 1e-2 still catches everything the stage exists for.

Both index leaves are written explicitly in `initialize_tensors`. That is not optional: the
default fill is random, and a garbage I32 index makes **CPU abort** on
`GGML_ASSERT(i01 >= 0 && i01 < ne01)` while **hexagon silently skips the row** and leaves
the output uninitialized.

### What was verified off-tree

The ggml chain was cross-checked against a direct double-precision implementation of the
formulation above — antidiagonal pairing built from explicit `(S-1-q, q)` index arithmetic,
`1/(√d·S)`, the permissive reduced mask, and the `view(NBq, P, NBk, P).sum((1,3))` pool —
run on the x86 CPU backend of `build-host`:

| `d` | `Hq` | `Hkv` | `Lq` | `Lk` | `S` | `Bl` | max&#124;ggml − ref&#124; | max&#124;rowsum − P&#124; |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 3 | 1 | 64 | 64 | 4 | 16 | 1.11e-05 | 1.19e-07 |
| 128 | 4 | 2 | 512 | 512 | 8 | 64 | 1.17e-05 | 2.38e-07 |
| 64 | 2 | 2 | 256 | 256 | 8 | 64 | 1.37e-05 | 2.38e-07 |
| 128 | 4 | 1 | 256 | 512 | 8 | 64 | 9.79e-06 | 2.38e-07 |
| 128 | 2 | 1 | 512 | 512 | 64 | 64 | 3.42e-06 | 5.22e-08 |
| 64 | 2 | 2 | 128 | 128 | 1 | 64 | 1.97e-05 | 1.91e-06 |
| 128 | 16 | 8 | 512 | 512 | 16 | 64 | 1.02e-05 | 3.58e-07 |

The residual is f16 K rounding plus f32 accumulation; the reference side runs in double.
The `S = Bl` row exercises `P == 1`, the `S = 1` row is the identity control (no reduction,
no reversal), and the `Lq ≠ Lk` row exercises the chunked-prefill mask offset.

### What was NOT verified

**Nothing in this file has been run on a device.** No HTP or OpenCL number exists for this
arm. Every claim above is either read out of source or measured on an x86 CPU backend.

A direct diff against the reference's own `xattn_estimate` was not run either. If someone
does it: use a **single chunk** (`chunk_size ≥ k_len`). `Xattention.py:201` is
`causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")`, and when
`k_num_to_pad == 0` the slice `[..., -0:]` is `[..., 0:]` — the *entire* mask goes to
`-inf`, and only the current chunk's columns are written back, so all previous chunks stay
masked out. Multi-chunk comparisons will "fail" against that bug. Do not port that line.

## What is deliberately not here

**`find_blocks` is not in the graph, and cannot be.** The selector turns each row of
`attn_sum` into a variable-length list of block indices: always keep the sink and the
diagonal, then take the smallest prefix of the remaining scores, sorted descending, whose
cumulative mass reaches `tau · total`. Three independent blockers:

1. **No gather along `ne0`.** `ggml_argsort` returns indices; nothing applies them to
   produce sorted *values*, because `ggml_get_rows` gathers along `ne1`.
2. **`ggml_step` is not supported on hexagon**, so the `cumsum < required` indicator would
   fall back to CPU and split the graph.
3. **The fatal one: the output length is data**, and every ggml tensor length is fixed at
   graph-build time. Turning a boolean mask into an index list is a stream compaction. ggml
   has no such op, and this is not a missing kernel — it is a property of the framework.

So the selector belongs on the host, over the read-back `[NBk, NBq, Hq]` tensor. In
`test-backend-ops` that is exactly right: all the arithmetic worth testing is in
`attn_sum`. In llama.cpp proper it means a device→host sync mid-layer, whose cost is
unmeasured.

**No flash-attention wiring.** ~~`ggml_hexagon_supported_fa_sparse` requires
`sel->ne[1] == 1`~~ — **this is no longer true.** `sel` now carries a query-block axis:
`ne = [n_sel, NBq, n_kv_heads or 1, n_seqs or 1]` with the query-block size in
`op_params[5]`, and the kernel reads row `q_start / BQ` (`fa_sel_row` →
`fa_chunk_block_start`). `ne[1] == 1` remains the shared-selection layout and is
bit-identical to before. The scoring graph is still not *wired* to a sparse FA op in
this repo — the host-side selector below still has to run — but the kernel can now
express what the scorer produces.

Two constraints came with it. `n_sel` must be **fixed** across query blocks (top-k, not
the paper's variable-count `tau` threshold): the chunk decomposition, the pipelining
decision and the VTCM budget are all derived from `sel->ne[0]`, and the kernel's untagged
DMA FIFO relies on `fa_chunk_nblk` being query-block-independent. And a kernel query tile
must not straddle a query block, i.e. **`Br` must divide `BQ`**, which pins `Br` far below
what the tiling search would otherwise pick. At `neq1=512, BQ=64` that takes `q_blocks`
from 1 to 8 — modelled cost 8 800 → 32 000 — so the selection has to buy back a ~3.6×
tiling penalty (~1.8× at `BQ=128`) before per-query-block selection is net positive.

Landing scoring alone buys a correctness-anchored, CPU-comparable implementation of the
real estimator and a measurable per-layer scoring cost. **It buys no speedup**, and the
density that would decide whether a speedup exists is data-dependent and unmeasured here.

**The cheapest unmeasured number, and it decides the design.** How much do adjacent query
blocks' selections actually overlap on a real model? If they overlap heavily, staging the
*union* of the query blocks inside one large `Br` and carving the per-query-block
restriction into the FA mask (`src[3]`, already `[kv, nq, heads]` and already staged per
query row) computes the same attention with **no kernel change at all** and keeps `Br`
large. It is never worse than per-query-block in K/V DMA, and it degenerates to dense only
when the selections are disjoint. That number can be computed offline from `attn_sum`
with no kernel work. Nothing in this repo measures it.

**Neither fused block-score kernel can serve this graph.** `try_fuse_xattn_score` rejects
any `SOFT_MAX` carrying a mask (`ggml-hexagon.cpp:3724-3726`, comment: *"no mask and no
ALiBi slope — the kernel implements a plain softmax"*), and the OpenCL matcher does the
same. Real XAttention *requires* a mask, so the 1.12–1.58× fused speedup recorded in
[xattn-block-selection.md](xattn-block-selection.md) **does not carry over** until
`op_xattn_score` grows mask support.

**GQA is unresolved, and the reference is no help.** It forbids GQA outright
(`Xattention.py:31` and `:53` both assert `num_q_head == num_kv_head`) and its callers
materialize `repeat_kv` before calling. So the reference scores per Q head, which this
graph reproduces for free — but `src[5]` is per *KV* head, and reducing `Hq` per-head
selections to one per KV head (union, max-of-scores, or score per KV head directly) is a
policy the reference does not supply. None of the three is the reference's; whichever gets
picked must be named in the code and must not be called "XAttention for GQA".

The reduction is **forced, not chosen**: the kernel stages one K/V chunk per
`(sequence, query block, kv_head)` and multiplies all `G = Hq/Hkv` query heads against it
in a single `g_br = align_up(G·Br, 32)` row tile. Honouring a per-Q-head selection would
mean `G` separate stagings per query block, i.e. discarding the GQA amortisation the tile
exists for. Of the candidate policies only SUM (equivalently MEAN — a uniform positive
scale is order-preserving within a row) is expressible in ggml: there is no row-max-value
op (`SUM_ROWS`/`CUMSUM`/`MEAN`/`ARGMAX` only, and `ARGMAX` returns an index), and a union
of per-head top-k has data-dependent length, which is the same stream-compaction blocker
as `find_blocks` and violates fixed `n_sel`. The SUM chain, for the record:

```c
a4  = ggml_reshape_4d(ctx, attn_sum, NBk, NBq, G, Hkv);   // free view; h = kv_head*G + g
ap  = ggml_permute   (ctx, a4, 1, 2, 0, 3);               // [G, NBk, NBq, Hkv]
ac  = ggml_cont      (ctx, ap);                           // LOAD-BEARING: hexagon argsort
                                                          // has no contiguity gate and
                                                          // addresses rows as data + r*nb[1]
ah  = ggml_sum_rows  (ctx, ac);                           // [1, NBk, NBq, Hkv]
as  = ggml_reshape_3d(ctx, ah, NBk, NBq, Hkv);
sel = ggml_argsort_top_k(ctx, as, n_sel);                 // I32 [n_sel, NBq, Hkv, 1]
fa->src[5] = sel; fa->op_params[4] = Bl; fa->op_params[5] = Bl;
```

`ggml_top_k` is not usable — `GGML_OP_TOP_K` has no case in the hexagon `supports_op`
switch, so it would split the graph. This chain is **untested here**: it can never be
NMSE-checked against CPU (CPU flash attention ignores `src[5]` and computes dense
attention), and argsort tie-breaking differences flip ranks. Note also that argsort over a
full `NBk` row hands an early query block *future* key blocks as soon as `n_sel` exceeds
its causally-allowed count — the scorer's reduced causal mask only drives those scores to
`-1e30f`, it does not remove them from the ranking. Only the FA mask `src[3]` stops them
being attended.

## Reproduce

```sh
cmake --build build-sparse --target test-backend-ops -j 24
adb push build-sparse/bin/test-backend-ops /data/local/tmp/llama.cpp/bin/

D=/data/local/tmp/llama.cpp
R="cd $D && ADSP_LIBRARY_PATH=$D/lib LD_LIBRARY_PATH=./lib"

adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_SCORE"       # stage 0, vs CPU
adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_ROWSUM"      # stage 3, invariant
adb shell "$R ./bin/test-backend-ops perf -b HTP0 -o XATTN_SCORE_MM"    # stage 1, replicated
adb shell "$R ./bin/test-backend-ops perf -b HTP0 -o XATTN_REVERSE"     # stage 2, GET_ROWS
adb shell "$R ./bin/test-backend-ops perf -b HTP0 -o XATTN_SCORE_MM_WG" # stage 4, per-call constant
```

## Open questions

- **`GET_ROWS` cost on HTP** for `[d, Lq, Hq]` F32. It takes the HVX row-copy path, not DMA
  (`use_dma` needs `ne00 ≥ 2048`; `d = 128`). If it dominates, the escape hatch is the
  reference's own copy-free form: `S` plain `d`-deep GEMMs over strided views,
  `A = Σ_q Qview_{S-1-q} @ Kview_qᵀ` (`xattn/src/kernels.py:217-228`), which needs no
  `get_rows`, no `cont`, and no contiguity on the packed axis — and so also sidesteps the
  KV-cache reshape hazard. It costs `3S-1` nodes instead of 4 and `ne00 = d` instead of
  `S·d`. Whether the HMX kernel honours a non-unit `nb[1]` on its operands is unverified;
  the host gate accepts it, the kernel was not read. Stage 2 measures the thing this
  decision turns on.
- **The permute+cont** is a scalar per-element copy on HTP (a permuted source fails both
  vectorized paths in `cpy_thread_f32_reshape`). Bounded at `Nq·NBk·Hq` f32, unmeasured,
  and the strongest argument for eventually fusing the softmax-through-pool tail.
- **Selection recall at `Bl = 64`.** The paper only ever uses `Bl = 128`; the `bs = 64`
  used everywhere in this tree is half its smallest value, and pooling is proportionally
  more aggressive. Unmeasured here.
- **fp16 accumulation.** This graph is F16 K × F32 Q with F32 accumulate, so it is safe.
  Any HMX-staged kernel that accumulates in fp16 over `S·d = 1024` is not: mllm measured
  raw logits reaching ~1e7 against fp16's 65504 and fixed it by pre-scaling both operands
  by `√temp`.
- **Ragged `L`.** `L % Bl != 0` or `L % S != 0` is not expressible — all three reshapes
  assert exact element counts. Zero padding is *not* neutral (a zero row gives logit 0, not
  `-inf`, changing every softmax denominator), so a padding path must mask explicitly.

## Expectation setting

mllm's end-to-end verdict on the same class of design was **negative**: with both paths
correct on one device, dense beat block-sparse + XAttention by 1.2–1.34× at `Sq` = 512 and
1024, with a crossover only at `Sq ≥ 2048`. What beat them was not the scoring FLOPs — it
was the per-query-block structure forcing attention out of one fused graph into many small
dispatches. That is exactly the structure a faithful implementation has. Landing scoring
alone is therefore not timidity; it is the only step whose value does not depend on that
unresolved question.

## Optimising the scorer: the reversal is the wrong operand

Profiled at the shape a chunked prefill actually scores -- `Lq=512` against every key so far,
not the square `Lq=Lk` used earlier. That changes the picture completely (S=16,
replicated-equivalent µs):

| Lk | total | matmul | **reversal** | epilogue |
|--:|--:|--:|--:|--:|
| 512 | 620 | 233 (38%) | **264 (43%)** | 123 (20%) |
| 1024 | 779 | 329 (42%) | **264 (34%)** | 187 (24%) |
| 2048 | 1052 | 534 (51%) | **261 (25%)** | 257 (24%) |

The antidiagonal reversal -- `ggml_get_rows(q, antidiag_idx)` -- is a flat ~262 µs and the
LARGEST term at short context. The epilogue is only 20-24% here, not the 61-74% measured on
the square shape, because `Lq` is pinned so the epilogue's `Nq` axis never grows.

Also worth recording: **`try_fuse_xattn_score` can never fire on this graph.** It rejects any
softmax with a mask (`ggml-hexagon.cpp`, `sm->src[1] != nullptr`) and the real scorer passes a
causal mask, so the HMX-staged fusion and its 1.12-1.58x only ever applied to the QSUB stand-in.

### Either operand can carry the reversal

Entry `(rq, rk)` must be `sum_j Q[a_j + rq*S] . K[b_j + rk*S]` with `a_j + b_j == S-1`.
Reversing Q gives `(S-1-j, j)`; reversing K gives `(j, S-1-j)`. Substituting `j' = S-1-j` maps
one onto the other exactly -- the matrices are **identical**, not merely equivalent. Confirmed
on device: `XATTN_SCORE` 8/8 with the portable K-side variant.

| Lk | reversal alone: revQ / revK portable / revK direct-F16 | full scoring: revQ / revK-F16 | |
|--:|--:|--:|--:|
| 512 | 262 / 91 / **48** | 605 / **472** | **1.28x** |
| 1024 | 264 / 220 / **102** | 778 / **671** | **1.16x** |
| 2048 | 262 / 465 / **261** | 1049 / 1065 | 1.00x |

Two things fall out.

**A type-taking `get_rows` is worth 2-4x on this op alone.** `ggml_get_rows` hardcodes an F32
result, so the portable K-side form pays a cast back to F16 and is 1.8-2.2x slower than the
hand-built F16 node (which ggml-hexagon's F16->F16 GET_ROWS already supports). The direct form
is perf-only here because ggml-cpu's `get_rows_f16` writes F32
(`ggml_cpu_fp16_to_fp32` into a `float *`, `ops.cpp:4925`), so a CPU reference miscomputes it
and an eval row would fail on the harness rather than the kernel. `rev_k=1` validates the
identical algebra portably; `rev_k=2` measures its speed.

**The K-side reversal scales with `Lk` while the Q-side is fixed**, so it wins outright only
below `Lk ~ 2048`. But that is the wrong way to deploy it: **K is the KV cache -- written once,
read by every later chunk -- so the permutation belongs at cache-write time**, amortised across
every chunk that reads it, instead of being redone per chunk. Q then needs no reversal at all:

| Lk | per-chunk scoring, reversal amortised away | |
|--:|--:|--:|
| 512 | 605 -> 343 | **1.76x** |
| 1024 | 778 -> 514 | **1.51x** |
| 2048 | 1049 -> 787 | **1.33x** |

That is the largest single win available in the scorer, and it needs no kernel work -- only
storing K in antidiagonal-packed order.

### What is left after that

At `Lk=2048` with the reversal gone: matmul 534 (68%), epilogue 257 (32%). The matmul runs at
**502 GFLOP/s against the 1.71 TFLOP/s the same op reaches on the square shape** -- `Nq = Lq/S`
is only 32 rows at `Lq=512, S=16`, so it is tall-skinny and starved, the same shape pathology
the original stand-in had. Raising `S` makes it worse, not better (`Nq` shrinks); `S=8` gets
926 GFLOP/s but loses overall on 2x the FLOPs.

## Relocating the reversal to CPU: measured, and it loses

The scorer splits cleanly into work each unit is good at -- HTP wins the reduced matmul 3.7x
(233 vs 862 µs), CPU wins the antidiagonal reversal 2.5x (107 vs 272 µs, both 8-threaded and
bandwidth-bound). That looks like an obvious split, so `examples/xattn-split` was built to
measure it: every tensor allocated in ONE hexagon rpcmem region, the two backends dispatched by
hand, so placement is proven by construction rather than by scheduler debug output.

**It loses, and not narrowly.** Device 87b3a4aa, Lq=512, S=16, min of 20 whole-graph iterations:

| Lk | all-HTP | A: cpu(rev) then htp(rest) | | overlap ceiling* | correct pipeline | |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 993 | 1560 | **0.64x** | 593 | 1470 | 0.68x |
| 1024 | 1254 | 1765 | **0.71x** | 1148 | 2457 | 0.51x |
| 2048 | 1697 | 2629 | **0.65x** | 1397 | 2753 | 0.62x |

\* dependency deliberately broken, so its results are garbage -- an upper bound only.

NMSE of both the split and the pipeline against the all-HTP result is **0.0**, so the algebra and
the CPU/DSP coherency are both correct. This is purely a performance result.

> **CORRECTION (measured after the fact).** The magnitudes below are inflated roughly 2.2x by
> an uncontrolled variable: CPU thread placement. This SoC has 6 cores at 2.78 GHz and 2 prime
> cores at 4.09 GHz, all on the `walt` governor and observed idling at 1.0-2.4 GHz. The CPU
> backend's 8 threads spread across the little cores; pinning the process to the two prime cores
> -- with 2 threads instead of 8 -- more than halves the reversal:
>
> | | cpu reversal | split (A) | all-HTP | |
> |:--|--:|--:|--:|--:|
> | default, 8 threads | 797 us | 1538 | 1013 | 0.66x |
> | `taskset c0`, 2 prime cores | **366 us** | 1030 | 1003 | **0.97x** |
>
> Stable to +-1% over three runs each. So the split is **break-even, not a 1.5x loss**, and the
> "rpcmem is intrinsically slow for the CPU" reading is mostly wrong: with threads on prime
> cores the R1 ratio (cpu-over-rpcmem vs cpu-over-malloc) falls from 1.89 to **1.05**.
>
> What survives the correction is narrower but still real: a **post-DSP degradation**. Even
> pinned, the reversal costs 101 us on a buffer the DSP has never touched and 366 us on one it
> has -- 3.6x, persistent across iterations with no DSP activity in between. That effect is
> genuine; its previously reported size was not.
>
> The wider lesson is the measurement one. Every CPU number in this document taken without
> pinning is a measurement of the Linux scheduler as much as of the code.

### Why: the CPU cannot cheaply compute over rpcmem

The zero-copy premise was that hexagon buffers are host memory -- `rpcmem_alloc2` with
`RPCMEM_HEAP_ID_SYSTEM`, `buffer_type_is_host` returns true -- so the CPU can work over them with
no boundary copy. It can, correctly. It just cannot do it *fast*:

| CPU reversal, identical graph | time | |
|:--|--:|--:|
| over malloc'd memory | **105 µs** | baseline |
| over hexagon rpcmem, HTP idle | **242 µs** | 2.3x |
| over hexagon rpcmem, after HTP has executed on it | **793-1068 µs** | 7.5-10x |

Two effects stack: rpcmem is intrinsically slower for CPU access than ordinary memory, and it
degrades a further 3-4x once the DSP has run over the same region. The second effect is the one
that kills it, and it is invisible to any benchmark that measures the CPU in isolation -- which
is exactly how the original 107 µs figure was obtained.

At 793-1068 µs the CPU is no longer faster than HTP's 272 µs at this op; it is 3-4x slower. The
premise of the whole split evaporates.

The alternative -- allocate on CPU and copy across the boundary -- was priced before building:
`q_antidiag` is 4.19 MB, so the copy moves 8.39 MB single-threaded and needs >50 GB/s to fit
inside the 165 µs of headroom, against ~10-20 GB/s realistic single-core. That route loses too,
by 3-5x.

### What this means more generally

This is the second heterogeneous-pipeline idea in this work to die on measurement -- the first
was offloading the block gather, priced at a 1.02-1.11x ceiling. Both died for the same
underlying reason: **on this platform the cost of sharing a buffer between CPU and DSP exceeds
the compute being offloaded.** Any future split should be gated on the R1 measurement in
`examples/xattn-split` -- CPU-over-rpcmem versus CPU-over-malloc, taken *after* the DSP has run --
before any implementation work. The harness prints STOP on its own when that ratio is bad.
