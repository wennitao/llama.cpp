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
exists for.

> **Correction (superseded).** An earlier version of this section claimed that "of the
> candidate policies only SUM is expressible in ggml". That is wrong, and the policy it
> forced was the wrong one. **MAX is expressible, and MAX is what the kernel's addressing
> actually calls for.** The implemented pipeline is described in
> [the section below](#from-scores-to-a-selection-implemented).

The reduction is forced because `sel` is read as
`sel + qb*sel_nb1 + kv_head*sel_nb2 + ib3*sel_nb3`, then `list[c*m + j]`
(`htp/flash-attn-ops.c:242-256`) — there is **no query-head term**. One list therefore
serves all `G` query heads of a KV head, and the object it has to name is the *union* of
the `G` per-head top-k sets. SUM (equivalently MEAN — a uniform positive scale is
order-preserving within a row) ranks by *average* demand, which is a different and worse
surrogate for that union: a block taking half of one head's mass and nothing from the
other `G-1` loses to a block taking a tenth from each, yet dropping the first costs that
head half its attention while dropping the second costs every head a tenth. The marginal
*cost* of keeping a block is identical either way — it is staged once and shared — so
ranking by max is ranking by the worst per-head loss avoided, which is the metric
XAttention's own recall is defined on.

Max is well-posed rather than dominated by whichever row happens to be largest because
every `(head, query block)` row of `attn_sum` sums to exactly `P = Bl/S` — the invariant
`XATTN_ROWSUM` checks against the analytic constant — so all `Hq*NBq` rows sit on one
common scale.

What is true is that ggml has no *op* for it: the reductions are `SUM`/`SUM_ROWS`/
`CUMSUM`/`MEAN`/`ARGMAX` (`ggml.h:497-501`, and `ARGMAX` returns an index),
`GGML_OP_POOL_MAX` has no case in hexagon's `supports_op`, and its unary switch has no
`RELU` or `ABS` (`ggml-hexagon.cpp:4471-4487`), so `relu(a-b)+b` and
`0.5(a+b+|a-b|)` both split the graph. But it does have `CLAMP`, and

```c
max(a, b) == ggml_add(ctx, ggml_clamp(ctx, ggml_sub(ctx, a, b), 0.0f, FLT_MAX), b)
```

is three HTP-resident nodes (`SUB` `:3649`, `CLAMP` `:3665`, `ADD` `:3651`). A
`log2(G)`-step halving tree over `ne3` then reduces the head axis. Note `ggml_clamp` is
unconditionally **in-place** (`ggml.c:4443-4457` builds its result with
`ggml_view_tensor`, with a `TODO` admitting it), so the `CLAMP` node aliases the `SUB`
output — safe here only because that output is dead the instant it is clamped.

`ggml_top_k` is still not usable — `GGML_OP_TOP_K` has no case in the hexagon
`supports_op` switch, so it would split the graph; `ggml_argsort_top_k` is
`ggml_argsort` plus a strided view and stays on device.

Note that argsort over a full `NBk` row hands an early query block *future* key blocks as
soon as `u` exceeds its causally-allowed count — the scorer's reduced causal mask only
drives those scores to `-1e30f`, it does not remove them from the ranking. Only the FA
mask `src[3]` stops them being attended, and the `bias` leaf described below is what keeps
them out of the top-`u` in the first place.

## From scores to a selection, implemented

`tests/test-backend-ops.cpp` now carries the full chain: `xattn_build_selection()` builds
`attn_sum -> mh -> mr -> +bias -> argsort -> sel` and `test_xattn_e2e` hands `sel` to
`ggml_flash_attn_ext` through `src[5]` as a **computed node**, not a leaf.

```
attn_sum  F32 [NBk, NBq_s, Hq]           (test_xattn_score stage 0, rev_k = 0)
stage A   max over the G query heads     -> mh  [NBk, NBq_s, Hkv, 1]
stage B   max over the R scorer blocks   -> mr  [NBk, NBq_a, Hkv, 1]   (R = Bq/Bl)
stage C   + bias, argsort DESC, view u   -> sel I32 [u, NBq_a, Hkv, 1], nb[0] = 4
```

Both reduces split their axis as `(fast = the thing being reduced)`, which makes the
reshape free: the query-head index is the fast half of `Hq` (the kernel takes head
`kv_head*G + j`, `htp/flash-attn-ops.c:2887`) and the scorer block index is the fast half
of `NBq_s` (`a_s = a_a*R + jl`). `G` and `R` must be powers of two, which costs nothing
real: `br_unit = ceil(32/G)` must divide `Bq`, and for `G ∈ {3,5,6,7}` it gives 11/7/6/5,
none of which divides 64 or 128, so those `G` already cannot run per-query-block sparse FA
at all.

A contiguous argsort source is **load-bearing**, and the bias `ADD` is what supplies it —
its dst is freshly allocated, which is the only reason no explicit `ggml_cont` sits there.
`ggml_hexagon_supported_argsort` checks only F32-in / I32-out / `ne0 <= 16K`
(`:3369-3389`) while both device kernels address row `r` as `data + r*nb[1]` over the
flattened `ne1*ne2*ne3`, so a permuted or strided source is silently **mis-sorted rather
than rejected**. Do not restructure the chain so the argsort's source becomes a view.

`NBk = 16` (e.g. `Lk = 2048, Bl = 128`) misses the HVX bitonic dispatch table
`{32,64,128,256,512,1024}` and takes the scalar quicksort — still on device, still no
graph split, and a scalar sort of 16 floats across `NBq_a * Hkv` rows is noise next to the
FA op. Do **not** extend the table (a new bitonic network is easy to get subtly wrong and
argsort has no contiguity gate to fail closed against), and do **not** pad the row: a pad
column would produce an index `>= NBk`, which the FA kernel clamps to `NBk-1`, i.e. a
silent duplicate.

### The `bias` leaf is not test scaffolding

`bias` is `F32 [NBk, NBq_a, 1, 1]`, broadcast over the KV-head axis by
`ggml_can_repeat`. It carries three deployment jobs:

1. `-BIG` on causally impossible blocks (`b >= (Lk-Lq)/Bl + (a+1)*R`). Masked cells are
   **present** in the ranking, not absent: HTP's scorer softmax clamps its exponent at
   `-88`, so a `-1e30` logit becomes `~6e-39` — subnormal, effectively zero, but sortable;
   CPU gives exactly `0`. Either way the future-block tail is a mass of near-exact ties,
   and without this term a real block with genuinely small mass can lose its slot to one.
2. `+BIG` on block 0 (the sink) and on the last causally available block (the diagonal).
   This is what the reference's `find_blocks` does unconditionally, and it is what
   guarantees no query row's mask comes out all `-INF` — the FA kernel's `-inf` sentinel
   is a *finite* `-65504` and a fully-masked FIRST chunk gives every column
   `p = exp2(0) = 1`, inflating `l` by the chunk width until a later real chunk rescales it.
3. A strictly decreasing ramp `-eps*b` (`eps = 1e-5`), far above the `~1e-39` masked floor
   and far below any real score separation, so the residual ordering is deterministic and
   identical on both backends and ties break sink-ward.

### Choosing `u`

`hmx_fa_find_chunk_size` requires `m = Bc/bs` to divide `sel->ne[0]`, with `m <= 8`
(`htp/flash-attn-ops.h:440-446`), so the achievable chunk counts are exactly
`{u/m : m | u, m <= 8}`. A **prime `u` collapses that set to `{u}`** — one chunk per
block. Measured with a leaf selection, `u=17` costs 2.48x `u=18` and `u=29` does 10% less
work than `u=32` while taking 2.83x longer. **Choose `u ≡ 0 (mod 8)`**, which leaves
`m ∈ {1,2,4,8}` all legal. Note the search does *not* simply take the largest legal `Bc`:
its cost model prices thread loss (`actual_threads` collapses to 1 below 3 KV blocks), so
at `G=2, Br=64` four chunks beats two. Never predict `Bc`/`Br` — read them from
`HEX_VERBOSE`.

Two further couplings: `kv_effective = sel->ne[0] * bs` drives the entire chunk-size
search (`ggml-hexagon.cpp:2094`), so **a `u` sweep is not a one-variable experiment** —
`Br`, `Bc`, thread count, pipelining and VTCM all move together; and `u > NBk` silently
drops the whole FA op to a dense CPU backend (`:2271`), which would time something else
entirely.

### When a query block has fewer than `u` causally available blocks

`n_avail(a) = (Lk-Lq)/Bl + (a+1)*R`. Per-query-block clamping is **impossible**: `n_sel`
is one `uint16_t` for the whole op, and `fa_chunk_nblk` deliberately does not take the
query block, because the untagged DMA FIFO's push and pop sites need not see the same one
(`htp/flash-attn-ops.c:217-231`). A computed `sel` must be fixed-width *by construction* —
and padding is no escape either, since an out-of-range index is clamped to
`n_blk_total-1`, i.e. becomes a duplicate.

It binds more narrowly than it looks. At the primary shape (`Lq=512, Lk=2048, Bl=64,
R=8`) the offset term is `1536/64 = 24`, so `n_avail(0) = 32 = NBk` and there is no
shortage at all. It binds only when `Lk ≈ Lq` — the first ubatch of a prefill — which is
exactly when `NBk` is small and sparsity has nothing to offer anyway.

When it does bind, the disposition is quiet degradation, not failure: the `-BIG` term
makes the selection prefer every real block over every future one, the `+BIG` sink term
guarantees at least two causally valid blocks per row, and the residual slots go to future
blocks that `src[3]` fully masks. **Every eval arm is registered outside this regime**
(`u <= n_avail(0)`, asserted at `initialize_tensors`); an arm inside it would be checking
undefined tie-break behaviour.

### Correctness: four arms, and what the conjunction does *not* prove

`ggml-cpu`'s flash attention reads no `src[5]` at all, so a CPU reference computes
**dense** attention — which is exactly why the QSUB stage-0 arm is perf-only at 0/3. Each
arm below says how it closes that gap.

| arm | `-o` | what it decides |
|---|---|---|
| 0 | `XATTN_MERGE` | the merged score vs CPU by NMSE, no argsort, zero tie exposure. Plus a structural invariant checked on each backend *alone*: `mr` is `> 0` exactly on the causally reachable blocks. |
| 0b | `XATTN_MERGE_MASS` | `sum_rows(mh)` against the analytic constant `P` on **both** backends (sharp only at `G == 1`; above that the invariant is the band `[P, G*P]`). |
| 1 | `XATTN_SELECT` | the full argsort, checked as a SET. Planted rows check both backends against an analytically known set; unplanted rows check the only thing decidable without a plant — that the device's own prefix is a true top-`u` of the device's own scores. Registered at `NBk = 32` **and** `NBk = 16`, which are two entirely different device implementations (HVX bitonic vs scalar quicksort, `htp/argsort-ops.c:487-508`). |
| 2 | `XATTN_E2E` (`plant=0`, `u = NBk`) | sparse attention over *all* blocks is dense attention, so a dense CPU reference agrees for **any** permutation the argsort emits. Tie-immune, end-to-end, and the only arm that runs the fully shared graph. Also the first exercise of `sel->ne[2] == Hkv` with `nr > 1`. |
| 3 | `XATTN_E2E` (`plant=1`) | the deployable density, ranked by the real scorer, with the `-INF` mask built from the analytically known set. `G == 1`, because a per-KV-head mask is only expressible when `nr == 1`. |

The plant rigs `Q[:, l, hq] = sqrt(d)·e_dir` (`dir = (l/Bl)·Hq + hq`) and makes `K`
constant within a key block, so the reduced dot is `S·sqrt(d)·g[b][dir]` and
`soft_max_ext`'s `1/(sqrt(d)·S)` scale turns it into exactly `g[b][dir]`: the
`(scorer query block, key block, query head)` logit matrix is directly programmable and
`attn_sum` is a closed form. `g` is a geometric ramp of total range `1e4`, so adjacent
ranks are separated by 1.35x at `NBk = 32` — three orders of magnitude above any backend
disagreement, which is what makes the winning SET independent of tie-breaking. The
construction *asserts* its own margin at `initialize_tensors` time and aborts rather than
ship a flaky row.

Verified locally, CPU-only, by breaking each reduce and confirming the arm goes red:
maxing the KV-head axis instead of the query-head axis, and maxing the query-block axis
instead of the scorer-block axis, are both caught by `XATTN_SELECT` with `plant=1`.

**Not proved by any of them:** whether the selection is any *good* (recall, perplexity,
needle-in-haystack are unreachable from `test-backend-ops`); index-for-index agreement
under unplanted scores (undecidable — rank is discontinuous, the two backends' scores
differ, and all three sort implementations are unstable); the antidiagonal reversal in the
planted arm (`K` is constant within a block, so the reversal is invisible there — stages 0
and 2 of `XATTN_SCORE` cover it); the estimator's own correctness (the reduced causal mask
deliberately admits `floor(S/2)` non-causal products per diagonal tile); or the measured
*perf* configuration, which is maskless while every eval arm carries a mask or runs at
full density.

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

# scores -> selection -> sparse FA. Run in this order: XATTN_MERGE first, because it is
# the cheapest confirmation that the max tree runs on HTP at all (the in-place CLAMP
# aliases its source, which no other in-tree graph does on this backend).
adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_MERGE"       # arm 0
adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_MERGE_MASS"  # arm 0b
adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_SELECT"      # arm 1, both argsorts
adb shell "$R ./bin/test-backend-ops test -b HTP0 -o XATTN_E2E"         # arms 2 and 3

# the D / P / P' / E / S sweep. GGML_HEXAGON_VERBOSE on every row: without
# Br / Bc / n_kv_blocks / n_threads / pipeline / vtcm the u sweep is uninterpretable,
# because kv_effective = u*bs changes the whole kernel configuration.
adb shell "$R GGML_HEXAGON_VERBOSE=1 ./bin/test-backend-ops perf -b HTP0"
```

Judge every correctness run by failing-case **class**, never by count: the count churns
+-16 run to run and this device's A/A gate shows up to 21% timing drift. The ledger to
hold is `FLASH_ATTN_EXT` ~2166-2183/2196 with **all** failures `sinks=1`,
`FLASH_ATTN_EXT_SPARSE 46/46`, `XATTN_SCORE 8/8`, `FLASH_ATTN_EXT_GATHER 2/2`,
`QSUB_BLOCK_SCORES 3/3`. Capture the failing `vars()` strings before and after and diff
the identity set — counting passes cannot distinguish "broke two, fixed two" from
"changed nothing".

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
not the square `Lq=Lk` used earlier. That changes the picture completely (S=16, whole-graph
minus the measured C = 620 µs per-call constant; matmul and reversal are replicated rows):

| Lk | total | matmul | **reversal** | epilogue |
|--:|--:|--:|--:|--:|
| 512 | 702 | 233 (33%) | **261 (37%)** | 208 (30%) |
| 1024 | 863 | 332 (38%) | 262 (30%) | 269 (31%) |
| 2048 | 1355 | 533 (39%) | 276 (20%) | **546 (40%)** |
| 4096 | 2366 | 931 (39%) | 262 (11%) | **1173 (50%)** |

The antidiagonal reversal -- `ggml_get_rows(q, antidiag_idx)` -- is a flat ~262 µs, so it is
the largest single term at `Lk=512` and irrelevant by `Lk=4096`. The matmul holds a steady
~39%. The epilogue (softmax + both `sum_rows`) is the term that grows, and past `Lk=2048` it
is the largest.

> This table was previously published with the totals divided by the 1.605 slope, which
> understated the epilogue at every row (123/187/257 instead of 208/269/546) and made the
> reversal look like the dominant cost throughout. The slope does not exist at these scales;
> see `xattn-block-selection.md`, "CORRECTION: the 1.605 slope does not exist at scale". The
> matmul and reversal columns were always replicated rows and are unchanged.

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

## The context-length curve, 512 to 4096

The first table in this work where every column is a directly measured quantity at a
matched shape, with the per-call constant handled by subtraction rather than by a fit.

Configuration: `d=128`, `Hq=16`, `Hkv=8` (G=2), `S=16`, `Bl=64`, density held at **50%**
(`u = NBk/2` = 4/8/16/32), `R = Lq/Bl` so `Bq = Lq` -- one selection for the whole chunk.
`FLASH_ATTN_EXT` (dense) is a replicated row; `XATTN_E2E` and `XATTN_SELECT` are
whole-graph and have C = 620 µs subtracted. Dense is quoted as `D_wg - C` so both sides
sit on the same basis. `attn = E - S`, in which C cancels exactly.

| kv | Lq | u | dense | sparse e2e | selection | attention | **D/E** | D/attn | sel share |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 512 | 4 | 919 | 1929 | 921 | 1008 | **0.48x** | 0.91x | 48% |
| 1024 | 512 | 8 | 1218 | 2207 | 1144 | 1063 | **0.55x** | 1.15x | 52% |
| 1024 | 1024 | 8 | 2388 | 3802 | 1927 | 1875 | **0.63x** | 1.27x | 51% |
| 2048 | 512 | 16 | 2001 | 3003 | 1548 | 1455 | **0.67x** | 1.37x | 52% |
| 2048 | 2048 | 16 | 8628 | 9726 | 4647 | 5079 | **0.89x** | 1.70x | 48% |
| 4096 | 512 | 32 | 4216 | 4712 | 2560 | 2152 | **0.89x** | 1.96x | 54% |
| 4096 | 2048 | 32 | 15962 | 15949 | 7518 | 8431 | **1.00x** | 1.89x | 47% |

At 25% density (`u=16`) at kv=4096, which the measured union saturation supports:

| kv | Lq | u | dense | sparse e2e | selection | attention | **D/E** | D/attn |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4096 | 512 | 16 | 4216 | 3947 | 2560 | 1387 | **1.07x** | 3.04x |
| 4096 | 2048 | 16 | 15962 | 12671 | 7518 | 5153 | **1.26x** | 3.10x |

Three things this settles:

1. **The attention kernel is not the problem.** `D/attn` reaches 1.9x at 50% density and
   3.1x at 25% -- at 50% density 1.9x is already past the 2.0x the FLOP count allows once
   the per-query-block tiling penalty is paid, and it improves monotonically with kv.
2. **Selection costs about half the pass at every context length**, 47-54%, and the share
   is remarkably flat. It is the whole reason `D/E` is below 1 for most of the table.
3. **Break-even is at kv=4096 for 50% density and kv~2048-4096 for 25%.** Below that,
   dense wins end-to-end. This matches mllm's own crossover at `Sq >= 2048`.

### Where the selection time goes

Nested differences from the stage ladder (`XATTN_REVERSE` and `XATTN_SCORE_MM` replicated;
`XATTN_SCORE`, `XATTN_MERGE`, `XATTN_SELECT` whole-graph, C subtracted once at the
innermost step and cancelling in every difference above it):

| kv | Lq | reversal | matmul | softmax+sum | head/block merge | argsort | total |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 512 | 261 | 233 | 204 | 142 | 81 | 921 |
| 1024 | 512 | 262 | 332 | 264 | 277 | 9 | 1144 |
| 1024 | 1024 | 558 | 489 | 638 | 221 | 20 | 1927 |
| 2048 | 512 | 276 | 533 | 541 | 178 | 20 | 1548 |
| 2048 | 2048 | 1053 | 992 | 2294 | 351 | -44 | 4647 |
| 4096 | 512 | 262 | 931 | 1168 | 162 | 37 | 2560 |
| 4096 | 2048 | 1071 | 1455 | 4674 | 357 | -39 | 7518 |

- **`softmax + sum_rows` is the largest term** at every shape past kv=512, and at the
  largest shape it is **3.2x the matmul**. The matmul is 19-32% of selection and never
  the bottleneck -- which is what the fusion work already implied and this now prices
  end-to-end.
- The **reversal** is flat ~262 µs at `Lq=512` regardless of `Lk` (it touches only Q) and
  ~1060 µs at `Lq=2048`; it scales with `Lq` alone.
- **argsort is free**, 0-81 µs, and at the two largest shapes it measures slightly
  negative, i.e. below run-to-run noise. The `find_blocks` threshold that XAttention uses
  was replaced by a fixed top-k for simplicity, and this says nothing was lost by it.
- **head/block merge** (Hq->Hkv max, then the R-block max) is 150-360 µs and roughly
  independent of everything.

So the one thing worth optimising in the selection leg is the softmax + reduction
epilogue: it is ~50% of selection, which is ~50% of the pass, i.e. ~25% of the whole
end-to-end cost. `try_fuse_xattn_score` was written to remove exactly this and measured
1.12-1.58x on the stand-in, but it **cannot fire on the real scorer** -- it rejects any
masked softmax (`ggml-hexagon.cpp:3823`) and real XAttention's softmax carries the reduced
causal mask. Teaching the fusion to accept a mask is the highest-value remaining item,
and it is worth roughly 12-15% of the end-to-end pass.

### The attention kernel alone

Scoring excluded entirely. Every number below is a **replicated** row on both sides, so
there is no per-call constant anywhere in this table and nothing is subtracted or
modelled. It is the cleanest comparison in this work, and it is the one that says
whether the kernel itself is finished.

`IDEAL` is dense FA at `kv_eff = u*64` -- what the selected KV would cost if it were
simply a shorter contiguous cache: no index list, no indirection, no query-block
constraint. `ceil = DENSE/IDEAL` is therefore the most any block-sparse kernel can
achieve at that density, and `%ceil` is how much of it the kernel gets.

`shared` = one selection for the whole chunk. `bq=128`/`bq=64` = per-query-block
selection at that query-block size; **bq=64 is the faithful XAttention shape**.

| kv | nb | u | density | DENSE | IDEAL | ceil | shared | | bq=128 | | bq=64 | | %ceil sh | %ceil 64 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 512 | 4 | 50% | 933 | 930 | 1.00x | 922 | 1.01x | 1370 | 0.68x | 1697 | 0.55x | 101% | 55% |
| 1024 | 512 | 8 | 50% | 1247 | 933 | 1.34x | 988 | 1.26x | 1486 | 0.84x | 1873 | 0.67x | 95% | 50% |
| 2048 | 512 | 16 | 50% | 2082 | 1247 | 1.67x | 1365 | 1.52x | 2153 | 0.97x | 2618 | 0.80x | 91% | 48% |
| 4096 | 512 | 32 | 50% | 4295 | 2082 | 2.06x | 2100 | 2.04x | 3312 | 1.30x | 4131 | 1.04x | 99% | 50% |
| 2048 | 2048 | 16 | 50% | 8796 | 4989 | 1.76x | 5181 | 1.70x | 8856 | 0.99x | 10718 | 0.82x | 96% | 47% |
| 4096 | 2048 | 32 | 50% | 16302 | 8796 | 1.85x | 8654 | 1.88x | 13483 | 1.21x | 16834 | 0.97x | 102% | 52% |
| 1024 | 512 | 4 | 25% | 1247 | 930 | 1.34x | 935 | 1.33x | - | - | 1683 | 0.74x | 99% | 55% |
| 2048 | 512 | 8 | 25% | 2082 | 933 | 2.23x | 985 | 2.11x | 1490 | 1.40x | 1875 | 1.11x | 95% | 50% |
| 4096 | 512 | 16 | 25% | 4295 | 1247 | 3.44x | 1358 | 3.16x | 2153 | 2.00x | 2627 | 1.63x | 92% | 47% |
| 2048 | 2048 | 8 | 25% | 8796 | 3549 | 2.48x | 3690 | 2.38x | 6142 | 1.43x | 7704 | 1.14x | 96% | 46% |
| 4096 | 2048 | 16 | 25% | 16302 | 4989 | 3.27x | 5243 | 3.11x | 8870 | 1.84x | 10696 | 1.52x | 95% | 47% |

**1. Block-sparse indexing is free.** With a shared selection the kernel runs at
**91-102% of the contiguous-dense ceiling** at every one of the eleven points -- twice
slightly *above* it, which is within run-to-run spread. The `src[5]` index list is
consumed as a DMA source offset and never materialises a gather, and this table is the
price of that design: nothing. There is no remaining work in the indexing path.

**2. Per-query-block selection costs a flat ~1.95x, and that is the entire problem.**

| | bq=128 | bq=64 |
|:--|--:|--:|
| tax over shared, all 10 points | 1.49-1.71x | 1.84-2.09x |

It does not move with context length, density, or chunk size -- it is a pure scheduling
tax, exactly as the earlier attribution said (rendezvous stall 43%, softmax thread loss
32%, extra K/V staging 17%). At `bq=64` the kernel gets **46-55% of ceiling** everywhere.
Recovering that factor of two is worth more than any other single item in this project:
it is worth as much as halving the density, and it costs no accuracy at all.

**3. The ceiling is lower than the FLOP ratio, because the dense kernel has a
Q-proportional floor.** Dense FA at `nb=512` measures 930 µs at `kv=256` and 933 µs at
`kv=512` -- 2x the KV work for 0.4% more time. At `nb=2048` the same floor is ~3378 µs.
Normalised that is ~1.7 µs per query token, and it is irreducible: it is Q staging, the
softmax/output pass and the writeback, none of which sparsity touches. So 50% density
buys 1.00x at kv=512 and only 2.06x at kv=4096, never the 2x the FLOP count suggests at
short context. Choosing density against this table rather than against the FLOP ratio is
the difference between a real 3.4x ceiling (25% at kv=4096) and an imagined 4x.

Below `kv=256` it gets worse, not better: `kv=128` costs 1360 µs and `kv=64` costs 1266,
both *slower* than `kv=256`, because `FA_MIN_KV_BLOCKS 3` and the chunk-count rule leave
no legal tiling. Nothing is to be gained by selecting fewer than ~4 blocks.

### What this means for the end-to-end table

The `XATTN_E2E` arm above is registered with `R = Lq/Bl`, i.e. `Bq = Lq`: **one selection
for the whole chunk**, which is the `shared` column here. Cross-checking the two methods,
`E - S` against a directly measured replicated row, they agree to 2-9% (2% at the large
shapes, 9% at kv=512), which independently validates the difference-of-whole-graph-rows
method used throughout this doc.

But it also means the end-to-end figures are the optimistic case. Composing this table's
`bq` tax with the measured selection cost, at kv=4096 / Lq=2048 / 25% density:

| bq | selection | attention | total | vs dense (15962) |
|--:|--:|--:|--:|--:|
| Lq (=2048) | 7518 | 5153 | 12671 | **1.26x** |
| 128 | ~7518 | 8870 | ~16388 | ~0.97x |
| 64 | ~7518 | 10696 | ~18214 | ~0.88x |

So the 1.26x end-to-end result depends on a query block as large as the whole prefill
chunk. Whether a selection that coarse is acceptable is an accuracy question, currently
deferred; the measured union saturation (u/k = 1.20/1.56/1.93 at R = 2/4/8) is evidence
that it might be, but that was measured to R=8 and `Bq = Lq` at Lq=2048 is R=32.

## Fixed 25% sparsity, kv = 512 -> 4096

One density, the whole stack, every context length. `u = NBk/4` = 2/4/8/16, `S=16`,
`Hq=16`, `Hkv=8`, `d=128`, mask off. `C` is derived at each shape from its own
`FLASH_ATTN_EXT_WG` / `FLASH_ATTN_EXT` pair rather than carried over.

| kv | Lq | u | kv_eff | C | dense | ideal | ceil | shared | D/sh | %ceil | bq=128 | bq=64 | D/64 | sel | e2e | D/E | sel% |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 512 | 2 | 128 | 618 | 921 | 1356 | 0.68x* | 1362 | 0.68x | 100% | 1511 | 1707 | 0.54x | 925 | 2375 | **0.39x** | 39% |
| 1024 | 512 | 4 | 256 | 626 | 1221 | 915 | 1.33x | 927 | 1.32x | 99% | 1364 | 1686 | 0.72x | 1144 | 2144 | **0.57x** | 53% |
| 1024 | 1024 | 4 | 256 | 585 | 2430 | 1715 | 1.42x | 1719 | 100% | 2830 | 3496 | 0.69x | 1968 | 3761 | **0.65x** | 52% |
| 2048 | 512 | 8 | 512 | 629 | 2025 | 921 | 2.20x | 973 | 2.08x | 95% | 1474 | 1859 | 1.09x | 1573 | 2638 | **0.77x** | 60% |
| 2048 | 2048 | 8 | 512 | 694 | 8562 | 3496 | 2.45x | 3632 | 2.36x | 96% | 6109 | 7663 | 1.12x | 4611 | 8271 | **1.04x** | 56% |
| 4096 | 512 | 16 | 1024 | 664 | 4184 | 1219 | 3.43x | 1324 | 3.16x | 92% | 2128 | 2587 | 1.62x | 2503 | 3918 | **1.07x** | 64% |
| 4096 | 2048 | 16 | 1024 | 751 | 15850 | 4877 | 3.25x | 5056 | 3.14x | 96% | 8745 | 10668 | 1.49x | 7477 | 12534 | **1.26x** | 60% |

`ceil = dense/ideal` is what this FA kernel reaches at `kv_eff` on this backend, and it is
a ceiling **only where both sides get the same thread count**. `%ceil = ideal/shared`.
`e2e` uses `Bq = Lq`, i.e. the `shared` column.

### * The kv=512 row's ceiling is an artifact, and the artifact is the real finding

`kv_eff = 128` is the only point in the table below `FA_MIN_KV_BLOCKS * 64 = 192`. The
chunk-size search does **not** reject it -- `flash-attn-ops.h:405` relaxes `Bc_search` to
`align_down(kv_len, 64)` when `can_pipeline` is false, and the op runs on HMX (the
`fa-params` line is emitted only on search success):

```
kv_eff  128 : Br 512 Bc 128 n_kv_blocks 1 n_threads 1 pipeline 0   <- kv=512, u=2
kv_eff  256 : Br 512 Bc  64 n_kv_blocks 4 n_threads 6 pipeline 1
kv_eff 1024 : Br 512 Bc 256 n_kv_blocks 4 n_threads 6 pipeline 1
```

What degrades is threading, not kernel selection: `n_kv_blocks = 1` makes
`ggml-hexagon.cpp:2139` set `n_threads = 1` and `:2142` set `pipeline = 0`. So the 0.68x
divides a **6-thread** dense numerator by a **1-thread** denominator. It is not a limit of
block sparsity, and it is beatable by doing *more* work -- in the same log, same op, same
shape:

| kv=512, nb=512, bs=64 | selected KV | measured |
|:--|--:|--:|
| `n_sel=4` (50%) | 256 | **919-924 µs** |
| `n_sel=2` (25%) | 128 | 1364-1366 µs |

**Halving the density makes the kernel 1.48x slower.** The dense sweep shows the same
cliff and shows it is not monotone either: kv=64 -> 1266, kv=128 -> 1358, **kv=192 -> 738**,
kv=256 -> 921, kv=320 -> 757 µs. The floor at `nb=512` is ~740 µs, and everything below
3 KV blocks is on the wrong side of it.

The design consequence outranks the row it came from: **density cannot be a constant.**
A selection budget must be `max(fraction * NBk, ~4 blocks)`, because below ~4 blocks of 64
the kernel loses all six threads and the pipeline, and selecting less costs more. The
actionable fix is the `n_threads`/`pipeline` coupling at `ggml-hexagon.cpp:2139` --
parallelising over the query axis when `n_kv_blocks < 3` -- not `hmx_fa_find_chunk_size`,
whose dense bound `flash-attn-ops.h:417-419` documents as deliberately not loosened.

Rows 2-7 all have `kv_eff >= 256` (4+ blocks, 6 threads) and their `ceil` values stand.
Row 1's end-to-end verdict also stands: even with the kernel fixed, 925 µs of selection
against 921 µs of dense keeps it a loss.

## What does the softmax cost? Almost nothing — and that relocates the target

Measured on device **41c5710f** (also SM8750 / Hexagon v79; the earlier sweeps were on
87b3a4aa). The two units agree to within 1.4% on every row measured on both — e.g.
`XATTN_SELECT(Lq=2048,Lk=4096,u=16)` 8212 vs 8229 µs, `XATTN_SCORE_MM(Lq=2048,Lk=4096)`
1459 vs 1455 — so the tables above carry over.

`xattn_geom::no_sm` replaces `ggml_soft_max_ext` with a plain `ggml_add` of the causal
mask, and drops the `1/(√d·S)` scale with it (a single positive factor cannot reorder a
top-k). Blocks are then ranked by summed raw logits. `ggml_hexagon_supported_binary`
accepts the broadcast add (F32, contiguous, `ggml_can_repeat`), and `test -o XATTN_SCORE,
XATTN_SELECT` passes 13/13 with zero unsupported ops, so this is an HTP-to-HTP comparison
with no CPU fallback on either side. Each pair is registered adjacently and both members
are whole-graph, so the difference is exact with no per-call constant to remove.

| Lq | Lk | scorer | no-softmax | saved | | selection | no-softmax | saved | |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 1024 | 1572 | 1493 | 80 | 5.1% | 1774 | 1721 | 53 | 3.0% |
| 1024 | 1024 | 2288 | 2226 | 62 | 2.7% | 2579 | 2454 | 125 | 4.8% |
| 512 | 2048 | 1979 | 1962 | 17 | 0.9% | 2183 | 2151 | 32 | 1.5% |
| 2048 | 2048 | 5009 | 4824 | 185 | 3.7% | 5307 | 5268 | 38 | 0.7% |
| 512 | 4096 | 2983 | 2912 | 72 | 2.4% | 3150 | 3089 | 61 | 1.9% |
| 2048 | 4096 | 7854 | 7694 | 160 | 2.0% | 8212 | 8084 | 128 | 1.6% |

**0.7–4.8% of the selection leg, 17–185 µs.** At the configuration that currently wins
end-to-end (kv=4096, Lq=2048) it is 128 µs out of 8212 — about 1% of the whole pass.
Run-to-run spread on repeated rows in the same log is 0.2–0.9%, so the small entries are
barely above noise. Dropping the softmax is not a performance lever.

### Where the epilogue actually goes

`XATTN_SCORE_HALF` (stage 5) stops after the key-axis reduction, so the term splits three
ways. `XATTN_SCORE_MM_WG` (stage 4) is the matmul timed whole-graph; all three carry the
same C.

| Lq | Lk | epilogue | softmax | | reduce-1 | | reduce-2 | | reduce-1 ns/elem |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 1024 | 387 | 69 | 18% | 186 | 48% | 133 | 34% | 5.67 |
| 1024 | 1024 | 649 | 69 | 11% | 353 | 54% | 227 | 35% | 5.39 |
| 512 | 2048 | 573 | 38 | 7% | 331 | 58% | 203 | 36% | 5.05 |
| 2048 | 2048 | 2349 | 216 | 9% | 1350 | 57% | 783 | 33% | 5.15 |
| 512 | 4096 | 1221 | 76 | 6% | 666 | 55% | 478 | 39% | 5.08 |
| 2048 | 4096 | **4754** | **221** | **5%** | **2719** | **57%** | **1815** | **38%** | 5.19 |

- **reduce-1** — `ggml_sum_rows` over the key sub-block axis — is the largest single term
  in the whole scoring pass, 57% of the epilogue and **12× the softmax**.
- Its cost is a flat **~5.2 ns per input element at every shape**, varying only 5.05-5.67
  across a 16× range of tensor size. A cost that tracks element count and nothing else is
  a per-row overhead, and the row here is `ne0 = P = Bl/S = 4` floats — **16 bytes against
  a 128-byte HVX vector**. The kernel performs 131k-524k separate 4-element reductions.
- **reduce-2** — permute + `cont` + the query-axis `sum_rows` — is another 38%, on a
  quarter as many elements, because it carries the chain's only transposing copy.
- Nothing here is a fallback: `ggml_hexagon_supported_sum_rows` requires only F32 and
  contiguity (:3196-3215), both hold, and there is no `ne0` gate.

So the epilogue is **95% block pooling and 5% softmax**. Two consequences:

1. `try_fuse_xattn_score` is aimed correctly but described wrongly in this doc: its value
   is in eliminating the `sum_rows` pair and their DRAM round trips, not the softmax. That
   also explains why the fused stand-in measured 1.12-1.58x rather than the small factor a
   softmax-only saving would predict.
2. **`P = Bl/S` is a first-class performance parameter**, not just a granularity choice.
   At `Bl=64, S=16` it is 4, the worst case. `Bl=128` at the same S doubles it and halves
   the row count while leaving the score matmul untouched, and `bs=128` is already legal
   in the attention kernel. Untested, and the obvious next measurement.

### Would CPU or GPU be better at the epilogue? Measured: no

The obvious split — HMX keeps the GEMM, the reductions move to a unit that likes short
rows — is a losing trade here, and it loses on the compute alone, before any transfer or
synchronisation cost is counted. Same graph, same shapes, `-b CPU` (8 threads on 8 cores,
which measured best: `taskset c0` and `f0` are 3-7x worse, since the thread count is not
reduced with the core mask).

`Lq=2048, Lk=4096, S=16`, whole-graph µs, C cancelling in every difference:

| | HTP | CPU | HTP advantage |
|:--|--:|--:|--:|
| score matmul | 3100 | 20402 | 6.6x |
| softmax | 221 | 1054 | 4.8x |
| reduce-1 (key-axis `sum_rows`) | 2719 | 5058 | 1.9x |
| reduce-2 (permute + cont + `sum_rows`) | 1815 | 7365 | 4.1x |
| **whole epilogue** | **4754** | **13477** | **2.8x** |

**The NPU is faster at every sub-part, including the one it is bad at.** reduce-1 costs
5.2 ns per element on HTP and 9.6 ns on the CPU — ggml's `sum_rows` is a per-row
`ggml_vec_sum_f32` call, so a 4-element row is as unfriendly to NEON as it is to HVX, and
the CPU has less bandwidth to hide it. There is no sub-piece of this pass worth relocating,
which makes the transfer question moot; the prior relocation work had already priced the
best case at 0.97-1.11x (`xattn-split`, `rpcmem-cpu-access.md`).

GPU is **not** measured here: this build has no `libggml-opencl.so` and the earlier OpenCL
work ran on a different device (b4bd0901, Adreno 830). The one relevant prior data point is
the de-overheaded scoring comparison in `xattn-block-selection.md`, where the GPU lost to
the NPU at small R and drew at R=64 — on the whole pass, not the epilogue specifically.

### The implication: without the softmax, the pooling is not an epilogue at all

Not measured — recorded because it follows from the same bilinearity the `no_sm` arm
already assumes. With the softmax gone the block score is a sum of raw logits, and a sum
factorises where a softmax cannot:

```
attn_sum[bq,bk] = Σ_{i∈bq} Σ_{j∈bk} (rq_i · rk_j) = (Σ_{i∈bq} rq_i) · (Σ_{j∈bk} rk_j)
```

So the pooling can move to the *operands*: pre-sum P adjacent reduced-Q rows and P adjacent
reduced-K rows, then matmul the pooled operands directly to `[NBq, NBk]`. That deletes both
reductions outright and shrinks the matmul by `P² = 16x` (`[128,256]` → `[32,64]` at
Lq=2048/Lk=4096). The pre-sums run over rows of length `S·d = 2048` — the opposite of the
4-element rows that make `sum_rows` slow here.

The block-level causal mask still applies afterwards, and it has to: the identity holds
exactly for fully-unmasked block pairs and fails for the partially-masked diagonal block —
which `find_blocks` force-includes regardless, so its score never decides anything.

Scale, from the numbers above: today the scorer is 3100 (mm) + 4754 (epilogue) = 7854 µs.
The rank-1 form is a ~134 MFLOP matmul at Nq=32, which at the measured 430 GF/s for that
Nq is ~310 µs, plus two pre-sums. **This is the reason to care about dropping the softmax
— not the 2% the node itself costs.** It is a different estimator with unmeasured accuracy,
and that, not the latency, is what would decide it.
