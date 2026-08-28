# Block selection: how much do adjacent query blocks share, and what does a large query block cost?

The block-sparse FA kernel wants a **large query block**: one shared KV-block list per `Bq`
queries, so the HMX tile stays full and the K/V staging is amortised. The XAttention
estimator wants a **small** one: it scores per query block, and the finer that block, the
better the list fits the rows it serves. Everything between those two pulls is decided by
one quantity -- how much do adjacent query blocks' selections already have in common.

[sparse-attn-inkernel-vs-gather.md](sparse-attn-inkernel-vs-gather.md) measured exactly one
point of that surface: `Bl = 64`, `Bq = Bl`, 25% density, 2314 tokens, and reported
`f = 0.779`. This sweeps it: block size 32 to 256 on **both** axes, four context lengths,
three densities, and -- the part that turns the overlap number into a decision -- the true
attention mass each policy actually captures.

## Method

`examples/xattn-overlap/block_overlap.py`. Qwen3-1.7B (28 layers, `Hq=16`, `Hkv=8`, `G=2`,
`d=128`) in bf16, a 16384-token pg19 prefill, post-RoPE `Q`/`K` captured per layer by
patching `apply_rotary_pos_emb`. All 28 layers, all 8 KV heads. Shorter contexts are
prefixes of the same capture, so `T` is a clean axis.

The estimator is the real one -- antidiagonal packing at stride `S`, `1/(sqrt(d)*S)` scale,
the permissive reduced causal mask, `view(NBq, Pq, NBk, Pk).sum` pooling. It was checked
against a direct transcription of the paper's index arithmetic in float64
(`max|diff| = 3.8e-08`), and every pooled block row sums to `Bq/S` exactly.

Two axes that the reference ties together are **decoupled here**, because the kernel already
decouples them: `Bq` is the query block one list serves, `Bk` is the KV block the list names.
The reduced score matrix does not depend on either, so the whole grid comes from one pass.

Defaults: `S = 16`, density 25%, per-head merge by MAX over the `G` query heads sharing a KV
head, sink and diagonal blocks forced (both match the shipped pipeline). A query block is
counted only where the top-k is a real choice, `n_avail >= k`.

```sh
P=/mnt/data/wentao/uv-env/kv_reuse/bin/python
$P examples/xattn-overlap/block_overlap.py --contexts 2048 4096 8192 16384 \
     --density 0.25 0.125 0.5 --out overlap.json
```

## 1. The intersection grid

Mean `|S(a) n S(a-1)| / k` between adjacent query blocks, `T = 16384`, 25% density. Each
cell averages 1.1e4 to 8.8e4 adjacent pairs (28 layers x 8 KV heads x the eligible rows):

| Bq \ Bk | 32 | 64 | 128 | 256 |
|--:|--:|--:|--:|--:|
| **32** | 0.828 | 0.844 | 0.860 | 0.876 |
| **64** | 0.809 | 0.819 | 0.838 | 0.860 |
| **128** | 0.785 | 0.791 | 0.805 | 0.833 |
| **256** | 0.760 | 0.763 | 0.772 | 0.790 |

**The surface is remarkably flat.** Across an 8x range on both axes the intersection only
moves from 0.876 to 0.760. Two monotone trends, both in the direction structure predicts:

- **larger `Bq` lowers it** -- adjacent query blocks sit further apart, so their attention
  patterns diverge. Going `Bq` 32 -> 256 at `Bk = 32` costs 0.828 -> 0.760, only 8%.
- **larger `Bk` raises it** -- a coarser KV block absorbs disagreements that a finer one
  would expose as different picks.

The diagonal (`Bq = Bk = B`, the setting XAttention actually specifies) with the distance
decay, the Jaccard, and the union growth:

| B | NB | k | d=1 | d=2 | d=4 | d=8 | Jaccard | u/k R=2 | R=4 | R=8 | per-layer min..max |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 32 | 512 | 128 | 0.828 | 0.779 | 0.730 | 0.682 | 0.719 | 1.171 | 1.382 | 1.622 | 0.749 .. 0.910 |
| 64 | 256 | 64 | 0.819 | 0.758 | 0.702 | 0.638 | 0.706 | 1.183 | 1.412 | 1.672 | 0.734 .. 0.909 |
| 128 | 128 | 32 | 0.805 | 0.735 | 0.665 | 0.568 | 0.687 | 1.195 | 1.442 | 1.748 | 0.720 .. 0.892 |
| 256 | 64 | 16 | 0.790 | 0.705 | 0.597 | 0.413 | 0.666 | 1.210 | 1.493 | 1.879 | 0.713 .. 0.895 |

The per-layer spread is tight and shows no depth structure, which matches the earlier
single-point result.

### Normalised by token distance, the block-size effect inverts

`d` counts *blocks*, so a row of that table compares distances that differ by 8x in tokens.
Re-indexing by token distance:

| B | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | 0.828 | 0.779 | 0.730 | 0.682 | . | . | . |
| 64 | . | 0.819 | 0.758 | 0.702 | 0.638 | . | . |
| 128 | . | . | 0.805 | 0.735 | 0.665 | 0.568 | . |
| 256 | . | . | . | 0.790 | 0.705 | 0.597 | 0.413 |

Read a **column**: at a fixed 256-token separation the intersection is 0.682 at `B = 32` and
0.790 at `B = 256`. Coarse blocks agree *more* at equal token distance -- the divergence in
the table's rows is a property of the distance, not of the block size. The decay is slow and
has a long tail (0.68 still shared at 8 blocks apart), which is the sink and the recent
window being shared by *every* query block rather than just neighbours.

### Context and density both raise it

`Bq = Bk = 64`, adjacent:

| | T=2048 | 4096 | 8192 | 16384 |
|:--|--:|--:|--:|--:|
| 12.5% density | 0.679 | 0.701 | 0.724 | 0.754 |
| 25% | 0.742 | 0.755 | 0.785 | 0.819 |
| 50% | 0.828 | 0.846 | 0.875 | 0.891 |

Both trends favour deployment: the longer the context and the higher the density, the more
adjacent query blocks agree. The regime where a large `Bq` is cheapest is the regime where
sparse attention is worth doing at all.

## 2. What one list over a large query block has to cost

The union of the `R = Bq/Bk` fine selections is the smallest list that is a true superset of
every constituent query block's picks. As a density (`T = 16384`, 25% fine density):

| Bq | Bk=32 | Bk=64 | Bk=128 | Bk=256 |
|--:|:--|:--|:--|:--|
| 64 | 0.293 (1.17x) | 0.250 (1.0x) | - | - |
| 128 | 0.345 (1.38x) | 0.296 (1.18x) | 0.250 (1.0x) | - |
| 256 | 0.406 (1.62x) | 0.353 (1.41x) | 0.299 (1.20x) | 0.250 (1.0x) |

Sub-linear and saturating, as reported before. `Bq = 256` over `Bk = 32` blocks needs 1.62x
the block count, not 8x. mllm's independent number for the same shape -- union of 8
sub-blocks at `BQ = 256` needing 45-48% density -- lands on ours at the context where they
measured it (49% at `T = 2048`, falling to 41% by `T = 16384`).

## 3. The part that decides it: recall

Overlap says how *big* a shared list must be. It does not say whether the shared list is any
good. So the same run computes exact causal attention and measures the **true probability
mass** each policy captures, averaged over query rows and heads. Four policies at the same
large `Bq`, plus an oracle that ranks by the true mass instead of the estimator:

- **N** native -- pool the reduced scores straight to `(Bq, Bk)`, top-k. One list.
- **M** max-merge -- score at `(Bk, Bk)`, max over the `R` rows, top-k. **This is what the
  shipped pipeline does** (stage B in [xattn-scoring.md](xattn-scoring.md)).
- **U** union -- score at `(Bk, Bk)`, top-k per fine row, union. Variable size.
- **F** fine-exact -- the per-fine-block lists, unmerged. What XAttention specifies, and what
  needs `Bq = Bk`.

### The price of a large query block, at fixed density

Native recall at 25%, `T = 16384`:

| Bq \ Bk | 32 | 64 | 128 | 256 |
|--:|--:|--:|--:|--:|
| **32** | 0.9386 | - | - | - |
| **64** | 0.9356 | 0.9335 | - | - |
| **128** | 0.9317 | 0.9297 | 0.9269 | - |
| **256** | 0.9274 | 0.9257 | 0.9233 | 0.9200 |

Coarsening the query block 32 -> 256 at `Bk = 32` costs **1.1 points of recall** (0.9386 ->
0.9274). Coarsening the KV block 32 -> 256 at `Bq = 256` costs 0.7 more (0.9274 -> 0.9200).
The two axes are comparable in size and roughly additive, so if `Bq` has to be 256, keeping
`bs` at 32 buys back about 40% of what the coarse query block cost.

### Iso-quality: the real number

Recall differences are hard to price. The useful form is the inverse: **at what density does
a shared-list policy match the recall that the exact per-fine-block selection gets at 25%?**

| T | Bq | Bk | R | F recall | N density | penalty | U density | penalty |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16384 | 64 | 32 | 2 | 0.9387 | 0.264 | **1.05x** | 0.302 | 1.21x |
| 16384 | 128 | 64 | 2 | 0.9337 | 0.267 | **1.07x** | 0.305 | 1.22x |
| 16384 | 256 | 128 | 2 | 0.9272 | 0.266 | **1.06x** | 0.306 | 1.22x |
| 16384 | 128 | 32 | 4 | 0.9389 | 0.281 | **1.13x** | 0.370 | 1.48x |
| 16384 | 256 | 64 | 4 | 0.9341 | 0.285 | **1.14x** | 0.375 | 1.50x |
| 16384 | 256 | 32 | 8 | 0.9392 | 0.300 | **1.20x** | 0.449 | 1.80x |
| 8192 | 256 | 32 | 8 | 0.9269 | 0.305 | 1.22x | 0.475 | 1.90x |
| 4096 | 256 | 32 | 8 | 0.9106 | 0.317 | 1.27x | 0.510 | 2.04x |
| 2048 | 256 | 32 | 8 | 0.9138 | 0.337 | 1.35x | 0.535 | 2.14x |

Three things fall out, and they are the findings.

**The penalty is a function of `R = Bq/Bk` alone.** At `T = 16384`, every `R = 2` row is
1.05-1.07x and every `R = 4` row is 1.13-1.14x, regardless of whether `R = 2` means
(64, 32) or (256, 128). `Bq` and `Bk` do not enter separately. So the design question is not
"how large may `Bq` be" but "how many KV blocks wide is it".

**A large query block is cheap.** Holding XAttention's own quality, `Bq = 8*Bk` costs 1.20x
the KV traffic at `T = 16384`, and the penalty *falls* with context (1.35x at 2048, 1.20x at
16384) because overlap rises with context. `R = 2` is essentially free at 1.05x.

**The union is the wrong way to build the list, by ~1.5x.** At every shape U needs far more
density than N for the same quality, and the reason is visible in the curves: U and N sit on
nearly the *same* recall-vs-density curve (at matched budget U leads N by only 0.2-0.5
points, and never beats the oracle). The union is not a better-shaped set -- it is the same
tradeoff pinned at an operating point nobody chose. Native scoring at a slightly larger `k`
reaches the same quality for 1.20x instead of 1.80x.

That is a correction to
[sparse-attn-inkernel-vs-gather.md](sparse-attn-inkernel-vs-gather.md), which recommended
the union. Its comparison was against the *per-query-block kernel path*, and against that
the union does win. It was never compared against simply scoring at the coarse `Bq`, which
costs nothing extra and dominates it.

### The shipped max-merge is very slightly worse than not merging

`N >= M` at every shape and every budget, by 0.001-0.002 recall:

| blocks | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| M max-merge | 0.8814 | 0.9072 | 0.9262 | 0.9413 | 0.9539 | 0.9644 | 0.9733 |
| N native | 0.8832 | 0.9086 | 0.9274 | 0.9423 | 0.9545 | 0.9648 | 0.9736 |

(`T = 16384`, `Bq = 256`, `Bk = 32`.) The difference is small, but the direction is
consistent and **N is also the cheaper graph**: both pool the same reduced score matrix, so
pooling the query axis directly to `Bq` simply *deletes* stage B (the `log2(R)`-step
`SUB`/`CLAMP`/`ADD` max tree over the scorer-block axis) rather than replacing it. The
per-KV-head max tree over `G` (stage A) is unaffected and still needed.

### The estimator is not the bottleneck; the granularity is

An oracle ranking by the true block mass beats the estimator by only **0.3-0.8 points** of
recall at the same budget (0.9358 vs 0.9274 at `Bq=256, Bk=32, 25%`; 0.9226 vs 0.9200 at
`Bq=Bk=256`). Two consequences: a cheaper scorer that is nearly as accurate loses almost
nothing, and no scorer improvement can recover the 1.1 points that coarsening `Bq` costs.
The oracle ranked by SUM over the `G` heads beats the one ranked by MAX by under 0.001, so
the MAX policy argued for in [xattn-scoring.md](xattn-scoring.md) costs nothing measurable
either way on this metric.

## Robustness

Every control moves the headline by less than the prompt does.

| control | Bq=Bk=64, T=16384, 25% |
|:--|--:|
| default (MAX merge, sink+diagonal forced) | 0.819 |
| SUM merge over the `G` heads | 0.823 |
| no forced sink/diagonal | 0.817 |
| SUM merge and no forcing (the earlier tool's setting) | 0.821 |
| stride `S = 8` | 0.818 |
| stride `S = 32` | 0.819 |
| pg19 book 1 | 0.826 |
| pg19 book 7 | 0.828 |

The estimator's own stride is irrelevant to the overlap (+-0.002 over a 4x range), which is
expected: overlap is a property of the attention pattern, not of the reduction used to guess
it. Prompt is the largest single source of variation and it is still under 0.01.

**On the earlier `f = 0.779`.** At the earlier tool's exact configuration (`T = 2304`,
`Bl = 64`, 25%, SUM merge, no forcing) the three books give 0.734, 0.752 and 0.736. The old
number sits above that spread; the gap is prompt, not method. It changes no conclusion here,
all of which are about *differences* measured within one capture.

**Population.** The recall pass has to restrict to query blocks whose causal pool fits the
largest priced budget, which selects later query blocks -- and those overlap less. Measured
directly: raising the bar from `n_avail >= k` to `n_avail >= 0.625*NBk` moves the `B = 32`
intersection 0.828 -> 0.790 and `u/k` at `R = 8` from 1.622 to 1.802. That is exactly the
gap between the union densities in sections 2 and 3 (0.406 vs 0.449). All policies inside
the iso-quality table are measured on **identical rows**, so the penalties compare; the
absolute densities there are the late-context, least-favourable ones.

## 4. The phone regime: kv 512-4096, a query tile of 2x or 4x the selection block

Everything above is measured where a sweep is well conditioned. On device the context is
512 to 4096, and that regime behaves differently enough to need its own table. The question
is the deployable one: **selection happens per query block `B`, the kernel stages one KV list
per query tile of `R = 2` or `4` such blocks. How big is that list, and how much of it does
each sub-block actually want?**

`--tiles 1`, 25% sparsity, `S = 16`. For a tile, let `c(b)` be how many of its `R`
sub-blocks selected key block `b`. Then everything below is a statistic of `c`:

| symbol | definition | what it is |
|:--|:--|:--|
| `k` | `round(0.25 * NB)` | blocks ONE sub-block selects, the 25% budget |
| `u` | `#{b : c(b) >= 1}` | **`u` and `\|union\|` are the same number** -- the size of the tile's list. This is the value `n_sel` (`sel->ne[0]`) has to take, which is why it is called `u` in the older docs. |
| `\|core\|` | `#{b : c(b) = R}` | blocks EVERY sub-block wants |
| `pair` | `sum_b C(c,2) / C(R,2)` | mean intersection size over the `C(R,2)` sub-block pairs |
| `dens` | `u / NB` | fraction of the KV cache the tile stages |
| `waste` | `1 - mean(c)/R` | share of (staged block x sub-block) pairs a per-row mask would switch off |

At `R = 2` there is only one pair, so `pair` and `|core|` are the same quantity and those two
columns are identical by construction. They separate only at `R = 4`.

| T | B | NB | k | R | tile | u = \|union\| | u/k | mean dens | p99 | max | dens@p99 | \|core\| | core/k | pair | pair/k | mask waste |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 32 | 16 | 4 | 2 | 64 | 5.31 | 1.33 | 0.332 | 7 | 7 | 0.438 | 2.69 | 0.674 | 2.69 | 0.674 | 0.246 |
| 512 | 32 | 16 | 4 | 4 | 128 | 7.62 | 1.90 | 0.476 | 11 | 12 | 0.688 | 1.10 | 0.276 | 2.20 | 0.549 | 0.475 |
| 1024 | 32 | 32 | 8 | 2 | 64 | 10.01 | 1.25 | 0.313 | 13 | 14 | 0.406 | 5.99 | 0.749 | 5.99 | 0.749 | 0.201 |
| 1024 | 32 | 32 | 8 | 4 | 128 | 13.20 | 1.65 | 0.413 | 19 | 22 | 0.594 | 3.47 | 0.434 | 5.30 | 0.662 | 0.394 |
| 1024 | 64 | 16 | 4 | 2 | 128 | 5.35 | 1.34 | 0.335 | 7 | 7 | 0.438 | 2.65 | 0.662 | 2.65 | 0.662 | 0.253 |
| 1024 | 64 | 16 | 4 | 4 | 256 | 7.57 | 1.89 | 0.473 | 10 | 11 | 0.625 | 1.17 | 0.291 | 2.19 | 0.548 | 0.471 |
| 2048 | 32 | 64 | 16 | 2 | 64 | 19.67 | 1.23 | 0.307 | 25 | 29 | 0.391 | 12.33 | 0.770 | 12.33 | 0.770 | 0.187 |
| 2048 | 32 | 64 | 16 | 4 | 128 | 24.80 | 1.55 | 0.387 | 35 | 41 | 0.547 | 8.26 | 0.516 | 11.41 | 0.713 | 0.355 |
| 2048 | 64 | 32 | 8 | 2 | 128 | 10.09 | 1.26 | 0.315 | 13 | 15 | 0.406 | 5.91 | 0.738 | 5.91 | 0.738 | 0.207 |
| 2048 | 64 | 32 | 8 | 4 | 256 | 13.20 | 1.65 | 0.413 | 18 | 21 | 0.562 | 3.45 | 0.431 | 5.29 | 0.661 | 0.394 |
| 2048 | 128 | 16 | 4 | 2 | 256 | 5.17 | 1.29 | 0.323 | 6 | 7 | 0.375 | 2.83 | 0.706 | 2.83 | 0.706 | 0.227 |
| 2048 | 128 | 16 | 4 | 4 | 512 | 7.29 | 1.82 | 0.455 | 10 | 11 | 0.625 | 1.15 | 0.289 | 2.28 | 0.569 | 0.451 |
| 4096 | 32 | 128 | 32 | 2 | 64 | 38.88 | 1.22 | 0.304 | 49 | 57 | 0.383 | 25.12 | 0.785 | 25.12 | 0.785 | 0.177 |
| 4096 | 32 | 128 | 32 | 4 | 128 | 47.85 | 1.50 | 0.374 | 71 | 81 | 0.555 | 17.74 | 0.554 | 23.68 | 0.740 | 0.331 |
| 4096 | 64 | 64 | 16 | 2 | 128 | 19.87 | 1.24 | 0.310 | 26 | 30 | 0.406 | 12.13 | 0.758 | 12.13 | 0.758 | 0.195 |
| 4096 | 64 | 64 | 16 | 4 | 256 | 25.01 | 1.56 | 0.391 | 37 | 41 | 0.578 | 8.02 | 0.501 | 11.28 | 0.705 | 0.360 |
| 4096 | 128 | 32 | 8 | 2 | 256 | 10.14 | 1.27 | 0.317 | 13 | 14 | 0.406 | 5.86 | 0.732 | 5.86 | 0.732 | 0.211 |
| 4096 | 128 | 32 | 8 | 4 | 512 | 13.19 | 1.65 | 0.412 | 18 | 20 | 0.562 | 3.47 | 0.434 | 5.31 | 0.664 | 0.394 |
| 4096 | 256 | 16 | 4 | 2 | 512 | 5.20 | 1.30 | 0.325 | 6 | 7 | 0.375 | 2.80 | 0.700 | 2.80 | 0.700 | 0.231 |
| 4096 | 256 | 16 | 4 | 4 | 1024 | 7.58 | 1.90 | 0.474 | 10 | 12 | 0.625 | 1.05 | 0.263 | 2.14 | 0.534 | 0.472 |

`p99`, `max` and `dens@p99` are of `u` across tiles. The tool prints `p50` and `p90` as well,
and prints `|core|` in the column of that name -- these are the `tiles_25.txt` rows headed
`B NB k R tile kbar |uni| u/k dens p50 p90 p99 max |core| c/k pair/k`.

Rows with `k <= 2` are omitted: at 25% sparsity `k = T/(4B)`, so `T=512, B>=64` and
`T<=1024, B>=128` give `k` of 1 or 2, where the forced sink and diagonal *are* the whole
selection and every statistic is an artifact of the forcing rather than a measurement.
`mask waste` is `1 - mean(c)/R`, the share of (staged block x sub-block) pairs that a
per-row mask would have to switch off -- the cost of turning the superset back into the
exact selection.

**`u/k` is a function of `(R, k)` and essentially nothing else.** Not of `B`, not of `T`.
Read the `R=4, k=8` rows: 1.650 at (1024, 32), 1.651 at (2048, 64), 1.649 at (4096, 128).
Three different block sizes and a 4x context range agree to three decimals. Pooling all
cells:

| k | 4 | 8 | 16 | 32 |
|:--|--:|--:|--:|--:|
| u/k at R=2 | 1.31 | 1.26 | 1.24 | 1.22 |
| u/k at R=4 | 1.88 | 1.65 | 1.56 | 1.50 |

So the selection granularity does not influence the union *ratio* directly. It influences it
only through `k = T/(4B)`: **a smaller selection block raises `k`, and a larger `k` unions
better.** That inverts the usual intuition and it is the practical rule for this regime --
at fixed 25% sparsity, pick `B <= T/32` so that `k >= 8`. At `T=2048` that means `B <= 64`;
at `T=4096`, `B <= 128`; at `T=512` no choice in range reaches it (`B=32` gives `k=4`).

**`R=2` is nearly free, `R=4` is not.** Doubling the query tile over the selection block
costs 1.22-1.34x the block list (density 0.30-0.34 instead of 0.25) and leaves 18-25% of the
staged rows unwanted. Quadrupling costs 1.50-1.90x (density 0.37-0.48) and 33-48% of what it
stages is unwanted by any given sub-block. The `core` column says the same thing from the
other side: at `R=4` only 26-55% of a sub-block's picks are shared by all four, while at
`R=2` it is 66-79%.

**The tail is what actually sizes the kernel, and it is much worse than the mean.** `n_sel`
is one `uint16_t` for the whole op and cannot vary by query block
(`ggml-hexagon.cpp:2227`), so a union that is a true superset everywhere has to be sized at
the *maximum*, not the mean. At `T=4096, B=32, R=4` the mean union is 47.85 of 128 blocks
(37%) but the max is 81 (63%); at `R=2` the mean is 38.88 (30%) and the max 57 (45%). Sizing
at p99 instead of max still costs 0.38-0.63 density. **Quoting a mean union ratio understates
the deployed cost by about 1.3x at p99 and 1.5x at the max.** The alternative -- fix `n_sel`
at something near the mean and let over-full tiles drop their lowest-ranked blocks -- is no
longer a superset, and then there is no reason to prefer it to scoring at the tile
granularity directly (section 3).

**Below `T=2048` the density floor binds first.** `FA_MIN_KV_BLOCKS = 3`
(`htp/flash-attn-ops.h:349`) gates the pipelined path, and `hmx_fa_find_chunk_size` needs
`m = Bc/bs` to divide `n_sel` with `m <= 8`. At `T=512` the largest `k` available in this
block-size range is 4, and the union at `R=4` is 7.62 of 16 blocks -- 48% density before any
tail allowance. That matches the existing measurement that `kv=512` loses to dense for every
scorer, and it is not a scoring problem: there are not enough KV blocks for a block-sparse
kernel to have anything to skip.

- **One model, one corpus.** Qwen3-1.7B on pg19. `G = 2` is small; a model with `G = 8`
  stresses the per-KV-head merge much harder, and that is where the MAX-vs-SUM policy could
  start to matter.
- **Recall is not quality.** Attention-mass recall is the metric XAttention is tuned on, but
  no perplexity or downstream number is measured here. A 1.1-point recall drop has no
  established exchange rate.
- **Density is not time.** The iso-quality table prices KV traffic in blocks. The kernel's
  actual cost is not linear in `n_sel`: `hmx_fa_find_chunk_size` needs `m = Bc/bs` to divide
  `sel->ne[0]` with `m <= 8`, which makes the cost a sawtooth in `u` (a prime `u` is 2.5x a
  neighbouring smooth one). A 1.20x density penalty must still be rounded to a `u` with a
  large divisor before it is a time.
- **`Bq < Bk` is unmeasured for recall.** The grid is complete for the intersection but the
  recall pass only runs `Bq >= Bk`, since M/U need `R >= 1`.
- **Nothing here ran on a device.** These are activations from an H100 and analysis in
  float32. No HTP timing is claimed.

## 5. Crossing this with the kernel: is there a good naive-union operating point?

The overlap grid says how big a shared list must be; `xattn-scoring.md` says what a small
query block costs. Multiplying them answers the deployment question directly.

### A cost model, calibrated and validated

Two measured curves at `kv=2048, nb=512, bs=64` (`FLASH_ATTN_EXT_SPARSE`, replicated):

```
cost_shared(u) = 610 + 46.9*u  µs        (u=8 -> 985, u=16 -> 1360)
tax(Bq):  64 -> 1.89x   128 -> 1.51x   256 -> 1.10x   512 -> 1.00x
```

The tax is multiplicative and independent of `u` — measured at both `u=8` (1.89×) and `u=16`
(1.92×) — so `cost(Bq, u) = tax(Bq) * cost_shared(u)`. Validated against the `n_share` sweep,
which measures *real* unions of two fine selections at `bq=128`:

| u | predicted | measured | |
|--:|--:|--:|:--|
| 16 | 2059 | 2153 | +4.6% |
| 14 | 1917 | 2454 | **+28.0% — sawtooth** |
| 12 | 1775 | 1651 | −7.0% |
| 10 | 1633 | 1806 | +10.6% |
| 8 | 1491 | 1486 | −0.4% |

### The answer

`Bk = 64`, 25% fine density (`k = 8` of 32 blocks). Union sizes from §2's diagonal;
native-at-coarse-`Bq` budgets from §3's iso-quality table.

| R | Bq | union u | tax | cost | vs R=1 | native u | cost | vs R=1 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 64 | 8.0 | 1.89× | 1865 | 1.00× | 8.0 | 1865 | 1.00× |
| 2 | 128 | 9.5 | 1.51× | 1595 | 1.17× | 8.6 | 1531 | 1.22× |
| 4 | 256 | 11.3 | 1.10× | **1255** | **1.49×** | 9.1 | 1143 | 1.63× |
| 8 | 512 | 13.4 | 1.00× | **1234** | **1.51×** | 9.6 | **1057** | **1.76×** |

**Yes, naive union has a good point, and it is `R = 4`.** The tax collapses (1.89 → 1.10)
faster than the union grows (1.00 → 1.41), so the product falls hard to `R=4` and is then
flat — `R=8` buys 1.3% more for 19% more KV traffic. There is no interior optimum to find:
the curve is monotone and saturates, which is what "the overlap surface is remarkably flat"
looks like once it is priced.

Three qualifications, and the last one is specific to *naive* union:

1. **Native scoring at the coarse `Bq` dominates it by 1.17×** at equal quality (1057 vs
   1234), and needs no union step at all. §3 established this on recall grounds; the kernel
   numbers agree. Naive union is the *second* best thing to do.
2. **Most of the win is available by `R = 2`–`4`.** If a fine query block is wanted for
   other reasons, `R = 4` captures 1.49× of the available 1.51×.
3. **The union size is data-dependent, and the kernel is not smooth in it.** A union landing
   on `u = 14` measured **2454 µs against 2149 µs for `u = 16`** — 12% less work, 14% more
   time — because `n_kv_blocks = u / (largest divisor of u <= 8)` is a sawtooth. Confirmed
   twice in the same sweep (`u=10` at 1806 against `u=12` at 1651). A naive union produces
   whatever `u` the data gives, per query block and per KV head, so it will land on bad
   divisors routinely. Any deployment must **round the union up** to a divisor-friendly size,
   which erodes part of the 1.49×.

### Short context: the model breaks, and the sawtooth dominates

`cost_shared(u) = 610 + 46.9u` must not be extrapolated below its `u = 8..16` calibration.
Measured directly (`nb=512`, `bs=64`, shared selection, device `eb49fb9d`, whose dense rows
match the other units to 0.5%):

| u | kv_eff | µs | what the chunk rule gives |
|--:|--:|--:|:--|
| 2 | 128 | **1360** | below `FA_MIN_KV_BLOCKS*64` → 1 thread, no pipeline |
| 3 | 192 | **746** | `m=1` → 3 chunks, 6 threads — the fastest point measured |
| 4 | 256 | 927 | 4 chunks |
| 5 | 320 | 1112 | prime → 5 chunks |
| 6 | 384 | **784** | 3 or 6 chunks |
| 7 | 448 | **1492** | prime → 7 chunks |

The curve is **not monotone in the work done**. `u=6` costs 784 µs and `u=7` costs 1492 —
17% more work for 90% more time. `u=5` costs 1112 and `u=6` costs 784 — 20% more work for
30% *less* time. The fit would have predicted 704/751/798/845/892/938.

The per-query-block tax, by contrast, is completely stable across all of them and matches the
long-context values: **1.10–1.16× at `bq=256`, 1.43–1.53× at `bq=128`, 1.78–1.89× at `bq=64`.**
It is a property of the schedule, not of the shape.

### Against dense at short context

Dense `nb=512`: kv=512 → 933 µs, kv=1024 → 1244 µs.

| kv | u | density | shared | **vs dense** | bq=256 | bq=128 | bq=64 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 2 | 25% | 1360 | **0.69×** | 0.66× | 0.62× | 0.55× |
| 512 | **3** | 37.5% | **746** | **1.25×** | 1.12× | 0.88× | 0.70× |
| 512 | 4 | 50% | 927 | 1.01× | 0.89× | 0.68× | 0.55× |
| 1024 | 4 | 25% | 930 | **1.34×** | 1.19× | 0.90× | 0.73× |
| 1024 | 5 | 31% | 1113 | 1.12× | 0.97× | 0.74× | 0.60× |
| 1024 | **6** | 37.5% | **784** | **1.59×** | 1.45× | 1.08× | 0.85× |
| 1024 | 7 | 44% | 1492 | 0.83× | 0.72× | 0.54× | 0.44× |

**At kv=512, selecting *more* blocks is both faster and better.** `u=3` beats `u=2` by 1.82×
while being a strict superset — 37.5% density is 1.25× dense where 25% density is 0.69×. The
25% budget is simply on the wrong side of the threading cliff.

**At kv=1024 the best point is `u=6`, not the nominal `u=4`**: 1.59× against 1.34×, again
for *more* selected KV.

**Per-query-block loses to dense everywhere below kv=2048.** The best `bq=64` cell in the
whole table is 0.85×. A faithful XAttention query block is not deployable at short context on
this kernel at any density.

So the short-context rule is the opposite of the intuition the long-context tables build: do
not shrink the budget. Pick the smallest `u` that is **≥ 3 blocks and has a divisor ≤ 8 that
leaves 3–6 chunks**, and take the extra density for free. For a naive union this is a
sharper constraint than the density penalty itself — the union hands you `u = 2.4` or `6.7`,
and rounding to 3 or to 6 is the difference between 1.25× and 0.83×.

### A predictor for `u`, and what its coefficients mean

`u` is the **size of the selection list** handed to the kernel — `sel->ne[0]`, so
`kv_eff = u * bs`. In union terms it is the size of the *result*, not the number of lists
combined: unioning `R` fine selections of `k` blocks each gives `k <= u <= R*k`.

Cost is **not** a function of `u` alone, which is why the linear fit failed at short context.
It is a function of two things `u` determines: the KV chunk count and the effective KV length.
`n_kv_blocks(u)` is deterministic — it is what `hmx_fa_find_chunk_size` returns, and it can be
computed on the host without touching the device:

```
kv      = u * bs
cap     = align_down((kv - 1) / (FA_MIN_KV_BLOCKS - 1), bs)     # pipelined
m       = max{ m : m | u,  m <= FA_SPARSE_MAX_M,  m*bs <= cap }
n_kv_blocks = ceil(kv / (m*bs))
```

Fitting the seven measured pipelined points against that:

```
cost(u) = 121 + 161 * n_kv_blocks(u) + 0.554 * kv_eff(u)      µs, nb=512
```

| u | m | chunks | kv_eff | measured | predicted | err |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 1 | 3 | 192 | 746 | 710 | −4.9% |
| 4 | 1 | 4 | 256 | 927 | 906 | −2.3% |
| 5 | 1 | 5 | 320 | 1112 | 1102 | −0.9% |
| 6 | 2 | 3 | 384 | 784 | 816 | +4.1% |
| 7 | 1 | 7 | 448 | 1492 | 1494 | +0.1% |
| 8 | 2 | 4 | 512 | 985 | 1048 | +6.4% |
| 16 | 4 | 4 | 1024 | 1360 | 1331 | −2.1% |

`R² = 0.984`, max error 6.4%. **Held out** — the five real two-way unions measured at
`kv=2048, bq=128` (multiply by the 1.514× tax), none of which was used in the fit:

| u | chunks | predicted | measured | err |
|--:|--:|--:|--:|--:|
| 16 | 4 | 2015 | 2153 | +6.8% |
| 14 | 7 | 2638 | 2454 | −7.0% |
| 12 | 3 | 1557 | 1651 | +6.0% |
| 10 | 5 | 1937 | 1806 | −6.7% |
| 8 | 4 | 1586 | 1486 | −6.3% |

Within ±7% everywhere, **including `u = 14`, which the naive `610 + 46.9u` line missed by
28%.** The sawtooth is not noise and not a special case; it is `n_kv_blocks` moving.

Two coefficients worth reading:

- **161 µs per KV chunk.** This is a *fixed* cost per chunk, independent of how much KV the
  chunk holds — the fork/join and HMX-queue handoff structure priced in
  `flash-attn-htp-anatomy.md`. It arrives independently at the `~190 µs` figure already
  recorded in `flash-attn-ops.h:401` from a different measurement. At `u=7` you pay it seven
  times (1127 of 1492 µs); at `u=6` three times (483 of 784). **That is the whole sawtooth.**
- **0.554 µs per KV row** is the actual streaming and arithmetic — a third of the cost at
  `u=16`, and under a fifth at `u=3`.

Below three chunks the model does not apply at all: `u=2` gives one chunk, one thread, no
pipeline, and costs **3.85×** what the pipelined fit predicts.

So the design rule is mechanical. Enumerate `u`, compute `n_kv_blocks(u)`, and take the
smallest `u` meeting the quality budget that also minimises `161*n_kv_blocks + 0.554*u*bs`.
For `bs=64` the good sizes are those with a divisor near `u/3`: 3, 6, 8, 12, 16, 24, 32 —
and the ones to avoid are the primes and near-primes: 5, 7, 11, 13, 14.
