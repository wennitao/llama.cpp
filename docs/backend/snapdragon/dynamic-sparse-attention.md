# Per-row selection lengths: the dynamic block-sparse kernel, and the threshold policy

The fixed-`u` pipeline in `sparse-attention.md` selects the same number of KV blocks for every
query block of every head. This document covers what replaced it: a kernel where each query
tile attends to **its own** number of blocks, decided at run time from the scores the scorer
already computes, and the policy that feeds it.

Everything here is measured on SM8750 / Hexagon v79 (6 HVX threads, 1 HMX unit, 8 MB VTCM) with
Qwen3-1.7B Q4_0, wikitext-2, `-fa 1 -ngl 99 -dev HTP0`. Speed rows alternate arms three times
(this device drifts >25% across sequential same-arm runs — see `README.md`).

**Headline:** at 4k context the attention op runs at **0.32× dense** and end-to-end prefill is
**1.33× dense / 1.20× the fixed-`u` pipeline**, at **+0.15% perplexity** where fixed-`u` costs
+0.43%. The dynamic kernel is better on both axes than the pipeline it replaces.

## 1. Why fixed `u` was the wrong shape

The fixed pipeline had to answer "how many blocks?" once, at graph-build time, for the whole
op. Two measured facts made every answer bad:

1. **Demand is not uniform.** Per-query-block demand at a fixed mass target spans roughly 2×
   between the median and the tail, and the spread is driven by *position* (a late query block
   reaches more KV) rather than by content. Sizing for the tail wastes the median; sizing for
   the median truncates the tail.
2. **The threshold beats any fixed budget at equal density**, which is the part I got wrong for
   a while. Matching an adaptive rule's *average* density with a fixed `u` does not recover its
   quality — see the grid in §5.

The obvious objection — "serve each query block its own demand, i.e. the union" — is a
*different* policy, and a poor one on its own. The union of four query blocks' fixed-`u`
selections is larger and no better. What wins is the union of four *thresholded* selections,
where each contributing row was sized by its own scores (§5).

## 2. The kernel contract: `src[6]`

`FLASH_ATTN_EXT` gains an optional seventh input
([ggml.c:5511](../../../ggml/src/ggml.c#L5511)):

```
src[5]: I32 [u_max, NBq, n_kv_heads|1, n_seqs|1]   the block-index lists
src[6]: F32 [NBq,   n_kv_heads|1, n_seqs|1, 1]     per-row LENGTHS (optional)
```

With `src[6]` present, `sel->ne[0]` becomes an **upper bound** `u_max`; row `(qb, kv_head, seq)`
attends to the first `cnt[qb, kv_head, seq]` entries of its list. `cnt` is F32 because a count is
the natural output of an in-graph reduction (`sum_rows` over a 0/1 membership row); the kernel
truncates toward zero and clamps to `[1, u_max]`
([flash-attn-ops.c:228](../../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c#L228)).

### 2.1 The invariant that looked like a blocker, and wasn't

The old kernel documented `fa_chunk_nblk` as *"deliberately NOT a function of the query block"*,
because the untagged DMA FIFO requires the push count to equal the pop count one loop iteration
later, and those two sites do not necessarily see the same query block.

That reasoning is real but the invariant it protects is **weaker than "constant"**: what is
required is that every push and pop site derive the length from the `q_start` of *the tile the
chunk serves*. `fa_push_chunk` already did exactly this for the selection **row** — it computes
`qb = fa_sel_row(factx, q_start)` itself rather than accepting it as an argument, precisely so
the tail prefetch (which stages the *next* tile's chunk 0 while the consumer still pops the
current tile's) cannot desync. Threading the length through the same channel inherits the same
guarantee:

```c
static inline uint32_t fa_chunk_nblk(const struct hmx_fa_context * factx, uint32_t c,
                                     uint32_t row_nsel);
```

The length is re-derived at every site from `(qb, kv_head, ib3)` and **never cached across the
loop seam** ([flash-attn-ops.c:257](../../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c#L257)).

### 2.2 What else had to change — and what didn't

| | change |
|:--|:--|
| loop bound | per tile: `row_chunks = ceil(row_nsel / m)`, replacing `factx.n_kv_blocks` in the pipelined path, the fallback path, both prefetch horizons and the epilogue ([:2559](../../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c#L2559)) |
| partial final chunk | **already existed** — `fa_chunk_nblk` returns `min(m, n − c·m)` and `fa_chunk_rows` sums real block rows |
| `m \| u` search rule | **skipped** when counts are present: the host passes `sel_blocks = 0` ([ggml-hexagon.cpp:2136](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2136)) |
| VTCM | **unchanged** — it holds one `Bc`-wide chunk, not the selection |
| host sizing | unchanged: threads, pipelining and VTCM are still functions of `u_max` alone |

The `m | u` rule is what produced the notorious sawtooth in the fixed kernel: `u=47` was forced
to `m=1` (47 chunks) while `u=48` got `m=8` (6 chunks), making `47 → 48` **3.12× faster with one
more block**. With the rule gone, cost is monotone in each row's count — there is no longer any
value in asking for *more* blocks than you want.

## 3. The policy: a per-element threshold, unioned per tile

`LLAMA_SPARSE_ATTN=thr:<c>` ([llama-sparse-attn.h:52](../../../src/llama-sparse-attn.h#L52)).

Scoring is the same subsampled meanpool as the fixed pipeline — per 64-row KV block, K
subsampled 8-of-64 and mean-pooled to one centroid per (block, KV head); per 64-query block, Q
subsampled 4-of-64 and pooled over the `G` query heads of each KV group; score `= q̄·k̄/√d`, per
(query block, KV block, **KV head**).

The rule, per 64-query block: **keep block `j` iff `softmax(scores)_j · avail > c`** — keep any
block carrying more than `c×` the uniform share of attention mass over the reachable blocks.
The 256-query tile then serves the **union** of its four rows' kept sets, sized per tile.

Chosen over XAttention's cumulative-mass cut because it needs no sort/scatter in the graph
(membership is a pointwise comparison), and the two sit on the same measured quality/cost curve
(§5). Every op runs on the NPU
([llama-graph.cpp:2776](../../../src/llama-graph.cpp#L2776)):

| step | op |
|:--|:--|
| membership | `soft_max` → `mul` (avail) → `scale_bias` (×BIG, −c·BIG) → `add` (force) → `clamp(0,1)` |
| union over the tile's 4 rows | 3 × `add` of strided views → `clamp(0,1)` |
| per-row length → `src[6]` | `sum_rows` |
| packed list → `src[5]` | `argsort` desc of the 0/1 row |

Two details that are not free choices:

- **Forced blocks ride in a separate post-threshold channel.** The sink and each row's own block
  must always be selected. A `+BIG` in the *pre-softmax* bias would eat the entire distribution
  and drive every other block below any `c`. So the bias leaf carries two slices — reach bias
  (pre-softmax) and force (post-comparison) — plus an `avail` column
  ([llama-kv-cache.cpp:1806](../../../src/llama-kv-cache.cpp#L1806)).
- **`argsort` of a 0/1 row is sufficient.** Order *among* members is irrelevant (the kernel
  attends to all of them) and every index appears exactly once, which is the kernel's
  disjointness invariant — it clamps out-of-range indices but never deduplicates, and a repeated
  index shifts that block's logits by `+ln 2` with no assert and no NaN.

`avail` rescales the softmax so `c` is length-independent: without it the same `c` would mean
different things for a query block reaching 4 KV blocks and one reaching 64.

## 4. Measured: device

### 4.1 Quality and speed, ctx=4096

Perplexity is wikitext-2, 8 chunks, `-c 4096 -ub 2048`. Speed is `llama-bench -p 4096`,
alternated ×3.

| arm | pp4096 t/s | PPL | Δ vs dense |
|:--|--:|--:|--:|
| dense | 1742 | 17.713 | — |
| fixed `u` = 62% | 1925 (1.11×) | 17.789 | +0.43% |
| **`thr:1.0`** | **2339 (1.33×)** | **17.740** | **+0.15%** |
| `thr:0.5` | 2141 (1.23×) | 17.791 | +0.44% |

`thr:1.0` dominates the fixed pipeline on both axes: **1.20× faster and closer to dense.**
`thr:0.5` keeps more blocks and does *not* improve quality — the extra blocks are below the
noise, so `c = 1.0` ("keep above-average blocks") is the default.

### 4.2 Context sweep

`llama-bench`, alternated ×3; totals are prefill wall time.

| pp | dense | fixed 62% | **`thr:1.0`** | thr/dense | thr/fixed |
|--:|--:|--:|--:|--:|--:|
| 512 | 2159 (237 ms) | 2116 | 2146 (239 ms) | 0.99× | 1.01× |
| 1024 | 2211 (463 ms) | 2196 | 2318 (442 ms) | **1.05×** | 1.06× |
| 2048 | 2012 (1.02 s) | 2122 | 2448 (0.84 s) | **1.22×** | 1.15× |
| 4096 | 1753 (2.34 s) | 1923 | 2339 (1.75 s) | **1.33×** | 1.22× |

**The break-even point moved from ~2k to ~512.** The fixed pipeline is a net *loss* below 2k
(0.98× at 512) because it pays for a rounded-up `u` whether or not the tiles want it. The
threshold is at parity at 512 and ahead from 1024 on. The gain is still growing at 4k.

### 4.3 Where the time goes

`GGML_HEXAGON_PROFILE=1`, one prefill per cell, warmup batch excluded. The scorer is isolated by
subtracting the dense arm's non-attention op time — the rest of the model is identical between
arms.

| ctx | dense attn | sparse attn | **sparse/dense** | scorer | rest of model | attn share of dense |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 33.5 ms | 26.3 ms | **0.78×** | 7.8 ms | 178.7 ms | 16% |
| 1024 | 89.4 ms | 50.5 ms | **0.57×** | 10.7 ms | 347.9 ms | 20% |
| 2048 | 323.6 ms | 116.9 ms | **0.36×** | 18.7 ms | 667.8 ms | 33% |
| 4096 | 945.1 ms | 300.7 ms | **0.32×** | 41.0 ms | 1335.4 ms | 41% |

Each row reproduces the end-to-end bench within ~3% (at 4096: `(945+1335)/(301+41+1335)` =
1.36× predicted vs 1.33× measured).

- **The scorer costs 2–3% of end-to-end at every context** and never approaches the savings: at
  4096 it costs 41 ms against 644 ms saved.
- At 512 there are only 8 KV blocks and sink + own-block are force-kept, so there is little left
  to skip — hence 0.78× and the wash end-to-end.
- **All of it is on the NPU.** Verified in the execute-op log: `FLASH_ATTN_EXT` consumes
  `sparse_ranked` and `sparse_cnt`, both produced in-graph, on the `hmx-pipe`.

### 4.4 The headroom bound

Running `thr:99` collapses the selection to the forced blocks, so attention is near-free and
what remains is *rest-of-model + scorer*:

| arm, pp4096 | t/s | total |
|:--|--:|--:|
| dense | 1750 | 2.34 s |
| `thr:1.0` | 2339 | 1.75 s |
| `thr:99` (floor) | 2653 | 1.54 s |

Attention at `c=1.0` costs **0.21 s** above the floor versus **~0.80 s** for dense. A *perfect*
selector could reach at most 1.52× at this context; `thr:1.0` already realises 1.33× of it.

## 5. Measured: the algorithm, in isolation

`examples/xattn-overlap/sparse_ppl.py`, Qwen3-1.7B bf16 on GPU, wikitext-2 ctx=4096, 8 chunks,
one attention hook so the *algorithm* is the only variable. Dense = 15.083.

### 5.1 A harness bug that nearly cost the right design

The harness applied per-row causal `−inf` **before** group-averaging scores. The device does the
opposite: it scores raw, cuts one list at the group-last row's reach, and leaves per-row
causality to the kernel's mask. Averaging `−inf`-masked rows makes any KV block *recent* to a
256-query group `−inf` for the whole group — so blocks a row could legitimately reach became
unselectable.

The artifact said `bq=256` costs **+13.6%** perplexity where the device measures **+2.4%** at the
same geometry and density. It nearly drove a switch to `bq=64`, which is **1.7× slower on this
kernel** (§5.3). *Anchor a research harness against a device-measured point before trusting its
cross-configuration ranking.*

### 5.2 Geometry grid, after the fix

| geometry | threshold | fixed `u` at matched density |
|:--|:--|:--|
| `bq=64` per-head | **15.071** @ 41% | 15.551 @ 47% |
| `bq=64` shared | 15.375 @ 45% | 15.707 @ 51% |
| `bq=128` per-head | 15.812 @ 41% | 16.066 @ 47% |
| `bq=256` shared | 16.674 @ 46% | 17.890 @ 51% |

Two readings. **The threshold beats fixed `u` in every cell** — at *lower* density. And
granularity dominates quality: finer `bq` is worth more than anything else on this table.

### 5.3 …but fine granularity loses on this kernel

`bq` pins `Br`, and this kernel wants `Br` large (see `sparse-attention.md` §5). Device A/B at
equal `u`, kv=4096, nb=2048:

| `u` | `bq=256` | `bq=64` | penalty |
|--:|--:|--:|--:|
| 24 | 7570 µs | 12926 µs | 1.71× |
| 28 | 9331 | 15841 | 1.70× |
| 32 | 9664 | 16865 | 1.75× |
| 40 | 11663 | 20599 | 1.77× |

KV-block residency (`GGML_HEXAGON_FA_KV_RESIDENCY=2`) recovers only ~5%. So the fine-granularity
path loses end-to-end despite winning on quality-per-block.

### 5.4 The resolution: union-threshold at `bq=256`

Each 64-query block thresholds its own per-head list; the 256-query tile serves the union, sized
per tile. Every row gets a **superset** of its own picks while `Br` stays 256:

| policy | PPL | union size (of avail) |
|:--|--:|--:|
| union-thr 0.9 | **15.235** | 57.7% |
| fixed `u` at matched density | 15.636 | 66.0% |
| union-thr 0.8 | 15.694 | 41.4% |
| fixed `u` at matched density | 17.865 | 51.8% |

And the per-element rule sits on the same curve while needing no sort (per-head, `bq=256`):

| `c` | PPL | union size |
|--:|--:|--:|
| 0.3 | **15.072** (= dense) | 66.5% |
| 0.5 | 15.120 | 55.9% |
| 1.0 | 15.441 | 40.2% |

`c=0.3` reaches dense quality at roughly the deployed cost; `c=1.0` reaches the deployed
pipeline's quality at ~0.68× its cost.

## 6. Negative result: chunk-boundary round-up

A row with 26 blocks and `m=8` pays four chunk overheads but fills 3¼. The partial chunk's fixed
cost (~358 µs, against ~93 µs per marginal block) is already paid, so extending the row to the
chunk boundary should be the cheapest quality available — provided the list entries past the
count are score-ranked so the extension pulls in *next-best* blocks.

Implemented (round-up flag through `op_params[6]`, plus an `argsort` key of
`membership·1e6 + pooled_mass` so non-members rank by score), then **reverted**: I could not
demonstrate it engages, and at `c=1.0` it was speed-neutral and **bit-identical** in perplexity.

The instruments were the problem, and this is worth recording so the next attempt does not
repeat it:

- the `dyn` eval rows in `test-backend-ops` mask everything past each row's count to `−INF`, so
  rounded-in blocks contribute exactly zero — **the reference cannot diverge**, whether or not
  the kernel rounds;
- `perf` mode does not register those rows, so timing could not observe it either;
- and a 6% gap I initially cited from an *unalternated* `llama-bench` pair at `c=3.0` was
  thermal — the controlled profiled measurement in the same run showed 186.1 vs 184.2 ms.

A real test must observe the **effective** row length rather than infer it: expose the
host-chosen `Bc` (hence `m`) so the reference can be built over `ceil(n/m)·m`, or have the
kernel report the lengths it used. The prize is bounded by §4.4 — at most the 0.21 s that
attention still costs above the floor at 4k.

## 7. Constraints and gotchas

- **Single KV stream only.** With a non-unified cache each sequence gets its own stream, and a
  per-sequence ubatch has `n_seqs_unq == 1` while the cache still has `n_seq_max` streams —
  `get_k_pool_src` pools a single-stream layout and asserts. Hit in `llama-perplexity` whenever
  `n_batch > n_ctx`, which packs `n_batch/n_ctx` sequences. Now a clean bail to dense.
- **`ubatch > 256` required.** At `n_tokens <= bq` there is one query block and the selection
  degenerates to the shared layout. Decode always lands here — this is a **prefill** optimisation.
- **Graph node budget.** The threshold scorer adds ~25 nodes per attention layer; without a
  budget bump `graph_reserve` aborts in `ggml_view_4d` with the metadata pool exhausted.
- **`G` capability cliff** (inherited): `br_unit = ceil(32/G)`, and `bq=256` needs a legal `Br`
  dividing 256, so `G ∈ {3,5,6,7,11..15}` cannot run this at all — it fails closed to dense.
  Llama-3.2-3B is `G=3`.

## 8. What is still not measured

- **Retrieval / long-context tasks.** Everything here is wikitext perplexity. A threshold is
  exactly the rule that could truncate a needle in a passkey task, and `c` is the knob that
  would do it. This is the most important gap.
- **Context beyond 8k**, where the cost model says the advantage keeps growing.
- **Per-layer `c`.** The per-layer budget spread in the harness suggests a further win.
- **Decode.** Not applicable as built (see §7).
