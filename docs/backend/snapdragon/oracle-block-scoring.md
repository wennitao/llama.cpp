# How close can a cheap block scorer get to the oracle?

Block-sparse attention picks `u` KV blocks per (query block, KV head). The **oracle** ranks
those blocks by the attention mass they actually carry — the best any selector could do at
that budget, and not implementable, since it needs the attention it exists to avoid. Every
real scorer is an approximation to it, and this measures the gap.

`examples/xattn-overlap/oracle_scoring.py`. Qwen3-1.7B bf16 on an H100, post-RoPE Q/K
captured from a real prefill by patching `apply_rotary_pos_emb`, so the scorers see exactly
what the kernel would. All 28 layers, all 8 KV heads, `bs = 64` on both axes, `T = 8192`,
LongBench narrativeqa.

**Metric: recall** — the fraction of true attention mass landing inside the selected blocks,
averaged over query rows and heads. It is what block-sparse attention actually loses: a row
whose mass is 95% inside its selection computes a softmax over 95% of its own distribution.
Query blocks with fewer than `2u` causally available blocks are skipped; at `avail = u+1`
every policy picks `u` of `u+1`, scores ~1.0, and those rows would otherwise dominate the
average and compress every difference toward the oracle.

## The result

Recall as a fraction of the oracle's, sink and diagonal blocks force-included for every
scorer (see below for why):

| density | u | **oracle** | xattn | meanpool | recent | maxpool | quoka | random |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 6.25% | 8 | **0.842** | 99.4% | **99.4%** | 98.8% | 96.4% | 94.4% | 85.2% |
| 12.5% | 16 | **0.884** | 99.4% | **99.4%** | 98.3% | 96.9% | 95.0% | 83.8% |
| 25% | 32 | **0.922** | 99.4% | **99.3%** | 97.6% | 97.6% | 96.4% | 84.4% |
| 50% † | 64 | **0.958** | 99.4% | **99.3%** | 98.4% | 98.3% | 97.5% | 84.7% |

> † **The 50% row is one query block.** `recall_at` skips blocks with `avail < 2u`; at
> `u = NB/2` only `a = NB-1` survives — and none do when `NB` is odd, which dropped
> `niah_single_1` from the cell entirely. 6.25 / 12.5 / 25% score 113 / 97 / 65 of 128 blocks
> and are sound. Do not quote the 50% row; measuring it needs a longer context, not a
> different scorer.

- **xattn** — XAttention: antidiagonal packing at stride 16, softmax at `1/(√d·S)`, block-pool,
  max over the GQA group.
- **meanpool** — block-mean of Q dotted with block-mean of K. One `d`-vector per block per
  side. The cheapest scorer that uses the data at all.
- **recent** — position only: sink + the most recent blocks. No data dependence whatsoever.
- **maxpool** — block-max of the raw logits.
- **quoka** — QUOKA: keep the `N_Q=16` lowest-cosine-similarity-to-mean queries, GQA
  pre-aggregate, cosine similarity, max over queries.
- **random** — random fill. With sink+diagonal forced on top, this isolates what those two
  blocks buy on their own.

### Three findings

**1. The cheapest scorer is already at the oracle.** `meanpool` — one mean vector per block
per side, a single `[NBq, d] × [NBk, d]` matmul, no reversal, no softmax, no per-query work —
ties XAttention to within 0.1 points at every density, and does so with a tight per-layer
spread (worst layer 0.972 at 6.25%, 0.982 at 25%). XAttention's antidiagonal machinery buys
nothing measurable here over a block mean.

**2. Sink + diagonal is most of the game.** `random` with those two forced reaches **84–85%**
of oracle. Everything data-dependent scoring achieves is the remaining ~15 points, and the
spread among all real scorers is under 3 of them. A purely positional policy (`recent`)
takes 97.6–98.8%.

**3. Nothing works without the sink.** Removing the force (`--force 0`) collapses XAttention
from 99.4% to **80.6%** and meanpool from 99.3% to **77.5%**, while `recent` and `maxpool` are
almost unmoved because they find the sink on their own. XAttention's `find_blocks`
force-includes sink and diagonal unconditionally and the llama.cpp pipeline carries them in
its bias leaf, so this is what deployments do — but it means any scorer comparison that omits
it is mostly measuring whether the sink was found, not scoring quality.

### What limits recall is the budget, not the scorer

The oracle itself is only **0.842 / 0.884 / 0.922** at these densities (the 50% figure is the
single-row cell flagged above). The distance
from a good cheap scorer to the oracle is 0.6% of oracle; the distance from the oracle to 1.0
is 4–16%. **Raising `u` buys an order of magnitude more recall than improving the scorer** —
which is the same conclusion the kernel measurements reached from the other side, where the
cost of extra blocks is sub-linear and the cost of a bad `u` is a 2.9× cliff.

Layer 2 is the hardest layer for every policy, and it is the oracle that is hard there
(0.704 at 6.25%), not the scorers: their ratios to it hold up.

## Generalisation: two models, 11 datasets

Extended to **RULER** (real instances from `simonjegou/ruler` at 8192, not regenerated) and
**LongBench**, two instances each, all layers, four densities — 44 (dataset, density) cells
per model.

| | Qwen3-1.7B | | | Llama-3.2-1B-Instruct | | |
|:--|--:|--:|:--|--:|--:|:--|
| scorer | mean | min | worst case | mean | min | worst case |
| **xattn** | **99.3%** | 98.9% | cwe @ 25% | **99.5%** | 99.1% | cwe @ 25% |
| **meanpool** | **99.1%** | 98.6% | multiquery @ 50% | **99.2%** | 98.6% | cwe @ 50% |
| maxpool | 97.2% | 95.2% | cwe @ 6.25% | 98.3% | 97.5% | cwe @ 12.5% |
| recent | 96.4% | **81.3%** | **vt @ 50%** | 97.1% | **89.2%** | **vt @ 50%** |
| quoka | 95.8% | 92.3% | lcc @ 6.25% | 98.5% | 97.8% | lcc @ 6.25% |
| random | 84.7% | 80.5% | lcc @ 12.5% | 87.1% | 83.9% | lcc @ 12.5% |

Datasets: `niah_single_1`, `niah_multikey_2`, `niah_multiquery`, `vt`, `cwe`, `qa_1` from
RULER; `narrativeqa`, `qasper`, `gov_report`, `lcc`, `passage_retrieval_en` from LongBench.
Oracle absolute recall spans 0.798–0.979 (Qwen) and 0.827–0.979 (Llama).

**The tie holds everywhere.** `meanpool` is within **0.2 points of XAttention on average and
0.9 at its worst**, across two architectures, retrieval and aggregation and summarisation and
code, and an 8× density range. XAttention is consistently very slightly ahead — it never
loses — but the margin never exceeds 0.9 points of oracle.

**Positional selection is not safe.** `recent` looks fine on average (96–97%) and collapses to
**81.3%** on RULER `vt`, where the answer is a chain of variable assignments scattered through
the context — exactly the structure a recency prior cannot see. Any policy without data
dependence carries that tail risk.

**QUOKA is architecture-sensitive**: 95.8% mean on Qwen3 against 98.5% on Llama, and the gap
is widest at low density. Qwen3 applies QK-norm before RoPE, which changes the cosine geometry
its query sub-selection depends on. That is a hypothesis, not a measurement.

**`cwe` and `vt` are the hard tasks** for every scorer, and `passage_retrieval_en` the easiest
— which is the expected ordering: aggregation over the whole context versus a single localised
target.

## Limits

- **Two models, both small.** 1–1.7B. Larger models and different GQA ratios are unmeasured.
- **Two instances per dataset**, 8192 tokens.
- **Recall is a proxy.** No downstream quality number is attached to any row here. A 99.4%
  ratio is not a claim about accuracy; it is a claim about mass.
- **`bs = 64` on both axes**, matching the kernel. The block-size surface is measured
  separately in `block-selection-overlap.md`.
- **50% density is unmeasurable here** at `T = 8192` with `slack = 2` — one query block
  survives the filter. Everything at 50% in this document should be read as unsupported.
- **The density label is per-document, not per-op.** `u = density × NB` is taken over the
  whole 8192-token capture and applied to every query block, so the harness models one
  flash-attention op over the entire prompt. The kernel's density is per-op
  (`kv_eff / nek1`, `ggml-hexagon.cpp:2095`) and a chunked prefill scales `u` with each
  chunk's cache. The mean *local* budget fraction the harness evaluates is 0.150 / 0.232 /
  0.347 against labels 0.0625 / 0.125 / 0.25 — those factors are exact; the resulting recall
  overstatement is estimated at ~3 / 2 / 1 points and has **not** been measured.

## meanpool on the NPU

Implemented as `test_meanpool_block` and measured on SM8750. 2/2 against CPU, zero
`NO-SUPPORT` fallbacks. `bs=64`, `Hq=16`, `Hkv=8`, `d=128`, K arriving pre-pooled (its block
means are a cache-write cost, the same amortisation applied to XAttention's K-side reversal).

| Lq | Lk | score | +argsort | **selection leg** | K pooled inline | extra | XAttention | cheaper by |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 2048 | 664 | 33 | **697** | 1515 | +817 | 1553 | 2.23× |
| 2048 | 2048 | 2383 | 56 | **2440** | 3205 | +766 | 4652 | 1.91× |
| 512 | 4096 | 656 | 46 | **702** | 2143 | +1440 | 2530 | **3.60×** |
| 2048 | 4096 | 2362 | 82 | **2444** | 3837 | +1393 | 7477 | **3.06×** |

All values whole-graph minus the ~620 µs per-call constant.

**meanpool is 1.9–3.6× cheaper than the XAttention scorer** at the same shapes, and the
argsort is free (33–82 µs) as it is for every block-granularity selector.

### But the cost is reading Q, not scoring it

Two readings of the table matter more than the ratio:

- **The cost does not depend on `Lk` at all.** 702 µs at Lk=4096 against 697 at Lk=2048 — a
  2× cache for 0.7% more time. The matmul is 4.19 MFLOP; it is invisible.
- **The cost scales with `Lq`.** 702 → 2444 µs for a 4× chunk.

So essentially all of it is the Q block-mean: a halving-add tree over the token axis, touching
every element of a `[d, Lq, Hq]` tensor about twice. The 512× FLOP advantage over XAttention
buys only 1.9–3.6×, because neither scorer was ever FLOP-bound — this backend charges for
*touching Q*, and meanpool touches all of it.

Pooling K inline instead of at cache-write time adds **766–1440 µs**, roughly doubling the
leg and confirming the amortisation is load-bearing rather than a modelling convenience.

### Which is why it still loses to a subsampled scorer

At Lq=2048, Lk=4096 the selection legs line up as:

| scorer | selection leg | quality (ratio to oracle) |
|:--|--:|--:|
| XAttention | 7477 | 99.3–99.5% |
| **meanpool** | **2444** | **99.1–99.2%** |
| QUOKA + strided sample + block selection | **690** | not measured at this configuration |

QUOKA-strided is 3.5× cheaper than meanpool for the same reason meanpool is 3× cheaper than
XAttention: it never reads most of Q. It scores `N_Q = 16` sampled query rows per chunk
instead of all `Lq`, so its cost is independent of the chunk as well as of the cache.

**The synthesis is the obvious one and is not yet measured: meanpool over a subsample.** Take
the mean of 4 or 8 rows per block rather than all 64 — an 8–16× cut in the only term that
costs anything — and check on GPU whether the ratio-to-oracle holds. That is one parameter in
the harness on each side, and it is the next experiment.

> The `MEANPOOL_E2E` rows are *not* comparable to the QUOKA end-to-end figures: meanpool
> emits one selection row per 64-token query block, so `op_params[5] = 64` and `Br` is pinned
> to 64, paying the full 1.9× query-block tax. Merging its score rows over `k` adjacent blocks
> — the `(k, u)` surface in `block-selection-overlap.md` — is what makes it comparable, and
> `k = 4` is the optimum there.

## The strided sample, and what a chunk-wide selection costs

Two configurations the NPU work depends on had never been measured for quality: the
**evenly spaced query sample** that replaced QUOKA's cosine ranking, and the **chunk-wide
selection** that makes `Br` large. Both are measured here, plus `meanpool_s8` — meanpool over
8 sampled rows per block instead of all 64, the synthesis the NPU cost model pointed at.

`merge = k` collapses `k` adjacent query blocks onto one shared list, as a chunk-wide
selection does. The oracle is merged too (by summed mass — the best list a group could
share), so a merged scorer is not charged for coarsening twice. Recall is still scored per
original query block against that block's own mass.

### First: what a shared list actually costs (`--faithful 1`)

Getting the *cost* of merging right needed two corrections, both found by auditing the
harness against the kernel rather than by looking at the numbers. Both made merging look
cheaper than it is, and both grow with `k`.

**The kernel cuts the list once, over the group's reach.** `n_sel` is "deliberately NOT a
function of the query block" (`ggml/src/ggml-hexagon/htp/flash-attn-ops.c:219-230`): every row
of the group gets the same `u` indices, all of them staged and computed, and entries past an
individual row's diagonal are killed only by the per-row mask (`:1477-1493`). Those slots are
spent and return nothing. `recall_at` used to re-cut a fresh top-`u` inside each fine block's
*own* causal prefix, which silently refunded them.

**The bias leaf is indexed by the coarse query block.** It forces exactly two entries per
group — block 0 and the group's last reachable block
(`tests/test-backend-ops.cpp:8478`). Applying `force_sink_diag` per fine block *before* the
merge instead put `k` forced diagonals into the shared row; at `k=32` that is 32 forced
entries competing for a 32-slot budget.

`--faithful 1` does what the kernel does: cut once over the group reach, force sink and
group-last only. No causal mask is needed to charge for the waste — `mass[a,:,j]` is exactly
0 for `j > a`, so a slot spent on a future block gathers nothing on its own. At `k=1` the two
paths are identical by construction and reproduce to four decimals on both models at every
density, which is the check that nothing else moved.

What moved, absolute oracle recall at 25%:

| | `k=4` | `k=32` |
|:--|--:|--:|
| Qwen3-1.7B, refunding metric | .915 | .878 |
| Qwen3-1.7B, **faithful** | **.911** | **.831** |
| Llama-3.2-1B, refunding metric | .915 | .875 |
| Llama-3.2-1B, **faithful** | **.911** | **.830** |

`k=4` was roughly right. **`k=32` was credited with about 4.5 points it does not have** — and
chunk-wide selection is what the end-to-end NPU result runs. Everything below is the faithful
metric.

### The scorer table

Ratio to the merged oracle, 11 datasets × 3 densities, all layers, 2 instances:

| | | Qwen3-1.7B | | | | Llama-3.2-1B | | | |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| k | oracle | xattn | meanpool | **meanpool_s8** | **quoka_str** | xattn | meanpool | **meanpool_s8** | **quoka_str** |
| 1 | .881/.887 | 99.3% | 99.1% | **99.0%** | 96.4% | 99.5% | 99.2% | **99.2%** | 98.7% |
| 4 | .864/.871 | 99.0% | 98.7% | **98.6%** | 94.2% | 99.4% | 99.1% | **99.0%** | 98.7% |
| 32 | .676/.706 | 96.1% | 96.0% | 95.9% | 92.5% | 97.7% | 97.0% | 96.9% | 97.4% |

> 50% density is **not measurable on this harness at T=8192**. `recall_at` skips query blocks
> with `avail < 2u`, and at `u = NB/2` that leaves exactly one block (`a = NB-1`) per
> layer-instance — and none at all when `NB` is odd, which silently dropped `niah_single_1`
> from every 50% cell. The four-density tables earlier in this document carry that defect in
> their 50% row; 6.25 / 12.5 / 25% score 113 / 97 / 65 of 128 blocks and are sound.

### 1. Subsampling Q is free

`meanpool_s8` reads **1/8 of Q** and lands within **0.1–0.2 points** of full meanpool on both
models at every merge factor. On the NPU meanpool's entire cost is the Q block-mean — 702 µs
at Lq=512 and 2444 at Lq=2048, independent of `Lk` — so this is an ~8× cut in the only term
that costs anything, for no measurable quality.

### 2. The strided sample beats QUOKA's own ranking

`quoka_str` scores **96.8% against QUOKA's 95.8%** on Qwen3 and **98.8% against 98.5%** on
Llama. Replacing the cosine-dissimilarity query selection with an evenly spaced sample was
adopted purely because it made the scorer chunk-independent (15227 → 1399 µs); it turns out
to be *better*, on both architectures. The paper's central claim — that low
cosine-similarity-to-mean queries are the informative ones — does not reproduce as a ranking
signal at block granularity here.

### 3. Chunk-wide selection is a density decision, and this is the number that was missing

Merging costs about a point at `k = 4` — which is exactly where the `(k, u)` cost surface
puts its optimum. At `k = 32` it depends entirely on the budget. Oracle absolute recall,
Qwen3-1.7B / Llama-3.2-1B:

| density | u | oracle, k=1 | oracle, k=32 | the merge costs | best scorer @ k=32 |
|--:|--:|--:|--:|--:|--:|
| 6.25% | 8 | .840 / .852 | **.542 / .598** | **29.8 / 25.4 pts** | 95.8% / 97.2% |
| 12.5% | 16 | .883 / .888 | **.655 / .690** | **22.8 / 19.8 pts** | 95.2% / 96.9% |
| 25% | 32 | .921 / .922 | **.831 / .830** | **8.9 / 9.2 pts** | 98.0% / 99.1% |

Thirty-two query blocks sharing one list of 8 cannot be served at all: the oracle itself falls
from 0.840 to 0.542, and no scorer can recover what the budget never had. **The 2.67×
end-to-end NPU result uses chunk-wide selection at 25% density**, the benign end of this
curve — but it gives up **8.9 points** of absolute recall to the coarsening alone, and the
same configuration at 6.25% would give up 30.

**The earlier claim that the scorer stops mattering at `k = 32` does not survive the metric
fix.** It was an artifact of the refund. On Qwen3 the real scorers now spread **7.3 points**
at `k=32`/25% (xattn 98.0%, meanpool 97.6%, recent 96.6%, maxpool 93.3%, quoka_str 92.5%)
against 1.2 before. On Llama the spread is 1.7 points and the old reading broadly holds. So:
coarsening still dominates, but on at least one architecture a bad scorer compounds it rather
than being masked by it.

## Subsampling K, and why it decides the deployment

Every `meanpool_sN` above subsamples **Q only** and takes a full 64-row K mean. That
asymmetry was never deliberate, and it matters: Q pooling costs `O(n_tokens)` per layer per
ubatch while K pooling costs `O(n_kv)`, so on a long prefill the K side is the whole cost.
Measured inline on the HTP it is **+1366 to +1991 us** per call at `Lk` = 2048-4096, which is
more than the entire `meanpool_s4` scorer it feeds (1570 us at `Lq=2048, Lk=4096`).

There is a real reason to expect K to subsample *worse* than Q. On the Q side the 64 rows are
a redundant view of the same ranking question, which is why 8-of-64 costs 0.1 points. On the
K side the block **centroid is the object being ranked**, so a sample is a noisy estimate of
it rather than a redundant view of it. The symmetry argument is not valid a priori.

Absolute recall, `k=4`, 25% density, faithful metric, Q pooled at `bq`, 11 datasets x all
layers x 2 instances:

| Q / K rows | Qwen3-1.7B | vs K64 | Llama-3.2-1B | vs K64 |
|:--|--:|--:|--:|--:|
| Q64 / K64 | .902 | | .904 | |
| **Q4 / K64** | **.901** | — | **.903** | — |
| **Q4 / K16** | **.899** | −0.002 | **.901** | −0.002 |
| Q4 / K8 | .896 | −0.005 | .899 | −0.004 |
| Q4 / K4 | .892 | −0.009 | .896 | −0.007 |
| Q4 / K2 | .882 | −0.019 | .889 | −0.014 |
| Q4 / K1 | .866 | −0.036 | .876 | −0.027 |
| `recent`, no scorer | .882 | −0.019 | .887 | −0.016 |

**The asymmetry is real but mild.** Q is free to subsample (Q64 → Q4 costs 0.001); K is
roughly 5-9x more sensitive per unit of subsample. But the absolute cost stays small down to
K8: **K16 costs 0.2 points and K8 costs 0.5**, against a 4x and 8x cut in the only term that
grows with context.

Two consequences for the deployment:

- **K8 or K16 is the operating point**, and it makes inline pooling affordable: the +1839 us
  measured at K64 scales with rows read, so K8 is roughly +230 us — below the scorer itself.
- **No pooled-K cache is needed.** Its whole justification was that K pooling is O(n_kv) per
  layer per ubatch. Subsampling removes most of that, and a strided
  `[n_embd_k_gqa, KSUB, NBk]` view over the cache tensor has strictly increasing strides, so
  it stays on the NPU like the full-row version.

> `Q4/K4` (.892) still beats `recent` (.882) on Qwen by a point of mean recall, and by more
> in the tail — but it is the *first* configuration where the margin over having no scorer at
> all gets thin. K8 keeps a 1.4-point margin; K1 falls below `recent` entirely.

## The table: quality against cost

Quality is the ratio to the merged oracle at `k=4`, 25% density, 11 RULER/LongBench
datasets × all layers × 2 instances. Cost is the **selection leg on the HTP** at
`Lq=2048, Lk=4096, u=16, bs=64` — the whole-graph time minus the ~620 µs per-call constant,
K pre-pooled, measured in one invocation so every row shares `C` and the thermal window.
Both axes are for the same algorithm; nothing is modelled.

| scorer | Qwen3-1.7B | Llama-3.2-1B | **NPU µs** | vs XAttention |
|:--|--:|--:|--:|--:|
| xattn | **99.2%** | **99.3%** | 7505 | 1.0× |
| meanpool | 99.0% | 99.0% | 2489 | 3.0× |
| **meanpool_s8** | 98.9% | 99.0% | **727** | **10.3×** |
| **meanpool_s4** | 98.8% | 98.9% | **378** | **19.9×** |
| meanpool_s2 | 98.5% | 98.8% | not built | — |
| quoka_str | 96.2% | 98.6% | 799 | 9.4× |
| quoka | 94.6% | 98.3% | — | — |
| maxpool | 96.5% | 98.0% | — | — |
| recent | 96.8% | 97.3% | ~80 (argsort only) | — |
| random | 70.6% | 75.8% | ~80 (argsort only) | — |

`meanpool_sN` = block mean over `N` evenly spaced rows of each 64-token block instead of all
64. `quoka_str` = QUOKA with the same substitution on its query axis.

**`meanpool_s4` is the operating point: 98.8 / 98.9% of oracle, 378 µs, 19.9× cheaper than
the XAttention scorer.** It dominates `quoka_str` on both axes — better quality *and* half the
cost — and buying the last 0.4 points of quality by going to XAttention costs **20×**. Under
the faithful metric `quoka_str` is 2.6 points behind on Qwen3 rather than 1.7, so the margin
widened; the ordering did not change.

Two structural facts make the frontier this steep:

- **Cost is Q-bound, so it tracks the subsample directly.** meanpool 2489 → 727 → 378 µs at
  64 → 8 → 4 rows, while the matmul (4.19 MFLOP) and the argsort (~80 µs) never move. At
  `Lq=512` the `s8` and `s4` graphs measure 614 and 547 µs whole-graph, i.e. *at or below the
  per-call constant* — the scorer has become too cheap for this method to resolve.
- **Quality is flat in the subsample.** 64 → 2 rows costs 0.4 points on Qwen and 0.2 on
  Llama. A block's 64 query rows are near-redundant for the purpose of ranking KV blocks,
  which is the same redundancy QUOKA exploits — but an evenly spaced sample extracts it more
  cheaply, and better, than a cosine ranking.

### Where the remaining loss actually is

One configuration, Qwen3-1.7B at 25% density, `bs=64`, `k=4`, meanpool_s4 — each step
measured against the one above it, so the terms add:

| | recall | this step costs |
|:--|--:|--:|
| exact attention | 1.000 | — |
| per-block oracle at `u = NBk/4` | 0.921 | **7.9 pts** — the budget |
| merged oracle at `k=4` | 0.911 | **1.0 pts** — the merge |
| meanpool_s4, 98.8% of it | 0.900 | **1.1 pts** — the scorer |
| | | **10.0 pts total** |

**The budget is 79% of the loss.** The scorer and the merge together are 2.1 points. Llama
gives 7.8 / 1.1 / 1.0 for 9.9 — the same shape.

Chunk-wide is a different accounting entirely. At `k=32`, same density and scorer:

| | recall | this step costs |
|:--|--:|--:|
| per-block oracle | 0.921 | **7.9 pts** — the budget |
| merged oracle at `k=32` | 0.831 | **8.9 pts** — the merge |
| meanpool_s4 | 0.811 | **2.0 pts** — the scorer |
| | | **18.9 pts total** |

The merge now costs as much as the budget, and the scorer term doubles because a shared list
is harder to rank well. Chunk-wide is the fastest configuration measured and the most
expensive in recall by a factor of two.

**And a mean over 28 layers hides a lot.** Per-layer recall is now persisted (`bylayer` in the
json). At `k=4`/25%, `meanpool_s4` averages 0.900 on Qwen3 but its worst layer (6) is
**0.829**; on Llama the mean is 0.901 and layer 0 is **0.781**. At `k=32` the worst layers are
0.702 and 0.631. Quote the mean and you are quoting something no layer necessarily
experiences.

> Two earlier versions of this section were wrong. The first said "~15 points total: 1 scorer,
> 3 merge, 11 budget" — the 3-point merge came from a density-*averaged* column rather than
> the 25% row, and the 11-point budget was `1 − 0.892`, which already contained the merge and
> so double-counted it. The correction to "9.4 total: 7.9 / 0.6 / 0.9" fixed the arithmetic
> but still used the refunding merge metric; with `--faithful 1` the merge term is 1.0, not
> 0.6, and the scorer term 1.1, not 0.9.

### Note on scope: why `u` and `k` appear in a scorer comparison

`u` has to. Recall is *defined* at a budget — "what fraction of the mass does the top-`u`
capture" — so there is no scorer comparison without one. It is a parameter of the metric,
not a variable under study, and it is swept over four densities only to confirm the scorer
ranking does not depend on it. It does not.

`k` did **not** have to. It was added because the NPU result rests on a chunk-wide selection
whose quality had never been measured, and folding it into the same sweep was the cheapest
way to close that gap. But it is a *deployment* question, not a scoring one, and the table
above is a deployment accounting rather than a result of the scorer comparison — it imports
a density, a merge factor and a scorer that the scoring question does not fix. Read the
quality/cost table for "which scorer"; read this section only as "what one particular
configuration gives up".
