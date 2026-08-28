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
| 50% | 64 | **0.958** | 99.4% | **99.3%** | 98.4% | 98.3% | 97.5% | 84.7% |

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

The oracle itself is only **0.842 / 0.884 / 0.922 / 0.958** at these densities. The distance
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
