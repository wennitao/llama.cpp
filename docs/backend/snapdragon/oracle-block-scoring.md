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

## Limits

- **One document, one model, one context length.** LongBench narrativeqa[0] at T=8192 on
  Qwen3-1.7B. The per-layer spread is tight, but document-to-document and model-to-model
  variation is unmeasured.
- **Recall is a proxy.** No downstream quality number is attached to any row here. A 99.4%
  ratio is not a claim about accuracy; it is a claim about mass.
- **`bs = 64` on both axes**, matching the kernel. The block-size surface is measured
  separately in `block-selection-overlap.md`.
