# QUOKA on the Hexagon HTP

QUOKA — *Query-oriented KV selection for efficient LLM prefill*, Jones et al., Qualcomm AI
Research, ICLR 2026 ([arXiv:2602.08722](https://arxiv.org/abs/2602.08722)) — implemented
faithfully from Algorithm 1 and measured on SM8750 / Hexagon v79, against the XAttention
scorer already characterised in `xattn-scoring.md`.

## What QUOKA is, and the thing to know first

```
 1: if B_CP > N_Q:  M_Q <- mean(Q, dim=2); S_Q <- CosSim(Q, M_Q)
 4:                 Q   <- gather(topk(-S_Q, N_Q), Q)
 6: Q <- Q/norm(Q, -1)                                        (b, n_q,  N_Q, d)
 7: K <- K/norm(K, -1)                                        (b, n_KV, T,   d)
 8: Qbar <- mean(Q.reshape(b, n_KV, n_q/n_KV, N_Q, d), dim=2) (b, n_KV, N_Q, d)
 9: S    <- Qbar K^T                                          (b, n_KV, N_Q, T)
10: Shat <- max(S, dim=2)                                     (b, n_KV, T)
11: I    <- topk(Shat, B_SA)
12: K*, V* <- gather(K, I), gather(V, I)
```

**QUOKA is not block-sparse.** `Shat` carries one score per *token*, step 11 picks
individual keys, and step 12 materialises a compacted cache that a **dense** kernel then
consumes — the paper's stated selling point ("fully compatible with standard dense kernels").
So the request "its block-sparse config / block size" has no referent: there is no KV block
size. Its knobs are `B_CP` (prefill chunk, 128 primary, ablated 128/256/512), `B_SA` (key
budget, 512/1024/2048) and `N_Q` (16).

Two structural facts follow, and they pull in opposite directions on this device:

- **Its selection is chunk-wide.** Step 10 maxes over the query axis, leaving one row per KV
  head. That is the `bq = B_CP` shape this kernel is *fastest* at — QUOKA pays none of the
  1.84–2.09× per-query-block tax that a faithful XAttention does.
- **Its selection granularity is a token.** The top-k sorts `T` elements where XAttention
  sorts `NBk = T/64`. That 64× difference in the sort domain is what decides it here.

## Measured

`QUOKA_SCORE` / `QUOKA_SELECT` / `QUOKA_KNORM` in `test-backend-ops`, whole-graph µs,
Hq=16 Hkv=8 d=128 N_Q=16, **zero `NO-SUPPORT` fallbacks** (see the plumbing notes below) and
3/3 against CPU on the deterministic core.

| Lk | B_CP | score | select | **top-k** | sub-select | K-norm |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 128 | 1528 | 1643 | 115 | 695 | 1091 |
| 1024 | 512 | 2685 | 2770 | 85 | – | – |
| 2048 | 128 | 1815 | 7343 | **5528** | 542 | 1635 |
| 2048 | 512 | 2937 | 8479 | **5542** | – | – |
| 2048 | 2048 | 14225 | 19811 | **5586** | – | – |
| 4096 | 128 | 2413 | 14675 | **12262** | 771 | 2657 |
| 4096 | 512 | 3524 | 15707 | **12183** | – | – |
| 4096 | 2048 | 14799 | 26976 | **12177** | – | – |

`top-k` = `select − score`, i.e. step 11 alone. (`Lk=1024` is degenerate: `B_SA = Lk`, so
the budget is the whole cache and nothing is sorted.)

**Step 11 is the entire story: 12.2 ms at Lk=4096, ~80% of the whole selection pass.** The
same operation in the XAttention path — an argsort over 64 *blocks* rather than 4096 *tokens*
— measured **0–81 µs**, and negative at the largest shapes, i.e. below noise. Reducing the
sort domain 64× is worth 12 ms per chunk on this hardware.

Two secondary costs, both real:

- **Step 7 normalises the whole cache**: a f16→f32 CPY plus an L2_NORM over `T·d·n_KV`, 48 MB
  of traffic at Lk=4096, measured 1091–2657 µs. It is a *cache-write* cost — K is written
  once and read by every later chunk — so the table above assumes K arrives pre-normalised,
  exactly as the XAttention K-side antidiagonal argument does. Charged per chunk it would
  roughly double the scorer.
- **Steps 1–5 scale with `B_CP`**, not with `N_Q`: 542–771 µs at B_CP=128, but the score
  column rises 2413 → 3524 → 14799 as B_CP goes 128 → 512 → 2048 (~6.4 µs per extra query).
  QUOKA's cost is therefore *not* chunk-independent, which is why the paper prescribes 128.

## End-to-end, and why the chunk size decides it

QUOKA's attention leg is dense FA at `kv = B_SA` and `nb = B_CP`. Measured:

| | nb=128 | nb=256 | nb=512 | nb=2048 |
|:--|--:|--:|--:|--:|
| kv=1024 | 462 | 689 | 1242 | 4990 |
| µs per query token | **3.61** | 2.69 | 2.43 | 2.44 |

At its own design point (B_CP=128, B_SA=1024), prefilling 2048 tokens at a final context of
4096 costs 16 chunks × (14675 + 462) ≈ **242 ms**, against dense at **15.9 ms** and
XAttention at 25% density at **12.5 ms**. Even granting a free top-k it is 16 × (2413 + 462)
≈ **46 ms**. (A real chunked prefill sees `Lk` growing, so the triangular sum is ~4× smaller
than the at-final-context figure — the conclusion does not change.)

The multiplier is structural: a 128-token chunk runs the whole selection pass 16× per 2048
tokens, and QUOKA's per-pass cost is dominated by terms that scale with the **cache**, not
with the chunk. On an A100 that is affordable because scoring is negligible against dense
attention; here each pass also pays the ~620 µs per-`graph_compute` constant and the
~1.4 µs/token Q-side floor.

## What is worth taking from it

The scoring *idea* is cheap and orthogonal to the selection granularity:

- **GQA pre-aggregation (step 8)** collapses `n_q` to `n_KV` before the matmul — a free 2×
  here, and XAttention does not do it (it scores per query head).
- **Query sub-selection (steps 1–5)** replaces `Lq/S` scored rows with a fixed `N_Q`, so the
  score matmul stops growing with the chunk. At Lk=4096 QUOKA's matmul is 134 MFLOP against
  XAttention's 2.15 GFLOP.
- **No antidiagonal reversal**, which costs XAttention a flat ~1071 µs `GET_ROWS`.

The obvious hybrid is QUOKA scoring with a *block* selection: pool `Shat` over 64-token
blocks and top-k over `NBk`, which makes step 11 free and keeps the `src[5]` DMA-offset path
so step 12's gather disappears too. Not measured.

## Plumbing found on the way (all backend gaps, not QUOKA's)

Three ops passed the host `supports_op` gate and then failed on the DSP with
`HTP_STATUS_NO_SUPPORT`, each silently dropping one op per graph to a fallback — visible only
as a slower number, since `perf` reports no error:

1. **f32 × f32 `MUL_MAT`.** Every matmul in these graphs elsewhere is f16 × f32. Replaced
   with a broadcast `MUL` + `SUM_ROWS` over `d = 128` (a 512-byte row, the good regime).
2. **A transposing `CONT`** on `[d, Lq, Hq] → [Lq, d, Hq]`, used to bring the query axis to
   `ne0` for `SUM_ROWS`. Replaced with halving adds over `ne1`; **3600 → 1017 µs**.
3. **An i32 `CONT`.** `cpy-ops.c` has no i32 path. `ggml_argsort_top_k` returns a strided
   view, so reshaping its result to feed `get_rows` needed a copy. Fixed by keeping the
   scores 2-D so the view is already the rank `get_rows` wants.

Also worth recording: `GGML_HEXAGON_OPBATCH=0` does not disable batching, it sets the batch
size to zero and nothing executes (`n-ops 0`, no `execute-op` lines, no results). It is not a
usable A/B control.

## The hybrid: QUOKA scoring, block selection, block-sparse FA

Implemented as `test_quoka_block` (`QUOKA_BLK_SCORE` / `_SELECT` / `_E2E`). QUOKA's steps 1–9
unchanged, then `Ŝ` is pooled over KV blocks *before* the query-axis max, top-k over `NBk`,
and the block list goes to `flash_attn_ext` as `src[5]` — no gather. 3/3 against CPU on the
deterministic core, zero fallbacks.

**Reduction order is the trick.** The token index is `ne0`, so `[Lk, N_Q, Hkv] → [Bl, NBk,
N_Q, Hkv]` is a free reshape; pooling that axis first costs one `sum_rows` over **64-element
rows** (16× longer than the `P = Bl/S = 4` rows that make XAttention's reduce run at 1.0
GB/s) and shrinks the query-axis max tree by `Bl` — from 524288 elements to 8192 at Lk=4096.

| Lk | B_CP | u | pool | score | **top-k** | attn | e2e | dense | D/e2e |
|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|
| 2048 | 128 | 8 | max | 2873 | −22 | 476 | 3326 | 751 | 0.23× |
| 2048 | 128 | 8 | **sum** | 1618 | −70 | 542 | 2089 | 751 | 0.36× |
| 2048 | 512 | 8 | **sum** | 2773 | 24 | 1162 | 3959 | 2076 | 0.52× |
| 4096 | 128 | 16 | **sum** | 1967 | 16 | 610 | 2593 | 751 | 0.29× |
| 4096 | 512 | 16 | max | 5309 | 63 | 1368 | 6740 | 4295 | 0.64× |
| 4096 | 512 | 16 | **sum** | 3115 | 42 | 1386 | 4543 | 4295 | **0.95×** |

Three results:

1. **The top-k vanished.** −70 to +63 µs — below noise at every point — against QUOKA's
   **12262 µs** token-level top-k at Lk=4096. Pooling into blocks before ranking is worth
   ~12 ms per chunk, and it removes step 12's gather as a bonus, since `src[5]` is consumed
   as a DMA source offset.
2. **Sum-pool beats max-pool 1.6–1.7×** (1967 vs 4103 at Lk=4096/B_CP=128). One `sum_rows`
   over 64-element rows against a 6-step max bisection. QUOKA's argument for `max` is about
   the *query* axis, where it is preserved; on the block axis `sum` is both cheaper and the
   choice XAttention already makes.
3. **The attention leg is correct and cheap** — 1386 µs at Lk=4096/B_CP=512/u=16 against
   1352 µs for the equivalent shared-selection `FLASH_ATTN_EXT_SPARSE` row in
   `xattn-scoring.md`, 2.5% apart. The selection shape is chunk-wide, so no per-query-block
   tax.

**But it still does not beat dense**, best 0.95×, because the scorer is now the whole pass
(69% at the best point). And the reason is the one this project keeps rediscovering: the
16× FLOP advantage does not translate, because the pass is bound by the `[Lk, ·, Hkv]` score
tensor and its reductions, not by arithmetic. Per query token the two scorers are
**identical**: QUOKA-block 3115/512 = 6.1 µs/token against XAttention 3150/512 = 6.2.

### Where the remaining lever is

QUOKA's steps 1–5 are linear in `B_CP` — measured 15227 µs of score at B_CP=2048 against
3982 at 512 — and that is exactly what stops the chunk being enlarged. Everything *else* in
the scorer is independent of `B_CP` (a fixed `N_Q` matmul plus a `Lk`-sized reduction). So
the configuration worth measuring next is **the QUOKA scorer without query sub-selection at a
large chunk**: a fixed ~2 ms scorer amortised over 2048 query tokens, plus the ~5 ms
attention leg, against 16302 µs of dense. That is the only arrangement in which QUOKA's
cheap-matmul property actually pays here, and it trades away the paper's accuracy argument,
so it is a quality question before it is a latency one.
