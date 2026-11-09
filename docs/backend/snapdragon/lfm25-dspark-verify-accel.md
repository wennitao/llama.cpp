# Faster HMX verification for LFM2.5-1.2B with DSpark, bit-exact

File paths below (`plugins/`, `impl/`, `generation/`, `v8val-*`, ...) are relative to the experiment directory `phone-agent-bench-20261106/lfm25-verify-accel-a226de75/`, which is not in git. The code is in this branch, one commit per switch.

Follow-up to [HMX correctness on LFM2.5-1.2B with DSpark](lfm25-dspark-hmx-consistency.md). Two rounds of changes speed up DSpark (max 9) speculative decoding with the same consistent HMX arithmetic. Every logit hash and every generated token equals the HMX reference.

- **Round 1** (`plugins/v7`, board `198e72da`): DSpark decodes 512 output tokens at **102.5 tok/s from a 512-token prompt (was 72.5)** and **57.7 tok/s from a 4096-token prompt (was 42.5)**. Verification plus the rest of each speculative call drops from 33.7 to 23.0 ms (512) and from 44.6 to 32.3 ms (4096).
- **Round 2** (`plugins/v8`, both attention query tiles share each K/V block; measured on board `ae7378cd`, a faster SM8750 unit): another 1.03x (512) and 1.09x (4096). On that board DSpark runs at **118.8 tok/s (512) and 72.0 tok/s (4096)**, against 79.8 / 47.1 with the original plugin (1.49x / 1.53x) and 115.7 / 66.1 after round 1.

Ordinary HMX decoding speeds up in round 1: 37.6 to 57.7 tok/s (512) and 33.7 to 49.4 tok/s (4096) on `198e72da`, at or above the stock mixed backend's 54.5 / 49.3 tok/s from the previous study, with the consistent HMX numerics. Round 2 does not change 1-token decode (fixed-phase harness); on `ae7378cd`, ordinary decoding with the original plugin and with the final libraries is 41.8 and 67.4 tok/s (512) and 37.4 and 57.7 tok/s (4096).

## Round 2: both query tiles share each K/V block

With G = 4 query heads per KV head, one HMX attention tile holds 8 positions (32 rows). A 10-position verify and the 9-token DSpark draft block therefore ran two query tiles one after the other, and each tile streamed and interleaved every 256-key K/V block itself. `GGML_HEXAGON_FA_DUALTILE=1` runs both tiles through one (KV head, block) loop: the K/V DMA and both interleaves happen once per block. Each tile keeps its own Q DMA buffer and Q tile, S/P/D/O tiles, diag(1/l) tile, m/l, row buffers, slopes, mask cache, zero-mask bits, active-block list and job structs, and calls the same compute functions with the same arguments in the same per-tile order. One operand changes: the padded rows of tile 1's P tiles (24 of 32 rows at 10 positions) are zero instead of tile 0's leftover values; valid rows see them only as 0 x finite. Rows keep their tile and lane, so the lane asymmetry that ruled out Br = 16 cannot occur. The host sets the flag only with `GGML_HEXAGON_FA_FIXED=1`, 9-16 positions (exactly two tiles), at least 3 KV blocks and 2 threads (the pipelined loop), a KV view that is a multiple of 256 keys, no mask or a broadcast mask, and a dual layout that fits VTCM (486,016 B instead of 370,304 B, of 8 MB). When the two tiles need different key blocks, the call runs the unchanged per-tile loop; with a causal mask this happens only while the KV view has at most 4096 keys, at past mod 256 = 247 or 248 for 10 positions and 248 for 9 (past = cached tokens). The drafter's mask rows are identical across its block (non-causal), so a 9-position draft (width 9) takes the dual path from 3 KV blocks up; widths 1-8 run one tile.

Labels below: **original** = the HMX-consistency plugin (lib-base); **round 1** = lib-v7 with round 1's eight switches; **now** = lib-v8 with those switches plus `GGML_HEXAGON_FA_DUALTILE=1`; **switch off** = lib-v8 with round 1's switches only. All round-2 measurements are on board `ae7378cd` (see the board note below).

Server cohort (same protocol as round 1: two passes with reversed order, cooled starts, 512 output tokens, greedy; `generation/gen4-*`). All 20 outputs equal the HMX reference token sequences.

| Prompt | HMX ordinary, original | HMX ordinary, now | DSpark, original | DSpark, round 1 | DSpark, now | DSpark now / round 1 | DSpark now / original |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 41.76 | 67.44 | 79.79 | 115.67 | **118.78** | 1.03x | 1.49x |
| 4096 | 37.38 | 57.68 | 47.11 | 66.09 | **72.01** | 1.09x | 1.53x |

| Prompt | Accepted + bonus / call | Draft ms/call: original / round 1 / now | Verification and other ms/call: original / round 1 / now |
|---:|---:|---:|---:|
| 512 | 3.79 (all three) | 16.64 / 13.02 / 12.68 | 30.80 / 19.71 / 19.19 |
| 4096 | 3.09 (all three) | 25.27 / 19.23 / 17.45 | 40.48 / 27.64 / 25.56 |

The cohort has no round-1 ordinary-decoding arm: "HMX ordinary, now" covers rounds 1 and 2 together, and round 2's effect on 1-token decode comes from the fixed-phase harness below (unchanged).

Per forward, fixed-phase harness (ABCD DCBA order, mirrored between contexts, two passes of 4 measured repetitions, `v8ab-summary.json`):

| ms per forward | 512 | 4096 | 8192 |
|---|---:|---:|---:|
| verify, 10 positions: original / round 1 / switch off / now | 28.81 / 17.91 / 17.96 / **17.37** | 38.38 / 25.77 / 25.61 / **23.50** | 48.50 / 34.29 / 34.37 / **30.05** |
| DSpark draft, 9 positions: original / round 1 / switch off / now | 17.07 / 13.02 / 12.89 / **12.54** | 24.34 / 18.73 / 18.53 / **16.96** | 32.42 / 25.18 / 25.22 / **21.99** |
| 1-token decode: original / round 1 / now | 25.57 / 14.70 / 14.73 | 28.02 / 17.26 / 17.26 | 30.85 / 19.88 / 19.93 |

The switch itself (switch off to now) takes 2.11 ms off a 10-position verify and 1.57 ms off a 9-position draft at 4096, 4.32 and 3.23 ms at 8192, and 0.59 and 0.35 ms at 512 (against round 1: -2.27 / -1.78, -4.24 / -3.19, -0.54 / -0.48 ms). At 4096 and 8192 the verify gain repeats in both passes with a pass-to-pass spread of at most 0.5 ms (verify-10 at 4096: 23.46 / 23.55 ms now, 25.74 / 25.48 switch off, 25.70 / 25.85 round 1). Calls the switch cannot change (verify at 4 and 5 positions, 1-token decode) stay within 0.13 ms of round 1 and within 0.28 ms of switch off; that is the arm-to-arm noise, so the 512 gains (0.35-0.59 ms) are near it, and the server cohort (1.03x, pass ranges 115.5-115.8 vs 118.8 tok/s) carries the 512 result. The implementer's cost model (`impl/fa2/NOTES.md`, fitted to an `a226de75` profile) predicted -1.2 to -2.4 ms per verify-10 and -1.0 to -1.9 ms per draft at 4096, and -0.2 to -0.45 ms per verify at 512; the measured -0.59 ms at 512 is above that range.

The per-op profile (`GGML_HEXAGON_PROFILE=1`, `v8prof-*`, switch off to now, one run each) puts the gain in the attention op. FLASH_ATTN_EXT per 10-position verify drops from 9.76 to 7.65 ms at 4096 and from 18.34 to 14.35 ms at 8192 (-22% at both lengths; measured repetitions only: 9.69 to 7.60 and 18.33 to 14.30). In the 10-position verify no other op moves by more than 0.15 ms, and 1-token decode attention is unchanged. The drafter's 9-position attention is 1.64 ms (4096) and 3.20 ms (8192) faster per draft call. The same logs show that the dual path ran: the 9- and 10-position attention ops get the dual VTCM layout (486,016 B) with the switch and 370,304 B without it.

Correctness evidence for round 2:

- Device sweep (`v8val-*`, `v8val-dualchecks.json`): "now" ran on 24 contexts (13 boundary contexts and 11 dual-tile edge prefixes) with every proposal width 1-9 (verify 2-10), every rollback slot and two sequential passes. **456/456 logit hash files equal** the references. About 123 of these files depend on a dual-path call: w9, w10 and their rollback files where that call is dual, and all 19 files of the three prefill-chunk contexts. The remaining files check that nothing else changed. The edge prefixes (context label = past + 1) cover:
  - the fallback: 760 at 10 positions and 761 at 9 and 10;
  - per-tile zero-mask bits that differ while the active blocks match: 763, 4088, 4089, 4091, 8184 and 8186, and also the boundary contexts 511 and 767 and 760 at 9 positions;
  - a fully masked trailing block for tile 0 above 4096 keys: 4088 and 8184 at 10 positions, 4089 at 9 and 10;
  - 12- and 14-token last prefill chunks that take the dual path: 525, 4111 and 7693.

  The original plugin made the references of the 11 new prefixes on the same board in the same session (209 of the 456 files). Its cross-board reproduction was checked at one context: at 4096 it reproduces the `a226de75` reference hashes (19/19). Controls: switch off (512, 4096) and lib-v8 with every switch off (4096) give 57/57.
- In every arm, batched verify logits and rollback logits equal sequential decoding (max_abs 0). In the 28 arms that have a reference, the drafter's proposals at widths 1-9 equal the reference run's. Proposals are the only drafter check the harness makes: a drafter change is seen only if it flips an argmax (the 10 candidate logits per position that reach the host from the backend top-k sampler were not compared). The same proposal check passes retroactively on round 1's `v7val` sweep (15/15 arms).
- Timing arms 24/24 and profile arms 4/4 hash-equal; server cohort 20/20 outputs equal the reference, with acceptance identical for original, round 1 and now.
- Off device, the binary diff: the merged v8 DSP library differs from v7 in `hmx_flash_attn_ext`, the new dual-tile functions, three helpers now emitted out of line (`hmx_fa_vtcm_layout_build`, `dma_cache_fa_push`, `fa_push_mask_dma_gqa`) and one DMA-push LTO clone (`dma_ring_push_single_1d`, which now computes its return value). Every other function is instruction-identical after address masking. The 20 attention compute functions are instruction-identical with byte-identical constant tables (`impl/fa2/strict-v7-vs-v8.log`); libllama is byte-identical. With the switch off, v8 runs the same operations as v7 but not byte-identical code, and the `hmx_flash_attn_ext` stack frame is 640 B larger.
- Off device, the orchestration models (verbatim orchestration code, stubbed DMA/HMX engines, every job logged):
  - The implementer's model passed 143 cases and caught 17 of 17 injected bugs.
  - The exactness reviewer's independent model passed 987 runs and caught 31 of 31 mutations; a missing zero fill of tile 1's padded P rows is among them.
  - The safety reviewer's model passed 1620 runs under ASan/UBSan and 1332 under MSan, and caught 15 of 16 mutations.
  - The exact device shapes ran through the implementer's model: verify 9/10 and the 9-position draft at 22 of the 24 sweep contexts, plus the 3 prefill chunks. All 69 cases passed (57 dual, 3 fallback, 9 not eligible), each on the predicted path (`impl/fa2/simtest/devcases/`).

  That valid rows do not depend on padded-row content rests on HMX semantics and the device sweep.

Limits:

- **Board.** The new QDC session handed out `ae7378cd`, an SM8750 unit whose prime cores run at 4.32 GHz and whose NPU runs at about 2112 MHz in the profile counters. `a226de75` and `198e72da` run at 4.09 GHz and about 1709 MHz. Absolute numbers are higher on `ae7378cd` (the original plugin decodes DSpark at 79.8 vs 72.5 tok/s from a 512-token prompt). Round 2 was measured only on this board, so every round-2 comparison is within it. The controllers now read the prime cores' `cpuinfo_max_freq` for the cooling gate and the minimum-frequency pin.
- **Not run on the device.** 11-, 13-, 15- and 16-position calls, which the switch accepts, never ran on the device (DSpark with max 9 verifies at most 10 positions); the implementer's orchestration model covers 9 to 16 positions. The path each call took was not counted on the device; the fallback coverage rests on shape analysis and the simulator replay of the exact shapes.
- **Cooling gate.** Every arm and measurement started below the gate (it never had to wait). The gate reads CPU zones and skin, not the NPU.

Found during round 2, pre-existing and unchanged (LFM2.5 is not affected):

1. **Per-head mask race.** On the pipelined HMX attention path with a per-head mask (`mask->ne[2] > 1`), every block's mask DMA goes into one VTCM buffer, so block s+1's mask can land before block s's softmax reads it. The implementer's model found it and both reviews confirmed it; the dual mode refuses per-head masks.
2. **K/V broadcast gap** (found by the safety review). The HMX attention host branch leaves `broadcast_rk3/rv3` at {0, 0}, so K/V broadcast over dimension 3 would read the wrong batch (not hit: one batch).

## Round 1 end-to-end result

Server cohort, board `198e72da` (SM8750, same SoC and clocks as `a226de75`; the original `base` arm reproduces the previous study's numbers on it). Two passes with reversed mode/context order, cooled starts, 512 output tokens, greedy. All 20 outputs equal the HMX reference token sequences.

| Prompt | HMX ordinary, before | HMX ordinary, now | DSpark, before | DSpark, small-m pipe only | DSpark, now | DSpark now / before |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 37.58 | 57.71 | 72.53 | 91.75 | **102.47** | 1.41x |
| 4096 | 33.65 | 49.37 | 42.48 | 49.99 | **57.69** | 1.36x |

| Prompt | Accepted + bonus / call | Draft ms/call, before / now | Verification and other ms/call, before / now |
|---:|---:|---:|---:|
| 512 | 3.79 (unchanged) | 18.52 / 13.97 | 33.66 / 22.96 |
| 4096 | 3.09 (unchanged) | 28.26 / 21.41 | 44.65 / 32.28 |

Long prefill also drops: 4096-token DSpark prefill 3.57 s to 2.97 s. Evidence: `generation/gen1-analysis.json`, `generation/gen1-output-comparisons.json`.

The same cohort protocol with the final libraries (`plugins/v7`, after the partial-flush hardening; `generation/gen3-*`) gives DSpark 102.59 / 57.95 tok/s and HMX ordinary 57.48 / 49.47 tok/s (512 / 4096), with all 8 outputs equal to the reference.

Per forward, fixed-phase harness at a 4096 prefix (ABCCBA, two passes, `v3ab-summary.json`):

| ms per forward | before | small-m pipe | now |
|---|---:|---:|---:|
| 1-token decode | 31.18 | 24.88 | 20.28 |
| verify, 4 positions | 34.11 | 28.03 | 22.34 |
| verify, 10 positions | 42.36 | 36.31 | 29.97 |
| DSpark draft, 9 positions | 27.56 | 23.45 | 20.57 |

At a 512 prefix: 28.20 / 21.81 / 17.39 (decode), 31.59 / 25.21 / 20.72 (verify 10), 19.49 / 14.17 / 13.69 (draft).

## Where the time went, and what changed

Profiles (`GGML_HEXAGON_PROFILE=1` with host timestamps) of the original configuration showed, for a 10-position verify at 4096 (42.5 ms): HMX matmul 22.2 ms, attention 13.2 ms, small ops 3.6 ms, and about 3.4 ms of host round trips between 12 DSP batches per forward. Note: OPBATCH `start`/`cycles` are pcycles, which stop while the DSP idles, so inter-batch gaps must be measured with host timestamps.

All changes are host env switches, default off; off-path builds were checked to run exactly today's code. The final configuration:

```sh
GGML_HEXAGON_MM_HMX_ALL=1 GGML_HEXAGON_FA_SELECT=3 GGML_HEXAGON_FA_FIXED=1 GGML_HEXAGON_SET_ROWS_PACK=1   # as before
GGML_HEXAGON_MM_HMX_PIPE_SMALL=1    # pipelined HMX matmul loop at m <= 32
LLAMA_RS_SLOT_INPUT=1               # rollback slot order is one graph input, not a per-layer CPU ARANGE
LLAMA_EARLY_LAYER_INP_EVENT=1       # the layer-input fence rides in the compute batch
GGML_HEXAGON_FLUSH_SPLIT=64         # send the batch in parts while the host is still building it
GGML_HEXAGON_FA_ZMASK=1             # softmax fast path for all-zero mask blocks
GGML_HEXAGON_MM_INLINE_STORE=1      # small HMX chunk stores on the calling thread
GGML_HEXAGON_SET_ROWS_WIN=1         # rollback conv-state SET_ROWS: one load, VTCM gathers, one store
GGML_HEXAGON_GLU_RETILE=1           # SWIGLU rows split so all 6 HVX threads work
```

| Change | Where | What it fixes | Measured effect |
|---|---|---|---|
| Small-m pipelined HMX matmul | host kparams | `htp_mm_hmx_pipeline(m) = m > 32`, so every decode/verify/draft matmul serialized weight DMA and HVX dequant | -6.3 ms per forward (decode, verify and draft) |
| Slot order as graph input | libllama | each short-conv layer built `{K-1..0}` with ARANGE, which the HTP cannot run: 21 graph splits, 12 DSP batches and 10 host round trips per target forward | 21 splits to 1; -1.6 to -1.9 ms per forward together with the early fence (profiled) |
| Partial batch flush | host | with one split, the host built all ~225 ops (about 9 us each) before the DSP started: ~2 ms idle per forward. A part ends only before an op whose opcode `htp_fusion_try_fuse` can never fuse; fusion and limit-driven cuts follow an exact mirror of the unsplit batch's accounting; no part flush while the session has peers; `event_synchronize` flushes whenever the unsplit flow would. So ops and fusions are those of the unsplit batch | -1.8 ms decode, -2.2 ms verify-10, -1.1 ms draft (4096) |
| Zero-mask softmax fast path | DSP | 16 of 17 key blocks at 4352 keys have an all-+0.0 mask; the mask compare/select path is skipped while the identical add instruction is kept | attention -1.7 ms per verify-10 at 4096; -3.5 ms per 511-token prefill chunk |
| Inline small stores | DSP | a 6-thread pool dispatch for every HMX chunk store, even 704 floats at m=1 | -1.2 ms decode, -1.3 ms verify-4 (4096) |
| Rollback SET_ROWS rewrite | DSP | 20480 DMA rows of 8 B per layer at width 10 | 0.79 to 0.13 ms per verify-10 |
| SWIGLU re-tiling | host | 10 rows on 6 threads left one idle | -0.15 to -0.4 ms per verify-10 |
| Early layer-input fence | libllama | the event fence went out as its own batch after the logits read | the fence joins the last compute batch (one round trip less per forward) |

After round 1, attention at long context was the largest remaining item: about 11.5 ms of a 30 ms verify-10 at 4096 on `198e72da` (matmul plus LM head 15.8 ms, about 2 ms above the weight-streaming floor; small ops under 2.2 ms; host about 0.6 ms). Round 2 (above) takes 22% off the attention of a 10-position verify, measured on `ae7378cd` (9.76 to 7.65 ms at 4096).

## Tried and rejected

- **Contiguous Q4_0 weight DMA** (576-B packed tiles, packed dequant; `GGML_HEXAGON_MM_DMA_CONTIG`): bit-exact in simulation (389.6 M elements) but **1.4-1.6 ms slower per forward** on the device. The premise that the 576-byte row shape limits the DMA rate did not hold; the packed dequant adds instructions.
- **Br = 16 attention tile for 9-16 queries**: not bit-exact. The qf32-pair to fp16 conversion behind `hvx_vec_f32_to_f16(v0, v1)` rounds exact fp16 ties differently in lanes 0-31 and 32-63; moving rows 32-39 into the high half changes `l` in about 14 of 9792 realistic row-blocks. This probably also explains the earlier rejected adaptive-Br prefill divergence. Rows must keep their lanes.
- **Asynchronous residual-bias load**: exact, neutral on time.
- **Multi-entry graph op cache** (`GGML_HEXAGON_GRAPH_CACHE`): no host gain measured, and review found a replay bug after a failed GLU fusion (not reachable for this model).
- **Eager flush at graph end**: splits the early fence into its own batch; no gain.

## Correctness evidence

- Boundary sweep (13 contexts up to 8192 tokens, every proposal width 1-9 so verify widths 2-10, every usable rollback slot, sequential replay twice): final libraries **285/285 logit hash files equal** to the HMX references (`v7val-*`; every intermediate build v3-v6 also 285/285), including off-switch controls; the all-switches build (including the rejected items) 323/323 (`v2val-*`); the small-m pipeline alone 247/247 (`pipeval-*`).
- Partial flush, x86 host harness of the real backend code (stubbed FastRPC, every batch and op logged by value): op streams identical modulo batch cuts for part sizes 1-4096 in four scenarios, in 48 runs with binding per-batch vmem limits, and in the reviewer's three corner-case harnesses (back-to-back graphs under a limit, two peer sessions, event synchronize after an early-sent fence), each of which broke an earlier version (`merged/review-flushsplit/`). The reviewer found and verified the fixes over four rounds and approved the final version. Pre-existing and unchanged: when a fence was sent by an ordinary shared flush and has not run yet, whether `event_synchronize` flushes already depends on timing.
- Generation: 20/20 outputs equal the HMX reference token sequences (`generation/gen1-output-comparisons.json`).
- Cross-board: the `a226de75` reference hashes reproduce exactly on `198e72da`.
- Hexagon simulator (on this host, device compiler flags): zero-mask softmax 97,768 comparisons; inline store 2,184 configurations plus 8,640 control-flow runs of the verbatim matmul loop; SET_ROWS 4,656 cases; GLU re-tiling 308 comparisons; all bit-identical.
- Adversarial code reviews of every item and of the merged tree; no open blocking issue in the final configuration.

## Reproduction and scope

Source: upstream `80d2c0c2c` plus private changes in `merged/source` (diffs per item in `impl/<item>/change.diff`, the partial flush in `impl/flushsplit.diff`). Libraries: `plugins/v7` (host plugin, v79 skel and libllama, built together; deploy all three). `plugins/v3`-`v6` are earlier builds of the partial flush; for this model they run the same op streams at the same speed (v3 was used for the gen1 cohort, v4 for gen2). Final switch set: `v3-final.json`. Controllers: `phase.py` (fixed-phase harness, protocols `*-protocol.json`), `generation/run.py` (server cohort), `deploy_board.py` (hash-verified deployment). The board changed from `a226de75` to `198e72da` during the study; every timing comparison above is within one board.

Round 2: `impl/fa2/change.diff` (made on `merged/source.v3`; its three files are byte-identical in the round-1 tree `merged/source.v7`; the merged tree is `merged/source`), libraries `plugins/v8` (libllama equals v7's), switch set `v8-final.json` (round 1's set plus `GGML_HEXAGON_FA_DUALTILE=1`), protocols written by `mkv8.py`, device chain `run_v8_chain.sh`, extra checks `dual_checks.py`, edge-prefix inputs `inputs-extra/`, reviews `impl/fa2/REVIEW-*.md`.

Greedy decoding only; one target and drafter pair; cooled starts, not sustained load or energy. Nothing was pushed or submitted upstream.
