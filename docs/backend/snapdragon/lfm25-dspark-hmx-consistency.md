# HMX correctness on LFM2.5-1.2B with DSpark

Links to `phone-agent-bench-20261106/` point to the local experiment directory next to the repository checkout; it is not in git. Follow-up: [faster bit-exact HMX verification](lfm25-dspark-verify-accel.md).

The private HMX configuration now preserves ordinary greedy output in the tested cases on phone `a226de75`. Matrix multiplication uses HMX at every token count, and HMX attention uses fixed tiles. Verification and rollback agree exactly with sequential execution, including long prefixes and cached turns.

The reference is ordinary decoding with the same HMX arithmetic. Its output can differ from the stock mixed HMX/HVX backend and the earlier all-HVX reference. This establishes consistency for this configuration, not equivalence to those numerical references or a general proof for other models and samplers.

## What changed

The default backend switches matrix implementations at four rows. Its HVX quantized matrix path quantizes activations to Q8, while HMX uses FP16 activation arithmetic. Attention also switches implementation or tile geometry with shape. Those decisions make one-token decoding, batched verification and differently split prefill execute different arithmetic.

The final change is confined to host dispatch in [htp-kparams.cpp](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/source/ggml/src/ggml-hexagon/htp-kparams.cpp); see the [diff](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/hmx-dispatch.diff). It permits existing HMX matrix kernels below the usual performance threshold and selects fixed attention tiles after checking VTCM capacity. No DSP arithmetic kernel, model weight, sampler or recurrent-state algorithm was changed in this step. The previous packed SET_ROWS optimization remains enabled.

```sh
GGML_HEXAGON_MM_HMX_ALL=1
GGML_HEXAGON_FA_SELECT=3
GGML_HEXAGON_FA_FIXED=1
GGML_HEXAGON_SET_ROWS_PACK=1
```

`MM_HMX_ALL` and `FA_FIXED` are private experimental controls introduced here. `FA_SELECT=3` is the existing force-HMX option. For the target's GQA factor of four, attention uses `Br=8` query positions and `Bc=256` KV positions. Existing pipelining and capacity checks remain active. Unsupported shapes can still fall back; this is not a general backend guarantee.

## Isolation evidence

Maximum absolute full-vocabulary logit difference between batched verification/post-rollback and sequential decoding, across tested verification widths 4, 5 and 10:

| Matrix policy | Attention policy | 512-token prefix | 4096-token prefix |
|---|---|---:|---:|
| Stock mixed | Stock adaptive | 1.3336655 | 3.4899797 |
| Always HMX | Always HVX | 0 | 0 |
| Always HVX | HMX, fixed tiles | 0 | 0 |
| Always HMX | HMX, adaptive tiles | 0.03515625 | 0.03125 |
| Always HMX | HMX, fixed tiles | 0 | 0 |

This separates the matrix and attention effects. Small residual differences in the adaptive HMX attention arm did not flip top-1 on this short forced window, which is insufficient for a long-generation guarantee. [Isolation data](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/isolate-analysis.json).

The [profile](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/profile-fixed-p0-4096-hmx-fixed-profile-analysis.json) confirms HMX for every marked matrix and attention operator during target prefill, single-token decoding and four/ten-position verification. The fix is not silently returning those operators to HVX.

## Correctness validation

- All seven paired generation cases match: 1728 output token IDs in total. The cases are two 512-token continuations from 512/4096-token inventory prompts, native-template weather tool selection, cached and cold tool-result follow-ups, invoice arithmetic, and Python bracket-checking code. Both cached follow-ups also match their respective cold recomputations, despite cache reuse differing by one token between ordinary and speculative execution.
- The final boundary sweep checks all verification widths 2 through 10 and every usable rollback position in 13 cases, including prefixes up to 8192 tokens and positions around KV padding/pipeline boundaries. All 1287 full-vocabulary comparison vectors have zero error. Sequential replay is deterministic.
- SHA-256 checks also establish byte equality for all 234 batched/rollback logit arrays against the corresponding sequential byte ranges. Full vectors for 512, 4096 and 8192 tokens are retained locally. [Certification](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/byte-equality-certification.json).
- Moving the final three prefix tokens out of bulk prefill and processing them individually preserves the sequential logit hashes at both 512 and 4096 tokens. Ordinary and speculative-context sequential references also match at those lengths.

These are diagnostic correctness probes, not a scored agent benchmark. The generation tests are greedy. Task quality, non-greedy sampling, other quantizations, other models and general numerical equivalence remain untested.

Evidence: [boundary checks](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/boundary-analysis.json), [task comparisons](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-tasks/output-comparisons.json), [long generation comparisons](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-long/output-comparisons.json), and [requests and references](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-tasks/references.json).

## Performance

The final cohort has two passes with reversed mode/context order and 512 output tokens per request. Every HMX output matches the earlier HMX reference. Stock ordinary decoding is deterministic against its own reference. Values below are mean decode tokens/s; timing covers 511 steps after the first output token.

| Prompt tokens | Stock ordinary | HMX ordinary | HMX DSpark, max 9 | DSpark / HMX ordinary | DSpark / stock |
|---:|---:|---:|---:|---:|---:|
| 512 | 54.48 | 37.71 | 72.57 | 1.92x | 1.33x |
| 4096 | 49.32 | 33.72 | 42.55 | 1.26x | 0.86x |

Only the HMX-to-HMX comparison preserves the generated token sequence. Stock and HMX outputs first differ at zero-based indices 3 and 6 for the short/long prompts. The stock column is a throughput comparison with a different numerical reference, not a lossless replacement claim. Forcing HMX for single-token decoding slows the baseline, so quoting only 1.92x/1.26x would hide a material cost.

Mean server prefill plus decode time is 7.30 s for HMX DSpark versus 9.55 s for stock ordinary at 512 prompt tokens. At 4096 tokens it is 15.54 s versus 11.68 s. Long prefill alone costs 3.53 s with DSpark, 3.30 s with HMX ordinary, and 1.32 s with stock ordinary. These exclude model loading and HTTP/ADB overhead.

| Prompt tokens | Accepted draft tokens/call | Accepted + bonus/call | Draft ms/call | Verification and other ms/call |
|---:|---:|---:|---:|---:|
| 512 | 2.79 | 3.79 | 18.58 | 33.58 |
| 4096 | 2.09 | 3.09 | 28.18 | 44.61 |

Draft-call counts and durations come from runtime statistics after subtracting warm-up. The final column is total decode time minus draft time, divided by draft calls; it includes verification, state/features, sampling, runtime work and any standalone tail token, rather than isolating one kernel. The long request has one standalone tail token. The measured acceptance and growing per-call cost explain why long-context throughput still loses to the fast stock path.

All effective CPU affinity samples are `6-7`. PMU cycles/task-clock averages span 3.880-4.081 GHz; sampled sysfs clocks are 4089600 kHz. Sampled CPU temperature reaches 65.7 C in this final cohort. These are cooled-start comparisons, not sustained-load or energy measurements. Two passes do not establish broad statistical generality. [Complete timing, acceptance and control analysis](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-analysis.json), [protocol](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-balanced/protocol.json), and [output checks](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-balanced/output-comparisons.json).

Fixed small query tiles also make long prefill expensive. An attempted optimization kept `Bc=256` but enlarged `Br` to fit the query and VTCM. It changed the 4096-token sequential reference, so it was rejected before generation testing. The rejected source, binary and [diff](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/rejected-adaptive-br/change.diff) are retained separately; the final plugin and source use fixed `Br` again.

The remaining systems problem is to preserve numerical behavior across token grouping while making verification and prefill efficient enough to beat fast single-token decoding at measured acceptance. Packed recurrent-state writes and fixed HMX arithmetic are working pieces of a solution. Efficient, numerically consistent attention tiling and single-token matrix execution remain open; this study does not establish a novel general solution or full-agent speedup.

## Reproduction and scope

Device `a226de75`, SM8750, Hexagon v79, target LFM2.5-1.2B-Instruct Q4_0 and DSpark Q8_0. Pinned source `80d2c0c2c56e4a1bbf74be98f64b99927bf230bc`. Both models fit in memory. This is the Instruct model, not Thinking.

The fastest cores were rechecked: cores 6/7 at 4.0896 GHz maximum, versus 2.784 GHz for the other cores. Every inference process starts under `taskset c0` before creating threads, with two CPU decode and two batch workers for both target and draft. Effective thread affinity is checked. WALT minimum frequency is temporarily raised to 4089600 kHz under a restore watchdog; original WALT minimum is 1017600 kHz. Clocks, thermal readings, PMU cycles/task-clock, environment and mapped libraries are retained. Binding and frequency control are separate.

Use the [tested server launcher](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/generation-long/start-server.sh) or [phase launcher](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/start.sh) with the paired host/DSP libraries in `/data/local/tmp/agent-bench/lfm25-hmx-consistency/lib`. The DSP skeleton is unchanged from the packed SET_ROWS experiment and must stay paired with its host interface. Original core libraries and model files remain under `/data/local/tmp/agent-bench/lfm25-verify`.

The [build command](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/build-protocol.json), [deployment hashes](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/deployment.json), [study protocol](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/study-protocol.json), [boundary protocol](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/boundary-protocol.json) and all controllers are retained in the experiment directory. The code is in the `GGML_HEXAGON_MM_HMX_ALL`, `GGML_HEXAGON_FA_FIXED` and `GGML_HEXAGON_SET_ROWS_PACK` commits of this branch.

The phone's original governors and minimum frequencies are restored. No inference or profiler process remains. All 21 original runtime/model/script hashes are unchanged, and the final plugin matches the initially validated fixed-tile binary. [Cleanup and preservation checks](../../../../phone-agent-bench-20261106/lfm25-hmx-consistency-a226de75/cleanup-complete.json).
