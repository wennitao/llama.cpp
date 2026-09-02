# Heterogeneous NPU + GPU: measured, and why it does not pay on this SoC today

Question asked: the Adreno GPU sits idle while the HTP runs the whole model — can part of the
compute move to it, and can the data movement be arranged so the two do not fight over DRAM?

Short answer: **not with the current scheduler, and not by much even with a better one.** The
GPU delivers ~1/5 of the NPU's prefill throughput on this model, the scheduler can only run the
two *serially* (neither backend implements events), and decode — where a second unit would be
most welcome — is bandwidth-bound on DRAM the two units share, so the GPU measures the **same**
decode rate as the NPU and can add nothing.

Everything below is SM8750 (Hexagon v79 + Adreno 830), Qwen3-1.7B Q4_0, `llama-bench`
`-fa 1 -ngl 99`, one binary carrying both backends (`GGML_HEXAGON=ON GGML_OPENCL=ON`).

## 1. Calibration: what the GPU actually delivers

| pp | GPU only (Adreno 830) | HTP only | GPU / HTP |
|--:|--:|--:|--:|
| 512 | 559 t/s | 2159 | 0.26 |
| 1024 | 515 | 2211 | 0.23 |
| 2048 | 433 | 2012 | 0.22 |
| 4096 | 333 | 1767† | 0.19 |

† `ub=1024`; see §3 for why.

The GPU's share falls with context because its flash-attention is weaker than the HMX kernel
and it cannot run the sparse one. The "~1.5 TFLOPS peak" is not the operative number: on the
GEMM phase the HTP realises ~8.6 effective TFLOPS (11.5 TFLOP of projections in 1.335 s at 4k,
`dynamic-sparse-attention.md` §4.3), so the GPU is a **~15–20% compute partner**, not a peer.

Two constraints surfaced on the way:

- **OpenCL cannot allocate the `ub=2048` compute buffer.** It needs 1187 MiB in one buffer and
  the device's max single allocation is 1024 MB (`ggml_opencl: max mem alloc size: 1024 MB`), so
  any configuration with GPU layers is forced to `ub ≤ 1024`.
- **`ub=1024` costs the HTP 14%** on the sparse path (2006 vs 2325 t/s at pp4096) — the tile
  amortisation the kernel is built around. Every heterogeneous arm therefore starts from a
  handicapped NPU.

## 2. The scheduler runs the two units serially

`pipeline_parallel` in `llama-context.cpp` requires `caps.async && caps.events` on every
non-CPU device. Neither backend qualifies:

| | `caps.async` | `caps.events` | `synchronize` |
|:--|:--|:--|:--|
| Hexagon | true | **false** | flush the dspqueue |
| OpenCL | **false** | **false** | `clEnqueueBarrierWithWaitList` + `clWaitForEvents` |

So with two devices the scheduler falls back to: copy split inputs, compute split, synchronize,
next split — strictly in graph order. A layer split therefore executes **NPU layers, then GPU
layers, per ubatch**, and can only be slower than the NPU alone. Row/tensor split is not an
alternative: `LLAMA_SPLIT_MODE_ROW` needs a backend-provided `ggml_backend_split_buffer_type`
(CUDA only), and `LLAMA_SPLIT_MODE_TENSOR` is architecture-gated.

## 3. Measured: layer split, `ub=1024`, pp4096

| arm | t/s | vs HTP-only |
|:--|--:|--:|
| HTP only, dense | 1767 | 1.00 |
| HTP 24 layers / GPU 4, dense | 1235 | **0.70** |
| HTP 26 / GPU 2, dense | 1593 | 0.90 |
| HTP only, `thr:1.0` | 2006 | 1.00 |
| HTP 24 / GPU 4, `thr:1.0` on HTP layers | 1316 | **0.66** |

Exactly the serial picture: each GPU layer costs ~5× an HTP layer (439 vs 83 ms per layer over
the full 4k prefill), so moving four layers loses 30%. Under the sparse kernel the NPU is faster
still and the GPU's relative cost is worse.

### 3.1 What perfect pipelining would buy

Suppose events existed on both sides and the scheduler overlapped ubatch *i* on the GPU with
ubatch *i+1* on the NPU. Balancing stages puts `k ≈ 4` layers on the GPU (`83·(28−k) ≈ 439·k`),
each stage ≈ 1.95 s per 4k prefill, and a two-stage pipeline over `N` ubatches costs
≈ `(N+1)/N × 1.95 s` against 2.32 s NPU-alone:

| ubatches in flight | pipelined | NPU alone | gain |
|--:|--:|--:|--:|
| 4 (4k at `ub=1024`) | 2.44 s | 2.32 s | **loss** |
| 8 (8k) | 2.19 s | 2.32 s | 1.06× |
| ∞ | 1.95 s | 2.32 s | 1.19× ceiling |

And that ceiling is against the *dense* NPU at `ub=1024`; against `thr:1.0` at `ub=2048`
(2325 t/s) it is under 1.0×. **The GPU is too slow relative to the HMX for pipelining to
recover the ubatch penalty it forces.**

## 4. Bandwidth: the concern is right, but for decode, where nothing can help

Estimated DRAM traffic for a 4k prefill (2 × 2048 ubatches, f32 activations round-tripping
between ops, weights read once per ubatch, KV written once):

| | GB | over | GB/s |
|:--|--:|--:|--:|
| dense | 39.3 | 2.34 s | 16.8 |
| `thr:1.0` | 38.8 | 1.75 s | 22.2 |
| LPDDR5X peak (SM8750) | | | ~77 |

Prefill sits at 20–30% of peak: **compute-bound, with room for a second unit's traffic.**
Contention is not what kills prefill heterogeneity; the GPU's speed is.

Decode is the opposite — and the measurement is unambiguous:

| tg64 | t/s |
|:--|--:|
| HTP only | 30.6 |
| GPU only | 31.4 |

Two very different compute units land on the **same** number, because both are streaming the
same 1.14 GB of weights per token from the same DRAM. A second unit adds compute, not
bandwidth. Splitting decode across them would at best match this and at worst thrash.

## 4b. Decode, measured per operator with bus counters — and a correction

The first pass at this section attributed ~47% of each decode token to "host/dispatch with the
device idle". That was wrong in an instructive way, so both the wrong number and the fix are kept.

### What the device profile showed (`GGML_HEXAGON_PROFILE=1`, tg32 on HTP0)

| per decode token | |
|:--|--:|
| wall (unprofiled tg64) | 32.7 ms (30.6 t/s) |
| device busy | 17.2 ms — MUL_MAT 14.3 ms, attention 1.6 ms, 396 ops |
| **weights streamed by the HTP** | **0.79 GB**, not the model's 1.14 GB |
| unaccounted | ~15.5 ms |

The missing 0.35 GB is the lm-head. `output.weight` is Q6_K in a Q4_0 quantisation, and even
re-quantised to Q8_0 it stays on the CPU (`CPU_REPACK` buffer 243 → 315 MiB) — not because of
the type but because of an explicit cut in `ggml_hexagon_supported_mul_mat`:
`// hardcoded limit to refuse the lm-head for now` at `src0->ne[1] > 32768`. So every decode
token the CPU streams a quarter-gigabyte for one GEMV, and that — not dispatch — was the
"host half".

### Lifting the cut (`GGML_HEXAGON_LM_HEAD=1`, Q8_0 lm-head)

| | tg64 | device busy / token | host + dispatch |
|:--|--:|--:|--:|
| lm-head on CPU | 29.7 t/s | 17.2 ms | ~15.5 ms |
| **lm-head on HTP** | **39.5 t/s (1.33×)** | 23.9 ms (lm-head 5.8 ms @ 57 GB/s) | **~1.4 ms (5%)** |

Prefill is unchanged (1766 vs 1769 t/s at pp4096). The Q6_K → Q8_0 output type is
quality-neutral or better; the perplexity check is in the commit message.

### Per-operator bus bandwidth, hardware-counted

`GGML_HEXAGON_PROFILE=0x3,0x41,0xce,0x43,0xcf,0x7d,0x8c,0x40` puts eight PMU counters on every
op. On v79 the v60-era 32B/64B line counters read ~0, but `AXI_READ_REQUEST` (0x40) calibrates
to a **constant 191 B per request across every weight-streaming op (p10 = p90)**, which makes it a
clean linear proxy for bytes on the bus — including for ops that read no weights. Per token,
lm-head still on CPU:

| op class | µs/token | weights MB | **bus MB** | **bus GB/s** |
|:--|--:|--:|--:|--:|
| `MUL_MAT+MUL_MAT` (gate/up) | 7358 | 396 | 418 | **56.9** |
| `MUL_MAT+ADD` (down, o-proj) | 5137 | 262 | 277 | 53.9 |
| `MUL_MAT+MUL_MAT+MUL_MAT` (qkv) | 2558 | 132 | 140 | 54.6 |
| `FLASH_ATTN_EXT` (KV reads) | 1645 | 0 | 44 | 26.8 |
| `RMS_NORM+MUL`, `SWIGLU`, `ROPE`, `SET_ROWS` | ≈950 | ~0 | ~3 | 1–4 |
| **total** | **17.7 ms** | **793** | **884** | **50.0** |

Reading it:

- **The GEMVs run at 54–57 GB/s**, ~70–75% of the ~77 GB/s theoretical LPDDR5X peak and close to
  what is practically achievable. This is the number that was previously *inferred* at "~85 GB/s"
  from the wrong byte count; the counters settle it.
- **Non-weight traffic is 91 MB/token** (~10%): KV reads for attention, plus tiny activations.
- **L2 miss counters see almost none of it** (815 misses/token): the weight stream bypasses L2
  via DMA, as designed, so cache counters are the wrong instrument here — the AXI ones are right.

### The decode picture, corrected

With the lm-head on the HTP a token is ~25 ms, of which ~24 ms is the device streaming 1.11 GB
of weights at ~50 GB/s and ~1.4 ms is everything else. Decode is **DRAM-bound with a small
dispatch tail**, not dispatch-bound. The remaining single-stream headroom is therefore bytes and
bandwidth utilisation, not scheduling: a smaller lm-head type (the Q8_0 one is now 28% of all
bytes), and closing the ~25% gap to peak. A second compute unit still adds nothing to either.

## 5. What would change the answer

- **Events in both backends.** OpenCL already uses `cl_event` throughout (its `synchronize` is a
  barrier + wait), so `event_new/record/wait` is plumbing. Hexagon would need a completion
  marker on the dspqueue. This unlocks §3.1 — a ≤1.19× ceiling that needs ≥8k context and a
  smaller ubatch than the sparse kernel wants.
- **A faster GPU path.** The ratio in §1 is the whole story. A GPU kernel at ≥50% of the HMX
  rate would make a ~1.4× pipelined ceiling plausible; Adreno 830 through the current OpenCL
  backend is not that.
- **Something that is not attention or GEMM.** After the lm-head moves to the HTP (§4b) the
  non-device share of a decode token is ~5%; there is no host-side cost left for a GPU to absorb.

## 6. Operational notes (they cost runs)

- `llama-bench` separators: `,` separates **test configurations**; `/` separates devices or
  split proportions **within** one. `-dev HTP0,GPUOpenCL -ts 24,4` runs HTP0-only twice and a
  GPU-only config — it never runs a split. The heterogeneous form is
  `-dev HTP0/GPUOpenCL -ts 24/4`.
- OpenCL's `FLASH_ATTN_EXT` `supports_op` checks dtypes only, never `src[5]`/`src[6]`: GPU
  layers under a sparse configuration silently run **dense** attention. Correct, and the
  documented fallback, but it means a split never runs the sparse kernel on the GPU side.
- The QDC device reprovisions on reboot and clears `/data/local/tmp`; rebuild-and-push is the
  recovery, everything is reproducible from `build-sparse`.
