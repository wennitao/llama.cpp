# Heterogeneous NPU + GPU: measured, and why it does not pay on this SoC today

Question asked: the Adreno GPU sits idle while the HTP runs the whole model — can part of the
compute move to it, and can the data movement be arranged so the two do not fight over DRAM?

Short answer: **not with the current scheduler, and not by much even with a better one.** The
GPU delivers ~1/5 of the NPU's prefill throughput on this model, the scheduler can only run the
two *serially* (neither backend implements events), and decode at depth ~0 is bandwidth-bound on
DRAM the two units share, so the GPU measures the **same** decode rate as the NPU. At real
context depth decode is bound by the NPU's own attention kernel instead (§4c) — and moving
that kernel to the GPU loses to crossings before it starts (§4e); the fix is on the NPU.

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
| **lm-head on HTP** | **32–39.6 t/s (1.10–1.34×, bimodal across runs†)** | 23.9 ms (lm-head 5.8 ms @ 57 GB/s) | **~1.4 ms (5%)** |

† Five runs of the HTP arm over two device sessions: 39.5, 39.5, 32.2, 33.1, 39.6 t/s — two
clusters, nothing in between; the CPU arm is steady at 29.5–29.7. Not understood (a DSP clock
state is the obvious suspect); reported as the range, not the best case.

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

### 4c. Context length: every decode number above is at depth ~0 — and depth changes the story

`llama-bench -p 0 -n 64` sets `n_ctx = n_prompt + n_gen`, so the KV cache never exceeded 64
tokens in §4/§4b. That is the best case for the weight-streaming picture and the worst
representation of attention. With a real context (`-d`, lm-head on HTP):

| depth | tg64 | ms/token | bytes model @ 50 GB/s | measured vs model |
|--:|--:|--:|--:|:--|
| 0 | 32–39.6* | 25–31 | 1.11 GB → 22 ms | at bus rate |
| 4096 | 19.9 | 50 | 1.58 GB → 32 ms | **1.6× slower than bytes** |
| 8192 | 14.6 | 69 | 2.05 GB → 41 ms | 1.7× |
| 16384 | 9.1 | 110 | 2.99 GB → 60 ms | 1.8× |

\* bimodal across runs, see §4b.

The gap is attention, and the PMU profile at depth 4096 says why:

| op class, decode @ d4096 | µs/token | bus MB | bus GB/s | kernel |
|:--|--:|--:|--:|:--|
| **`FLASH_ATTN_EXT`** | **28 010 (46%)** | 741 | **26.4** | `hvx Br 1 Bc 64 nkvb 68` |
| lm-head `MUL_MAT` | 10 167 | 359 | 35.3 | |
| `MUL_MAT+ADD` | 7 751 | 328 | 42.4 | |
| `MUL_MAT+MUL_MAT` | 7 227 | 396 | 54.8 | |
| qkv | 2 507 | 132 | 52.7 | |
| total device-busy | 60.3 ms | 2033 | 33.7 | |

Decode attention runs on the **HVX path** (`Br 1`, 64-token KV chunks, 68 chunks per layer at
this depth), not the HMX pipeline prefill uses (`hmx-pipe Br 192 Bc 1024`), and it moves KV at
**26 GB/s — half the GEMV rate**. So at 4k depth attention is already 46% of the token, and the
token as a whole is no longer bandwidth-bound: total bus traffic is 33.7 GB/s against the 50+
the GEMVs alone sustain. Past ~8k the KV stream is larger than the weights and this path is the
whole story.

Consequences for the two questions this document asks:

- **Decode has a real kernel target after all** — the HVX decode-attention path. Two levers,
  both on the NPU: fewer KV bytes (a decode-side block selection; today's sparse kernel only
  engages for `ubatch > 256`), and a faster path (the HMX pipeline, or a wider-chunk HVX one)
  to bring 26 GB/s toward 50.
- **It still does not reopen the GPU case.** Within one token the layer chain is serial —
  attention *L* must finish before o-proj *L* — so GPU attention cannot overlap with NPU GEMVs;
  it would have to be *faster* than 28 ms for 470 MB of KV on its own, which is not where the
  measured GPU sits.

### 4d. Decode attention on the GPU: the kernel-only A/B says no before anything else does

`test-backend-ops perf`, the Qwen3-1.7B decode shape (hs=128, 8 KV heads, GQA 2, one query,
causal mask, f16 KV), same rows on both backends, two alternations each (spread ≤ 0.2%):

| per op (×28 layers per token) | HTP0 (HVX path) | GPUOpenCL | GPU / HTP |
|--:|--:|--:|--:|
| kv = 1024 | 192 µs | 434 µs | 2.3× slower |
| kv = 4096 | 739 µs → 20.7 ms/token | 2777 µs → 78 ms/token | **3.8×** |
| kv = 8192 | 1470 µs | 5480 µs | 3.7× |
| kv = 16384 | 2957 µs | 10690 µs | 3.6× |

At kv=4096 the op reads 16.8 MB of KV: the HTP does it at ~23 GB/s (the 26 GB/s in §4c
includes mask and padding), the Adreno at ~6 GB/s. The GPU's time grows 6.4× for 4× KV, so
this is not launch overhead — the kernel is per-byte slow, i.e. under-parallelised over the KV
axis for a single query. **Attention on the GPU would make decode slower at every depth even
with free crossings and a zero-copy KV cache.** A GPU kernel would have to be ~4× faster than
ggml-opencl's at this shape merely to tie the HVX path, and then also pay 2×28 crossings.

What this does not settle is the *other* comparison in §4c: the HTP's own decode-attention
path is at ~23 GB/s where its GEMVs reach 55, so the honest target for decode at depth is that
kernel, on the NPU.

#### 4d.1 …but the GPU's 6 GB/s is the kernel, not the silicon

A plain memory-bound GEMV (`MUL_MAT` m=4096, k=14336, n=1) on both backends, two alternations:

| | weight bytes | GPUOpenCL | HTP0 |
|:--|--:|--:|--:|
| f16 | 117.4 MB | 1798 µs → **65 GB/s** | 1938 µs → 61 GB/s |
| q8_0 | 62.4 MB | 1008 µs → 62 GB/s | (perf row aborts on HTP — see commit) |
| q4_0 | 33.0 MB | 501 µs → 66 GB/s | (idem) |

The Adreno streams ~65 GB/s when the kernel is shaped for it — *above* the HTP's GEMV rate.
So ggml-opencl's flash-attention at 6 GB/s is under-parallelised by ~10× against its own
hardware, and the ceiling for a well-written single-query GPU attention at kv=4096 is
16.8 MB / 65 GB/s ≈ **260 µs/op ≈ 7.2 ms/token**, versus the HVX path's 20.7 ms/token.

This is what makes the user's mllm kernel the operative question rather than a footnote. The
gate arithmetic at depth 4096, per token:

| | attention | crossings (2 × 28) | total |
|:--|--:|--:|--:|
| HTP HVX path today | 20.7 ms | — | 20.7 ms |
| GPU at its streaming ceiling | 7.2 ms | 56 × *c* | 7.2 + 56*c* |
| HTP HVX path fixed to its GEMV rate (~57 GB/s) | ~8.3 ms | — | **~8.3 ms** |

The GPU route breaks even against *today's* HTP path at *c* ≈ 240 µs per crossing and gets half
the gain at *c* ≈ 120 µs — but it never beats the **on-NPU fix**, which reaches the same
bandwidth with no crossings, no shared KV buffer, and no second runtime. A GPU decode-attention
kernel is only worth building if the HVX path cannot be brought toward its GEMV rate; §5 of the
workflow synthesis addresses why it sits at 23 GB/s.

#### 4d.2 The mllm decode kernel: it exists, it is good, and as-is it ties the HVX path

`/mnt/raid0_ssd/wentao/mllm`, branch `opencl-flash-attention`, commit `5e9fef1a` "dedicated
FlashAttention decode kernel — 5.5 → 43.7 GB/s (8.1×@4096)": `flash_attention_fp16_decode` +
`flash_attention_fp16_decode_merge` in `mllm/backends/opencl/kernels/flash_attention.cl`,
dispatched by `OpenCLFlashAttention2Op` for `S_q == 1`. It is the shape ggml-opencl's kernel is
missing: no LDS staging (each K/V element is used once), lane *t* owns key row *t* for a full-D
dot in registers, lane *t* owns output dim *t* for P·V, and **split-K over the KV axis**
(`nsplit = ceil(S_kv/256)` capped at 16) with an unnormalised `(o, m, l)` partial per partition
and a tiny merge kernel. Measured by its author on **this same unit (eb49fb9d)**:

| S_kv | ms/op | GB/s | vs ggml-opencl FA (§4d) | vs HTP HVX (§4d) |
|--:|--:|--:|--:|--:|
| 1024 | 0.397 | 21.2 | 434 µs → 1.09× faster | 192 µs → 2.1× slower |
| 2048 | 0.543 | 30.9 | | |
| 4096 | **0.768** | **43.7** | 2777 µs → **3.6× faster** | 739 µs → **tie (0.96×)** |

Two caveats decide how to read the tie:

- **Its shape is MHA (`B=1, H=16, D=128`, "GQA pre-expanded, kernel sees `H_q == H_kv`").**
  So 43.7 GB/s is for 33.5 MB of KV per op; our model's 8 KV heads under GQA 2 are 16.8 MB. The
  kernel indexes K/V by *query* head with no group mapping, so on our cache as-is it reads each
  KV head twice — same bytes, same time, hence the tie. A **GQA-aware variant** (one workgroup
  per KV head serving both query heads) halves the traffic at the same streaming rate:
  **~0.4 ms/op ≈ 11 ms/token at depth 4096**, against the HVX path's 20.7. That is the real
  prize on the GPU side, and it is ~2×, not ~3×.
- Its author's phase ablation puts the residual bottleneck in the **P·V phase** (strided V
  reads; scores alone reach 48 GB/s). A packed-V layout is the next ~25%, and it is not free —
  it needs a V repack the KV-cache writer does not do today.

**Porting cost into ggml-opencl's `FLASH_ATTN_EXT` is modest.** Strides are generic and in
elements (`K_h_stride`, `K_s_stride`), so ggml's cache layout — `[d, kv, n_kv_head]` f16 with
the head *inside* the row, i.e. `K_s_stride = d·n_kv_heads`, `K_h_stride = d` — is passed
directly; D is fixed at 128 by `FA_D`. Needed: an f32→f16 Q adapter (4 KB), an f16→f32 O
adapter, the causal mask made a no-op for `S_q == 1` (already the kernel's assumption), the
GQA head mapping, and the persistent split-K scratch buffer. Roughly a day.

**What this does not change:** with the GQA-aware variant the GPU saves ~10 ms/token at depth
4096 *before* 2×28 crossings, so the break-even crossing is ~170 µs and half the gain needs
~85 µs; and the KV cache must be readable by both units without copies. Both are the
scheduler/buffer questions in the synthesis below — and the on-NPU HVX fix (§4d.1) reaches the
same ~8–11 ms/token with neither.

## 4e. Decision: GPU decode attention — no; the HVX decode path — yes

Four read-only investigations (scheduler placement, ggml-opencl's FA, the HVX decode path,
the mllm kernel) plus the measurements above. Their adversarial-verification stage did not run
(session limit), so claims below are single-source unless marked *verified*; the ones the
decision rests on were re-read in source.

### Why the GPU route cannot pay through `ggml_backend_sched`, regardless of kernel

- **Placement.** Making ggml-hexagon reject decode FA would send it to the **CPU**, not the
  GPU: pass 3 assigns an unassigned node to the backend with the most *buffer-supported*
  inputs, hexagon's rpcmem buffers are host-visible, the CPU accepts any host buft, and
  OpenCL's `supports_buft` accepts only its own (*verified*, `ggml-backend.cpp:1199-1226`,
  `ggml-opencl.cpp:11711-11722`). The only route is the existing user-pin API
  `ggml_backend_sched_set_tensor_backend`, which llama-context already uses for `norm`/`l_last`
  (*verified*, `llama-context.cpp:2532`) — ~10 lines in `graph_get_cb`, gated on `n_tokens==1`.
- **Crossings.** A pinned FA turns each layer into HTP → GPU → HTP: **28 HTP graph_computes per
  token instead of ~1**, each a blocking dspqueue round trip measured at ~620 µs
  (`sparse-attention.md` §2) → ~17 ms/token before the GPU does anything; plus, per layer, three
  blocking 8 KB transfers and ≥3 barrier+wait round trips on the OpenCL side (no events, no
  async copies on either backend) at ~0.36 ms per small dispatch by mllm's own measurement on
  this SoC. **Crossing overhead alone is ~1–2 ms/layer = 28–56 ms/token — at or above the entire
  28 ms the HVX attention costs today.** The break-even crossing computed in §4d.1 was ~170 µs;
  the stack delivers ~1000+.
- **KV sharing is solvable, and is not the wall.** ggml-opencl has no host-pointer import at all
  (`buffer_from_host_ptr` returns nullptr), so today the scheduler would copy the K/V views —
  470 MB/token at 4k. But mllm's `het-pipeline` branch has a *validated* zero-copy recipe on this
  SoC: `clCreateBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM)` with a
  `cl_mem_ion_host_ptr{CL_MEM_ION_HOST_PTR_QCOM, CL_MEM_HOST_IOCOHERENT_QCOM, fd, ptr}` over an
  rpcmem allocation — GPU reads of DSP-written K/V byte-exact 30/30. ggml-hexagon's buffers
  already carry the `(base, fd)` pair it needs. (The generic `cl_khr_external_memory_dma_buf`
  import silently creates a separate buffer on Adreno 8xx; do not use it.)

So the GPU path needs, in order: events + async copies in *both* backends, an OpenCL-side
import of hexagon buffers, a GQA-aware port of the mllm kernel — and at the end of all that it
ties or modestly beats the on-NPU fix.

### Why the HVX path is at 23 GB/s, and how to double it

The investigation's arithmetic reproduces the PMU measurement: 16 heads × 4352 rows × 2 tensors
× 28 layers = 3.90 M AXI requests × 191 B = 745 MB against the measured 741 MB (0.5%). That
implies each 256 B K/V row (2 KB stride in the cache) is one bus request, so the 191 B/request
proxy *undercounts* here and the true traffic is ~1.0 GB/token — the DMA engine is running at
~45 GB/s, near the GEMV rate. The useful half is lost to structure, not to hardware:

1. **2× redundant fetch.** The HVX kernel's unit of work is one flattened (token, head) query row
   per thread, so both query heads of a GQA-2 group independently DMA the same KV head
   (`ggml-hexagon.cpp:2217`, `flash-attn-ops.c:468`). 33.6 MB moved per op for 16.8 MB unique.
2. **16 rows over 6 threads** = 3,3,3,3,3,1 — all six busy for only the first third.
3. **Two 64-row blocks in flight per thread**, busy-wait pop.
4. Decode takes HVX by an explicit gate (`DK <= 128 && neq1 < 5`, `ggml-hexagon.cpp:2030`); HMX at
   `Br=1` would fill 2 of 32 tile rows and is correctly rejected.

Ranked fixes, attention ms/token at depth 4096 (today 20.7 kernel-only / 28 in-model):

| fix | est. attention | what it is |
|:--|--:|:--|
| **#1 split-KV "flash-decoding" on HVX** | **~9** | work unit = (kv head, KV range) serving both query heads; per-split `(m, l, acc)` partials + combine. Unique bytes, perfect 6-way balance. Mirrors the mllm kernel's structure, on the NPU. |
| #2 head-grouping only | ~14 | iterate 8 KV-head rows, G dots per fetched block; 8 rows over 6 threads leaves two rounds |
| #3 decode-side block selection (25%) | ~5.5 alone, ~3 on top of #1 | needs `src[5]/src[6]` in the HVX path and a single-query scorer; quality unvalidated |
| #4 deeper prefetch / `Bc=128` | ≤10–15% | VTCM has room (0.5 of 8 MB used) but the aggregate is already near rate |

Fix #1 alone takes the token from ~50 to ~38 ms at depth 4096 (≈1.3×) with no second runtime,
and is the same idea the GPU route would need anyway.

### The one measurement that decides fix #1's size

The split between DMA-bound and HVX-compute-bound inside the decode kernel is inferred, not
measured. `GGML_HEXAGON_PROFILE=3` on one decode FA op at kv=4096 gives per-thread
`HVX_FA_QK + HVX_FA_SFM` busy vs DMA-event coverage: DMA ~100% and compute < 40% → #1 delivers
its full ~2×; compute > 60% → the HVX chain is the ceiling and #1 buys only the imbalance.

## 4f. Built: the split-KV, GQA-grouped HVX decode path (`GGML_HEXAGON_FA_DECODE`, default on)

### The sizing trace first

`GGML_HEXAGON_PROFILE=3` on one decode `FLASH_ATTN_EXT` at kv=4096 (Qwen3 geometry), per
thread, from the on-DSP event trace:

| thread | rows | span (kcycles) | HVX compute busy | of which QK / softmax+PV |
|--:|--:|--:|--:|:--|
| 0–4 | 3 each | 1310–1608 | **~65%** | 33% / 32% |
| 5 | 1 | 468 (36% of the others) | ~63% | |

DMA descriptors overlap ~3.6-deep. So the kernel was *not* DMA-starved — it was 65%
compute-bound per thread, with one idle thread and 2× redundant bytes. That caps what a
structural rewrite alone can return: balance (16 rows over 6 threads → 18 slots, 1.125×) plus
the DMA-wait share, not the full 2× the byte count suggested.

### What was built

`flash_attn_ext_f16_dec_thread` + `flash_attn_ext_f16_merge_thread`
(`ggml/src/ggml-hexagon/htp/flash-attn-ops.c`). The unit of work is **(sequence, KV head, KV
range)**: the range's 64-row K/V blocks are DMA'd once and every query row of that KV head — all
G GQA heads × all n_tokens — consumes them from VTCM. `n_split` ranges per KV head are chosen to
minimise `ceil(units/threads) × blocks_per_split` (3 for 8 heads on 6 threads at kv=4096: 24
units, 4 per thread, ~22 blocks each). Each unit leaves an unnormalised `(M, S, acc[DV])` partial
per row in shared VTCM; a second `work_queue_run` merges the splits with `w_s = exp(M_s − M)`,
applies sinks, normalises and stores. Every numeric step (dot, softcap, mask+ALiBi, online
softmax, P·V) is the row-per-thread kernel's HVX sequence, run once per row against a block
staged once — the flash-decoding structure of a split-K GPU kernel, on the NPU, with no
crossings. The row-per-thread kernel stays selectable (`GGML_HEXAGON_FA_DECODE=0`) for A/B and
as the fallback when rows-per-unit exceed the scratch bound.

Two bugs found by the eval suite on the way, both in the merge: an f32 `exp` of the empty-split
sentinel (`M = −1e4`, which the row kernel only ever fed to the saturating f16 `exp2`) now
short-circuits to weight 0; and the accumulate loop walked `DV/32` whole vectors, truncating head
dims that are not multiples of 32 (`hsv = 40, 72, 80` — 228 of the suite's 243 failures). It
now walks the padded `size_vkq_acc` span.

### Measured

Kernel-only, `test-backend-ops perf`, decode shape, two alternations (spread ≤ 0.5%):

| kv | row-per-thread | **split-KV** | gain |
|--:|--:|--:|--:|
| 1024 | 191 µs | 182 µs | 1.05× |
| 4096 | 740 µs | **619 µs** | **1.20×** |
| 8192 | 1474 µs | 1192 µs | 1.24× |
| 16384 | 2952 µs | 2370 µs | 1.25× |

End-to-end `llama-bench` tg64, lm-head on HTP, two alternations:

| depth | before | **after** | gain |
|--:|--:|--:|--:|
| 0 | 39.3 / 39.4 | 39.5 / 39.8 | flat (attention is ~5% of the token) |
| 4096 | 21.0 / 21.0 | **23.1 / 23.1** | **1.10×** |
| 8192 | 14.6 / 14.6 | **16.8 / 16.8** | **1.15×** |

Correctness: the full `FLASH_ATTN_EXT` eval suite (2198 rows: nr2 up to 32, nr3 ∈ {1,3},
nb ∈ {1,3,32,75}, mask/sinks/ALiBi/softcap, all KV types) run with the new path and with the old
path on the same binary; the failure sets differ only by the known sinks/large-head flappers,
which flip in both directions: new path 27 / 22 failing rows over two runs, old path 18 (24 and
23 on earlier builds), all with `sinks=1`, errors 5.1e-4 – 7.7e-3 against the 5e-4 gate versus
5.3e-4 – 2.3e-2 for the old path's own failures; the same binary flips 20 / 15 rows between two
consecutive runs. **No non-sinks row fails under the new path, and 1088 sinks rows are in the
suite.** Judge this family by the diff, never by the absolute count.

### Where the rest of the time is

At kv=4096 the new kernel spends ~3.6 µs per 64-key block per row — the same per-row cost as
before; the structure removed the imbalance and the redundant bytes, so it is now compute-bound
on the per-row HVX chain. Reading the primitives:

- **QK** (`hvx_dot_f16_f16_aa_rx32`) is eight 4-row dots, each ending in a horizontal
  reduction: for 64 keys × 128 dims (8 k MACs) it costs ~2.2 k cycles, ~17× off the HVX MAC rate.
- **P·V** (`hvx_mad_f32_f16_aa_rx2_vec` per key pair) pays a `vror`+`repl` broadcast per key and
  a V load + `vshuff` per row.

Two levers, in order of size:

1. **Fused two-query micro-kernels**: the GQA pair's rows already sit in one unit, so one K
   load can feed both dots and one V load + shuffle both accumulators. ~1.3× more on the kernel.
2. **Transposed-K dot**: stage each K block interleaved once per unit (what the HMX path's
   `fa_phase_k_interleave` does) so the dot is a broadcast-MAC over 64 keys with no reductions,
   amortised across the unit's rows. The ≥2× lever, and a real kernel design.

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
