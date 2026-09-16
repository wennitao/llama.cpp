# Heterogeneous NPU + GPU: measured, and why it does not pay on this SoC today

Question asked: the Adreno GPU sits idle while the HTP runs the whole model — can part of the
compute move to it, and can the data movement be arranged so the two do not fight over DRAM?

Short answer: **not with the current scheduler, and not by much even with a better one.** The
GPU delivers ~1/5 of the NPU's prefill throughput on this model, the scheduler can only run the
two *serially* (neither backend implements events), and decode at depth ~0 is bandwidth-bound on
DRAM the two units share, so the GPU measures the **same** decode rate as the NPU. At real
context depth decode is bound by the NPU's own attention kernel instead (§4c) — and moving
that kernel to the GPU loses to crossings before it starts (§4e); the fix is on the NPU.
For prefill, the natural split — the NPU takes a query tile's *shared* KV blocks, the GPU the
blocks only one or two sub-blocks wanted, merged by an online-softmax fold — was built end to end
and is exact (§4j), but the GPU spends 4-5x the time on a block pair that the NPU saves by
shedding it; the +2-4% the model showed with the split on was reproduced by a *dummy* GPU kernel
on the same duty cycle (the bus-clock lift of §4i). Taking the host out of the GPU's critical path
(§4k: a pre-enqueued gate that polls the DSP's `ready` word itself, compaction and staging on the
GPU, persistent work-groups that finish stages in order, kernel-written done words) and then
trimming the fold's own handling makes the attention op 1.17x faster than that control, worth ~2%
end to end and at the edge of what pp4096 throughput resolves through thermal drift; the split
also needs a sustained-load guard, because the Adreno throttles before the HTP does.
One decode asymmetry does hold up (§4l): the HTP gathers scattered KV rows at 9 GB/s, one DMA
descriptor per row, while the GPU gathers the same rows through index lists at 33-45 GB/s, so
whole-cluster sparse decode can drop its shadow copy by giving the scattered clusters to the GPU.

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

### 4f-bis. One pass on the HVX decode chain: QK reduction tree, no V shuffle (+12%), and where the cycles really are

Section 4f named two levers on the per-row chain. Reading the primitives changed the target:
sharing K/V loads between the GQA rows buys nothing (VTCM loads dual-issue), the waste is in
the QK reduction (4 keys at a time, serial rotate-add, ~17x off MAC rate) and in a `vshuff` of
every V vector for every row in P*V. Shipped (`GGML_HEXAGON_FA_DECODE` path only):

- `hvx_dot_f16_f16_aa_rx32_tree`: 32 keys reduced together by single-level `vshuff` exchanges
  (4/8/16/32/64 B) + adds, 31+31 ops, lands in key order; drop-in for `rx32`.
- P*V accumulates whole padded V vectors without the shuffle; the accumulator lives in the
  widened (even, odd) lane order and is un-permuted once at partial write-back
  (`hvx_unshuff_copy_f32_aa`). Accumulator = 2 f32 vectors per padded V vector.

| kv | before | after | | tg64 @ d4096 |
|--:|--:|--:|--|--:|
| 4096  | 636 us | 563 us (1.13x) | | 13.6 t/s vs 12.6 with the row kernel |
| 8192  | 1195 us | 1067 us (1.12x) | | |
| 16384 | 2349 us | 2099 us (1.12x) | | |

FLASH_ATTN_EXT suite: 0 non-`sinks` failures over 2196 cases including every head size;
the `sinks=1` near-threshold flappers moved (21 vs 16, same error class, all `sinks=1`).

**Why only 12%, per the per-thread trace at kv=4096 (DSP at 1710 MHz):** per row-block QK is
1879 cycles and softmax+P*V 2159, i.e. ~4000 traced of ~5440 wall per row-block; the DMA of a
64-key K+V block takes ~12k cycles per thread, about the compute of the block's 2 rows. So the
kernel now sits at the DMA limit of its access pattern -- 64 rows of 256 B with a 2 KB stride
per K/V block, 2 in flight per thread -- at ~30 GB/s, while the GPU streams the same layout at
45 GB/s. Both compute halves are latency-bound, not op-bound: P*V is one 64-long
load-FMA-store chain through VTCM per accumulator, QK has 4 FMA chains in flight.

**Tried and rejected:** register-resident 4-chain P*V with scalar-splat broadcasts plus an
8-chain DK=128 unrolled tree. It overflowed the 16 KB worker stack (DSP process died; 64 KB
stacks cure it), and with the spills it is *slower*: 682 us at 4k. Rolled back; the functions
stay in `hvx-fa-kernels.h` unused. The next real step on this path is the DMA pattern (contiguous
all-head blocks, deeper prefetch), not the arithmetic -- parked, since the heterogeneous split
(4h) is the research direction and already halves this op.

## 4g. Measured: the no-barrier handshake (`examples/hetero-sync-probe`)

The proposal that reopened the GPU question: let the GPU own some KV splits of the decode
attention and let the HTP *merge* consume its partials through shared memory, with no graph
barrier -- the HTP op raises `ready`, the GPU writes partials + `done`, the HTP op polls `done`.
Section 4f's split-KV kernel already produces `(M, S, acc)` partials and merges them, so the
merge side is the existing phase with more inputs. What decides the usable GPU share is the
fixed cost of that handshake per layer, so it was measured before anything was built on it.

### What was built

- `HTP_OP_SYNC_PROBE` (`htp/sync-probe-ops.c`), dev-only, reached through a `GGML_OP_CUSTOM`
  node tagged with `HTP_SYNC_PROBE_MAGIC` when `GGML_HEXAGON_SYNC_PROBE=1`. Inside a normal op
  batch it writes a payload and flushes it, publishes `ready`, spin-polls `done` with a cache
  invalidate per read, checks a payload the other agent wrote, and records qtimer stamps.
- `llama-hetero-sync-probe`: allocates one 1 MiB hexagon rpcmem buffer, aliases it into
  OpenCL with `CL_MEM_EXT_HOST_PTR_QCOM` + `cl_mem_ion_host_ptr{ION_HOST_PTR, IOCOHERENT, fd,
  base}` (the mllm recipe; `rpcmem_to_fd` on the buffer base), and drives the handshake in
  several modes. Host stamps are `cntvct_el0`; the HTP's `c31:30` qtimer turned out to be the
  same 19.2 MHz counter (offset bounded to 0.3 us by the handshake itself), so every
  cross-domain split below is direct. The HTP round trip itself is single-clock.

```
cmake --build build-sparse --target htp-v79 ggml-hexagon llama-hetero-sync-probe -j 32
adb push build-sparse/bin/llama-hetero-sync-probe build-sparse/bin/libggml-hexagon.so \
         build-sparse/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llama-hetero/
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./llama-hetero-sync-probe --iters 10 --cpu 7 \
         --modes ping,cpu,gpu1,gpusvm --b2b 10
```

### Flag mechanics, all directions (SM8750, Adreno 830 driver 0800.70)

| path | measured | note |
|---|---|---|
| HTP write -> CPU sees it | ~0.1 us | no CPU cache maintenance needed: 5/5 correct and faster with `--host-inval 0 --host-clean 0` (the CPU<->cDSP path is IO-coherent both ways) |
| CPU write -> HTP sees it | ~0.1 us | the HTP must invalidate its own L2 line before every poll: `Q6_dcinva_A` 0.2 us/poll; `qurt_mem_cache_clean(INVALIDATE)` 0.2-0.3; DMA (bypass) 0.4. Without an invalidate it never sees the update (5/5 timeouts) |
| HTP publish `ready` | 0.3-0.7 us | `qurt_mem_cache_clean(FLUSH)` on the line, or `dccleana` + `syncht` |
| HTP writes data, flushes, raises `ready`; CPU reads the data | correct 30/30 | 2 KB 3.5 us, 8 KB 3.9 us, 64 KB 6.1 us incl. the scalar write loop |
| same data read by a **pre-launched** GPU kernel after the flag | correct 30/30 | the Q / new-K/V direction: the kernel was already running when the HTP wrote |
| GPU kernel end -> HTP sees its stores | 2-4 us | GPU stores into the alias reach memory at kernel end |
| GPU polling the rpcmem alias mid-kernel | never sees an update (0/10, 5M loads) | Adreno caches ION buffers within a kernel, `IOCOHERENT` or not; its mid-kernel stores are invisible too |
| GPU polling a fine-grain SVM word | sees it 2-4 us after the host store; mid-kernel SVM stores visible to the host 10/10 | `CL_DEVICE_SVM_FINE_GRAIN_BUFFER` + `ATOMICS` are supported (no `FINE_GRAIN_SYSTEM`) |

### The round trip as the merge op would see it (`T_DONE - T_READY`, HTP clock; main thread on CPU 7)

| variant | HTP round trip | where it goes |
|---|---|---|
| CPU writes 64 KB + `done` | 4.9 us | all of it is the host writing |
| fresh GPU launch, one fused kernel (64 KB + flag) | **245 us** | enqueue + `clFlush` 36 us, launch 196 us, kernel 8 us, visibility 4 us |
| fresh GPU launch, two kernels | 264 us | on a little core: 320-460 us (enqueue alone 130-280 us) |
| `clFlush` skipped | never runs before the 300 ms timeout | the driver does not auto-submit |
| **pre-launched GPU kernel, host relays the flag through SVM**, 40-64 KB | **7.1-8.9 us** | ready->host 0.1, relay store 0.0, SVM->GPU 2-3, GPU writes, end->HTP 2.5-3.5 |
| same, main thread unpinned (little core) | 9.1 us | |
| pre-launched, GPU reads a 64 KB HTP payload first | 17.8 us | one work-group reading 64 KB scalar: the kernel, not the sync |
| back-to-back pre-enqueued kernels, A end -> B start | 0-1 us | in-order queue; a per-layer chain can be armed at token start |

Side finding, not this section's object: the op-batch dispatch (`compute_async` -> op entry)
is 90-130 us with the main thread on a big core and 160-650 us unpinned; a ping round trip is
233 us vs 356-834 us. Part of the ~620 us per-`graph_compute` constant quoted in Section 4e may
be little-core scheduling. Worth an A/B on `llama-bench` with `taskset` before it is quoted
again.

### What it means for the design

- The handshake is not the obstacle. With the GPU kernel pre-launched and the flag relayed
  through SVM, the per-layer fixed cost is ~15 us (flush Q and the new K/V rows ~4, relay
  7-9, done visibility ~3) instead of the >=250 us of a fresh launch that Section 4e's budget
  assumed. The usable GPU share is no longer dispatch-bound.
- Each layer's GPU kernel must be **already running** when the HTP raises `ready`: the GPU
  cannot poll HTP memory with plain loads (§4k later found device-scope *atomic* loads through the
  ION alias do see the DSP's writes), so a host thread on a big core relays the flag into SVM, and the
  per-layer kernels are enqueued as a chain (0-1 us between them). The kernel spins while it
  waits -- ~1 ms per layer at 4k -- which is power, not latency.
- The HTP side is the existing split-KV merge with extra partial slots, plus: flush Q and the
  token's K/V rows before `ready`, poll `done` with `dcinva` (0.2 us per read), invalidate the
  partial range before merging. On a late GPU the merge waits; on a timeout it can compute the
  missing splits itself, so correctness never depends on timing.
- **The ceiling is now DRAM, not sync.** At depth 4k a token reads ~470 MB of KV over the 28
  layers; the HVX path does that in ~17 ms (~27 GB/s, compute-bound, Section 4f). Two engines
  streaming it together sit against the ~55-60 GB/s the GEMVs already reach, so attention
  cannot drop below ~8 ms: <=2.1x on attention, <=~1.28x on the token at 4k, <=~1.45x at 16k
  where attention is 60% of the token. Those are the honest ceilings; the kernel-side levers of
  Section 4f (fused two-query micro-kernels, transposed-K dot) reach part of the same gain with
  no cross-device machinery, and the two compose.
- Still unmeasured: the GPU decode kernel itself in this setting (mllm's `flash_attention_fp16_decode`
  with GQA, f32 Q/O adapters and Section 4f's partial format), DRAM contention while both engines
  stream KV, and whether a spinning kernel disturbs the Adreno driver over a whole token.

## 4h. Measured: the GPU decode kernel, and both engines streaming the same KV (`examples/hetero-decode-attn`)

Section 4g left three things unmeasured. This closes two of them: the GPU decode kernel in
this setting, and DRAM contention when the HTP and the GPU stream the KV cache at once.

### What was built

- `fa_dec_gqa` (OpenCL, in the tool): one work-group per (KV head, KV split), 128 lanes, the
  G query heads of a KV head share every K/V read (GQA-aware). It reads Q (f32), K/V (f16) and
  the mask straight from the ggml layouts through the ION alias and emits the split-KV partial
  of Section 4f -- 128-byte header (M, S), then acc[DV] f32, scores in scale*q.k + mask units
  with natural exp -- so the HTP merge can combine partials from both engines. Correct against
  a CPU reference to 1e-8 after a host-side merge; the HTP FA output over its range to 2e-5.
- `llama-hetero-decode-attn`: one hexagon rpcmem buffer holds Q, K, V, mask, the partials and
  the sync-probe words; the HTP graph is [probe handshake] -> [FLASH_ATTN_EXT over K/V views of
  positions [gpu_kv, kv)] -> [probe ping], the GPU kernel is pre-launched over [0, gpu_kv) and
  released through SVM at the handshake, so both start within a few microseconds and the FA
  op's own start/end come from the probe stamps on the HTP qtimer.

```
./llama-hetero-decode-attn --kv 4096 --gpu-frac 0.6 --span 512 --iters 7   # Qwen3-1.7B shapes: nh 16, nkvh 8, d 128
```

### Results (median of 7, main thread on CPU 7, Qwen3-1.7B shapes, ~1% keys masked)

| kv | HTP alone, full range | GPU alone, full range | best split measured | wall (both) | vs HTP alone | HTP / GPU slowdown when both stream |
|--:|--:|--:|---|--:|--:|--:|
| 4096  | 636 us (26.4 GB/s)  | 436 us (38.5 GB/s)  | GPU 0.6, span 512: HTP 287 / GPU 299 | **299 us** | **2.13x** | x1.03 / x1.30 |
| 8192  | 1195 us (28.1 GB/s) | 774 us (43.3 GB/s)  | GPU 0.5, span 256: HTP 636 / GPU 617  | **638 us** | **1.87x** | x1.00 / x1.39 |
| 16384 | 2349 us (28.6 GB/s) | 1481 us (45.3 GB/s) | GPU 0.5, span 256: HTP 1203 / GPU 1178 | **1203 us** | **1.95x** | x1.00 / x1.49 |

- The HTP does not feel the GPU: it is compute-bound at ~27 GB/s (Section 4f), so a second
  reader costs it 0-3%. The GPU is the one that pays for contention (x1.2-1.5), which is why
  the balance point sits at a GPU share of 0.5-0.6 rather than at the ratio of the standalone
  speeds. Combined traffic at the balance point is 50-56 GB/s -- inside what the GEMVs already
  draw, below the ~77 GB/s peak.
- Span (keys per GPU work-group): 512 beats 256 beats 128 at 4k (GPU alone 229 / 255 / 292
  us); at 16k 512 and 256 tie and 1024 loses (too few, too long work-groups). 512 is the
  default to carry forward, or nsplit ~5-8 per KV head.
- The GPU kernel alone is already 1.45-1.6x faster than the HVX path over the full range
  (its 45 GB/s at 16k vs the HTP's 28.6). A GPU-only attention would still lose to the split,
  which uses both.

### What it means

- Per layer the attention wall halves at every depth measured. On Qwen3-1.7B (28 layers) that
  is 28 x (636 - 300) = 9.4 ms off a ~43 ms token at 4k (~1.28x) and 28 x (2349 - 1150) = 34 ms
  off a ~110 ms token at 16k (~1.44x), before the ~15 us/layer handshake of Section 4g. These
  are the same ceilings Section 4g projected from bandwidth; now they are measured per layer.
- What is left is integration (Section 4g's design, now with a validated kernel): the FA op
  starting at block `gpu_blocks`, raising `ready`, merging the GPU's partials from the shared
  buffer after polling `done`; a host relay thread that pre-enqueues one kernel per layer and
  copies each `ready` into SVM; and the token-level measurement with `llama-bench`. The third
  open item (a spinning kernel across a whole token) is answered below.

### Re-measured with the faster HTP kernel (4f-bis), span 512

| kv | HTP alone (tree kernel) | split wall (GPU share) | vs HTP alone | limiting engine |
|--:|--:|--:|--:|---|
| 4096  | 563 us  | 295 us (0.55; HTP 282 / GPU 295) | 1.91x | GPU (x1.33 under contention) |
| 8192  | 1067 us | 590 us (0.50; HTP 569 / GPU 590) | 1.81x | GPU (x1.38) |
| 16384 | 2099 us | 1204 us (0.55; HTP 983 / GPU 1204) | 1.74x | GPU (x1.40-1.48) |

The HTP side got 12% faster, so the balance point moved toward the HTP and the GPU kernel under
contention (26-31 GB/s) is now what bounds the wall. Against this morning's starting point
(636/1195/2349 us) the combined effect is 2.2x/2.0x/1.95x on the per-layer attention wall.

### Sustained across a whole token (`--layers 28`)

The third open item: does a chain of pre-launched spinning GPU kernels hold up over a token,
and does dual-streaming drift? The tool pre-enqueues 28 decode kernels on the in-order queue,
each spinning on its own SVM word, and relays one per "layer" while the HTP runs its FA op.

| kv | layers x passes | GPU chain | last-pass partials | HTP FA op / layer | per-layer drift (last/first third) |
|--:|---|---|---|--:|--:|
| 4096  | 28 x 5 | 28/28 ran each pass | OK | 289 us (282-292) | 1.000 |
| 16384 | 28 x 4 | 28/28 ran each pass | OK | 1103 us (1093-1124) | 1.003 |

The chain holds: every kernel runs, in order, correct, and the GPU busy span tracks the HTP
token wall. Per-layer HTP FA time is flat -- no thermal or driver drift over the token, and it
matches the single-shot numbers, so the spinning kernels do not disturb the HTP. (The GPU
kernel's own event span inflates to ~550 us/layer because it includes the spin waiting for its
relay; that is idle power, not added latency -- the compute is still ~300 us and it finishes
before the next relay.) The one cost that remains real is the spin itself: ~1 ms of GPU
occupancy per layer at 4k, pure power.

### Correction to the 4g dispatch side-finding: big-core pinning hurts real decode

Section 4g found the sync-probe dispatch ~3-5x cheaper with the main thread pinned to a big
core, and flagged the ~620 us per-`graph_compute` constant as possibly little-core scheduling.
The `llama-bench` A/B settles it the other way (Qwen3-1.7B, tg64 @ d4096, taskset, 3x4 reps,
interleaved):

| process affinity | tg64 @ d4096 |
|---|--:|
| big cores only (6,7) | 8.9 t/s |
| all 8 cores (default) | 13.3 t/s |

Forcing the process onto the two big cores is **1.5x slower**, because decode has real
CPU-resident work (the lm-head GEMV unless `GGML_HEXAGON_LM_HEAD=1`, sampling, graph build)
that then contends for two cores. The isolated dispatch latency does drop when a lone thread
owns a big core, but at the process level the scheduler's default spread wins. So the ~620 us
constant is not something process-level pinning fixes; a gain would need thread-level affinity
for the dispatch thread only, which is a separate change and not pursued here.

## 4i. Built: the integration prototype (`GGML_HEXAGON_HETERO_FRAC`)

The split of Sections 4g-4h wired into the real decode path, as a research prototype behind an
environment variable. Numerically correct against a CPU reference and faster on the real model at
every depth measured; several of the mechanisms it needed were not the ones the component
measurements suggested, and they are the findings of this section.

### What it is

- **HTP side** (`flash-attn-ops.c`, split-KV dec path): a tagged decode `FLASH_ATTN_EXT` reads
  `op_params[8..10]` (GPU blocks, control slot, GPU splits) and a control buffer attached as
  `src[7]` (`HTP_OP_MAX_INPUTS` is 8 now). It flushes Q, bumps its slot's ready sequence,
  computes KV blocks `[gpu_blocks, n)`, polls the slot's done sequence with `dcinva` (50 ms
  timeout, then it merges what it has and counts the miss), invalidates the GPU partials and
  merges them as extra splits of the existing merge.
- **Host side** (`ggml-hexagon.cpp`, compiled in when OpenCL is available): one 8 MB rpcmem
  control buffer per session; a fine-grain SVM block; ION aliases of the KV-cache and compute
  buffers (by fd); the GQA decode kernel of Section 4h; a relay thread pinned to a big core. Per
  `graph_compute` it tags eligible FA nodes (decode shape, D 64/128, HVX split-KV path, no sinks
  or sparse selection), pre-enqueues one kernel per tagged node, and for each in order: waits for
  the HTP's ready word, copies Q into SVM, releases the kernel through SVM, waits for the GPU's
  done word in SVM, copies the partials into the control buffer, sets the HTP's done word.
- **Share control** (`GGML_HEXAGON_HETERO_ADAPT`, default on): the HTP op records how long it
  waited for `done`; the relay moves that layer's GPU share per token toward a few microseconds
  of wait (see below).
- Tools: `llama-hetero-decode-attn --integrated` drives this path on a synthetic FA op and checks
  it against the CPU reference; `llama-completion`/`llama-bench` with `-dev HTP0` run it on the
  model.

### Findings that changed the design

1. **Writes through the ION alias are not visible per kernel inside an in-order chain.** With 28
   kernels pre-enqueued, a kernel's stores to the aliased rpcmem buffer (partials, done flag)
   reached memory only at the driver's command-buffer boundaries -- every fourth layer in
   practice -- while the kernels themselves executed promptly (the next kernel's spin count
   showed it). The single-kernel probes of Section 4g could not see this. Fine-grain SVM writes
   are visible immediately (the probe's mid-kernel `alive` store), so the GPU now writes Q-in,
   partials and done through SVM and the host relay copies ~60 KB per layer into rpcmem for the
   HTP: 4-13 us per layer.
2. **A GPU kernel that spins longer than the KGSL hang watchdog kills the context.** The first
   version spun unbounded; one missed release faulted the context (`gpu timeout ctx ... ts 1`),
   after which nothing GPU-side completes and the process becomes unkillable in the driver. The
   spin is now capped (~85 ms), so a missed release degrades one layer instead of the device.
3. **`clSVMAlloc` with a 4096-byte alignment argument fails on this driver** (default alignment
   works); the failure was silent apart from a WARN, and hetero simply never engaged in two runs.
4. **llama's default device split puts layers on the OpenCL GPU.** With both `HTP0` and
   `GPUOpenCL` registered, `-ngl 99` without `-dev HTP0` assigned layers 0-8 to the OpenCL backend
   in the 4k runs. Every all-on-HTP number needs `-dev HTP0`; this morning's default
   `llama-bench` runs (13.3 t/s at d4096) were mixed-device -- pinned to HTP0 the same
   configuration gives 20.2 t/s. The big-core taskset A/B redone with `-dev HTP0` still loses
   (12.1 vs 20.4 t/s), so the 4h correction stands.
5. **The device throttles after ~30-60 min of sustained runs**: the standalone bench went from
   847 to 1777 us (GPU alone) and 1079 to 3260 us (HTP) at 16k, and back after cooling. Perf
   numbers from that window were discarded; the final table alternates arms back to back.
6. **Common-tool logging hides the backend.** `llama-completion`, `llama-perplexity` and
   `llama-bench` drop ggml INFO/DEBUG lines unless `-v`; and without an explicit `-ngl`,
   `common_fit_params` offloads nothing to HTP0 ("did not report memory"), so a run can silently
   be CPU-only. Both cost a debugging round each.

### Correctness

| check | result |
|---|---|
| `--integrated`, backend hetero path vs CPU reference, share 0.5, kv 1024 / 4096 / 8192 | max abs err 1.6e-5 / 9.6e-6 / 7.8e-6 (HTP-only: 1.7e-5) |
| same, share 0.25, kv 4096 | 1.3e-5 |
| `llama-perplexity -b 1` (every token through the decode path), docs corpus, c=1024, 2 chunks | CPU 15.720 / 12.542; HTP-only 15.703 / 12.548; split 0.25: 15.808 / 12.597; 0.5: 15.787 / 12.638; 0.75: 15.525 / 12.460 |
| handshake counters over 47 tokens x 28 layers | 0 ready timeouts, 0 GPU timeouts, 0 DSP timeouts |

The perplexity moves +-1% and non-monotonically with the share (the 0.75 arm is better than the
CPU reference), which is the perturbation signature of a different but correct attention
(f32 exp on the GPU part, f16 on the HTP part), not of a defect; the direct comparison settles
it at the 1e-5 level. Greedy continuations diverge after ~10 identical tokens for the same
reason.

### Performance on the model (`llama-bench`, `-dev HTP0`, Qwen3-1.7B Q4_0)

Cool device (45-70 C during the runs), arms alternated back to back, first run after an idle
period discarded (it is a DVFS ramp for both arms: 8.5 and 13.4 t/s at d16384/d8192 with the
split off):

| depth | HTP only | split, fixed share 0.5 | gain |
|--:|--:|--:|--:|
| d4096  (tg64) | 20.2-20.3 t/s | 22.6-23.0 t/s | **1.12x** |
| d8192  (tg64) | 15.38 t/s     | 18.90 t/s     | **1.23x** |
| d16384 (tg32) | 10.52 t/s     | 13.50 t/s     | **1.28x** |

These are the token-level numbers the component measurements projected (Section 4h: ~1.28x
at 4k was an over-estimate because it used the pre-4f-bis HTP baseline; the per-layer wall
halves, and attention's share of the token grows with depth, hence the trend). Hot device
(70-80 C, GPU throttled, HTP not): d4096 1.11x, d8192 1.00x, d16384 0.98x with the same fixed
share -- see below.

### The fixed share is fragile; the HTP's wait time is the control signal

A fixed 0.5 share balanced the two engines only in the thermal state it was chosen for. When the
GPU throttles (70-80 C after ~30 min of runs; the HTP does not, its DCVS corners are pinned), the
GPU side becomes the long pole and the HTP idles waiting for `done`: the 1.23x at d8192 measured
cool dropped to 1.00x hot, and d16384 went slightly negative. The prototype therefore feeds back:
the HTP op writes how long it waited for `done` into its slot (`+8`), and the relay moves that
layer's GPU share one 64-key block per token -- shrink if the wait exceeded 40 us, grow if it was
under 5 us (`GGML_HEXAGON_HETERO_ADAPT=1`, default). Each layer converges on its own to a few
microseconds of wait, so a throttled GPU degrades toward HTP-only performance instead of below
it, and a cool GPU takes what it can.

Hot device (72-84 C, after ~10 min of back-to-back benchmarks; each arm twice, alternated):

| depth | HTP only | fixed share 0.5 | adaptive share (start 0.5) |
|--:|--:|--:|--:|
| d8192  (tg64) | 15.37 t/s | 15.40 t/s (1.00x) | 18.16-18.40 t/s (**1.18-1.20x**) |
| d16384 (tg32) | 10.59-10.67 t/s | 10.65 t/s (1.00x) | 11.52-12.78 t/s (**1.09-1.20x**, includes the convergence tokens) |

Convergence trace at d16384, hot: GPU blocks 128 -> 105 -> 89 -> ~60 of 260 over batches 1-5,
the HTP's average wait for the GPU 1206 -> 918 -> 510 -> 14 us, then 0-8 us for the rest of the
run; the per-token attention span drops from 81 to 62 ms. Steps are proportional to the wait
(`wait_us / 17` blocks, capped at 16, since an HTP block costs ~8.5 us) so a cold start from 0.5
settles in ~5 tokens; growth is one block per token. When the GPU is cool the controller lands
within ~2% of the best fixed share (d8192: 18.4 vs 18.8 t/s), so the cost of the feedback is
small and the protection is worth it: the split is never slower than HTP-only by more than the
convergence transient.

### Short context (d512-d4096): neutral below ~2k -- and a side effect that is not the split

Same procedure as above (cool device, arms alternated back to back, two rounds each, `-dev HTP0`,
tg64), lm-head on the CPU as in every table so far:

| depth | HTP only | fixed share 0.5 | adaptive (start 0.5) | adaptive gain |
|--:|--:|--:|--:|--:|
| d0    | 29.64 t/s | 29.70 | 29.94 | 1.01x |
| d512  | 27.08 | 27.06 | 28.23 | 1.04x |
| d1024 | 25.84 | 25.49 | 26.33 | 1.02x |
| d2048 | 23.62 | 24.59 | 24.98 | 1.06x |
| d4096 | 20.48 | 22.66 | 22.93 | 1.12x |

No depth loses, and nothing below d2048 gains more than the run-to-run noise (about +-1 t/s at
d512). The device profile (`GGML_HEXAGON_PROFILE=1`, tg16, per-token medians after the controller
settles) says why:

| depth (KV as padded) | FA/token, HTP only | FA/token, split | GPU blocks (of total) | HTP wait for GPU | FA share of token |
|--:|--:|--:|--:|--:|--:|
| d512  (768 keys, 12 blocks)  | 3.56 ms | 3.59 ms | 2-3 of 12   | 2 us  | 10% |
| d1024 (1280 keys, 20 blocks) | 5.78 ms | 5.04 ms | 4-8 of 20   | 10 us | 15% |
| d2048 (2304 keys, 36 blocks) | 9.27 ms | 6.70 ms | 15-16 of 36 | 19 us | 22% |
| d4096 (4352 keys, 68 blocks) | 16.6 ms | 10.1 ms | 31-35 of 68 | 16 us | 34% |

Two things cap the short-context gain. Attention is 10-22% of the token below d4096, so even
halving it is worth 5-11% at most. And the split has per-layer costs that do not shrink with the
KV: the HTP op pays ~25 us of its own (Q flush, `ready` publish, `done` poll, partial invalidate,
extra merge slots: 129 us measured vs 105 us modelled at d512), the GPU kernel has a ~45 us floor
however few keys it gets, and the relay adds ~10 us. An HTP block costs 8.4 us (FA/layer fits
26 us + 8.4 us x blocks over the four HTP-only points), so the GPU has to take 3-4 blocks before
the split breaks even, and it can only take blocks the HTP would otherwise still be working on
after the GPU's floor. At d512 the controller settles on its floor of 2-3 blocks of 12, the op is
unchanged (3.59 vs 3.56 ms) and so is the token; at d1024 the GPU gets 4-8 of 20 for 0.7 ms per
token, at d2048 15-16 of 36 for 2.6 ms. The 2-block floor plus the wait-driven controller is what
keeps the short arms from going negative -- a fixed 0.5 share is already 1% under HTP-only at d1024.

**With the lm-head on the HTP** (`GGML_HEXAGON_LM_HEAD=1`, Section 4b: 29.6 -> 38.7 t/s at d0) the
token is shorter and attention's share larger, and the same sweep reads 1.10x / 1.06x / 1.08x /
1.15x / 1.24x at d0 / 512 / 1024 / 2048 / 4096 (38.7 -> 42.5, 36.6 -> 38.7, 33.1 -> 35.9,
29.5 -> 33.9, 24.2 -> 30.1 t/s). The d0 row is the tell: attention is ~1 ms of a 26 ms token there,
so a 10% gain cannot be attention. The profile shows the *non-attention* DSP time per token dropping
from 22.6-23.5 to 20.9-21.9 ms (-1.7 ms, 7%) at every depth whenever the split is on. Two controls
isolate the cause (each arm twice, alternated, lm-head on the HTP):

| depth | HTP only | + busy loop pinned to CPU 7 | + pure-ALU GPU kernel, other process | split (adaptive) |
|--:|--:|--:|--:|--:|
| d0    | 39.5-39.8 t/s | 40.0-40.5 (1.01x) | **43.4-43.6 (1.10x)** | 42.2-42.7 (1.07x) |
| d2048 | 29.4-29.7     | 29.9-30.0 (1.01x) | **31.6 (1.07x)**      | 33.8-33.9 (1.15x) |
| d4096 | 24.2          | --                | **25.5 (1.05x)**      | 29.9-30.0 (1.24x) |

A busy prime core does nothing. A GPU kernel that touches no memory (`llama-gpu-keepalive`,
`examples/hetero-decode-attn/gpu-keepalive.c`: back-to-back 20 ms ALU loops, GPU busy 90-99%)
speeds HTP-only decode up by 2.0-2.3 ms per token at every depth -- the whole d0 effect, and more
than the split itself delivers there. Profiled at d0, the DSP batch per token shortens from 24.25
to 22.15 ms with that kernel running while attention stays at 1.5 ms: the saving is in the GEMVs,
on the device. The memory system is faster while the GPU is active. The HTP already votes
`HAP_DCVS_VCORNER_MAX` for its bus and its core clock is pinned (`htp/main.c`), and the GPU clock
reads 900 MHz in every arm, so what is left is the GPU's own DDR/NoC bandwidth vote or the
fabric's idle states between the HTP's DMA bursts. The readable DDR node
(`bus_dcvs/DDR/cur_freq`) cannot settle it: it shows only the CPU cluster's vote, which rises in
the split arm from the relay thread's copies and not at all in the GPU-kernel arm.

Net of that side effect the split's own contribution with the lm-head on the HTP is -2% / +7% /
+18% at d0 / d2048 / d4096 (split vs HTP-only-plus-GPU-kernel), which is exactly the attention
saving the profile shows (0 / 2.6 / 6.4 ms out of 23-41 ms tokens). With the lm-head on the CPU
the same GPU kernel moves the token by 1% or less (d0 30.0 -> 30.1, d4096 20.2 -> 20.4 t/s, with the
split at 1.13x at d4096 in the same run), although the DSP batch still shortens by 0.9 ms (17.7 ->
16.8 ms): the CPU's own lm-head GEMV keeps the memory system busy for half of every token and hides
the effect. So every table in this document with the lm-head on the CPU is an attention-only
result; the `GGML_HEXAGON_LM_HEAD=1` numbers are not, and any HTP+GPU claim in that configuration
has to be made against an HTP-only-plus-`llama-gpu-keepalive` baseline, not against HTP-only.

The answer for 512-4k, then: the split is never a loss, is worth 2-6% at 1k-2k and 12% at 4k with
the lm-head on the CPU, and net of the GPU-activity effect the same 0% / 7% / 18% at 1k / 2k / 4k
with the lm-head on the HTP. Below ~2k the token belongs to the GEMVs, and the way to make a second
engine pay there is to stream weights on it, not attention.

### Is there DRAM bandwidth left for a second engine? Measured with the weights streaming

The short-context token is GEMVs, so the heterogeneous question there is whether two masters get
more out of the DRAM than the HTP alone. Measured at d0 with the lm-head on the HTP (the HTP
streams 1124 MB of weights per token; the PMU counts 1084 MB of AXI reads at 191 B/request, so the
GEMV bandwidth below is hardware-counted) while a GPU streamer (`llama-gpu-stream`: float4 reads
over a 256 MB buffer, duty-cycled) runs in another process. The GPU rate is its per-second plateau
during the decode phase; solo, the same streamer reads 54 / 25 / 13 GB/s at duty 1 / 0.5 / 0.25.

| GPU traffic during decode | HTP GEMV bandwidth (AXI-counted) | GPU | aggregate | token (HTP-only decode) |
|---|--:|--:|--:|--:|
| none, GPU idle                     | 50.6 GB/s | --   | 50.6 GB/s | 39.5 t/s |
| none, GPU busy on ALU (keep-alive) | 55.7      | --   | 55.7      | 43.5 |
| streamer, duty 0.25                | ~50 (from the token rate) | 5.8  | ~56 | 39.5 |
| streamer, duty 0.5                 | 48.1      | 13.6 | 61.7      | 37.8 |
| streamer, duty 1                   | 39.2      | 27.5 | 66.7      | 32.0 |

Three things follow. The DRAM does have more than the HTP takes: two masters reach 67 GB/s (87%
of the 77 GB/s theoretical) against 56 for the HTP alone with the fabric awake, so a weight split's
raw ceiling on GEMV time is ~1.2x -- about 3 ms of a 23 ms token, ~1.15x, before any per-op cost;
with 113 fused GEMV ops per token, 25 us of split/merge per op would consume all of it. The HTP's
own stream is latency-sensitive rather than bandwidth-limited: it runs at a fixed DMA depth, so
co-streaming raises its latency and it loses bandwidth nearly one-for-one at light GPU traffic
(duty 0.25: the GPU takes 5.8 GB/s and the aggregate does not move) and 30% at full GPU traffic;
the extra 11 GB/s appears only once the GPU takes ~40% of the bytes. And the keep-alive effect
(50.6 -> 55.7 GB/s, 11%) is the same latency story from the other side: a faster fabric lets the
same DMA depth carry more, which is a larger and cheaper gain than a weight split would net.

So the short-context answer holds at the operator level too: attention below 2k is
fixed-cost-bound, and the GEMVs that own the token have ~1.15x of bandwidth to gain from a second
engine at a per-op crossing cost of the same size. On this SoC the second engine pays for decode
only where the work is large per crossing: attention at depth.

### The decode token phase by phase, hardware-counted (d0 and d4096)

The question "is the memory busy during the whole token" can be answered with the PMU rather than
inferred: `GGML_HEXAGON_PROFILE=0x3,0x41,0xce,0x43,0xcf,0x7d,0x8c,0x40` counts the cDSP's AXI read
requests on every op (Section 4b). The request size depends on the access pattern, so it has to be
calibrated per class against known bytes: weight rows come out at 191 B/request (1084 MB counted vs
1124 MB of weights), while the attention op's 256-byte K/V rows come out at ~256 B/request (385 MB
x 256/191 = 516 MB vs 503 MB of padded KV + mask at d4096). With that, tg32 with the lm-head on
the HTP (unit 55b03820, PMU logging on, so the token is ~15% slower than the plain runs):

| op class (per token) | ops | d0: ms | d0: GB/s | d4096: ms | d4096: MB read | d4096: GB/s |
|---|--:|--:|--:|--:|--:|--:|
| gate/up GEMV (`MUL_MAT+MUL_MAT`)        | 28  | 7.60 | 52.2 | 7.71  | 397 | 51.5 |
| lm-head GEMV                            | 1   | 5.86 | 49.5 | 5.80  | 290 | 50.0 |
| down / o-proj GEMV (`MUL_MAT+ADD`)      | 55  | 5.27 | 49.8 | 5.46  | 263 | 48.2 |
| QKV GEMV (`MUL_MAT+MUL_MAT+MUL_MAT`)    | 28  | 2.65 | 49.9 | 2.68  | 133 | 49.5 |
| attention (`FLASH_ATTN_EXT`, 256 B/req) | 28  | 1.53 | ~20  | 16.66 | 516 | 31.0 |
| norms, RoPE, SwiGLU, KV writes          | 253 | ~1.3 | 2-3  | 1.52  | 4   | 2-3 |
| gaps inside the DSP batch               |     | --   | 0    | 0.28  | 0   | 0 |
| host tail (plain run, wall minus batch) |     | ~1.0 | 0    | ~1.1  | 0   | 0 |

So during decode the DRAM is drawn on for the whole token except ~2.9 ms of small ops, dispatch
gaps and host tail (7% at d4096): the GEMVs pull ~50 GB/s for 21.5 ms, attention ~31 GB/s for
16.7 ms at d4096. Two more things the counters settle:

- **The attention op is not bound by DRAM latency or bandwidth.** With the GPU keep-alive kernel
  running, every GEMV class speeds up 8-12% (48-52 -> 54-58 GB/s) while attention does not move
  (16.66 -> 16.61 ms, 23.1 -> 23.2 GB/s at 191 B/req); under a full GPU stream the GEMVs lose 25%
  and attention 3%. Its limiter is the DMA engine's handling of its own descriptors (two 64-row
  x 256 B transfers in flight per thread, ~4.6 GB/s per thread), not the fabric -- the GEMV path
  drives the same engines 2.5x harder with 16 descriptors queued. Bigger descriptors (all 8 heads
  of a block, or several blocks per head) and deeper queues are the fix, and only that.
- **In the split, the HTP reads 54% of the KV** (208 MB vs 385 MB at 191 B/req) in 11.09 ms
  instead of 16.66: proportional would be 9.0 ms, the extra 2.1 ms is the wait for `done`, the
  per-layer hetero overhead and the merge. The GEMVs in the split arm run at the keep-alive rate
  (52-57 GB/s), because the spinning kernel keeps the GPU busy -- so the split's token-level gain
  with the lm-head on the HTP includes that effect, as Section "Short context" accounts for.

Writes are not counted (none of the eight events tracks AXI writes; the KV writes are 112 KB/token
and the outputs are small, <1% of traffic), the counter sees only the cDSP's own port (the GPU's
traffic is read from its own tool), and per-op numbers are op averages -- sub-op phases (DMA vs
QK vs softmax inside attention) need `GGML_HEXAGON_PROFILE=3` trace events, which the kernel emits.

### Status and what is left

The prototype is complete as a research result: a correct, self-balancing HTP+GPU decode
attention on the real model, 1.12x/1.23x/1.28x at 4k/8k/16k when the GPU is unthrottled and
~1.2x at depth when it is; neutral (never negative) below 2k, where attention is 10-15% of the
token and the per-layer fixed costs leave the GPU only its 2-block floor. It is not mergeable as is: it links OpenCL into the hexagon backend,
mutates the FA node's `op_params`/`src[7]` from the backend, and carries a per-layer spinning GPU
kernel (~1 ms of GPU occupancy per layer, pure power). Left open: the GPU kernel under
contention (26-31 GB/s) is now the long pole at every depth -- a faster decode kernel (half8 V
loads, fewer lanes per row) would move the balance point and the gain; the Section 4f-bis DMA
pattern on the HTP side would do the same from the other end; and the share controller could
use the GPU's own done latency as a second signal instead of only the HTP's wait. Separately, the
GPU-activity effect above is a lever of its own: with the lm-head on the HTP, HTP-only decode is
5-10% faster whenever the GPU merely stays busy, so a bandwidth vote that achieves the same
without a GPU kernel (the HTP's bus corner is already MAX, so the missing vote is elsewhere --
GPU or CPU side) would bank that for free; until then every HTP+GPU number in that configuration
is read net of it.


## 4j. Built and measured: the prefill split (shared blocks on the NPU, exceptions on the GPU)

The block-sparse prefill kernel (`docs/backend/snapdragon/sparse-attention.md`) serves one
256-row query tile with the UNION of what its four 64-row sub-blocks selected. The union costs
recall nothing but costs work: a block wanted by one sub-block is computed for all 256 rows. The
split idea: count, per tile and KV head, how many sub-blocks wanted each block, `c(b)`; blocks
with `c(b) >= c*` stay on the NPU as the tile's shared list, blocks below are *exceptions* that the
GPU computes for exactly the sub-blocks that wanted them, and the two partials are merged by the
online-softmax identity. `c* = 2` is the operating point: with the deployed threshold scorer
(`LLAMA_SPARSE_ATTN=thr:1.0`) the union is ~16 blocks per tile and the shared list ~12, so the NPU
sheds a quarter of its attention work — about 400-530 us of a 2.3-2.6 ms op at `ub=1024`,
kv=4096, Qwen3-1.7B — if the GPU's part is hidden and the merge is free.

### What was built

Everything below is on the branch, off by default, and exact where a check exists.

- **The fold** (`HTP_FA_FOLD_MAGIC` in `src[7]`, `ggml/src/ggml-hexagon/htp/flash-attn-ops.c`):
  the HMX op merges another engine's unnormalised `(m, l, acc)` partial into its own output. Three
  realisations were built and priced: re-aiming the final HMX normalisation diagonal at
  `w_htp/S_total` (the "diagonal fold"); merging in the store threads against the raw
  accumulator, whose column-major tile order the store now addresses directly
  (`HTP_FA_FOLD_F_INSTORE`); and, for comparison only, spilling the HTP's own partial and merging
  in a separate HVX pass (`HTP_FA_FOLD_F_SPILL`). A finding that decides correctness for any
  external merge: `vtcm_m_vec` does not hold `m`, it holds `(m + scale) * log2(e)`; the extra
  term cancels in `O/l` and is invisible to the shipped kernel.
- **The handshake** (`HTP_FA_FOLD_F_LIVE`, `HTP_FA_FOLD_F_STAGED`, `HTP_FA_FOLD_F_STAGE_QB`): the
  op publishes `ready` at entry and, instead of one wait before its first tile, takes one wait per
  *stage* (a 256-row query block across all heads) at the first tile that needs it, runs the
  tiles without exceptions first, skips the fold on them, and per-stage-invalidates only the rows
  it is about to read. A head start (`LLAMA_SPARSE_ATTN_HEADSTART=K`) leaves the first K tiles on
  the union so the GPU's launch latency lands in their shadow.
- **The GPU side**: mllm's `bs_fa2_fused` exception kernel ported to llama.cpp's tensor layouts
  (`examples/fa-exc-gpu`), reading Q, V and the mask in place through an ION alias of the hexagon
  buffers and a `image1d_buffer` view for V, staging only the exception blocks' K transposed, and
  emitting the C1 triple. Standalone harnesses (`examples/fa-fold-check`, `examples/fa-hetero-live`)
  run the NPU op, the GPU kernel and the merge over one rpcmem buffer against a double-precision
  reference, with arms for NPU-only, serial, single-wait live and staged live.
- **The graph** (`src/llama-graph.cpp`, `src/llama-sparse-attn.h`): `LLAMA_SPARSE_ATTN_CSTAR`
  cuts the shared list from `c(b)`, hands the exception *membership* (a 0/1 row per sub-block,
  no packing) to the attention node as `src[8]` (`ggml_flash_attn_ext_set_sparse_exc`), and
  implements head start and an optional per-tile cap (`LLAMA_SPARSE_ATTN_GPU_CAP`) with HTP-native
  ops only.
- **The backend sidecar** (`ggml_hexagon_hfold_*` in `ggml-hexagon.cpp`, `GGML_HEXAGON_FA_FOLD=1`):
  one fold buffer attached as `src[7]`, per-buffer ION aliases dropped when the buffer is freed,
  and a relay thread that, per attention op, waits for `ready`, scans the membership into
  per-head compact block tables, stages K^T through an image write, launches the exception kernel
  once per query block and publishes `done[stage]` as each completes. A host-reference check of
  GPU partial rows (`GGML_HEXAGON_FA_FOLD_CHECK=1`) and a keep-alive control
  (`GGML_HEXAGON_FA_FOLD_KEEPALIVE=<us>`) are built in. `HTP_OP_MAX_INPUTS` went from 8 to 12.

### Op level (harness, unit f3b4a4c5/9aed338b, lq=1024, kv=4096, 16 heads / 8 KV heads)

The merge. Cost over the dense op of the same shape, min of 3 interleaved:

| merge realisation | cost per op | exact |
|---|---|---|
| diagonal fold (in the HMX normalisation) | +290 to +350 us | yes, nmse 1.1e-7 |
| in-store fold, partial prefetched a tile ahead | +113 to +186 us | yes, nmse 7.0e-8 |
| in-store fold without the prefetch | +980 us | yes |
| explicit merge on the NPU (spill + HVX pass) | +890 to +920 us | yes |
| explicit merge on the GPU (earlier, `merge_bench`) | 0.38 ms quiet, 0.61 under HTP load | yes |
| HTP read-read-write floor for the same 24.6 MB (f32 ADD) | 278 us | — |

The in-store fold is about 4-6 us per tile and is only paid on tiles with exceptions. Its
prefetch had to be issued from the first chunk's softmax workers, one *linear* l2fetch each: the
64-bit l2fetch descriptor's stride field is 16 bits (v75 PRM `Rtt[47:32]`), so a 2D fetch cannot
span GQA heads (512 KB apart) or the two merge regions (8 MB apart) — the first attempts were
walking the wrong addresses — and requests queue three deep per thread rather than cancelling.

The overlap. Live arms at the deployed-like split (union 16 -> pool 12, exceptions on every head),
in-store fold, GPU staged per query block:

| arm | attention op | NPU blocked | note |
|---|---|---|---|
| A npu-only (union, real sparse kernel) | 3.59 to 3.87 ms | — | the baseline shape |
| C live, single wait | ~4.9 ms | 1.3 to 2.0 ms | the GPU hides 0% |
| D staged, head start 0 | 3.9 to 4.1 ms | 0.5 to 0.6 ms | first-stage stall: launch + first stage + event wake-up |
| D staged, head start 6 | 3.30 ms | 0 | GPU hidden 100%, 1.09x |
| D staged, half the heads exception-free | −8% to −16% | 0 | reorder buys a clean shadow |

The GPU exception kernel: 0.84-0.86 ms back-to-back for 128 pairs with mllm's mapping (32-row
work-groups, 2 rows x 8 keys per thread). A 64-row work-group ran 1.87 ms (12 float8 accumulators
per thread; occupancy collapses) and a 4-rows x 4-keys mapping 1.10 ms (half the texel reads per
FMA, still slower): the kernel is issue/latency-bound at these launch sizes, not texture-bound,
and the original mapping is the floor. Under a concurrent HTP prefill it runs 36-38% slower; the
HTP under a full GPU stream changes <1%.

### Model level (`llama-bench` pp4096, `llama-perplexity` wikitext-2, ctx 4096, ub 1024)

Correctness: 896 GPU partial rows against a double-precision reference, 0 mismatches; perplexity
on the same two chunks: union 19.6745, split with head start 12 at 19.6362, split with head start
0 at 19.7034 (the split is a slightly different policy, not bit-equal to the union by design;
`c* = 0` is bit-identical to the shipped path).

Speed, alternating arms in one session (thermal drift between consecutive runs is 10-20%; only
adjacent pairs count, see §6):

| configuration | pp4096 | attention op | SWIGLU | fused projection |
|---|---|---|---|---|
| union (`CSTAR=0`) | 2388-2397 t/s | 2318 us | 1248 us | 1618 us |
| union + keep-alive (dummy 1.2 ms GPU kernel / 18 ms, no attention on the GPU) | 2435-2531 t/s | 2242 us | 1147 us | 1563 us |
| split, head start 12, per-query-block staging | 2400-2408 t/s | 2235-2242 us | 1130 us | 1548 us |
| split, head start 8 | ≈ union | 2335 us | | |
| split, GPU cap 4, head start 4 / 6 | below union | 2605 / 2487 us | | |

The dummy kernel alone gives +2 to +6% and shortens every memory-bound DSP op — including the
union's own attention op — to exactly the numbers the split reaches. The split never beats that
control. Its own contribution is zero within noise. The mechanism is §4i's: a periodically busy
GPU raises the shared bus/DDR clocks. Before that control was run, two host-side effects had
masked and then mimicked a gain: the membership tensor marked as a graph output made
`ggml_backend_sched` cut one split per layer (76 ms per forward outside DSP ops), and
`GGML_HEXAGON_VERBOSE=1` costs ~85 ms per forward of logging.

### Why it does not pay

Per (sub-block, block) exception pair the NPU saves ~3 us by shedding it and the GPU spends
~12 us computing it under contention (~9 us quiet, 5-6.5 us in mllm's large-launch measurement).
The GPU can only run in the NPU's shadow — the exception work needs this layer's Q, K and V,
produced immediately before the attention op, and its output is needed immediately after — so it
can absorb at most ~60% of the pairs the deployed scorer produces (~5.4 per tile per head, ~173
per op, twice the harness's assumption), and the head start that buys its lead hands the rest of
the saving back. A per-tile cap does worse: pairs over the cap rejoin the shared list and grow
the pool. Memory contention is a third of the GPU's cost, small launches another third, the
kernel the rest; even at zero contention and full launches the GPU would need about twice the NPU
time it saves per pair. The kernel remaps confirmed the current mapping is the floor. The design
is complete, exact and shelved.

### Lessons that cost device time

1. ION aliases must be dropped when a hexagon buffer is freed: the compute buffer is reallocated
   when the ubatch grows, the dmabuf fd is recycled, and an alias keyed by fd keeps the OLD pages
   alive — the GPU read stale Q and mask from the third ubatch on, deterministically.
2. K^T written through a buffer alias and read through an image view returns the previous op's
   keys; the texture cache is only invalidated for image writes. Stage through `write_imageh`.
3. `ARANGE` and `SUB` are not HTP ops; one CPU node per layer splits the graph (-15%). Build
   index vectors from `scale_bias(0,1) -> cumsum` and differences from `mul`/`scale_bias`.
4. A tensor consumed by the backend must not be a graph output once a node depends on it: one
   `ggml_set_output` per layer = one scheduler split per layer.
5. Host reference checks after `done` are racy (the next layer overwrites Q at the same
   address); hold the done words until the reference ran.
6. The explicit pre-ready flushes of Q/K/V/mask/membership (~32 MB of `dccleaninva` per op) are
   redundant with the op-start dirty flush (`htp_tensor_flush_all`, whole-L2 above 4 MB).
7. The raw HMX accumulator is in column-major tile order; only the norm pass makes it row-major.
8. Staging by KV head reorders the tiles so the mask DMA cache no longer dedups the 8 heads of a
   query block: ~12 MB of mask per op instead of ~1.6. Stage by query block.
9. Decompose wall time into profiled DSP-op time and the remainder before blaming either engine;
   and run the keep-alive control before believing any HTP+GPU gain.

### Status

The prefill split is a working, exact prototype: `LLAMA_SPARSE_ATTN=thr:1.0
LLAMA_SPARSE_ATTN_CSTAR=2 LLAMA_SPARSE_ATTN_HEADSTART=12 GGML_HEXAGON_FA_FOLD=1
GGML_HEXAGON_FA_FOLD_PERF=0`. As built here it is not faster than the union baseline once the
GPU's bus-clock lift is controlled for; §4k removes the launch stall this section blamed and gets
the attention op to 1.09x over that control, which is the ceiling. What is worth
carrying forward from it: the in-store fold contract and its findings (the `m` domain, the
l2fetch descriptor limits, the accumulator layout), the per-stage handshake, and the keep-alive
observation — a GPU that merely stays busy is worth 2-4% of prefill throughput on any
configuration, dense included, at the cost of GPU power; whether a bus or DDR performance vote
gets the same lift without a GPU kernel is the open question.

## 4k. The launch overhead removed: a GPU chain that needs nothing from the host

§4j's ledger left one line the design could still move: the first-stage stall (GPU launch, host
tables, K^T staging, first stage, event wake-up) was 0.5-1.0 ms, so the head start had to be 12 of
32 tiles and handed back ~190 us of the ~500 us the pool saves. This section removes that stall.
Everything is measured on unit f3b4a4c5, Qwen3-1.7B Q4_0, pp4096 at `ub=1024`, the deployed
scorer (`thr:1.0`, `c* = 2`), and stays exact (perplexity below).

### What changed

- **A gate kernel in front of the chain.** Every attention op's GPU work is one in-order chain,
  `gate -> stage_k_p -> fa_exc`, enqueued at graph-post time for the whole graph (the first chain
  from the posting thread itself, the rest from the relay). The gate is a single work-item that
  spins until it reads the op's sequence number in the DSP's own `ready` word, through the ION
  alias with device-scope atomic loads, or in a fine-grain SVM mirror the relay writes as a
  fallback; it then zeroes the op's counters and publishes the sequence, and every kernel behind
  it checks that word and does nothing if the gate timed out (the relay then runs the old
  host-driven path). `GGML_HEXAGON_FA_FOLD_PREQ` selects when chains are enqueued (-2 whole graph
  at post, default; -1 after the previous op's chain; N us before the predicted ready; 0 at ready).
- **Compaction on the GPU.** The staging kernel builds each KV head's compact exception table in
  local memory from the membership rows the graph already produces, and the work-group that takes
  a head's first unit publishes the tables `fa_exc` reads. Nothing is computed on the host after
  `ready`.
- **Persistent work-groups.** Both kernels run a fixed number of 128-thread work-groups (64 for
  staging, 32 for attention, `GGML_HEXAGON_FA_FOLD_PWG*`) that pull units from a counter: staging
  units head-major, attention units stage-major. Attention stages are (256-row query block, KV
  head) pairs, `HTP_FA_FOLD_F_STAGE_QBH`, 32 per op at `ub=1024`; the done line moved out of the
  slot to `hdr.off_done` (up to 64 words). The last work-group of a stage (per-stage atomic count)
  writes an ION stamp, the ION done word the DSP polls, and an SVM copy for the relay's bookkeeping.
- **The relay thread is off the critical path.** It mirrors `ready` into SVM, copies SVM done
  words to the DSP's line (never needed, see below), keeps statistics, and runs the fallback.

### Findings, each measured

1. **GPU writes through the ION alias are visible mid-kernel.** The stamp the kernel writes just
   before its SVM done word was already visible on the host at the moment the SVM word flipped,
   for every one of 6528-9728 stages in every run. The DSP polls done words the kernel itself
   wrote; the host is not in the done path.
2. **The GPU can poll the DSP's `ready` word.** With device-scope atomic loads on the ION alias the
   gate opened on the DSP's own word in 194-203 of 204 ops, before the relay (20 us naps) had seen
   it. §4g's "the GPU cannot poll rpcmem" was true of plain loads, which stay cached. The host is
   not in the ready path either; gate open is 0-8 us after the relay's own detection.
3. **Launch shape decides progressive completion.** One plain launch per op (512 work-groups) had
   the first of 32 stages complete at 2.5 ms of 3.9: every work-group shares the GPU and finishes
   with the rest. One launch per stage cost ~65 us per boundary and the last stage landed 4.4-5.3
   ms after ready. Persistent work-groups pulling stage-ordered units complete a stage every 27-30
   us; 32 work-groups match 64, 128 is slower.
4. **The real bottleneck behind the gate was the K^T staging**, 1.6 ms when launched over the
   staging capacity with early exits: the trace showed `stage_k` starting at +115 us and `fa_exc`
   at +1700. The persistent, count-bounded form takes 110-190 us. Kernel boundaries inside the
   chain cost 30-100 us each, variable.
5. **The relay's wake-up matters only for the first op of a graph**, and it matters a lot when the
   CPU is busy: `llama-perplexity` (logits on the CPU between graphs) blocked the DSP 620 us per op
   because the first chain went in after `ready`. Enqueuing that chain from the posting thread,
   before the DSP is dispatched, brought it to 54 us per op (18-22 in `llama-bench`).
6. **K^T through buffer loads instead of the texture is 25% slower** in the attention units
   (stage spacing 40 vs 30 us), which rules out merging staging into the attention kernel with
   per-head readiness (image writes are not coherent with image reads inside one kernel).

Timeline per op (trace, us after `ready`, head start 6): gate open 0-8, `stage_k_p` starts 16-90,
`fa_exc` starts 175-300, first stage done 196-300, last stage done 1100-1300; the DSP's mean
blocked time 18-22 us per op.

### Results

Attention op, `GGML_HEXAGON_PROFILE=1`, mean of 224 ops per run, two interleaved repetitions:

| arm | FLASH_ATTN_EXT us | DSP blocked us/op |
|---|---|---|
| union (`CSTAR=0`) | 2311 / 2305 | - |
| union + keep-alive (dummy GPU kernel, §4j control) | 2254 / 2235 | - |
| split, chain, head start 3 | 2118 / 2134 | 116 |
| split, chain, head start 4 | 2080 / 2097 | 63 |
| split, chain, head start 5 | 2063 / 2073 | 38 |
| split, chain, head start 6 | 2054 / 2066 | 15 |

The split now beats the keep-alive control on the attention op by ~190 us, 1.09x (1.12x against the
plain union), which §4j's pipeline never did. It sits at the design's own ceiling for this GPU:
pool ~1800 + in-store fold ~150 + head start 6 ~90. Below 6 tiles the DSP blocks on the first
stage; above, each tile hands back ~15 us of pool saving.

Correctness: perplexity on the same two wikitext chunks, union 19.6745, split head start 4 19.6065,
head start 6 19.5747 (the sub-block policy; `c* = 0` stays bit-identical to the shipped path).

End to end (`llama-bench` pp4096, three interleaved rounds, thermal drift between rounds):

| round | union | union + keep-alive | split, head start 6 |
|---|---|---|---|
| 1 | 2441 | 2523 | 2470 |
| 2 | 2321 | 2307 | 2271 |
| 3 | 2173 | 2170 | 2173 |

The device cooled between the earlier sweeps and heated through this one (-11% over three rounds),
and the three arms are within that drift of each other in every round. The ~1.3% the profile
predicts for the split beyond the keep-alive lift is below what pp4096 throughput can resolve here;
the per-op profile above is the measurement.

### The ledger, measured (unit 9aed338b, every arm with the GPU keep-alive so the bus clock is equal)

`GGML_HEXAGON_FA_FOLD_NOTAG=1` initialises the sidecar (keep-alive) but tags no node, so the NPU
runs the pool alone: a timing floor whose output lacks the exceptions. Mean FLASH_ATTN_EXT per op,
two interleaved repetitions:

| arm | us | note |
|---|---|---|
| union | 2234 / 2260 | ~15.7 blocks per tile and KV head |
| pool only, c* = 2, head start 0 | 1614 / 1636 | the split's ceiling; 620 us below the union |
| pool only, head start 6 | 1730 / 1727 | the head start costs ~100 |
| split, chain, head start 6 | 2056 / 2056 | DSP blocked 18 us/op |
| split, chain, head start 0 | 2272 / 2290 | DSP blocked ~300 us/op |
| pool only, c* = 3 | 1173 / 1149 | |
| forced blocks only (`thr:99`, sink + own blocks, ~5 per tile) | 819 / 820 | |

From the union, pool and forced-only points: ~4.2 us per (256-token tile, KV head, 64-key block)
on the NPU, the harness's number, and ~150 us per op that does not scale with blocks. The split
at head start 6 decomposes as pool 1614 + head start 100 + fold and partial handling ~310 +
blocked 18. The fold's ~310-355 is well above the harness's 113-186. Timing-only probes
(`GGML_HEXAGON_FA_FOLD_PROBE=<HTP_FA_FOLD_F_PROBE_* bits>`) split it: skipping the per-stage line
invalidation of the partial (~8.3 MB of range `dcinva` per op) takes the op from 2063/2071 to
1947/1958 us, so that invalidation costs ~115 us per op; the rest, ~220 us, is the partial itself
(7 MB per op read through L2: weights, gathers, the accumulator add). The l2fetch is essential:
without it the store reads DDR synchronously and the op is 2538/2540 us. Since the op-start flush
is a whole-L2 flush-invalidate whenever more than 4 MB of inputs are dirty (Q alone is 8 MB at
`ub=1024`), the per-stage invalidation is redundant at this shape and only needs a cheap guard for
small ubatches.

### Correctness: the stale-read trap between the DSP and the GPU

Pushing the ledger further first cost a day of correctness work, and the findings matter more than
the microseconds. Every arm below is `llama-perplexity` on the same two wikitext chunks; the union
reproduces to four decimals on every unit, so the split has to as well.

- **The GPU read the previous layer's Q.** Pre-enqueued chains read the inputs of the *previous*
  op that had occupied the same address (the graph allocator reuses buffers layer to layer), for
  10-20% of the rows checked, and the fold's result changed run to run (19.53 to 19.72 across
  eight runs). Host reference reads of the same Q rows at `ready` were correct and stable, the
  values were plausible activations, and neither an acquire fence at all-SVM-devices scope, nor
  atomic loads, nor a 5 ms hold after `ready`, nor an in-order map/unmap or a migration, nor GPU
  idle time before the submission changed it. What did: **a submission made after the DSP wrote
  the data, referencing the buffer**. A one-work-item kernel that takes Q, the mask, the
  membership, K and V as arguments, submitted by the relay on a second queue when it sees
  `ready`, with the gate opening only on the relay's word written after that submission, gives
  0 wrong rows in 204 and a perplexity of 19.5647 on three consecutive runs
  (`GGML_HEXAGON_FA_FOLD_TRIG=2`, the default). The same kernel submitted but not waited for
  leaves the mismatches, so it is the submission's cache maintenance that does it. The host is
  back on the `ready` path for one tiny enqueue (~40 us after the relay's detection).
- **`hex_l2fetch_block` over-fetches by up to 16 KB.** It rounds every fetch up to whole 16 KB
  rows, so the partial prefetch pulled the first rows of the *next* query block's stage into the
  DSP's L2 before the GPU had written them. The per-stage range invalidation happened to clean
  that up; without it the stale rows were folded. `hex_l2fetch_rows` fetches exact row runs and is
  what every partial prefetch uses now.
- **The per-stage invalidation is redundant when a whole-L2 flush-invalidate ran since the last
  fold read.** The DSP context now carries an epoch bumped by every whole-L2 flush (batch start
  and end, the op-start flush above 4 MB of dirty inputs); the fold skips the 8 MB of range
  `dcinva` when the epoch moved and does one whole-L2 invalidate itself otherwise. Doing that
  whole-L2 flush unconditionally before `ready` cost 18% of pp4096 and fixed nothing, because the
  stale reads were on the GPU's side.
- **Determinism is the test.** A split that is "close to the union" can be wrong; two identical
  runs that disagree to the fourth decimal always are. The check that found the cause snapshots
  the membership row and the Q row at `ready` and recomputes a few exception rows on the host
  after the chain finishes (`GGML_HEXAGON_FA_FOLD_CHECK=2`); a reference computed later from the
  live buffers reads the next layer's inputs and is garbage itself.

Two of the ledger's items landed on top of that, both exact: the fold skips a 32-row group whose
partial is empty (about a fifth of the groups), and the GPU can write the partial's accumulator as
f16 (`GGML_HEXAGON_FA_FOLD_ACC16=1`, `HTP_FA_FOLD_F_ACC_F16`), halving the 7 MB per op the fold
reads, 0 wrong rows in 204 at the f16 tolerance and perplexity 19.5164 twice at head start 8.

### Pushing the ledger: what landed and what it measured

With the correctness settled, the ledger's items went in one by one (unit 9aed338b, attention op
per call, pp4096, ub 1024, every arm with GPU activity so the bus clock is equal):

- **Epoch-guarded invalidation** (above): 2063 to 1969 us at head start 6 on the same device state.
- **No K^T staging.** The exception kernel reads K in its natural `[key][d]` layout through an
  image view of the K cache, exactly as it already read V, and builds each work-group's block list
  from the membership row with atomic loads. The staging kernel and every table are gone, the
  chain is gate plus one persistent kernel, and the first stage lands 115-150 us after `ready`
  instead of 370-390. On an unthrottled GPU its stages also complete sooner (last done 1.24-1.28
  ms against 1.44-1.50). `GGML_HEXAGON_FA_FOLD_KNAT=1`, **opt-in**: its perplexity did not
  reproduce (19.6350 twice, then 19.7070) although the reference check passed on every row it
  probed, including a second-wave row added for it; the staged-K^T path reproduces to four
  decimals across five runs. One real race was found and fixed (the block list scanned by one
  thread was read by the other wave before a barrier); whatever remains is not in the rows the
  check covers. Gating the chain on the trigger's *completion* to rule out a read racing the
  trigger's maintenance deadlocks instead: the second command queue does not run concurrently
  with the first on this device, so the trigger lands behind the pre-enqueued chain and every op
  times out. The submission, not the trigger's execution, is what makes the data visible.
- **f16 partial accumulator**: 2021 to 1987-2001 us at head start 8, 0 wrong rows in 204 at the
  f16 tolerance. Default.
- **Empty 32-row groups skip the accumulator**, bit-exact.

One interleaved round on a device warming from cold (the MUL_MAT mean drifted 855 to 1257 us
across it, so the later arms carry a thermal handicap):

| arm | FLASH_ATTN_EXT us | DSP blocked us/op |
|---|---|---|
| union + keep-alive | 2263 | - |
| split, staged K^T, f32 partial, head start 6 | 2039 | 89 |
| split, staged K^T, f32 partial, head start 8 | 2021 | 37 |
| split, staged K^T, f16 partial, head start 8 | 2001 | 41 |
| pool only, head start 8 (the floor) | 1766 | - |
| split, natural K, f32 partial, head start 8 | 1987 | 12 |
| split, natural K, f32 partial, head start 6 | 1952 | 11 |

The natural-layout kernel with the f16 partial (trigger-gated chain, epoch-guarded invalidation,
empty groups skipped), two interleaved rounds after an eight-minute cool-down (the MUL_MAT mean
read 837 us at the start and 1126 at the end, so the later arms carry the handicap). These are
the fastest numbers measured, and they are the opt-in configuration for the reason above:

| arm | FLASH_ATTN_EXT us, round 1 / 2 | DSP blocked us/op | first stage done after ready |
|---|---|---|---|
| union + keep-alive | 2251 / 2251 | - | - |
| split, natural K, head start 6 | 1931 / 1925 | 16 / 7 | 111-114 us |
| split, natural K, head start 4 | 1911 / 1921 | 20 / 25 | 110-112 us |
| split, natural K, head start 3 | 1908 / 1905 | 31 / 27 | 111-113 us |

With the natural-layout kernel the attention op is 1.17x the bus-lifted union's, ~340 us per op;
end to end the adjacent pairs read +1.2% (2516 vs 2487 t/s) in the cool round and are inside the
drift in the hot one, as the ~2% the op gain predicts would be. The **shipped default** is the
staged-K^T path with the f16 partial at head start 8, the configuration that reproduces:
perplexity 19.5164 on three consecutive runs with the reference check clean, attention op 1995 us
at head start 8 (DSP blocked 42 us/op) and 2011 at head start 6 (94 us/op) against the union's
2251, i.e. 1.13x, with the first stage 372 us after `ready` and the last 1.47-1.52 ms.

**A deployment caveat the same sweep exposed.** In its second round the Adreno throttled harder
than the HTP: the GPU's own time per op went from 1.2-1.5 ms to 2.3-2.8 ms while the union's
attention op stayed at 2269 us, the NPU stalled behind the GPU for 470-1000 us per op and the
split's attention op became 2554-3036 us, far worse than the union. The split's gain depends on
the GPU keeping pace; a sustained-load guard (fall back to the union when the DSP's blocked time
climbs) would be needed before shipping it.

### Union against split over the prompt length

The shipped defaults (staged K^T, f16 partial, head start 8) against the union, `llama-bench`
pp512 to pp8192 at `ub=1024`, two interleaved rounds after a cool-down (round 1 / round 2). No
keep-alive control in this sweep, so the throughput column carries the bus-clock lift of §4i as
well as the attention gain; the attention-op column is the split's own contribution.

| prompt | union t/s | split t/s | union FLASH_ATTN_EXT us | split FLASH_ATTN_EXT us | split DSP blocked us/op | GPU first / last done after ready |
|---|---|---|---|---|---|---|
| 512 | 2382 / 2392 | 2501 / 2550 | 808 / 812 | 766 / 767 | - | - |
| 1024 | 2516 / 2546 | 2698 / 2687 | 1561 / 1554 | 1405 / 1407 | - | - |
| 2048 | 2457 / 2427 | 2673 / 2538 | 1793 / 1798 | 1596 / 1588 | 55 / 51 | 347 / 1270 us |
| 4096 | 2387 / 2263 | 2524 / 2287 | 2301 / 2304 | 1993 / 1994 | 38 / 42 | 371 / 1468 us |
| 8192 | 2203 / 2004 | 2194 / 1995 | 3345 / 3337 | 3115 / 3118 | 218 / 218 | 480 / 2205 us |

(The per-graph statistics print every eighth graph, so the two-graph runs at 512 and 1024 show
none.) The attention op gains 5% at 512, 10% at 1024, 11% at 2048, 13% at 4096 and 7% at 8192.
At 8192 the GPU stops hiding: its per-op time grows to 2.2 ms (more exception blocks per tile
with 128 KV blocks), the DSP blocks 218 us per op behind it at head start 8, and the union's
MUL_MAT runs 7% slower under the GPU's heavier K/V traffic (867 to 933 us), so the DSP's total
time drops 1.5% while throughput is flat. Throughput below 8192 is +5 to +9% in the cool round
and +1 to +6% in the hot one; the attention gain alone predicts +1 to +2% of that, the rest is
the bus-clock lift.

### Dense, naive block-64, union and split

The finest sparse granularity the kernel offers is one list per 64-query block with no union
(`LLAMA_SPARSE_ATTN_BQ` built as 64: `Br 64`, 128 rows per tile), which is the policy the split
reproduces with pool plus exceptions; dense is the shipped HMX flash attention with no selection.
Four arms interleaved, two rounds, same protocol (round 1 / round 2; the second round ran hot,
the dense arm at 8192 dropped 9% between rounds):

| prompt | dense FLASH_ATTN_EXT us | naive bq=64 | union | split |
|---|---|---|---|---|
| 512 | 1029 / 1031 | 775 / 780 | 809 / 814 | 766 / 765 |
| 1024 | 2713 / 2716 | 1716 / 1706 | 1554 / 1562 | 1417 / 1404 |
| 2048 | 3826 / 3820 | 2011 / 2017 | 1779 / 1785 | 1592 / 1589 |
| 4096 | 6185 / 6210 | 2658 / 2664 | 2305 / 2325 | 1999 / 1993 |
| 8192 | 11982 / 12028 | 3917 / 3919 | 3338 / 3343 | 3123 / 3120 |

| prompt | dense t/s | naive bq=64 | union | split |
|---|---|---|---|---|
| 512 | 2441 / 2310 | 2384 / 2350 | 2364 / 2320 | 2525 / 2472 |
| 1024 | 2455 / 2208 | 2488 / 2350 | 2512 / 2348 | 2715 / 2428 |
| 2048 | 2311 / 2001 | 2445 / 2188 | 2467 / 2202 | 2657 / 2256 |
| 4096 | 1996 / 1730 | 2310 / 2028 | 2290 / 2003 | 2327 / 2029 |
| 8192 | 1474 / 1340 | 1965 / 1779 | 1924 / 1764 | 1911 / 1771 |

Attention op relative to dense (round 1): naive 1.33x, 1.58x, 1.90x, 2.33x, 3.06x from 512 to
8192; union 1.27x, 1.75x, 2.15x, 2.68x, 3.59x; split 1.34x, 1.91x, 2.40x, 3.09x, 3.84x. The
split is 2% below the naive baseline at 512 and 18-25% below it from 1024 up (25% at 4096); the
union sits between the two (9-14% below naive from 1024 up, 4% above it at 512, where forced
blocks dominate and the union's over-compute is pure loss). The naive baseline costs only 1.15x
the union with the threshold scorer, not the 1.7x of the fixed-u measurement in
`dynamic-sparse-attention.md` §5.3: its per-block lists are exact and shorter, which pays back
part of the small-tile inefficiency. End to end, naive and union are equal (the R = 1 selection
has no union step, so it skips the per-tile argsort and buys back its slower attention); the
split leads both below 8192 and matches them at 8192, where its GPU stops hiding (see above).
Against dense, throughput is +3%, +11%, +15%, +17%, +30% for the split and -3%, +2%, +7%, +15%,
+31% for the union, from 512 to 8192 in the cool round. Perplexity on the two wikitext chunks:
dense 20.1955, naive 19.9329, union 19.6745, split 19.5164 (standard error about 1.1 on each;
two chunks are a smoke test, not a quality measurement); the naive arm was expected to match the
split's policy, so its gap points at a selection-side difference of the `BQ=64` build rather
than at the kernel, and was not chased.

### The controlled experiment: one set of wanted pairs, three executions

Everything above compares *policies* as much as executions: the union attends more pairs than the
block-64 build, and the split, whose rows also see the pool blocks that two or more *other*
sub-blocks selected, sits between them (its perplexity differs from the exact policy's for that
reason, 19.52 against 19.93-19.99 on the two chunks). To isolate execution, the wanted set is
held fixed (the same threshold scorer, one list per 64-query block) and executed three ways:

- **small-tile NPU**: the `bq=64` build, `Br 64`, computes exactly the wanted pairs;
- **large-tile NPU, unwanted pairs masked**: `LLAMA_SPARSE_ATTN_EXACT=1`, the 256-query tile runs
  its union list and the HMX softmax sets every (32-row group, KV block) the group's sub-block did
  not select to `-inf` (one bit per 64-key vector, the chunk's block ids hoisted), so it computes
  the union's pairs and keeps the exact result. Reproducible (19.9962 twice, 19.9887 after the
  hoist; the block-64 build reads 19.9329);
- **heterogeneous**: the split, pool on the NPU and the sub-blocks' own exceptions on the GPU.

Unit f3b4a4c5 after a redeploy, round 1 / round 2; the masked arm's hoisted numbers are one
round of a separate cool run against a union rerun (the unhoisted masking, 13-20% over the union,
was my bookkeeping and not the execution):

| prompt | small-tile NPU (bq=64) | large-tile NPU masked | union (same compute as masked) | heterogeneous split |
|---|---|---|---|---|
| 512 | 779 / 784 | 858 | 820 / 829 | 767 / 772 |
| 1024 | 1711 / 1713 | 1612 | 1542 / 1549 | 1409 / 1412 |
| 2048 | 2017 / 2019 | 1890 | 1785 / 1790 | 1585 / 1591 |
| 4096 | 2665 / 2668 | 2512 | 2324 / 2333 | 1995 / 2006 |
| 8192 | 3932 / 3933 | 3636 | 3353 / 3356 | 3238 / 3132 |

Throughput (t/s, round 1 / round 2): small-tile 2454 / 2403, 2524 / 2544, 2489 / 2425, 2332 /
2260, 2102 / 1985; union 2427 / 2397, 2586 / 2558, 2507 / 2401, 2401 / 2204, 2091 / 1951; split
2565 / 2566, 2762 / 2680, 2652 / 2483, 2500 / 2265, 2061 / 1968.

Reading: for the same wanted pairs, cooperation is the fastest execution on this SoC. The split
is 18-25% below the small-tile NPU from 1024 to 4096 and 18-20% below it at 8192, and 13-14%
below the large-tile NPU at 4096 and 8192 whether the unwanted pairs are masked (2512, 3636) or
simply kept (2324, 3353). The small tile loses to the large tile even though it computes 42%
fewer units: re-streaming K and V per 64 rows and running the HMX at four row tiles cost more than
the compute it saves, and the masked large tile cannot recover that because masking discards
results, not work. What the split removes is the union's over-compute on the c = 1 blocks, and it
does so at a GPU rate that only pays because the GPU runs in the NPU's shadow: at 8192 the shadow
is no longer long enough (DSP blocked 222-345 us per op) and the three executions converge.

### What it means

The launch overhead is gone in the sense that matters: between the DSP's `ready` and the GPU's
first stage there is one tiny host submission (the trigger the correctness section made
necessary), the first stage lands 115-150 us later, and the DSP stops blocking at a head start
of a few tiles instead of 12. With the fold's own handling trimmed (epoch-guarded invalidation,
f16 partial, empty groups skipped) the attention op is 1.16x faster than the bus-lifted union
(1911 vs 2251 us at head start 4 with the natural-layout kernel, cooled device, two rounds; the
reproducible default lands near 1.12x), against 1.09x for §4k's first chain and 1.0x for §4j.
That is ~340 us per op at best, ~38 ms per pp4096 forward, about +2% end to end beyond the
keep-alive lift; the pool-only floor (1766 us at head start 8, ~1700 at 4) says ~200 us of fold
and partial handling remain, and the GPU's per-pair rate is the same ~7 us against ~3 us of NPU saving, so
this is close to the whole prize on this SoC, as §4j predicted. The gate work-item spins for ~90%
of the op period; that is the same GPU duty the keep-alive control uses, and the same power
caveat, plus the throttling caveat above.

Settings: `LLAMA_SPARSE_ATTN=thr:1.0 LLAMA_SPARSE_ATTN_CSTAR=2 LLAMA_SPARSE_ATTN_HEADSTART=8
GGML_HEXAGON_FA_FOLD=1 GGML_HEXAGON_FA_FOLD_PERF=0`. Defaults: the chain with persistent
work-groups and whole-graph pre-enqueue, the gate opened by the relay's word after its trigger
submission (`TRIG=2`), staged K^T (`KNAT=0`; `KNAT=1` is the faster, not yet reproducible
variant), f16 partial (`ACC16=1`), device-scope per-group fences (`WGFENCE=0`). `GGML_HEXAGON_FA_FOLD_GPUDONE=0` restores §4j's host-driven path,
`GGML_HEXAGON_FA_FOLD_CHECK=2` runs the snapshot reference check, `GGML_HEXAGON_FA_FOLD_TRACE=N`
prints per-stage timelines, `GGML_HEXAGON_FA_FOLD_PROBE=<bits>` ORs `HTP_FA_FOLD_F_PROBE_*` into the
fold header.

## 4l. Decode: who should gather the scattered rows of a cluster -- measured (2026-09-15, unit f3b4a4c5)

Whole-cluster selection (cluster-sparse-decode.md, Stages 7-8) is the decode policy that keeps
retrieval quality, and it needs a cluster's keys fetched as one unit. On the HTP that meant the
cluster-ordered **shadow** copy, because a cluster's keys are scattered positions of llama's cache
and the HVX kernel fetches with DMA descriptors whose fixed cost is amortized only over contiguous
rows. The shadow doubles the KV footprint and does not fit at 16k. The prefill split of 4j-4k paid
because the NPU's compute tile was too coarse for singly wanted blocks; the decode analogue is a
**fetch-granularity** asymmetry: the NPU fetches efficiently only in contiguous runs, the GPU
gathers one row per work-item. Two measurements settle whether that asymmetry is real, both in
`llama-hetero-decode-attn` (Qwen3-1.7B shapes, 8 KV heads, 256-B f16 rows, clusters of 16-48 keys,
average 32, sorted by position within a cluster, sinks + a 256-key dense window as in runs mode).

**1. The HTP fetching a run's rows from the positional cache, one linked 1D descriptor per K and
V row** (`--cluster --runs --scatter`, header flag `HTP_FA_CLUSTER_HDR_SCATTER`; the kernel reads
the run's rows through `pos_map` instead of the shadow, the consumer pops `2 x bsz` descriptors
per block, the ring was raised to 1024 descriptors). Op-level exact (max err 1.3e-5 .. 3.2e-5,
the same run set as the shadow arm). FA op median us:

| kv, budget | rows/head | shadow runs (2D block per run) | scatter, chunk-local | scatter, identity perm | scatter, global random | scatter, 4 slots | in-place pages 16 / 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 4k, 12% | 700 | 174 (16.5 GB/s) | 307 (9.3) | 299 | 302 | 309 | 165 / 109 |
| 4k, 25% | 1205 | 249 (19.8) | 515 (9.6) | 522 | 539 | 515 | 254 / 154 |
| 16k, 12% | 2178 | 507 (17.6) | 1061 (8.4) | 1049 | 1044 | 1068 | 445 / 238 |
| 16k, 25% | 4279 | 831 (21.1) | 1997 (8.8) | 1978 | 1977 | 1995 | 826 / 425 |

The address pattern does not matter (identity, chunk-local and global random are within 2%) and
neither does the ring depth: the cost is the descriptor, about 0.1 us each, 0.2 us per row on top
of the ~0.06 us a contiguous row costs. Per-row gathers run at 8.4-9.6 GB/s on six threads, half
the rate of the shadow's runs and a quarter of the dense stream (36-39 GB/s). The NPU cannot do
shadow-free clusters alone at a useful rate; the shadow's justification was correct.

**2. The Adreno gathering the same kind of rows through per-head index lists** (`--gpu-gather`,
kernel `fa_dec_gather`: work-group (KV head, 256-key split), one K row per work-item through the
list, the V product over the listed rows, split-KV partials in the HVX format; OpenCL profiling
stamps; lists = clusters of 32 sorted positions taken in random order, either chunk-local (a
cluster inside one 1024-position window, the sidecar's pattern) or global). Host-merged partials
match the CPU reference to 1e-8. Kernel median us and GB/s over K+V bytes:

| kv, keys/head | streaming kernel over the same count | gather, identity list | gather, chunk-local clusters | gather, global random clusters |
|---|--:|--:|--:|--:|
| 4k, 491 (12%) | 75 (26.8) | 68 (29.5) | 70 (28.8) | 69 (29.1) |
| 4k, 1024 (25%) | 97 (43.2) | 93 (45.1) | 97 (43.3) | 97 (43.2) |
| 4k, 4096 (100%) | 371 (45.2) | 349 (48.1) | 372 (45.1) | 374 (44.9) |
| 16k, 1966 (12%) | 197 (40.9) | 184 (43.8) | 243 (33.1) | 266 (30.3) |
| 16k, 4096 (25%) | 370 (45.3) | 345 (48.7) | 464 (36.2) | 513 (32.7) |
| 16k, 16384 (100%) | 1216 (55.2) | 1138 (59.0) | 1651 (40.6) | 1901 (35.3) |

At 4k the scattered gather costs nothing over streaming. At 16k it costs 20-30% (the buffer is
four times larger; TLB or DRAM-page effects, not the kernel, since the identity list through the
same code is faster than the streaming kernel), and a 512-key split recovers part of it (222 us,
36.3 GB/s at 16k/12%). The first version of the kernel read the row index and the softmax weight
with scalar local loads inside the serial V loop and ran 40% slower than streaming even on an
identity list; 64-bit address math was not the cause, reading eight offsets and eight weights per
head as vectors was the fix. Compute added to this kernel is free: it is bandwidth-bound, as the
user expected.

**What the two measurements establish.** For the rows of a scattered cluster the GPU is 3.5-4x
faster than the HTP (33-45 GB/s against 8.4-9.6), about 2x faster than the HTP reading the same
clusters from the shadow, and as fast as the HTP streaming contiguous pages. So the shadow-free
whole-cluster design is: HTP for the sink page, the dense window, the not-yet-clustered tail and
whatever contiguous pages remain; GPU for the members of the selected clusters, gathered from
llama's own cache through the cluster member lists the sidecar already computes, merged by the HTP
as split-KV partials (the 4h/4i format and merge path). At 16k/12.5% that is a ~245 us gather plus
the ~80 us per-layer crossing, against 507 us for shadow runs (which do not fit at 16k), 445 us for
in-place 16-key positional pages (which lose the UUID retrieval) and 1061 us for the HTP gathering
the rows itself. The design removes the 1.9 GB shadow at 16k and lifts the 6592-position cap at 8k;
at 4k the token cannot gain (attention after selection is ~5 ms of 39 ms, the crossings ~2 ms), so
its value is quality and memory at depth, not speed. Not built: the sidecar emitting member lists
instead of the shadow, the DSP select pass publishing chosen runs to the GPU, the per-layer chain
(4k's) driving the gather kernel, the Q staleness rule of 4k (a post-write submission before the
GPU reads Q).

## 5. What would change the answer

- **Events in both backends.** OpenCL already uses `cl_event` throughout (its `synchronize` is a
  barrier + wait), so `event_new/record/wait` is plumbing. Hexagon would need a completion
  marker on the dspqueue. This unlocks §3.1 — a ≤1.19× ceiling that needs ≥8k context and a
  smaller ubatch than the sparse kernel wants.
- **A pre-launched GPU kernel with a flag relay** (§4g) bypasses the scheduler and the
  dispatch cost entirely for decode attention; what is left to prove is the GPU kernel and the
  shared DRAM budget, not the synchronization.
- **A faster GPU path.** The ratio in §1 is the whole story. A GPU kernel at ≥50% of the HMX
  rate would make a ~1.4× pipelined ceiling plausible; Adreno 830 through the current OpenCL
  backend is not that.
- **Something that is not attention or GEMM.** After the lm-head moves to the HTP (§4b) the
  non-device share of a decode token is ~5%; there is no host-side cost left for a GPU to absorb.
- **A weight split for the GEMVs** has a measured raw ceiling of ~1.15x on the token (two masters
  reach 67 GB/s vs 56 for the HTP alone, §4i) and the HTP's DMA stream loses bandwidth one-for-one
  to light co-traffic; it would pay only with a per-op crossing far under 25 us, which §4g's
  7-9 us relay is, but only for ops big enough to amortize the GPU's ~45 us kernel floor.

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
- `test-backend-ops -o MUL_MAT` on HTP0 scores 486/556 on this device with or without the hetero
  changes (checked with a consistent pre-hetero host+HTP pair): the failures are quantized `o=1`
  cases where the CPU reference itself is NaN. Pre-existing; not a regression signal.
- Host library and HTP skeleton must be pushed as a pair: the op descriptor grew from 7 to 8
  inputs, and a mismatched pair crashes the DSP (`dspqueue_read failed 0x2e`) on the first batch.
- Pass `-dev HTP0` to every llama tool, `-ngl 99` to `llama-perplexity`, `-v` to see backend
  logs, and discard the first `llama-bench` run after the device has idled (DVFS ramp: the off
  arm reads 34.5-35.1 instead of 39.5 t/s at d0 when it goes first; run a warm-up first).
- GPU activity alone speeds the HTP up. With the lm-head on the HTP (CPU idle) a GPU kernel that
  touches no memory makes HTP-only decode 5-10% faster (DSP batch 24.3 -> 22.2 ms/token at d0).
  Every HTP+GPU comparison in that configuration needs an HTP-only + `llama-gpu-keepalive` arm;
  with the lm-head on the CPU the token-level effect is ~1%. A busy CPU core does not do it.
- `GGML_HEXAGON_VERBOSE=1` costs ~85 ms per pp4096 forward of host logging; never leave it on in a
  timed arm, and read the per-op DSP profile (`GGML_HEXAGON_PROFILE=1`) for op times instead.
- Thermal drift between consecutive pp4096 runs on the QDC units is 10-20% (MUL_MAT 858 -> 1111 us
  across four runs), and memory-bound ops speed up while HMX ops slow down, which looks exactly
  like GPU contention. Only adjacent A/B pairs count; use `-r 2` or `-r 3`.
- The QDC pod's adb server is reached through an ssh forward on local 5037; `adb kill-server`
  kills the pod's server and only a new session brings it back. Units and their `/data/local/tmp`
  rotate between sessions (9aed338b <-> f3b4a4c5).
- Host library and HTP skeleton must be pushed as a pair (again): the op descriptor now carries
  12 inputs (`src[8]` is the exception membership).
- `/sys/devices/system/cpu/bus_dcvs/DDR/cur_freq` is the CPU cluster's DDR vote, not the DDR
  clock (no readable aggregate without root), and `kgsl-3d0/gpuclk` reads 900 MHz even with the
  GPU idle on this unit. The KV the decode op sees is the padded cache (768 keys at d512, 4352
  at d4096), not the prompt length.
