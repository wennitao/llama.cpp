# Block-sparse attention on HTP: in-kernel indirection vs. materialized gather

Does llama.cpp need mllm's heterogeneous "gather on CPU/GPU, attend on NPU"
pipeline? This measures the two kernel designs head-to-head on the same silicon,
in llama.cpp, on a Snapdragon 8 Elite.

**TL;DR.** mllm's central finding does not reproduce. A real index-list gather
costs **1-4% on top of attention** here, against **48.3% of a layer** in mllm, and
it gathers straight from a F16 KV cache after a backend fix in this work. But the
verdict between the two designs is **decided by block size, not by gather-vs-index**:
the in-kernel path pins the FA chunk size to `Bc = bs`, so at the deployment setting
(`bs=64`) it starves HMX and loses; at `bs=512` it matches dense attention over the
same rows to within **0.2%** and overtakes the gather. Below ~1k context, neither
design beats simply running dense attention.

---

## Setup

- Device `55b03820`, SM8750 / V79 HTP, `libggml-htp-v79.so`, 6 HVX threads, 8 MB VTCM.
- Qwen3-1.7B attention shape: `hs=128`, `Hkv=8`, `Hq=16` (`nr=2`), F16 K/V.
- Prefill: `nb=512` query tokens (one ubatch).
- All arms **maskless**, so neither is charged for mask traffic the other avoids.
- `test-backend-ops perf -b HTP0`. Every case verified to run on HTP -- confirmed
  with `GGML_HEXAGON_VERBOSE=1`, which shows `HTP0 x HTP0 -> HTP0` buffers on every
  node. Perf mode calls `ggml_backend_graph_compute` directly with no
  `ggml_backend_sched`, so there is no CPU-fallback path at all.

Four arms, all single-backend on HTP:

| Arm | Graph |
|---|---|
| `FLASH_ATTN_EXT_SPARSE` | one fused FA op, block indices in `src[5]` (`src[5]->ne[1] == 1` here -- one selection for the whole op; the per-query-block rows are a separate arm, see [flash-attn-htp-anatomy.md](flash-attn-htp-anatomy.md) §8) |
| `FLASH_ATTN_EXT_GATHER_ROWS` | `GET_ROWS`(+`CPY`) -> dense FA -- **real** run-time I32 index list |
| `FLASH_ATTN_EXT_GATHER` | strided view + `CONT` -> dense FA (regular stride; lower bound) |
| `FLASH_ATTN_EXT` | dense FA over the full KV -- the do-nothing baseline |

### Measurement caveat that invalidated a first pass

`test-backend-ops perf` duplicates only the graph's **output node**, not the graph:

```c
for (int i = 1; i < n_runs; i++) { ggml_graph_add_node(gf, out); }
```

A multi-op arm therefore amortizes every non-output op to `1/n_runs` -- in the
gather arm the gather ran **once** while attention ran up to 672 times. Every leg
is consequently timed as its own output node and the arm reported as the sum. The
single-op arms (SPARSE, dense FA) are unaffected and exact. The sum is an upper
bound in principle, but HTP dispatches ops synchronously (`htp/main.c:1041` is a
plain sequential `for` over ops), so the legs really are serial.

**This caveat applies to any multi-op `test-backend-ops perf` case, not just these.**

---

## Result 1 — the deployment setting: bs=64, 25% density, 512-4k

25% of blocks selected, block size 64, F16 KV cache.

| seq len | `n_sel` | eff. KV | **SPARSE** | gather (GET_ROWS+CPY) | **gather (F16-direct)** | dense (full KV) | winner |
|--:|--:|--:|--:|--:|--:|--:|:--|
|  512 | 2  |  128 | 2182.1 us | 1480.8 us | 1424.5 us | **984.0 us** | *dense beats both* |
| 1024 | 4  |  256 | **958.1**  | 1102.3    | 995.6     | 1473.0 | SPARSE 1.04x (tie) |
| 2048 | 8  |  512 | 1715.7     | 1314.1    | **1085.0**| 2076.2 | GATHER 1.58x |
| 4096 | 16 | 1024 | 3271.3     | 2291.7    | **1701.8**| 4299.1 | GATHER 1.92x |

Leg costs behind the gather columns (K+V, i.e. 2x each): `GET_ROWS` F16->F32
31.1 / 62.0 / 133.3 / 361.8 us; `CPY` 47.7 / 86.9 / 175.8 / 401.1 us; fused
F16->F16 `GET_ROWS` 22.5 / 42.3 / 80.0 / 173.0 us.

Three readings:

1. **The real gather is cheap.** 3-15% of the arm, against **48.3% of a layer** in
   mllm. Index-list addressing costs only **1-2%** over a regular strided view
   (the `FLASH_ATTN_EXT_GATHER` arm), so data-dependent addressing is essentially
   free on this DMA engine.
2. **The `CPY` costs more than the gather itself** and exists solely because
   `ggml_get_rows` cannot emit F16. Removing it (see below) is the single largest
   win available on the gather side.
3. **The sparse penalty tracks block *count*, not rows.** It matches dense at
   `n_sel=4` but falls 1.74x behind at `n_sel=8` and 2.22x at `n_sel=16`.

### 512 context: don't use sparse attention at all

Dense over the whole 512-row KV costs 984.0 us; sparse costs 2182.1 (**0.45x**) and
gather 1424.5 (**0.69x**). Two causes compound:

- **A ~940 us query-side floor** at `nb=512`, independent of KV length (dense FA is
  939.8 us at kv=256 and 984.0 us at kv=512). At kv=512 the KV-side work is already
  below that floor, so sparsity removes work nobody was paying for and adds overhead.
- **Only 2 blocks.** 25% of 512 at `bs=64` gives `n_sel=2`, and both `n_threads` and
  `hmx.pipeline` collapse when `n_kv_blocks < 3` (`ggml-hexagon.cpp:2052-2056`).
  The same cliff hits dense: **dense FA at kv=128 costs 1397.3 us, *worse* than
  939.8 us at kv=256.**

**Block-sparse attention has a minimum useful context. Below ~1k it loses to doing
nothing.** Note this row has `nb == kv`, i.e. a whole-prompt prefill; the floor is a
property of the query count, so 512-token ubatches of a longer prompt behave
differently.

---

## Result 2 — scatter distance is free

`kv` does not matter. Only the number of selected blocks does (`bs=128`):

| `n_sel` | eff. KV | kv=1024 | kv=2048 | kv=4096 |
|--:|--:|--:|--:|--:|
| 4  | 512  | 986.5 us | 986.2 us | 985.2 us |
| 8  | 1024 | 1790.6   | 1792.1   | 1796.9   |
| 16 | 2048 | —        | 3408.5   | 3420.9   |

Four blocks spread over 4096 rows cost the same as four spread over 1024.
**Spreading the selection across DRAM is not measurable.** This answers the question
neither repo had data for -- mllm never indexed in place, so it never asked.

---

## Result 3 — the gap is tiling, not memory

Same **effective KV (2048 rows)**, same spread, varying only the block size:

| `bs` | `n_sel` | SPARSE | GATHER | winner |
|--:|--:|--:|--:|:--|
| 128 | 16 | 3417.1 us | 2134.7 us | GATHER 1.60x |
| 256 | 8  | 2491.0    | 2132.2    | GATHER 1.17x |
| 512 | 4  | **2072.6**| 2132.9    | **SPARSE 1.03x** |

Dense FA over the same 2048 rows costs **2076.2 us** -- the sparse arm at `bs=512`
lands within **0.2%** of it. SPARSE improves **1.65x** as `bs` grows; GATHER is flat.

**In-place block indirection has zero cost once `Bc` is allowed to be wide.**

The mechanism: `ggml_hexagon_fa_sparse_bs()` feeds the graph's block size into
`hmx_fa_find_chunk_size()` as `Bc_fixed`, pinning the FA chunk to `Bc = bs` and
searching only `Br`. Dense FA searches `Bc` freely and picks larger. GATHER is flat
in `bs` precisely because, after materialization, its FA sees a contiguous KV and
recovers that freedom. **The gather was never buying better memory behaviour -- it
was buying back the tiling choice the sparse path gave away.**

---

## Result 4 — F16 `GET_ROWS` on HTP (backend change)

`ggml_get_rows` always returns F32, and the hexagon backend *also* required an F32
**source** -- so gathering used to force the entire KV cache to F32 (~940 MB instead
of ~470 MB for Qwen3-1.7B at 4k) purely to make the gather legal.

`htp/get-rows-ops.c` now covers the full `{F32,F16} x {F32,F16}` matrix: three
convert/copy workers generated from one macro, and the DMA path's hardcoded
`sizeof(float)` replaced by a carried element size so it is re-enabled whenever
source and destination types match (a same-type gather is a raw byte copy).
`ggml_hexagon_supported_get_rows()` accepts F16 sources and F16 destinations.

**F16 sources are also faster**, because the gather reads half the bytes:

| seq len | `GET_ROWS` leg, F32 KV | F16 KV | F16->F16 direct |
|--:|--:|--:|--:|
|  512 |  19.4 us |  15.6 us (1.24x) | **11.3 us** |
| 1024 |  39.9    |  31.0    (1.29x) | **21.1**    |
| 2048 |  84.8    |  66.6    (1.27x) | **40.0**    |
| 4096 | 254.6    | 180.9    (1.41x) | **86.5**    |

The F16->F16 column is the fused `GET_ROWS`+`CPY`: it removes the cast entirely and
roughly halves the remaining leg (115 GB/s vs 65-95 GB/s), cutting the gather-side
overhead **4.4x at 4k** (762.9 -> 173.0 us for K+V).

**Correctness caveat.** `test-backend-ops test -o GET_ROWS -b HTP0` passes with no
regressions, but that covers only **F16->F32** -- the eval cases go through
`ggml_get_rows`, which returns F32. The **F16->F16 and F32->F16 paths have no
CPU-reference coverage**, because ggml cannot build such a graph and the CPU kernel
writes `(float *)dst->data` unconditionally. The benchmark builds that node by hand.
Doing this properly upstream (a destination type on `ggml_get_rows`, or
`ggml_get_rows_cast`) would require a CPU implementation and would then be covered
automatically. **Do not ship the F16-out path without that or a bespoke check.**

Independently useful: MiniMax-M3's MSA decode path gathers the KV cache with
`ggml_get_rows` and previously could not run that op on HTP at all.

---

## Why mllm's conclusions do not transfer

| mllm finding | llama.cpp |
|---|---|
| Gather is slow on NPU (48.3% of a layer) | **Does not transfer** -- 1-4% here |
| Monolithic fused graph >> split graph | **Already banked** -- sparse FA is one op |
| Het pipeline: gather on CPU/GPU, attend on NPU | **Does not transfer; would hurt** |

mllm's gather cost is a **QNN lowering artifact**: a compiled QNN graph cannot
express indirection inside an op, so the selected KV must become a real tensor.
`monolithic_blocksparse_profile.md` lists the fix as its own top unimplemented
lever -- *"gather only indices and index inside a fused attention op"* -- which is
exactly what the llama.cpp `src[5]` design already does.

The heterogeneous pipeline is additionally blocked at the framework level:
`ggml_backend_sched_compute_splits` is a single sequential loop, and hexagon's
`cpy_tensor_async`, `event_record` and `event_wait` are all NULL with
`.events = false`. **Overlap is 0%**, so any work moved off the NPU is strictly
additive. mllm's split-pipeline win depends entirely on hiding CPU prep under NPU
compute.

For context on the ceiling: mllm measured attention at **16.5% of dense prefill at
Sq=512**, capping any sparse-attention win at ~1.20x end-to-end even with a free
kernel. **No equivalent end-to-end measurement exists for llama.cpp yet** -- every
number in this document is kernel-level.

---

## Result 5 — pricing the heterogeneous pipeline (gather on CPU, attend on NPU)

mllm's design hides the gather behind NPU compute. Measuring both sides on this
device (per-tensor legs, x2 for K+V; the CPU cannot emit F16 from `ggml_get_rows`
either, so its gather also needs the cast):

| seq len | CPU gather | NPU gather (fused F16) | all-NPU | het, serialized | het, **perfect** overlap | ceiling | what llama.cpp does today |
|--:|--:|--:|--:|--:|--:|--:|--:|
|  512 |  52.8 us |  22.5 us | 1424.5 | 1454.8 | 1402.0 | **1.02x** | 0.98x |
| 1024 |  99.8    |  42.3    |  995.6 | 1053.1 |  953.4 | **1.04x** | 0.95x |
| 2048 | 202.6    |  80.0    | 1085.0 | 1207.7 | 1005.1 | **1.08x** | 0.90x |
| 4096 | 402.0    | 173.0    | 1701.8 | 1930.8 | 1528.8 | **1.11x** | 0.88x |

**The ceiling is 1.02-1.11x**, and that assumes zero transfer cost, zero sync
overhead, and perfect overlap. **What llama.cpp can actually express today is
0.88-0.98x -- a slowdown** -- because `ggml_backend_sched_compute_splits` is a
single sequential loop and hexagon exposes no async copy or events, so work moved
off the NPU is strictly additive.

**The F16 gather fix (Result 4) is what collapsed the prize.** At kv=4096:

- before F16-direct: all-NPU 2291.7 us, perfect het 1528.8 us -> **1.50x** available
- after F16-direct:  all-NPU 1701.8 us, perfect het 1528.8 us -> **1.11x** available

And the premise inverts with it: **the NPU now gathers 2.32x faster than the CPU**
(173 vs 402 us for K+V at 4k). mllm moved the gather off the NPU because the NPU was
bad at it; on HTP, with a fused F16 gather, the NPU is the better gatherer.

### Which partner? CPU vs Adreno vs the NPU itself

mllm found the GPU gathers well. It does not reproduce here. Gather cost for K+V
(`2 x (get_rows + cast)`), all three backends on the same device:

| seq len | CPU | OpenCL (Adreno 830) | HTP | **HTP fused F16** |
|--:|--:|--:|--:|--:|
|  512 |  52.8 us |  71.8 us |  78.9 us | **22.5 us** |
| 1024 |  99.8    | 126.6    | 149.0    | **42.3**    |
| 2048 | 202.6    | 235.5    | 309.1    | **80.0**    |
| 4096 | 402.0    | 468.2    | 762.9    | **173.0**   |

Gather-kernel bandwidth (GB/s), by size:

|  | 512 | 1024 | 2048 | 4096 |
|---|--:|--:|--:|--:|
| CPU        | 161.5 | 161.9 | 159.0 | 153.1 |
| OpenCL     |  76.7 |  85.2 |  88.4 |  86.5 |
| HTP        |  94.3 |  94.7 |  88.2 |  65.0 |
| HTP fused  | 108.6 | 115.9 | 122.5 | 113.3 |

**The Adreno is the *slowest* of the three at gathering** -- about half the CPU's
throughput. Its `CPY` is actually faster than the CPU's (98.2 vs 124.2 us at 4k);
it is `GET_ROWS` specifically that is slow (135.9 vs 76.8 us). At ~86 GB/s on an
Adreno 830 this is almost certainly a limitation of llama.cpp's OpenCL `get_rows`
kernel rather than of the hardware, so it is a statement about the kernel, not
about the GPU.

**The fused F16 gather on the NPU beats all of them** -- 2.3x the CPU and 2.7x the
Adreno at 4k. The partner question is therefore moot: there is no third device that
gathers better than the NPU now does itself.

And the choice of partner does not move the ceiling anyway, because the ceiling is
bounded by *attention* time, not gather time: with either CPU or OpenCL as partner
it is 1.02 / 1.04 / 1.08 / 1.11x at 512 / 1k / 2k / 4k. The Adreno additionally
cannot read an HTP-resident KV cache at all without a full copy --
`ggml_hexagon_supported_buffers` requires every src and dst to live in the same
hexagon session buffer.

There is also no overlappable work *within* a layer -- gather -> attention is a
strict dependency. Realising even the 1.11x requires pipelining across layers or
ubatches with a **stale selection**, which is what mllm did (Jaccard ~0.92) and
whose end-to-end accuracy cost that project never measured.

**Verdict: not worth building.** It needs an upstream `ggml_backend_sched` change
plus hexagon async/events plus an accuracy compromise, to chase 1.11x -- against
1.65x available from `Bc` decoupling in a single file with no accuracy cost.

---

## The optimization this exposes

Selection granularity and kernel tiling are currently the same parameter, so a model
wanting fine-grained `bs=64` retrieval pays up to 1.65x. **Decouple them**: let
`Bc = m * bs` and have `fa_prefetch_block()` issue `m` DMA descriptors -- one per
selected block -- into a single `Bc`-sized VTCM chunk. The blocks already arrive via
independent 2D descriptors, so nothing in the DMA path forbids it; only
`fa_kv_block_start()`'s one-block-per-chunk assumption (`sel[b] * factx->Bc`) does.

Result 2 is what makes this safe: since scatter distance is free, the `m` blocks in
a chunk need not be adjacent. It would also lift the `n_kv_blocks < 3` collapse at
short context -- 8 blocks of 64 tiled at `Bc=512` is one wide chunk, not two narrow
ones.

**The in-kernel path is already the pipelined design.** `fa_prefetch_block()` DMAs
each selected block straight into a VTCM double buffer, prefetched two blocks ahead,
overlapping DMA with HVX softmax and HMX matmul. The gather arm, by contrast, is
three serialized whole-tensor ops with the data crossing DDR five times per tensor
(`14RH` bytes vs `2RH`, of which `8RH` is the F32 detour). Pipelining the gather
chain would require a descriptor ABI that expresses partial tensors, dependency
tracking (`finalize_ranges()` is an empty function body), concurrent execution
contexts, and a VTCM ownership model -- i.e. reimplementing what
`fa_kv_block_start` + `fa_prefetch_block` already do in ~12 lines.

---

## Test coverage

Added in this work, all passing (49/49 on HTP0):

- `FLASH_ATTN_EXT_SPARSE` with `nr=1` -- per-head selection, exercising the
  `sel_nb2` stride that every prior case left at zero.
- `per_head_sel` flag -- decouples a per-head *mask* from a per-head *selection*.
- Eval cases for both gather arms and all three legs, F32 and F16 KV, so a wrong
  stride or index list cannot hide as a plausible timing number.
- `FLASH_ATTN_EXT_PERMASK` -- dense FA with a per-head mask. See
  [fa-per-head-mask-bug.md](fa-per-head-mask-bug.md).

Known gaps:

- **Maskless sparse is unverifiable here** -- and it is what all perf rows use.
  Without a mask the CPU reference computes dense attention, so there is no
  reference. Structural limitation of the harness.
- **F16->F16 / F32->F16 `get_rows`** -- no CPU reference (see Result 4).
- **Batch > 1** (`sel_nb3`) and **out-of-range `sel`** are untested. The latter
  matters: five DMA sites compute `hex_smin(Bc, nek1 - start)` without the
  `start < n_kv` guard that `fa_kv_block_rows()` applies, so a bad index underflows
  to a huge row count and reads out of bounds.

## Notes on precision

The entire HMX FA data path is `__fp16` -- not just K/V but the scores, the
probabilities, the rescale factors, and **the output accumulator** (`vtcm_o_tiles`).
`GGML_PREC_F32` appears to be ignored by this backend: no `prec` field reaches the
DSP and `grep GGML_PREC` in `ggml-hexagon.cpp` finds only function names. Both arms
run identical precision so the comparisons are sound, but **fp16 accumulation of `O`
and `l` at long KV is a real numerical question that is unmeasured here** --
`test-backend-ops` checks `max_nmse_err = 5e-4` only at small eval sizes.

## Reproduce

```sh
cmake --build build-sparse --target htp-v79 ggml-hexagon test-backend-ops -j 32
adb push build-sparse/bin/test-backend-ops                          /data/local/tmp/llama.cpp/bin/
adb push build-sparse/ggml/src/ggml-hexagon/libggml-htp-v79.so      /data/local/tmp/llama.cpp/lib/
adb push build-sparse/bin/libggml-hexagon.so                        /data/local/tmp/llama.cpp/lib/

# perf: all arms, prefill sweep
D=HTP0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops perf \
    -o FLASH_ATTN_EXT_SPARSE,FLASH_ATTN_EXT_GATHER,FLASH_ATTN_EXT_GATHER_ROWS,\
GATHER_GET_ROWS_LEG,GATHER_GET_ROWS_F16_LEG,GATHER_CAST_LEG,FLASH_ATTN_EXT \
    -b HTP0 -p 'nb=512'

# correctness
D=HTP0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops test \
    -o FLASH_ATTN_EXT_SPARSE,FLASH_ATTN_EXT_PERMASK,FLASH_ATTN_EXT_GATHER,\
FLASH_ATTN_EXT_GATHER_ROWS,GATHER_GET_ROWS_LEG,GATHER_CAST_LEG,GET_ROWS -b HTP0
```

Cases live in `make_test_cases_perf()` / `make_test_cases_eval()` in
`tests/test-backend-ops.cpp`. Maskless sparse cases are perf-only by design.
`NHVX=1` forces the non-pipelined path, which is useful for isolating
pipelining-related behaviour.

## Per-query-block selection: where the 3x actually goes

Real XAttention selects a different KV block set per query block. The kernel now supports
that (`sel` gains a query-block axis), and it measured 2.99x slower at `Bq = 64`. The MAC
count is identical across every row below -- 2.15 GFLOP -- so every difference is schedule,
not math.

### The indexing itself is free

| | time | |
|:--|--:|:--|
| shared selection (mean of 2 runs) | 998.0 µs | baseline |
| per-query-block, `Bq = 512` | **995.2 µs** | **0.997x** |

At `Bq = 512` one kernel query tile still covers the whole 512-token chunk, so `Br` is
unchanged and the only difference is which row of `sel` is read. It costs nothing
measurable. **The mechanism has no intrinsic overhead.**

### The cost is the `Br <= Bq` constraint

A kernel query tile spans `[q_start, q_start + Br)` and resolves one selection, so it may
not straddle two scorer query blocks. Pinning `Bq` therefore pins `Br`, and that drives two
separate costs. Sweeping `Bq` with `bs` and `n_sel` held fixed (kv=2048, nb=512, bs=64,
n_sel=8, G=2):

| Bq | Br | q_blocks | softmax threads | time | vs Bq=512 |
|--:|--:|--:|--:|--:|--:|
| 512 | 512 | 1 | 6 | 995.2 | 1.00x |
| 256 | 256 | 2 | 6 | 1326.6 | 1.33x |
| 128 | 128 | 4 | 4 | 1571.3 | 1.58x |
| 64 | 64 | 8 | 2 | **2994.7** | **3.01x** |

- **(a) K/V re-staging.** Staging lives inside the `q_start` loop
  (`flash-attn-ops.c:2180`, with every `fa_phase_k/v_interleave` and `fa_push_chunk` call
  at `:2213`-`:2470` inside it), so each query tile re-streams its selected blocks.
  `q_blocks = nb/Br`, so `Bq=64` streams K/V 8x (2 MiB -> 16 MiB).
- **(b) Softmax thread starvation.** `fa_phase_softmax_and_build_d` (`:1646`) uses
  `n_use = min(n_threads, ceil(n_rows_g/64))` with `n_rows_g = n_rows_q * G`
  (`:2182`). At `G=2`: `Br=512` -> 1024 rows -> 16 -> **6 threads**; `Br=64` -> 128 rows
  -> 2 -> **2 threads**. Softmax is the serializing phase of the pipeline.

### Which dominates: the GQA control

Regroup the same 16 Q heads as `nh=2, nr=8`. Now `G=8`, so at `Br=64`
`ceil(8*64/64) = 8` and **all 6 softmax threads survive** -- while `q_blocks` is still 8,
i.e. the full re-staging penalty is still paid:

| | shared | per-qblock `Bq=64` | ratio |
|:--|--:|--:|--:|
| `nh=8, nr=2` (G=2, **2 threads**) | 998.0 | 2994.7 | **3.00x** |
| `nh=2, nr=8` (G=8, **6 threads**) | 917.7 | 1337.2 | **1.46x** |

Same `q_blocks = 8` in both. The only difference is thread count. So:

- **K/V re-staging costs 1.46x**
- **Softmax starvation costs a further 2.06x**, and is the dominant term

### The fix: 32-row softmax units

`fa_softmax_impl` split the query rows into **64-row** units, because the online-softmax
state is lane-packed one value per row and a 64-row group was treated as indivisible. It
is not. Every object it touches is already 32-row granular:

- `vtcm_m_vec` / `vtcm_l_vec` are **f32**, so one 128-byte vector holds 32 rows. The
  64-row unit read them as a `[i*2+0]`, `[i*2+1]` pair.
- The rescale matrix `D` is a 32x32 HMX tile, and `d_tile_scatter_offsets`
  (`hmx-fa-kernels.h:15`) has exactly 32 live entries. A 64-row unit scattered twice,
  the second time after `Q6_V_vror_VR(v_exp_m_diff, 64)` -- 64 *bytes*, i.e. 32 fp16
  lanes -- to bring rows 32..63 down.
- `fa_build_d_diag_inv_l` (`:1611`) already walks `vtcm_l_vec` one vector per 32 rows.

So the units were halved (`flash-attn-ops.c:1193` and `:1631`, which must move together)
and the paired indexing collapsed to a single vector. `n_row_vec_cnt` now equals
`n_row_tiles` exactly, the `r_vec_off < 32` branch in the innermost row loop is gone, and
one of the two `vscatter`s is gone with it.

**The output is bit-identical.** Row `r` maps to (`m`/`l` vector `r/32`, lane `r%32`) and
to D tile `r/32` under *both* schemes, and every operation between the load and the store
is lane-wise. What changes is only which thread does the work.

Only the fp16 accumulators (`rowmax_acc_v`, `rowsum_acc_v`, the slope vector) now half-fill
-- 32 of their 64 lanes -- which is exactly what a partial 64-row unit did already.

One real hazard, and it is silent. The ALiBi slopes are loaded with `hvx_vmem`, an
**aligned** load (`hvx-base.h:12`); at fp16 a 32-row stride is 64 bytes, so odd units would
be misaligned, and Hexagon's `vmem` masks the low 7 address bits rather than faulting -- it
would return the wrong 128 bytes with no NaN and no crash. `hvx_vmemu` is not the fix
either: at unit `u` it reads `[u*64, u*64+128)`, and `off_slopes` is the **last** VTCM
allocation with `total_bytes = off` (`flash-attn-ops.h:236`), so for an even unit count
that runs 64 bytes past a layout the device accepts at exactly `vtcm_size`. The kernel
instead loads the enclosing 128-byte block (`r_vec_idx & ~1`, provably in bounds for both
parities since `slopes_size = align_up(g_br*2, 128)` and `g_br` is 32-aligned) and folds
the odd-unit half into the per-row `vror`, whose amount stays under 128 bytes.

### Why not the key-axis split

The earlier version of this section recommended partitioning the softmax over the KV
column axis. That was priced against the wrong width. The softmax sees one **chunk**, not
the whole selection: at kv=2048/bs=64/n_sel=8 the search picks `Bc = 128`, i.e. **two**
64-column groups, not eight. At two groups a column split alone reaches the same 4-of-6
utilisation that 32-row units reach, and combining the two adds nothing
(`8/(6x2)` is the same 67%). It would cost per-thread partial `m`/`l`, a hand-rolled
barrier inside one `work_queue_run`, a sinks term that must be applied exactly once
against the *final* `m`, and a per-(row, column-group) rescale that is not a diagonal on
either side of `P.V` -- `hmx_fa_o_update_tile` (`hmx-fa-kernels.h:119`) takes one `d_diag`
for all column tiles. Zero marginal gain for all of that.

Two other candidates were also rejected: parallelising over kv_head (there is one HMX unit
-- `main.c:640` -- and `work_queue_run_async` publishes exactly one task, so it cannot
nest); and staging the union of the query blocks' selections and carving it with the mask
(zero kernel change, but this benchmark's selection is deliberately near-disjoint, so the
union is 2-4x more KV columns, and it forces the mask on).

### What it should be worth

Softmax phase time is proportional to (rounds) x (rows per unit). Correcting one number in
the table above first: the `G=8` control does **not** run 6 threads. `n_rows_g = 512` is 8
units of 64; `n_use = min(6, 8) = 6` but `vecs_per_t = ceil(8/6) = 2`, so threads 0-3 take
two units each and threads 4-5 hit the `vec_start >= n_row_vec_cnt` early return at
`:1204`. It runs **4 busy threads, 2 rounds**. That does not change the attribution -- it
is the per-row cost that matters:

| config | before (64-row) | after (32-row) | rows of work per row |
|:--|--:|--:|--:|
| `G=2, Br=64` (`n_rows_g=128`) | 2 units, 2 busy, 1 round = 64 | 4 units, 4 busy, 1 round = 32 | 0.50 -> **0.25** |
| `G=8, Br=64` control (`n_rows_g=512`) | 8 units, 4 busy, 2 rounds = 128 | 16 units, 6 busy, 3 rounds = 96 | 0.25 -> 0.1875 |
| shared selection (`n_rows_g=1024`) | 16 units, 6 busy, 3 rounds = 192 | 32 units, 6 busy, 6 rounds = 192 | 0.1875 -> **0.1875** |

**The phase gets exactly 2.00x at the target shape**, which is also the ceiling: 32 rows is
the D-tile atom, so `n_rows_g = 128` has only 4 atoms and 6-of-6 is unreachable. Sweeping
12000 `(n_rows_g, n_threads)` pairs with a makespan model that honours the early break,
4864 improve and **none regress**.

The prediction this makes is falsifiable: `G=2, Bq=64` moves to the same 0.25 rows-per-row
as the `G=8` control, so the **3.00x row should fall onto the control's band (~1.46x)**,
leaving K/V re-staging as the whole residual. Note the control speeds up too (0.25 ->
0.1875), so compare per-row softmax cost, not raw ratios against a moving control. The
shared-selection row has an identical makespan before and after, which makes it a clean
control for "did anything regress".

The **host cost model was deliberately left alone.** `hmx_fa_find_chunk_size` still prices
the softmax at 64 rows (`flash-attn-ops.h:383`). Updating it changes 4129 of 17160 dense
`(D, G, nb, kv)` tilings at T=6/8 MiB -- e.g. `D=64 G=2 nb=512 kv=2048` moves from
`(Br=512, Bc=512)` to `(288, 704)` -- and buys nothing here, because `br_align` caps `Br`
at `bq` *before* the cost model runs (`flash-attn-ops.h:324`), so every sparse
**per-query-block** configuration in the sweep above keeps its `(Br, Bc)` either way. The
one row that would move is the *shared-selection* baseline at `G=8`, `(288,128) ->
(264,128)` -- i.e. updating the host model would perturb the control the 1.46x is measured
against. Shipping the kernel change alone gets the full 2x with zero tiling risk.

### Not yet measured

Everything in this section is derived from the real chunk-size search and the real dispatch
arithmetic, compiled and run on the host. **No device run has been made.** Specifically
unverified: the 2.00x itself, the 3.00x -> ~1.46x prediction, and the bit-identity claim.
Bit-identity is the one to treat as a hard gate -- if any existing case shows *any* NMSE
delta, the reindexing is wrong somewhere and it is not tolerance noise. Because the
baseline carries 11 known failures (nearly all `sinks=1`), capture the failing `vars()`
strings before the change and diff the identity set; counting passes cannot distinguish
"broke two, fixed two" from "changed nothing".

## Decoupling the attention query block from the scorer block

Real XAttention re-selects every `Bl` tokens, and honouring that directly pins the kernel's query
tile to `Br <= Bq = Bl = 64`, which costs 1862 µs against 985 µs for a single shared selection.
The middle ground is to union the selections of the `R = Bq/Bl` scorer blocks an attention block
spans. `test_flash_attn_ext_sparse` gained `bl` and `n_share` for this: `n_share = s` is how many
of its `n_sel` blocks two adjacent scorer blocks hold in common, so the union size is a closed
form, `u = R*n_sel - (R-1)*s`, and the kernel's fixed-length slot list is that union enumerated
once. Nothing is padded, so no block can be attended twice -- which matters, because a repeated
index is a *silent* wrong answer here: nothing dedups it, the mask gives both copies identical
finite values, and the block's logits simply come out shifted by ln 2.

Measured, kv=2048, bs=64, n_sel=8, Bl=64. Baseline is exact per-scorer-block selection
(`Bq=64`) at **1862 µs**:

| Bq | R | s | f = s/n_sel | u | chunks | GFLOP | time µs | vs Bq=64 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 2 | 0 | 0.00 | 16 | 2 | 4.29 | 2154 | 0.86x |
| 128 | 2 | 2 | 0.25 | 14 | 2 | 3.76 | 2447 | 0.76x |
| 128 | 2 | 4 | 0.50 | 12 | 2 | 3.22 | **1651** | **1.13x** |
| 128 | 2 | 6 | 0.75 | 10 | 2 | 2.68 | 1794 | 1.04x |
| 128 | 2 | 8 | 1.00 | 8 | 1 | 2.15 | **1486** | **1.25x** |
| 256 | 4 | 0 | 0.00 | 32 | 4 | 8.59 | 2339 | 0.80x |
| 256 | 4 | 1 | 0.12 | 29 | **29** | 7.78 | **6614** | 0.28x |
| 256 | 4 | 2 | 0.25 | 26 | **13** | 6.98 | 3153 | 0.59x |
| 256 | 4 | 3 | 0.38 | 23 | **23** | 6.17 | 5283 | 0.35x |
| 256 | 4 | 4 | 0.50 | 20 | 4 | 5.37 | 1892 | 0.98x |
| 256 | 4 | 6 | 0.75 | 14 | 2 | 1780 | | **1.05x** |
| 256 | 4 | 8 | 1.00 | 8 | 1 | 2.15 | **1094** | **1.70x** |

### The union size must be chosen for smoothness, not minimality

`hmx_fa_find_chunk_size` requires `m = Bc/bs` to divide the selected-block count
(`flash-attn-ops.h:444`), with `m <= 8`. So `n_kv_blocks = u / (largest divisor of u that is
<= 8)` -- a number-theoretic sawtooth. A prime `u` runs one chunk per block:

| f | u | chunks | GFLOP | time µs |
|--:|--:|--:|--:|--:|
| 0.00 | 32 | 4 | 8.59 | 2339 |
| 0.12 | **29** | **29** | 7.78 | **6614** |
| 0.25 | 26 | 13 | 6.98 | 3153 |
| 0.38 | **23** | **23** | 6.17 | 5283 |
| 0.50 | 20 | 4 | 5.37 | 1892 |

**`u=29` does 10% LESS work than `u=32` and takes 2.83x longer.** Rounding the union *up* from 29
to 32 -- deliberately attending to *more* blocks -- is 2.83x faster. Any deployment that computes
a union must round `u` up to a value with a large divisor `<= 8`; minimising the union is
actively harmful.

Note the effect survives at equal chunk count: `u=14` (m=7, Bc=448) is slower than `u=16`
(m=8, Bc=512) despite less work, so the divisor's *value* matters too, not just the chunk count.

### Where the crossover actually sits

`Bq=128` beats exact per-scorer-block selection at **f >= 0.5**, and `Bq=256` at **f >= 0.75**.
That is a materially higher bar than the ~31% predicted earlier from a smooth
`cost ∝ u^0.4545` model -- that model is blind to chunk count and is least trustworthy exactly
where the answer is decided. The earlier 31% figure should be disregarded.

### Still unknown

The real overlap `f` between adjacent query blocks' XAttention selections. It cannot be measured
from random test tensors: random Q/K give near-uniform softmax scores and essentially random
top-k, which understates overlap badly. Structurally it should be high -- `find_blocks` forces
the sink and the diagonal block, and causal masking means adjacent query blocks draw from nearly
the same candidate pool -- but that is an argument, not a measurement, and the crossover is at
f=0.5, not at a bar that structure alone obviously clears. Measuring `f` on real model
activations is the deciding experiment.
