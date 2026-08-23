# mllm's OpenCL block-sparse kernels vs llama.cpp's `get_rows`

Follow-up to [sparse-attn-inkernel-vs-gather.md](sparse-attn-inkernel-vs-gather.md),
Result 5, which measured llama.cpp's OpenCL gather as the slowest of CPU/GPU/NPU.

## The measurement in Result 5 is not a like-for-like comparison

**mllm's OpenCL attention path has no materialized gather at all.** Not "a fast
one" -- none. There is no kernel under `mllm/backends/opencl/` that takes a block
index buffer and writes a K/V-shaped output, on any of its six branches.

`bs_qk_gemm` (`mllm/backends/opencl/kernels/block_sparse_two_pass.cl:96-111`) folds
the index into the address of a texture fetch:

```c
const int blk_id  = block_idx[bhqb * top_k + blk];
const int kg0     = (blk_id < 0) ? 0 : (blk_id * BK + key0);
const int kt_base = bh * FA_D * M_4kv + (kg0 >> 2);
...
B0.s0123 = read_imageh(Kt_img, t);      // indexed read, nothing copied
```

One index load feeds ~256 data fetches. Three techniques, and they are separable:

| Component | Scope |
|---|---|
| index-in-place | universal -- every block-sparse kernel in the repo |
| `image1d_buffer_t` textures | only the GEMM family; `block_sparse_attention.cl:206-216` is index-in-place with plain `__global` + LDS |
| block-contiguity | the index selects a *block* of `BK` keys, so each run is sequential |

The image is a **zero-copy view of the same `cl::Buffer`**, not a second copy.

**mllm's one real gather (`gg_k`/`gg_v`) lives in the QNN AOT runner**
(`examples/qwen3_qnn_aot/aot_run.cpp:114-131`), not the OpenCL backend, and exists
only because a compiled QNN graph cannot do data-dependent indexing. Its technique
is worth noting anyway: one work-item per 16 bytes, single `vload16`/`vstore16`, no
per-lane loop, flat NDRange, branchless padding.

## Why llama.cpp's `kernel_get_rows_f16` is slow

`ggml/src/ggml-opencl/kernels/get_rows.cl:130-147`. Four mechanical causes:

1. **Fully scalar** -- one 2-byte load and one 4-byte store per work-item. Host
   sizing makes the local size 128 at `ne00=128`, so the loop body runs **exactly
   once per lane**. Adreno wants 128-bit accesses; this uses 1/8 of the width.
2. **One work-group per row** -- a 128-lane group moves 256 B of useful reads.
3. **Index amortization of 1:1** -- all 128 lanes reload the same `r`, each
   resolving a dependent load before its single data load. mllm's ratio is ~256:1.
4. **F32 output is mandatory** (`ggml.c`: `// TODO: implement non F32 return`), so
   6 bytes move per element instead of 4.

Verified: the F16 path is the **only unvectorized gather in the backend** --
`kernel_get_rows_q4_0` already uses `float16` stores.

## A live correctness bug, unrelated to performance

**Verified:** the `GGML_OP_FLASH_ATTN_EXT` case in ggml-opencl's `supports_op`
never inspects `src[5]` (`grep 'src\[5\]' ggml-opencl.cpp` returns only the two SSM
uses). Hexagon guards this explicitly so the op falls back rather than silently
ignoring the selection. **Today a block-sparse FA node scheduled to OpenCL would
compute dense results with no error.** Fix is a few lines; land it before anything
else here.

## What is portable, ranked

**(a) Vectorize `kernel_get_rows_f16`** -- hours, blocked by nothing,
`moe_reorder_b.cl` is the in-repo template (`ne00=128` divides by 8). Ceiling is
capped: it cannot escape the 2x write amplification, and it optimizes a step mllm
does not have. Clean attribution: change the NDRange shape first, re-measure, *then*
vectorize -- that separates issue-rate from work-distribution.

**(b) Image/texture path** -- already in tree. `ggml_cl_img_pool_get_or_create`
wraps an existing `cl_mem` in a `CL_MEM_OBJECT_IMAGE1D_BUFFER`, zero-copy and
pooled; 56 kernel files already use `read_imageh`. **The buffer-type objection is
dead.** But a texture helps where an operand is *re-read*; a one-shot row gather has
no reuse, so treat this as an enabler for (c), not a win on its own.
*Free experiment:* the existing FA image-K path is **off by default**, gated on
`GGML_OPENCL_FA_K_IMG` / `GGML_OPENCL_FA_PREFILL_K_IMG` -- a ready-made A/B for
whether Adreno textures help dense FA on this device.

**(c) Block-sparse `FLASH_ATTN_EXT` on OpenCL** -- the actual technique, and smaller
than it looks. `flash_attn_f32_f16.cl:2299-2320` already has the structure of
`bs_qk_gemm`; **the only difference is that `k_idx` comes from a loop induction
variable rather than an index buffer**. The hexagon `src[5]` contract transfers
unchanged (I32 contiguous block indices, `op_params[4]` = block size); no ggml core
change needed. Effort is days-to-weeks, mostly V-side layout and padding/mask
correctness. Both ingredients already exist separately in llama.cpp -- index-in-place
in the MoE GEMVs, textures in the GEMM family; what has never been done is combining
a data-dependent index with an image fetch.

## Does this revive the heterogeneous pipeline?

**No.** The ceiling measured in Result 5 was 1.02-1.11x and is bounded by
**attention** time; a gather that goes to zero still leaves you there. (a) and (b)
cannot move it.

mllm's own data agrees from the other side --
`docs/qnn_backend/dense_vs_blocksparse_prefill.md:484-487`: *"The GPU gather **is**
2.8x faster in isolation ... but that win never reaches E2E: the gather isn't the
exposed bottleneck."* They built it, measured it faster, and it did not matter.

**Block-sparse attention *on* the GPU is a different proposition** -- it attacks the
term that actually bounds you. mllm measured, on Adreno 830 against their own dense
two-pass FA: top_k=4 -> 3.4x / 5.9x / ~8.5x at Sq 2048 / 4096 / 8192; top_k=16
(~25% density) -> 1.2x / 2.0x / ~3.6x.

Three caveats: those ratios are against **mllm's** dense FA, not llama.cpp's
heavily-tuned one, and do not transfer. mllm recorded a *negative* result for
register-tiling the QK (*"latency-hiding-bound, NOT arithmetic-intensity-bound"*) --
tune resident waves, not FLOP density. And `block_sparse_attention.cl` on both
branches is **superseded**; the live lineage is
`flash_attention.cl` -> `block_sparse_two_pass.cl`.
