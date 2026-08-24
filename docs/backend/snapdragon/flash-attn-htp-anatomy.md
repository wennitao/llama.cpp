# Anatomy of one `GGML_OP_FLASH_ATTN_EXT` on the Hexagon HTP

*Target: Snapdragon 8 Elite (SM8750), Hexagon V79 HTP — 6 HVX hardware threads, 1 HMX unit, 8 MB VTCM. All paths below are relative to `/mnt/raid0_ssd/wentao/llama.cpp/ggml/src/ggml-hexagon/`. This describes only the Hexagon backend; the CPU and OpenCL flash-attention paths are unrelated code.*

---

## 1. The hardware, in one page

The HTP is not a GPU. It is four loosely-coupled engines that share one scratchpad, and the entire design of this kernel is about keeping all four busy at once.

**HVX — the vector unit.** Six hardware threads, each with 128-byte (1024-bit) vector registers: `#define VLEN (128)`, `VLEN_FP16 = 64`, `VLEN_FP32 = 32` (`htp/hvx-types.h:11-13`). One instruction processes 64 fp16 lanes. Beyond plain arithmetic it has three families this kernel leans on hard:

- *permute/shuffle*: `Q6_W_vshuff_VVR` / `Q6_W_vdeal_VVR` (interleave/de-interleave a vector pair at a chosen granularity), `Q6_V_vror_VR` (byte rotate), `Q6_V_vdelta_VV` (arbitrary fixed permute, used to broadcast lane 0).
- *predication*: `Q6_Q_vsetq_R` / `Q6_Q_vsetq2_R` build byte-prefix predicates, `Q6_V_vmux_QVV` selects.
- *scatter/gather into VTCM*: `Q6_vscatter_RMVwV` / `Q6_vscatter_QRMVhV` write 32 lanes to 32 arbitrary offsets inside a bounded region. This only works against VTCM, and it is how the kernel does transposes that HVX cannot do in-register.

There is no integer divide. Every division in an index expression is shipped from the host as a magic-number reciprocal (`struct fastdiv_values {uint32_t mp; uint32_t l;}`, `htp/hex-fastdiv.h`).

**HMX — the matrix unit.** One per DSP (`htp/main.c:640` hardcodes `*n_hmx = 1`). It is *load-triggered*: there is no multiply opcode. Issuing an activation-load and a weight-load in the same VLIW packet causes a 32×32 fp16 tile MAC into an implicit accumulator. The whole flash-attention kernel is built from three asm macros:

```c
#define HMX_LOAD_MPY_F16(act, wt, range) \
    "{\n" \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n" \
    "}\n"
...
#define HMX_STORE_AFTER_F16(out, scale_reg) \
    "mxmem(" out ", " scale_reg "):after.hf = acc\n"
#define HMX_SET_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"
```
(`htp/hmx-utils.h:201-217`). In the SDK these are `LD/SLOT01`, `LD/SLOT0` and `ST/SLOT0`, which is why the two loads fit one packet. `range` is always `2047` here = one 2048-byte tile minus one. So **one packet = 32×32×32 = 32768 fp16 MACs**.

**VTCM — 8 MB of tightly-coupled memory.** Allocated as a single page at session start via `HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size)` (`htp/main.c:267`). It is not a cache: it is explicitly addressed, and whichever op is running owns all of it. HMX operands *must* live here in practice, and it is the only memory the scatter instructions can target.

**The user-DMA engine.** A hardware descriptor-chain walker. `dma_queue_push` fills a `dma_descriptor_2d` and `dmlink`s it onto the tail of a chain the engine is already walking; `dma_queue_pop` spins on `desc->done` with `dmpoll()` (`htp/dma-queue.h:243-248`). The FA kernel uses `ctx->dma[0]`, created with `nocache=1`, so descriptors set `src_bypass`/`dst_bypass` and K/V/mask streaming does not thrash L2 (`htp/dma-queue.h:200`, `htp/main.c:533`).

**How they are driven.** HVX work goes through `work_queue` — 5 spawned QuRT threads plus the calling thread running job 0 (`htp/work-queue.c:118-130, 141-167`), joined on an atomic barrier. HMX work goes through a *separate* dedicated QuRT thread with a 16-entry FIFO (`htp/hmx-queue.c:145-147`), which lazily takes `HAP_compute_res_hmx_lock` and holds it (`htp/hmx-queue.c:17-23`). That seventh thread is the entire reason HMX and HVX can overlap: `hmx_queue_push` returns immediately, `hmx_queue_pop` retires the *oldest* descriptor.

---

## 2. Crossing to the DSP

### Nothing is dispatched per-op

`ggml_backend_hexagon_graph_compute` (`ggml-hexagon.cpp:3755`) walks the graph once, converts each node to an `htp_opnode`, and appends it to a per-session **batch**. Only when the batch fills or the graph ends does one dspqueue packet go to the DSP.

```c
void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    if (!op_batch->fit_op(node)) { flush_batch(); }
    op_batch->add_op(node);
}
```
(`ggml-hexagon.cpp:1628-1634`). `fit_op` rejects on `n_ops_max` (default 1024), `HTP_OP_MAX_BUFS` = 16 distinct rpcmem fds, or the mapped-vmem budget. `flush()` at graph end blocks until the DSP responds, so `graph_compute` is synchronous end-to-end even though the queue is not.

### What actually crosses the boundary: ~300 bytes

The packet is a 24-byte `htp_opbatch_req` message plus exactly one `dspqueue_buffer` pointing into a pinned rpcmem block that holds three packed arrays:

```c
memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);   // htp_buf_desc[]  24 B each
memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);   // htp_tensor[]    56 B each
memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);   // htp_op_desc[]  224 B each
```
(`ggml-hexagon.cpp:1481-1489`). A dense FA op costs one 224-byte `htp_op_desc` (of which 128 B is the `kernel_params` blob), up to 5 × 56-byte `htp_tensor` for Q/K/V/mask/dst, and 1–2 buffer descriptors — **300 to 550 bytes, independent of KV-cache size.**

**Tensors are never copied.** `htp_tensor.data` is an *offset* into a shared rpcmem allocation (`ggml-hexagon.cpp:1283-1290`); `htp_buf_desc` carries the dma-buf fd. On the DSP, `prep_op_bufs` (`htp/main.c:865`) reuses or lazily `HAP_mmap2`s the fd, and `prep_tensor` rewrites the offset in place:

```c
t->data  = (uint32_t) (bufs[bi].base + offset);  // update data to the actual pointer
```
(`htp/main.c:908`). Q, K, V, mask and dst live in one physical DDR allocation that both sides map. Coherency is cache maintenance, not copying: a full `QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL` at batch entry and exit (`htp/main.c:1019, 1069`).

On arrival, `process_opbatch` (`htp/main.c:976`) wakes both pools once for the whole batch — `work_queue_wakeup` flips the 5 HVX workers from `qurt_futex_wait` to spin-polling, `hmx_queue_wakeup` makes the HMX thread take its lock (`htp/main.c:1035-1038`) — then runs a strictly serial loop `proc_op_req(octx, tens, i, &ops[i])` (`htp/main.c:1043-1046`). **There is no inter-op parallelism.** VTCM is acquired once around the whole drain loop (`htp/main.c:1110`), so its acquisition cost is amortised over a graph, not paid per op.

`execute_op` is a flat switch to `op_flash_attn_ext` (`htp/main.c:745`), which branches on a decision the host already made:

```c
if (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
    return hmx_flash_attn_ext(octx);
}
```
(`htp/flash-attn-ops.c:2565-2567`).

### What the host precomputes, and why it must

`ggml_hexagon_precompute_flash_attn_params` (`ggml-hexagon.cpp:1980-2135`) fills a 124-byte `htp_fa_kernel_params` (statically asserted ≤128, `htp/flash-attn-ops.h:90`). Four categories, each with a different reason to be host-side:

1. **Arithmetic the DSP shouldn't do.** ALiBi constants need libm: `n_head_log2 = 1u << floor(log2(n_head))`, `m0 = pow(2, -max_bias/n_head_log2)`, `m1 = pow(2, -(max_bias/2)/n_head_log2)` (`ggml-hexagon.cpp:2031-2033`). Softcap folding `scale /= logit_softcap` too (`ggml-hexagon.cpp:2018-2020`).
2. **Division-free index math.** Every broadcast/GQA divisor ships as a `fastdiv_values` magic reciprocal — `div_G`, `src3_div2/3`, `broadcast_rk2/rk3/rv2/rv3`.
3. **The tiling search.** `hmx_fa_find_chunk_size` (`htp/flash-attn-ops.h:288-405`) is a double loop that calls `hmx_fa_compute_vtcm_usage` — i.e. *builds the entire VTCM layout* — for every (Br, Bc) candidate. Dozens to hundreds of layout builds. Running that inside the op would burn DSP cycles per token.
4. **The admissibility answer — the structural reason.** `ggml_backend_hexagon_supports_op` calls the same function speculatively and rejects the op if the tiling doesn't fit: `if ((size_t) kparams.vtcm_size > sess->vtcm_size) ... return false;` (`ggml-hexagon.cpp:2227-2231`). The ggml scheduler needs a yes/no *before* graph partitioning, so this must be solvable without a round trip.

The cost is near-zero at steady state because of the graph cache on `graph->uid` (`ggml-hexagon.cpp:3764`): on a hit, the whole `std::vector<htp_opnode>` including the solved `kernel_params` is reused verbatim, so the search runs once per graph *shape*, not once per token.

### Which kernel gets chosen

A cascade, gated by `opt_fa_select` (default 2 = HMX → HVX → CPU):

```c
if (DK % 64 != 0 || DV % 64 != 0) { return false; }

// Fall back to HVX for small token counts if head dimension is small (DK <= 128)
const uint32_t neq1 = q->ne[1];
if (DK <= 128 && neq1 < 5) { return false; }
```
(`ggml-hexagon.cpp:1966-1973`). Plus `sess->n_hmx == 0` → HVX, and `k->type != F16 || v->type != F16` → HVX (HMX consumes fp16 tiles directly; Q may be F32 and is converted during tiling). The `DK % 64` rule keeps the K transpose on its fast even-tile path (§6). The `neq1 < 5` rule is the honest admission that **HMX here is a prefill/batched engine**: with one token and small G, a 32-row tile is almost all padding and the fixed Q-prep/epilogue cost dominates.

If `hmx_fa_find_chunk_size` cannot fit any tiling in VTCM, the code falls through to `kernel_type = HTP_FA_KERNEL_HVX; Br = 1; Bc = 64;` (`ggml-hexagon.cpp:2104-2106`) — a completely separate implementation where the unit of work is one flattened `(token, head, seq)` query row per thread, K/V re-streamed per head, an fp32 `VKQ32[DV]` accumulator, and ~485 KB of VTCM. Block-sparse attention is HMX-only and the op is rejected rather than silently ignoring `src[5]` (`ggml-hexagon.cpp:2222-2224`).

---

## 3. The VTCM budget

`hmx_fa_vtcm_layout_build` (`htp/flash-attn-ops.h:149-244`) is the single source of truth, called by both the host estimator and the device allocator with equivalent arguments. The header says why:

> *Building the layout once and reading offsets/total from it makes host estimate and device allocation impossible to desync -- previously they were duplicated formulas in two files and drifted.* (`htp/flash-attn-ops.h:97-101`)

The device still guards `if (L.total_bytes > ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;` (`htp/flash-attn-ops.c:2047-2049`) and carves from offset 0 — the FA op owns VTCM outright.

### Worked example

**DK = DV = 128, G = 2, Br = 512, Bc = 256, n_threads = 6, pipelined, broadcast mask, fp16 Q.** ⇒ `g_br = align_up(2·512, 32) = 1024`. I re-derived these offsets from the formulas in the header; the total matches the agents' compiled dump exactly.

```
region          offset      size      role
q_tiles              0    256 KB   Q in HMX tile layout, built once per (q_start, kv_head)
o_tiles[0]      262144    256 KB   O accumulator, ping
o_tiles[1]      524288    256 KB   O accumulator, pong
d_tiles[0]      786432     64 KB   diag(exp2(m_prev-m_new)), buffer 0
d_tiles[1]      851968     64 KB   ... buffer 1 (pipelined only)
d_inv_l         917504     64 KB   diag(1/l), epilogue
--- group B and group C both start at 983040 ---
k_tiles[0]      983040     64 KB  |  q_dma  983040  256 KB  ALIASED
k_tiles[1]     1048576     64 KB  |
v_tiles[0]     1114112     64 KB  |
v_tiles[1]     1179648     64 KB  |
s_tiles[0]     1245184    512 KB  |
s_tiles[1]     1769472    512 KB  |
p_tiles[0]     2293760    512 KB  |
p_tiles[1]     2818048    512 KB  |
s_rowmax       3342336      4 KB  |   (allocated, never referenced — see caveats)
p_rowsum       3346432      4 KB  |   (ditto)
row_bufs       3350528      6 KB  |   per-thread softmax scratch, 2 rows × 6 threads
--- group A part 2 ---
k_fp16[0]      3356672     64 KB   flat DMA landing zone for K, 128B-padded rows
k_fp16[1]      3422208     64 KB
v_fp16[0]      3487744     64 KB
v_fp16[1]      3553280     64 KB
m_vec          3618816      4 KB   running max, fp32[g_br]
l_vec          3622912      4 KB   running sum, fp32[g_br]
hmx_scales_id  3627008    256 B    HMX output scale = 1.0
hmx_scales_qk  3627264    256 B    HMX output scale = scale·log2(e)
mask_buf       3627520   1024 KB   4 slots × Br=512 rows × 512 B/line
slopes         4676096      2 KB   per-GQA-row ALiBi slopes
TOTAL          4,678,144 B = 4.461 MiB
```

**Where the money goes:** S + P = 2 MB (45%), four `[1024 × 256]` fp16 matrices. The mask buffer is 1 MB (22%). Q + O tiles 768 KB (17%). K/V — tiles *and* flat staging together — only 512 KB (11%). The score matrix and the mask dominate; K and V are nearly free. Since S/P scale as `g_br × Bc`, this is exactly what the search trades when it plays Br against Bc.

### The group B / group C union

This is the one thing in the layout that reads like a bug and isn't. Two cursors start at the same offset:

```c
    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_group_b_c = hex_align_up(off, HTP_FA_HMX_TILE_SIZE);
    size_t off_group_b = off_group_b_c;
    ... k/v/s/p tiles, rowmax, rowsum, row_bufs ...
    // Group C: Q fetch DMA buffer
    size_t off_group_c = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_c, off_q_dma,         q_dma_size);
    off = off_group_b_c + hex_smax(group_b_size, group_c_size);
```
(`htp/flash-attn-ops.h:191-216`). So `off_q_dma == off_k_tiles[0]` and the region is charged only `max(B, C)`.

It is safe by *lifetime*, not luck. The ordering per (q_start, kv_head) is: push Q DMA into `q_dma` and K/V DMA into `k_fp16`/`v_fp16` → wait for Q → `fa_phase_q_load` reads `q_dma` and writes `q_tiles` → *only then* does `fa_phase_k_interleave` write `k_tiles[0]`. The K/V *DMA landing zones* live in group A part 2 and never alias q_dma — that is what makes the concurrent DMA legal. It is also why the next iteration's Q DMA is only pushed at `htp/flash-attn-ops.c:2497`, after the epilogue's O-update has retired and P/V are dead. A corollary: F32 Q doubles `q_dma_size` to 512 KB and *still* doesn't change the total, because group C stays under group B.

### The (Br, Bc) search

```c
    const size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    const size_t c_iter_base  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    const size_t c_softmax    = 600;   // per 64-row vector chunk on HVX
    ...
    const size_t c_iter_actual  = c_iter_base + c_softmax * vecs_per_t;
    const size_t cost           = q_blocks * (c_q_fixed + kv_blocks * c_iter_actual);
```
(`htp/flash-attn-ops.h:337-367`). Note what that says: **HMX plus DMA is priced at 200 per iteration and the HVX softmax at 600 per 64-row chunk.** The calibrated model treats the softmax as the serialising phase, and the whole pipeline is arranged to hide everything else behind it.

The pipeline cap is derived, not guessed: `Bc_search = align_down((kv_len-1)/(FA_MIN_KV_BLOCKS-1), 64)` — the largest 64-multiple strictly below kv/2, so `ceil(kv/Bc) >= 3`. The comment records the measurement: *"kv=512 yielded Bc=128 and 4 blocks where Bc=192 gives 3 -- and each extra KV block costs a near-constant ~190us at nb=512, independent of Bc"* (`htp/flash-attn-ops.h:311-315`). Ties break first on less ragged-tail padding, then on larger Br·Bc.

Running the real function for G=2, DK=DV=128, qo_len=512, 8 MB, 6 threads gives a telling shape:

```
kv= 256 -> Br=512 Bc= 64   1.83 MiB
kv= 512 -> Br=512 Bc=192   3.59 MiB
kv=1024 -> Br=512 Bc=384   6.21 MiB
kv=2048 -> Br=512 Bc=512   7.97 MiB
kv=4096 -> Br=192 Bc=1024  6.88 MiB
```

Br stays pinned at the full token count and Bc grows until VTCM saturates near 8 MB; then the search is *forced to shrink Br* to buy more Bc. That crossover at kv=4096 is the VTCM budget directly rewriting the algorithm's blocking.

---

## 4. The loop nest and the tiling

```c
    for (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {                       // sequence / batch
        const uint32_t im3 = mask ? fastmodulo(ib3, mask->ne[3], &factx.src3_div3) : 0;
        for (uint32_t q_start = 0; q_start < neq1; q_start += Br) {   // Q block
            const uint32_t n_rows_q    = hex_smin(Br, neq1 - q_start);
            const size_t   n_rows_g    = n_rows_q * G;
            const size_t   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
            const size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;
            for (uint32_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                ...
                for (uint32_t kv_blk = 0; kv_blk < factx.n_kv_blocks; ++kv_blk) {
```
(`htp/flash-attn-ops.c:2127-2209`).

The whole nest — every level of it — runs on **one** thread, the one that called `execute_op`. There is no `work_queue_run` over Q blocks or heads. Parallelism happens strictly *inside* phases (§5).

**`kv_head` is nested inside `q_start`.** That's unusual, and no comment explains it, but the layout does: keeping only one `q_tiles` buffer resident means Q must be re-tiled per (q_start, kv_head) pair, and in exchange the entire KV double-buffered pipeline is sized for a single KV head. Q re-tiling is `c_q_fixed`-cheap; a second Q buffer would cost another 256 KB in the worked example.

**Br, Bc, g_br.**
- `Br` = Q **tokens** per block, before GQA expansion.
- `Bc` = KV rows staged per chunk.
- `g_br = hex_align_up(gqa_factor * Br, 32)` (`htp/flash-attn-ops.h:153`) = the row height of the matrices HMX actually multiplies.

The 32 is the HMX fp16 tile row count. Every `mxmem` moves exactly one 32×32 tile, so any buffer feeding HMX must be a whole number of them. The search respects this: `br_unit = ceil(32 / gqa_factor)` and `bc_unit = 32 * 2 = 64` (`htp/flash-attn-ops.h:302-304`) — Bc steps in *two* column tiles because the softmax's inner loops consume 64 fp16 columns per HVX vector.

Two row-tile counts are in flight and must not be confused: `n_row_tiles = g_br_actual/32` (the possibly-ragged actual Q block, a loop bound) and `n_row_tiles_g_br = g_br/32` (the allocated buffer, an address stride). Both are passed into every HMX job struct.

### GQA merging: heads become rows

There is no query-head loop anywhere in the HMX kernel. Row *r* of the merged `[g_br × DK]` Q matrix is the pair `(token q_start + r/G, head kv_head*G + r%G)`, computed with fastdiv:

```c
        const size_t q_idx0 = fastdiv(r + 0, div_G);
        const size_t h_idx0 = fastmodulo(r + 0, G, div_G);
```
(`htp/hmx-fa-kernels.h:334-341`), and inverted in the O store:

```c
        float * out = (float *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);
```
(`htp/flash-attn-ops.c:1016-1018`).

One QK-dot therefore multiplies a `[g_br × DK]` Q against the *same* K tiles and produces `[g_br × Bc]` S covering all G heads. G-fold repeated K/V traffic becomes a G-fold taller activation matrix — free, as long as g_br stays a multiple of 32. The math is unchanged because `m_vec` and `l_vec` are indexed per merged row, so every (token, head) keeps its own running max and denominator; and the rescale D is diagonal in merged-row space for exactly the same reason.

The mask follows. With a broadcast (head-independent) mask, merged row r reads staged line `fastdiv(r, div_G)`, with an explicit reuse when consecutive rows hit the same query index:

```c
                            const size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            ...
                                if (qi1 == qi0) { v_mask1 = v_mask0; }
```
(`htp/flash-attn-ops.c:1290-1302`). With a per-head mask, G separate 2D DMA descriptors are pushed with `dst_stride = G * m_line_bytes`, which *interleaves* the G head-lines so merged row r lands at `base + r*m_line_bytes` and the softmax indexes it directly (`htp/flash-attn-ops.c:1824-1829`).

---

## 5. One KV chunk, step by step

This is the heart of the kernel: `htp/flash-attn-ops.c:2209-2320`. `buf` = `buf_idx`, flipped at the bottom of each iteration (`:2319`).

Before the loop is entered the prologue has already: pushed DMA for chunks 0 and 1, waited for Q, tiled Q, waited for K(0), interleaved it, and **pushed QK(0)** (`:2156-2207`). So HMX is running before the first loop body executes.

### Steady state, iteration *i*

| # | Code | Busy unit | Reads / writes |
|---|------|-----------|----------------|
| 1 | `fa_pop_n(dma, …)` then `fa_phase_v_interleave` (`:2214-2217`) | DMA wait, then **all 6 HVX threads** | `v_fp16[buf]` → `v_tiles[buf]` |
| 2 | mask pop (`:2219-2231`) | DMA wait only | mask stays in its VTCM line layout; softmax reads it in place |
| 3 | fill `ou_job[prev_buf]`, `hmx_queue_push(O-update(i-1))` (`:2233-2255`) | nothing new — queues behind QK(i) | will read `p_tiles[1-buf]`, `v_tiles[1-buf]`, `d_tiles[1-buf]`, `o_prev` |
| 4 | `fa_pop_n` then `fa_phase_k_interleave`, `hmx_queue_push(QK(i+1))` (`:2256-2275`) | DMA wait, then **HVX** (the scatter transpose) | `k_fp16[1-buf]` → `k_tiles[1-buf]` |
| 5 | `hmx_queue_pop(hmx_q)` (`:2277-2278`) | **HMX** — retires QK(i); calling thread idles | `q_tiles` × `k_tiles[buf]` → `s_tiles[buf]` |
| 6 | `fa_phase_softmax_and_build_d` (`:2280-2305`) | **all 6 HVX threads**, blocking | reads `s_tiles[buf]` + mask, writes `p_tiles[buf]`, `d_tiles[buf]`, `m_vec`, `l_vec` |
| 7 | `hmx_queue_pop`, `hex_swap_ptr(&o_tile_curr, &o_tile_prev)` (`:2309-2312`) | **HMX** — retires O-update(i-1) | |
| 8 | `fa_prefetch_block(… kv_blk + 2 …)` (`:2313-2317`) | **DMA** | into `k_fp16[buf]`/`v_fp16[buf]`, just freed |

### The overlap, drawn

```
 iteration:        i-1                        i                        i+1
              ├───────────────┤        ├───────────────┤        ├───────────────┤

 DMA engine   [ fetch K/V/mask (i+1) ][ fetch K/V/mask (i+2) ][ fetch K/V/mask (i+3) ]
                    ↓ lands in *_fp16[1-buf]      ↓ lands in *_fp16[buf]

 HVX (x6)     V-int(i-1) K-int(i)  ██ softmax(i-1) ██   V-int(i) K-int(i+1)  ██ softmax(i) ██
                                    │                                          │
                                    │  reads s_tiles[1-buf]                     │  reads s_tiles[buf]
                                    │  writes p/d_tiles[1-buf]                  │  writes p/d_tiles[buf]

 HMX (1)      ▓ QK(i) ▓  ▓ OU(i-2) ▓          ▓ QK(i+1) ▓  ▓ OU(i-1) ▓          ▓ QK(i+2) ▓ …
                └ writes s_tiles[buf]              └ writes s_tiles[1-buf]
                            └ reads p/v/d_tiles[buf-ish], o_prev → o_curr

 hmx FIFO     push QK(i) ... push OU(i-2), push QK(i+1) ... pop QK(i) ... pop OU(i-2) ...
              (pop always retires the OLDEST descriptor)
```

At the instant the HVX softmax for chunk *i* is running:

- **HVX** owns `s_tiles[buf]` (read), `p_tiles[buf]`, `d_tiles[buf]`, `m_vec`, `l_vec` (write), and the chunk-*i* mask slot.
- **HMX** is executing O-update(i−1) — reading `p_tiles[1-buf]`, `v_tiles[1-buf]`, `d_tiles[1-buf]`, `o_prev`, writing `o_curr` — and then QK(i+1), reading `q_tiles` + `k_tiles[1-buf]`, writing `s_tiles[1-buf]`.
- **DMA** is landing V(i+1) and mask(i+1) into `v_fp16[1-buf]` and the other mask slot.
- `v_tiles[buf]` sits ready with chunk *i*'s V, waiting for O-update(i) next iteration.

Every pairing is disjoint across the `buf` / `1-buf` halves. That disjointness is precisely what the second copy of k/v/s/p/d buys, and it is why those five are `VTCM_LAYOUT_ALLOC_OPTIONAL(..., pipeline)` — the non-pipelined fallback (`:2350+`) pushes one HMX job and immediately pops it, strictly serial, using only the `[0]` copies.

### The push-order rule

This is the subtlest correctness constraint in the file, and it is documented in place:

```c
                        // ---- 3. Start HMX O update for block kv_blk - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relys on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
```
(`htp/flash-attn-ops.c:2233-2238`).

Concretely: correct order leaves the FIFO as `[QK(i), OU(i-1), QK(i+1)]`; step 5 retires QK(i), step 7 retires OU(i−1). Swapped, the FIFO is `[QK(i), QK(i+1), OU(i-1)]`; step 7 would retire QK(i+1) and leave OU(i−1) in flight into iteration i+1, where step 1's V-interleave writes `v_tiles[buf_idx]` — and at i+1, `buf_idx == (i-1)&1`, exactly the buffer OU(i−1) is reading. **Silent corruption. No hang, no assert.** (Fixed in commit `990e3bfee`.)

### Pipeline gating and the epilogue

Pipelining requires `n_kv_blocks >= 3 && n_threads >= 2` — three because the loop needs chunk *i* computing, *i+1* tiled, *i+2* prefetching simultaneously. The flag is decided on the host and is an *input* to VTCM sizing, not a consequence of it (`ggml-hexagon.cpp:2075-2078`).

The loop only ever pushes O-update for chunk `kv_blk-1`, so on exit chunk n−1's P/V/D are unconsumed. The epilogue pushes OU(n−1) and — instead of blocking — runs HVX work under it:

```c
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[0]));
                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                        hmx_queue_pop(hmx_q);
```
(`htp/flash-attn-ops.c:2336-2343`). Safe because `fa_build_d_diag_inv_l` reads only `l_vec` (finalised by the last softmax) and writes only `d_inv_l`, which the in-flight job neither reads nor writes.

Then the final normalise pushes one `hmx_fa_o_norm_worker` (diag(1/l) × O) and pops immediately — nothing left to overlap. Meanwhile `:2478-2521` has already enqueued the *next* (kv_head, q_start, ib3) iteration's Q and chunk-0 K/V/mask DMAs, so those fly under the o_norm job and the HVX O-store. The store is HVX writing straight to `dst->data` in DDR with unaligned vector stores — there is no DMA on the output path.

### Where the HVX threads actually fork

Each phase forks and joins independently, with a different shard unit:

| phase | fn | shard unit | formula |
|---|---|---|---|
| Q load + tile | `fa_phase_q_load` (`:946`) | padded Q rows | `align_up(ceil(g_br/n), 2)` (`:957`) |
| K interleave | `fa_phase_k_interleave` (`:724`) | KV rows | `align_up(ceil(kv_rows/n), 2)` (`:730`) |
| V interleave | `fa_phase_v_interleave` (`:772`) | KV rows | same (`:784`) |
| softmax + D | `fa_phase_softmax_and_build_d` (`:1609`) | **64-row vector chunks** | `vecs_per_t = fastdiv(n_row_vec_cnt + n - 1, &thread_div)` (`:1167-1169`) |
| O store | `fa_phase_o_store` (`:1099`) | GQA-expanded rows | `ceil(n_rows_g/n)` (`:1112`) |

Each gates on having enough work — `if (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) { n = factx->n_threads; }` (`:726-728`) — and runs inline below the gate, skipping the fork/join. The `align_up(..., 2)` is not cosmetic: the interleave and Q-prep kernels deal two rows out of one 128-byte load, so a shard boundary must never split a pair (*"Keep start/end even so r and r+1 are always in the same thread's range"*, `:813`). In the worked example the softmax has 16 chunks of 64 rows across 6 threads = 3 each, which is exactly the `vecs_per_t` the host cost model priced.

Because these forks happen several times per KV block, the workers must not sleep between them — `work_queue_wakeup` at batch entry puts them in a `hex_pause()` spin (`htp/work-queue.c:62-63`), and only `work_queue_suspend` at batch exit drops them back to futex.

---

## 6. How HMX actually multiplies

### The 2048-byte tile

`HMX_FP16_TILE_N_ROWS = N_COLS = 32`, `N_ELMS = 1024`, `TILE_SIZE = 2048` (`htp/hmx-utils.h:13-16`, mirrored in `htp/flash-attn-ops.h:17-21` so the host estimator can use them).

The layout inside a tile is **not** row-major. Writing `T[a][b]` for a 32×32 tile, the byte offset is

```
byte(a, b) = 128*(a/2) + 4*b + 2*(a%2)
```

i.e. 16 HVX vectors of 128 bytes, vector *j* holding the row pair (2j, 2j+1) interleaved across all 32 *b* values. The agents derived this from the prep code and confirmed it against an independent artefact in the source — the hand-written diagonal scatter offsets:

```c
// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
```
(`htp/hmx-fa-kernels.h:13-15`). Check: `128·(2i/2) + 4·(2i) + 0 = 136i`, and `128·i + 4·(2i+1) + 2 = 136i + 6`. Exact for all 32 rows.

The axes mean different things per operand:

| operand | `a` | `b` |
|---|---|---|
| activation | output row | reduction depth |
| weight | reduction depth | output column |
| output (what the store writes) | output row | output column |

**The consequence the kernel exploits:** an output tile is byte-identical to a weight tile whose depth axis is the output's row axis. So HMX can feed a tile it just wrote straight back in as a *weight* with zero repacking — which is what makes the O rescale possible (below).

Above the tile, grid order also differs: Q is `[row_tile][dot_tile]`, K is `[kv_tile][dk_tile]`, V is `[dv_tile][kv_tile]`, O is `[dv_tile][row_tile]`.

### Why K is transposed and V is not

Both K and V are *weights*, so both need depth on the 128-byte-strided axis. The asymmetry is structural, not a choice:

- **K's** reduction depth is the head dim, which is the *fastest* axis of the DMA'd `[kv_row][DK]` buffer. That is a real transpose.
- **V's** reduction depth is the KV row index, which is already the *slow* axis of `[kv_row][DV]`. No transpose needed.

HVX has no cheap 32-lane in-register transpose at this granularity, so K uses the **VTCM scatter engine**:

```c
                for (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmem(p1); p1 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                    tile_base += dst_step;
                }
```
(`htp/hmx-utils.h:88-95`, offsets built at `:77-78`, purpose stated at `:27-28`). The trick that makes it a 32-lane word scatter rather than 64 halfword scatters: the two adjacent depth values sharing a 32-bit word are exactly the `(a/2, a%2)` pair, so one scattered *word* is one tile pair-slot.

Cost per KV block, countable exactly: `Bc·DK/64` aligned vector loads + `Bc·DK/64` full-width scatters, on the fast path taken when `DK/32` is even. That is `256·128/64 = 512` loads + 512 scatters in the worked example, per KV block, per KV head, per Q block. The odd-tile fallback uses unaligned loads and *masked* 16-lane scatters at half the payload (`htp/hmx-utils.h:106-141`) — the existence of that specialisation, and the host's `DK % 64 != 0 → reject` rule, both say the authors treat scatter issue count as the cost driver.

V just needs a 2-row interleave, done with plain aligned loads and stores:

```c
                HVX_Vector     v0             = *pv_in0++;
                HVX_Vector     v1             = *pv_in1++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
```
(`htp/hmx-utils.h:176-182`). Same instruction count order, no scatter engine.

**K and V each cross into VTCM twice**: DMA writes raw row-major fp16 into `k_fp16`/`v_fp16` (rows padded to 128 bytes), then HVX reads that and writes tile format into `k_tiles`/`v_tiles`. Two writes and two reads of VTCM per element, plus a scatter — that is the true cost of "HMX wants a special layout", and it is why those staging buffers are separate regions in the layout rather than aliased onto the tiles.

### The tile loops

**QK-dot** (`hmx_fa_qk_dot_worker`, `htp/flash-attn-ops.c:1651-1680`):

```c
    const size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * row_tiles = q_tiles + r * dot_stride;
        const __fp16 * col_tiles = k_tiles;
        __fp16 *       out_tile  = s_tiles + r * n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;
        for (size_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, n_dot_tiles);
            col_tiles += dot_stride;
            out_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
```

Per (row tile, col tile): `n_dot_tiles = DK/32` MAC packets, then exactly one store. The `n_dot_tiles == 2/4/8` cases are fully unrolled straight-line asm (`htp/hmx-fa-kernels.h:55-109`). Note the row stride uses `n_tiles_per_bc = Bc/32` (allocated width), not `n_col_tiles` (actual width), so a partial trailing block leaves gaps — and the HVX softmax reads with the same stride, so they agree.

**O-update** (`hmx_fa_o_update_tile`, `htp/hmx-fa-kernels.h:119-192`) is the interesting one. Per (row tile r, DV tile c) it issues:

1. **one** packet with activation = `d_diag`, weight = `o_rc` — seeding the accumulator with `diag(exp2(m_prev − m_new)) · O_prev`;
2. `n_col_tiles` packets of `p_tile × v_tile`;
3. one store.

```c
    asm volatile(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    if (n_col_tiles == 2) { ...
```

So `O_new = D·O_prev + P·V` happens in **one accumulator run and one fp16 store**. No separate rescale pass, no read-modify-write of O. And note the operand roles: D is the *activation*, O_prev is the *weight* — only legal because of the output/weight layout identity above.

**Packet counts** for the worked example (`n_row_tiles = 32`, `n_col_tiles = 8`, `DK/32 = DV/32 = 4`), per KV chunk:

```
QK        32 · 8 · 4              = 1024 packets +  256 stores
O-update  32 · 4 · (1 + 8)        = 1152 packets +  128 stores
                                    ────────────
                                    2176 packets = 71.3 M fp16 MACs
```

Of the 1152, 128 are the D·O_prev packets, and inside each only 32 of 1024 products are useful — a 32× waste, but 128/2176 = **5.9% of the chunk's HMX packets**. That is the entire price of expressing a per-row scalar multiply as a matrix op.

**O-normalise** (`hmx_fa_o_norm_worker`, `:1738`) is the same shape with only the diagonal packet, run once per (Q block, KV head). It also *transposes the tile grid* from `[dv_tile][row_tile]` (what the update writes, advancing `o_stride = n_row_tiles_g_br·1024` per column) to `[row_tile][dv_tile]` (`o_out = o_curr + r*DV_tiles*1024`, advancing one tile per column) — which is exactly the layout `fa_o_store_thread_*` walks.

### Output scales

HMX has an output-conversion stage with per-column scale and bias, loaded as a 256-byte blob by `bias = mxmem2(...)`. Two blobs are built once per op:

```c
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    hmx_init_column_scales(factx.vtcm_hmx_scales_qk, hvx_vec_splat_f16(factx.scale));
```
(`htp/flash-attn-ops.c:2098-2100`; the `0x3c00` = "scale: 1.0, bias: 0.0 in FP16" reading comes from `htp/matmul-ops.c:2687`). The QK worker selects `scales_qk`, the O-update and O-norm workers `scales_id` (`:1664, :1709, :1749`).

**Why fold the attention scale here:** it is literally free — the convert-and-store stage runs anyway and the alternative value is 1.0. Doing it on HVX instead would cost `g_br·Bc/64` = 4096 vector multiplies per KV block in the worked example, on the unit the cost model already identifies as the bottleneck. It also cannot be folded into Q without changing Q's fp16 rounding; folding at the output applies it after accumulation. And critically it carries `log2(e)`, so S arrives at the softmax already in base-2 units (§7).

### Why the rescale is a diagonal matrix

The flash-attention rescale is one scalar per row. Three reasons it is a 32×32 matmul instead:

1. **Zero extra passes over O.** As a matrix it is one more packet on an accumulator run that is already happening, with one store. The HVX alternative is a read-modify-write of the whole O buffer (`g_br·DV·2` = 256 KB in the worked example) every KV block — *in the interleaved tile layout*, where each 128-byte vector mixes two rows' factors, so the multiplier vector would itself need a shuffle per vector.
2. **Zero HVX cycles**, on the unit that is the bottleneck, at the exact moment HVX is busy with a different chunk's softmax.
3. **Building the diagonal is nearly free** and already inside the softmax's inner loop: two 32-lane halfword scatters per 64 rows per KV block.

Storage: the conceptual matrix is `[g_br × g_br]` = `(g_br/32)²` tiles but block-diagonal, and the kernel only ever loads the diagonal ones (`d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS`, stride exactly one tile). So:

```c
    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    const size_t d_tile_size = (g_br / HMX_FP16_TILE_N_ROWS) * HTP_FA_HMX_TILE_SIZE;
```
(`htp/flash-attn-ops.h:160-164`). In the worked example that is 64 KB per buffer instead of 2 MB — which is what made D pipelineable at all.

Correctness detail: the off-diagonal entries must be zero, and they are — the entire D region (`d_tiles[0]`, optional `d_tiles[1]`, `d_inv_l`, allocated back to back) is zeroed once per Q block in `fa_q_load_thread`, and the scatter thereafter only ever writes the 32 diagonal positions, so the zeros persist across every KV block (`htp/flash-attn-ops.c:876-878`).

---

## 7. The online softmax on HVX

### State

Two persistent per-row accumulators, both **fp32**, one lane per GQA-expanded row:

```c
    HVX_Vector * vtcm_m_vec;           // Row max [g_br]
    HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
```
(`htp/flash-attn-ops.c:149-150`), sized `col_vec_size = align_up(g_br * sizeof(float), 256)`. One HVX vector = 32 fp32 lanes = exactly one 32-row HMX tile, which is why the softmax indexes `vtcm_m_vec[r_vec_idx*2 + 0]` and `[+1]` for its 64-row chunk and `fa_build_d_diag_inv_l` can read `vtcm_l_vec[i]` for tile *i* directly.

Initialisation happens in the Q-load phase: `l = 0`, `m = HTP_FA_M_INITIAL_VAL = -10000.0f` (`htp/flash-attn-ops.h:27`, `htp/flash-attn-ops.c:868-873`). A **large finite negative, not −inf** — so the first block's `exp2(m_prev − m_curr)` can never be `exp2(−inf)`. And −10000.0f is exactly fp16-representable (ULP at 10000 is 8; 10000/8 = 1250), so the fp32↔fp16 round trip in the rescale path is lossless.

### Dispatch and specialisation

`fa_softmax_impl` (`htp/flash-attn-ops.c:1145`) takes five `const bool` parameters (`has_mask, mask_broadcast, is_g1, has_alibi, has_softcap`) and is instantiated by four wrappers so the untaken branches vanish at compile time: `fa_softmax_thread_nomask` (`:1547`), `_mask_broadcast_g1` (`:1556`), `_mask_broadcast_gn` (`:1565`), and the generic `fa_softmax_thread` (`:1574`). ALiBi or softcap force the generic one.

Partitioning is by 64-row chunk and is **lock-free** — every piece of state is row-indexed, so the m/l update and D build are inlined into the per-thread loop rather than run as a serial pass (*"Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)"*, `:1480`). There is no barrier or atomic in the function.

### The inner algorithm

Row pairs, because of the tile layout. One 128-byte vector load gets 32 columns of *two* rows. Three passes over a Bc-wide per-thread row buffer (`vtcm_row_bufs + i*2*row_buf_stride`):

**Pass 1 — decode.** De-interleave two rows out of the HMX S tile; `+16` vectors = +2048 bytes = the next column tile, so two loads cover 64 columns for the row pair:

```c
                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_row_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_row_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
```
(`:1250-1256`). With softcap, `hvx_vec_tanh_f16` and the cap multiply happen here (and this branch loses the 2× unroll the plain path gets).

**Pass 2 — mask and rowmax.** The mask gate:

```c
            const HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0
            ...
                    HVX_VectorPred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                    ...
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_mask0_scaled), v_neg_inf);
```
(`:1268, 1314, 1328`). Two things worth naming: `q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16))` folds the ragged-KV tail into the same mux, and `v_neg_inf = Q6_Vh_vsplat_R(0xfbff)` (`:1184`) is **−65504, the most negative finite fp16, not −inf**. That is deliberate: a fully-masked row's max is then finite, `S − m` is a finite −55504 instead of NaN, and an ALiBi slope times it cannot produce `inf·0`. exp2's −24 clamp turns it into a hard zero. The threshold exists to funnel both true −inf masks and any "effectively −inf" finite bias into that one canonical sentinel — which does mean **any finite mask below −16 is treated as a hard mask, not a soft bias**. Correct for llama.cpp's −INFINITY masks; wrong for a model that used −20 as a penalty.

Then `Q6_Vhf_vmax_VhfVhf` accumulation and a 6-step rotate-and-max `hvx_vec_reduce_max_f16` (`htp/hvx-reduce.h:135`) leaving the row max broadcast in every lane.

**Pass 3 — exponentiate and re-interleave.** P is written straight back into HMX tile format, so the format conversion costs one shuffle that was needed anyway:

```c
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                ...
                HVX_VectorPair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
```
(`:1444-1453`), with the row sum accumulated in qf32.

**Per-row scalars ride in lanes.** Two idioms do all the plumbing: *extract-and-broadcast* — `hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4))` rotates lane r into lane 0 then splats it via a fixed `Q6_V_vdelta_VV` pattern (`:1364`, `htp/hvx-repl.h:29-42`); and *insert-one-lane* — two `Q6_Q_vsetq_R` prefix predicates differenced with `Q6_Q_and_QQn` to isolate one 2-byte lane, then `Q6_V_vmux_QVV` (`:1386-1395`).

### exp2, and no exp anywhere

There is no natural exponential in the HMX path. The base change is folded upstream so the inner loop pays nothing:

```c
    if (kparams->logit_softcap == 0.0f) {
        factx.scale = (__fp16) (kparams->scale * EXP_LOG2E_F);  // log2(e)
    } else {
        factx.scale = (__fp16) kparams->scale;
    }
    factx.logit_softcap = (__fp16) (kparams->logit_softcap * EXP_LOG2E_F);
```
(`:2015-2021`, `EXP_LOG2E_F 1.44269504f` at `htp/hvx-exp.h:19`). That fp16 scale goes into `hmx_scales_qk`, so **S as it lands in VTCM is already `scale·log2(e)·(Q·Kᵀ)`**. The mask is added after the HMX scale so it needs its own conversion — `hvx_vec_mul_f16_f16(v_mask0, v_log2e)` (`:1318-1320`), the one place `EXP_LOG2E_F` appears in the hot loop. With softcap, tanh is nonlinear so the fold moves onto the cap instead, giving `log2e · softcap · tanh(qk·scale/softcap)` — same base-2 result, additively compatible with the scaled mask.

`hvx_vec_exp2_f16` (`htp/hvx-exp.h:217-253`) is split-and-Horner in fp16: clamp at −24, subtract 0.5 and truncate to int16 (the −0.5 turns round-to-nearest into floor, needed because `Q6_Vh_equals_Vhf` *"does not do proper rounding"*, `htp/hvx-base.h:164-166`), 6-term Horner on the fraction, then fold the integer part into the exponent field by **integer add**, with an underflow-to-zero mux:

```c
    HVX_Vector y_exp           = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
    y_exp                      = Q6_Vh_vadd_VhVh(k_v, y_exp);
    HVX_VectorPred q_underflow = Q6_Q_vcmp_gt_VhVh(zero_v, y_exp);
    y                          = Q6_Vh_vaslacc_VhVhR(y, k_v, 10);
    return Q6_V_vmux_QVV(q_underflow, zero_v, y);
```

~15 vector ops for 64 exponentials, no table, no divide. Decoding the fp16 coefficient literals confirms they are the Taylor series of `2^f = exp(f·ln2)`: `0x398c` = 0.6934 = ln2, `0x33b0` = (ln2)²/2, `0x2b1b` = (ln2)³/6, `0x20ed` = (ln2)⁴/24, `0x157d` = (ln2)⁵/120.

### The rescale, and where the precision is spent

```c
        HVX_Vector v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        ...
        HVX_Vector v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_Vector v_m_diff_f16   = hvx_vec_f32_to_f16(v_m_diff0, v_m_diff1);
        HVX_Vector exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);
        ...
            v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
```
(`:1485-1514`). Two deliberate choices here:

- The max is **redone in fp32** against the original fp32 `m_prev`, guaranteeing `m_diff <= 0` exactly, so `exp2(m_diff) <= 1` can never overflow. That is the structural fix for the classic "rescale factor > 1" failure.
- The **subtraction is fp32 and only the small difference is narrowed** to fp16. Differencing two fp16 maxes near, say, 30 would be catastrophic; fp16 has excellent relative precision near 0.

Then the same fp16 vector becomes a matrix, via the `136i / 136i+6` scatter offsets, one masked 32-lane scatter per row tile, with `Q6_V_vror_VR(v_exp_m_diff, 64)` supplying rows 32–63 (`:1524-1541`). The final `1/l` uses the identical trick: `hvx_vec_inverse_f32` (a magic-constant seed plus three Newton steps, `htp/hvx-inverse.h:89-107`), narrow to fp16, scatter onto `d_inv_l`.

### The precision chain, end to end

| quantity | precision | where |
|---|---|---|
| Q / K / V tiles | fp16 | K,V required F16 by host; F32 Q converted during tiling |
| S = QKᵀ | fp16, pre-scaled by `scale·log2e` | `vtcm_s_tiles` |
| masked scores, rowmax, P | fp16 | `vtcm_row_bufs`, `vtcm_p_tiles` |
| **per-block row sum** | qf32 → fp32 → **rounded to fp16** | `:1457-1467` |
| running m, running l | **fp32** | `:1518-1521` |
| rescale D | fp16 diagonal tile | `:1492, 1534` |
| **O accumulator** | **fp16** | `vtcm_o_tiles`, HMX `:after.hf` store |
| 1/l | computed fp32, stored fp16 | `:1597-1599` |

The genuine fp32 accumulation is the per-block row sum, reduced by a 5-step tree so error grows like log(Bc) — but it is **immediately re-narrowed to fp16** to be parked in lane r (`rowsum_acc_v` is an fp16 vector, because one fp16 vector holds 64 rows and an fp32 vector only 32). So the running sum is fp32 but each *increment* carries ~11 mantissa bits. That is a lane-packing artefact, not an arithmetic necessity, and it is the sharpest compromise in the softmax proper. For contrast, the HVX-only fallback keeps a genuine fp32 `VKQ32[DV]` accumulator (`:437`) and is strictly more accurate per element — just far slower.

### ALiBi and sinks, briefly

**ALiBi** slopes are computed once per (Q block, KV head) outside the loop. For `G <= 32` all G slopes come out of *one* HVX vector: `hvx_alibi_slopes` builds `h = kv_head*G + ramp32` in fp32, evaluates both `m0^(h+1)` and `m1^(2(h−n_head_log2)+1)` with `hvx_vec_pow_const_base_f32`, and selects with `Q6_Q_vcmp_gt_VsfVsf` (`htp/hvx-flash-attn.h:13-45`). They are tiled out via a *stack* buffer "to avoid scalar writes to VTCM (which generates L2 misses)" (`:1798`), indexed by GQA-expanded row. In the loop, one load per 64-row chunk and a rotate-broadcast per row (`:1264`), then `slope × (mask · log2e)` added. Note ALiBi therefore **requires a mask** — the slope is only ever multiplied into a mask lane — and the −16 gate applies to the *un-sloped* mask.

**Attention sinks** need no extra pass. The Q-load phase seeds `m` with `sinks[head] * EXP_LOG2E_F` instead of −10000 (`:840-862`), so the ordinary `exp2(m_prev − m_curr)` *is* the sink's normalised contribution. The softmax special-cases only the first block's l update:

```c
        if (args->is_first_block && factx->sinks != NULL) {
            // First KV block with sinks: m_prev holds the seeded sink value (not -inf),
            // so exp_m_diff = exp2(sink - m_curr) is the sink's contribution to the
            // denominator. l_prev is 0 here, so add exp_m_diff directly instead of
            // multiplying the (uninitialized) l_prev term.
            v_l_curr0 = HVX_OP_ADD_F32(exp_m_diff0, v_rowsum_acc_f32_0);
```
(`:1504-1510`). The numerator is untouched because a sink has no V row and block 0's D multiplies a freshly-zeroed `o_tile_prev`. Branch-free and vectorised, because seeding m makes the `max` do the branching — versus the HVX fallback, which handles sinks scalar-side at the end with an explicit `s > M` branch (`:648-666`).

---

## 8. Where block-sparsity plugs in

An optional `src[5]` (I32, `ne = [n_sel, 1, n_kv_heads or 1, n_seqs or 1]`, with block size BS in `op_params[4]`) turns the dense KV range into a selected block list. It is **HMX-only** and the op is rejected otherwise (`ggml-hexagon.cpp:2222-2224`).

Host-side, `kv_effective = sel->ne[0] * bs` replaces `nek1` everywhere the search reasons about KV length, and Bc must be a multiple of bs — `m = Bc/bs` selected blocks fold into one chunk, capped at `FA_SPARSE_MAX_M = 8` because "the extra in-flight DMA descriptors (m per tensor) stay well inside the 256-entry queue" (`htp/flash-attn-ops.h:277-279`). The search additionally requires `sel_blocks % (Bc/bc_step) == 0` so no chunk is ragged: *"A ragged final chunk leaves stale rows in the K/V/mask staging buffers past its true width, and nothing downstream re-derives that width per chunk"* (`htp/flash-attn-ops.h:348-351`).

**Device-side, sparsity exists only in the DMA address computation.** `fa_chunk_block_start` is the entire indirection:

```c
    if (__builtin_expect(factx->sel == NULL, true)) {
        return c * factx->Bc;
    }
    const int32_t * list = (const int32_t *) ((const uint8_t *) factx->sel + kv_head * factx->sel_nb2 + ib3 * factx->sel_nb3);
    uint32_t idx = (uint32_t) list[c * factx->m + j];
    if (idx >= factx->n_blk_total) {
        idx = factx->n_blk_total - 1;   // clamp the INDEX, never the row count: a zero
    }                                   // row count would break n_col_tiles > 0 downstream
    return idx * factx->sparse_bs;
```
(`htp/flash-attn-ops.c:193-211`). `fa_push_chunk` then pushes m descriptors per tensor, each writing block j into slot `j*bs*row_stride` of the same Bc-wide VTCM buffer — and does the **same for the mask**, from the identical block starts in the identical order (`:1885-1894`).

That is why nothing downstream needs to know. The invariant is stated in the source:

```c
// INVARIANT: the kernel only ever works in chunk-LOCAL column indices. A chunk's m
// blocks are not contiguous in KV, so absolute KV position reaches the softmax solely
// through which mask columns were staged alongside them. Never reintroduce a
// (kv_start + column) term -- that is exactly the assumption chunking invalidates.
```
(`htp/flash-attn-ops.c:173-176`).

Grepping `fa_softmax_impl` for `kv_start`, `kv_head`, `sel` or `sparse` returns nothing. `sargs.kv_start` and `sargs.kv_head` are populated and never read. Every column reference — score loads, mask loads, P stores — uses a chunk-local `c` running `0..kv_rows`. The VTCM layout is unchanged too: Bc-wide buffers are Bc-wide whether the rows came from one run or m scattered blocks. **m, l, the rescale, exp2, the −16 threshold, ALiBi, sink seeding, and every HMX packet are byte-identical between dense and sparse.**

One FIFO discipline rule comes with it:

```c
// Push order is grouped PER TENSOR (K*nblk, V*nblk, mask*nblk), never per sub-block.
// The queue is FIFO and K's pop site sits one loop iteration away from V's, so a run
// of nblk K pops happens at a single site. Interleaving would not assert or hang --
// it would silently read K out of the V buffer.
```
(`htp/flash-attn-ops.c:1838-1845`). Grouping (m > 1) is additionally gated off when the mask is per-head or `nek1 % bs != 0` (`ggml-hexagon.cpp:2049-2062`), and `factx.mask_use_cache = (factx.m <= 1) && factx.mask_broadcast` (`:2087`) — the 4-slot dma_cache fast path pushes a single descriptor and cannot serve a multi-block chunk.

---

## Caveats: what could not be determined, and one suspected bug

Flagged rather than smoothed over.

**Hardware behaviour not visible from source.**
- **HMX internal accumulator precision is unknown.** The kernels only show `activation.hf` / `weight.hf` loads and an `:after.hf` store. Whether `D·O_prev + Σ P·V` accumulates at fp32 or something narrower determines the true precision of both QKᵀ and P·V, and nothing in the repo or the available SDK headers says.
- **HMX issue rate is unknown**, so the 2176-packets-per-chunk figure cannot be converted to microseconds. `docs/backend/snapdragon/gemm-latency-qwen3-1.7b.md:157-159` states the backend does not reach fp16 HMX peak and the author could not determine why.
- **VTCM scatter throughput is unknown**, so the 512 scatters per K chunk likewise cannot be priced. Trace events `HTP_TRACE_EVT_HVX_FA_K_PREP` / `_V_PREP` exist (`htp/htp-ops.h:189-190`) but no captured numbers are checked in.
- **The 256-byte scale blob's bit layout is not determinable, and the two initialisers are built inconsistently.** `Q6_V_vsplat_R(0x3c00)` splats 32-bit *words* (each lane `0x00003c00`), while `hvx_vec_splat_f16(scale)` splats *halfwords* (each lane `(s<<16)|s`). Under the reading "vector 0 = 32 per-column scales in the low half of each word, vector 1 = biases", both are correct and QK's high halves are ignored. Under "low half = scale, high half = bias within vector 0", the QK blob would also set a per-column bias equal to `scale` — a latent bug. The first reading is far more likely (the FA eval tests pass) but could not be confirmed from code or SDK.
- **The store's second operand** (`scale_reg` in `HMX_STORE_AFTER_F16`) is passed `0` at every call site in the repo; its ISA meaning is unknown.
- **That the non-retain store clears the accumulator is an inference**, from `mxclracc` appearing only in the matmul kernels (never in FA) plus the existence of a distinct `:after:retain.hf` variant in the ISA. If it did not clear, consecutive output tiles would accumulate into each other and be visibly wrong.
- **The v79 HMX thread's HVX context.** `HMX_QUEUE_POLL_COUNT` is 2000 only for `__HVX_ARCH__ > 79` and 1 otherwise (`htp/hmx-queue.h:20-24`); the build targets v79 (`CMakeLists.txt:81`). So on this part the HMX thread falls straight through to `qurt_futex_wait` between descriptors — every `hmx_queue_push` pays a futex wake, and the thread never reaches the `qurt_hvx_unlock()` spin path. Whether it implicitly holds an HVX context while sleeping depends on QuRT and is not determinable here.
- **qf16/qf32 bit layout** is undefined anywhere in this tree. One consequence: the 6th exp2 coefficient `Q6_Vh_vsplat_R(0x5082)` is consumed by `Q6_Vqf16_vmpy_Vqf16Vqf16`, i.e. it is a raw qf16 literal that does not decode as IEEE fp16. Under a plausible reading it comes out ≈1.535e-4 against an exact (ln2)⁶/720 = 1.540e-4, but this is unconfirmed.

**A suspected bug in the pipelined sparse path.** The two KV loops compute a chunk's width differently:

```c
// pipelined, :2211
const uint32_t kv_rows = hex_smin(Bc, nek1 - kv_start);
// fallback, :2352
const uint32_t kv_rows = fa_chunk_rows(&factx, kv_blk, kv_head, ib3, nek1);
```

`fa_kv_block_start` returns only the *first* selected block's offset, while `fa_chunk_rows` sums all m blocks. For dense (m = 1) they agree. For m > 1, if a chunk's first selected block sits near the end of the KV range, `nek1 - kv_start` under-reports the chunk width — and the QK job's `n_col_tiles`, the V-interleave row count and the softmax `kv_rows` all shrink, while the *O-update in the same iteration* takes its `n_col_tiles` from `fa_kv_block_rows` (the summing helper, `:2249-2250`). Inconsistent widths within one iteration. Nothing visible forces m = 1 when pipelining: the host allows m up to 8 whenever the mask is broadcast and `nek1 % bs == 0`, and sets `pipeline` independently on `n_kv_blocks >= 3`. The prologue at `:2153-2156` has the same shape. Relatedly, the helper comment at `:242` still reads *"Back-compat shims: today m == 1"*, which is now stale. **This was reasoned from source, not reproduced.**

**Numerical risks that are real and unmitigated.**
- **The O accumulator is fp16 and holds the un-normalised numerator** `Σ 2^(s−m)·v` across the entire KV loop; the divide by l happens once at the very end. O grows roughly with l. There is no clamp and no periodic renormalisation, so a long KV with large |V| can overflow the numerator past 65504 before the final divide. `docs/backend/snapdragon/sparse-attn-inkernel-vs-gather.md:340-345` flags this as "a real numerical question that is unmeasured here".
- **A row masked out across every KV block leaves l = 0**, and `fa_build_d_diag_inv_l` calls `hvx_vec_inverse_f32(0)` unconditionally (`:1597`) — the Newton iteration diverges. The HVX fallback *does* guard (`const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;`, `:665`); the HMX path has no equivalent. With a standard causal mask every query row attends to at least itself, so this may be unreachable in practice; not tested either way. The same applies to the padding rows between `n_rows_g` and `g_br`, whose l stays 0 — those produce inf/NaN in the O tile but are never stored, and D being diagonal means the NaN cannot bleed into a valid row.
- **Sinks + fp16 consistency.** P is exponentiated against the fp16 `m_new` while l and O are rescaled by `exp2` of the fp32 `m_prev − m_curr`. These agree exactly whenever `m_prev` is fp16-representable, which holds by induction from the fp16-exact −10000 seed. The exception is the first block with sinks, where `m_prev = sink·log2e` in fp32 and may not be fp16-exact; the block's contribution can be misweighted by up to `2^(0.5 ULP_fp16(m))`. Unmeasured.
- **Softcap accuracy.** `hvx_vec_tanh_f16` is `2·sigmoid(2x) − 1` over an fp16 exp2 and a polynomial reciprocal whose own header quotes *"Peak Error: 1.1295e-04"* (`htp/hvx-inverse.h:18`), on top of fp16 rounding at each of ~6 steps. Softcapped models get materially noisier logits on this backend.

**Dead or stale code, noted so a reader doesn't chase it.**
- `vtcm_s_rowmax` and `vtcm_p_rowsum` are allocated (8 KB in the worked example) and pointer-assigned but never read or written — the current code keeps rowmax/rowsum in vector registers.
- `const uint32_t im3 = ...` at `:1163` inside `fa_softmax_impl` is computed and never used (the file suppresses `-Wunused-variable`); the mask has already been resolved to a VTCM pointer by the caller.
- The comment at `:934` — *"Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm"* — is stale: per the layout, `q_dma` aliases group B, not `o_tiles[0]`. The zeroing is still required because `o_tiles[0]` is the initial O accumulator, so the code is correct either way.
- The `hmx_fa_vtcm_layout` struct comment says `m_buf_slot_bytes` is `align_up(Br * m_line_bytes, 4096)` while the code uses 256 (`htp/flash-attn-ops.h:132` vs `:172`). Harmless at the shapes above, where the value is already 4096-aligned.
- FA never uses `HMX_LOAD_MPY_DEEP_F16`, even though `q_tiles + r*dot_stride` and `k_tiles + c*dot_stride` are contiguous runs exactly as the matmul kernels require for it (`htp/hmx-mm-kernels-tiled.h:613-627`). Whether that is a deliberate constraint (`:deep` may require an explicit `mxclracc`, which FA does not issue, and may interact with the D·O_prev seed packet) or simply an unexploited optimisation is not determinable from source or the commits checked.
- `worker-pool.h` no longer exists; `work-queue.h:34-36` provides `worker_pool_run_func` as a `#define` alias for `work_queue_run`, so the two names in the tree are the same function.