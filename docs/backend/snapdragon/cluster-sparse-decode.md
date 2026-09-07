# Cluster-selected sparse decode attention on Snapdragon (HTP + Adreno sidecar)

Research prototype, branch `hexagon-sparse-fa`. Companion to `heterogeneous-npu-gpu.md`, whose
Section 4i established the starting point: decode on the SM8750 HTP is one weight stream, the only
DRAM slack is the attention phase at depth, the HVX decode kernel runs at 31 GB/s bound by its DMA
pattern, and the heterogeneous split takes part of that slack (1.12x/1.23x/1.28x at 4k/8k/16k).
Sparse decode attention takes the same slack first and by more. The stack had no sparse decode:
llama builds the block selection only for prefill ubatches and the HVX decode kernel had no
selection path.

## Design (decided 2026-09-06)

- **Shadow copy owned by the hexagon backend.** The positional KV cache is untouched; the backend
  keeps a cluster-ordered copy of K/V per layer in rpcmem (`htp-ops.h`, `struct
  htp_fa_cluster_header`), attached to decode `FLASH_ATTN_EXT` nodes as `src[7]` with static tags in
  `op_params[11..15]` (magic, layer, density, window, flags). Everything that changes per token is
  data in the buffer, because the host caches graph nodes and re-runs neither precompute nor
  prepare on a reused graph.
- **Selection unit = 64-key page in cluster order.** Keys are k-means-clustered per prefill chunk
  (C = N/64 centroids), sorted by (cluster, distance) and packed densely into pages of 64; each
  page's descriptor is the f16 mean of its keys. One page = one DMA descriptor, zero tail waste,
  identical page counts across heads.
- **Coverage prefix + dense tail.** Per layer the shadow publishes `covered_end`; the kernel picks
  `dense_start` = the last chunk boundary that leaves at least W positions, attends the listed
  pages of the chunks before it (no mask) and the positional range `[dense_start, n_kv)` with the
  normal mask. No key is attended twice; `covered_end == 0` is the dense kernel bit-for-bit.
- **Chunk-local clustering on a GPU sidecar**, one prefill ubatch behind the HTP (Stage 3);
  fixed page budget per KV head plus the recent window at decode (Stage 4); measurement-first
  staging.

## Stage 1 -- page-list decode attention on the HTP (built, measured 2026-09-06)

What changed: the HVX split-KV decode thread now walks a per-unit *list* of blocks through one
block-source function (`hvx_fa_dec_blk_src`) instead of a contiguous block range; the staging ring
has a variable depth (`dec_nslots`) with per-slot bookkeeping so the consumer pops exactly what
was pushed; units are handed out dynamically (atomic counter, longest heads first) when lists
differ per head; shadow pages are fetched as contiguous rows (2D at row stride, or one 1D
descriptor under a flag). The op entry reads the header, directory and chunk table with explicit
invalidates, derives `dense_start`, copies the host-written lists into VTCM, and echoes the lists
it used when asked. The dense path is the same code with `cl_on == false`.

Tool: `llama-hetero-decode-attn --cluster` builds a one-layer shadow from a permutation of a
random positional K/V, writes per-head lists (`--density`, `--sel contig|scatter`, `--skew`), tags
the node the way the backend does, runs dense and page-list arms on HTP0 and checks each against a
CPU reference over exactly the keys the kernel was told to attend (`--pmu` captures the op's AXI
read requests).

### Results (unit eb49fb9d, Qwen3-1.7B shapes: 8 KV heads, D 128, 16 query heads, W 256, median of 7)

| kv | dense positional | pages, 100% listed | 50% | 25% | 12% | 6% |
|--:|--:|--:|--:|--:|--:|--:|
| 4096 (60 candidate pages/head)   | 570 us, 29.4 GB/s | 491 us, 34.2 GB/s | 284 us, 31.4 | 178 us, 28.0 | 126 us, 22.9 | 121 us, 17.3 |
| 16384 (252 candidate pages/head) | 2116 us, 31.8 GB/s | 1822 us, 36.8 GB/s | 952 us, 35.8 | 513 us, 34.2 | 283 us, 31.5 | 176 us, 28.3 |

Every arm matched the CPU reference to 1e-5..3e-5, and the echoed lists matched the host lists.
The AXI counter reads 250-255 B per request for pages (vs 191 B for weight rows), and the counted
bytes equal the listed bytes.

What the sweep says:

- **The gather is free.** Scattered pages cost exactly what the same number of contiguous pages
  cost at every density (4k/25%: 178 vs 177 us; 16k/25%: 513 vs 514 us). A random permutation of
  keys inside pages costs nothing either.
- **Per-head imbalance is free** with dynamic dispatch: lists skewed 1.75x/0.25x around the mean
  run in the same time as uniform lists (179 vs 178 us).
- **Ring depth and descriptor shape are not the lever.** 2, 4 and 8 staging slots, and 1D vs 2D
  page descriptors, are identical to the microsecond at both depths. The DMA engine's per-thread
  throughput (~5.7 GB/s per thread x 6 threads) is the ceiling for this access pattern; the earlier
  hypothesis that queue depth would lift the op toward the GEMV path's 55 GB/s is refuted for the
  attention pattern.
- **The cluster-ordered layout is itself worth 16-20%.** With every page listed (same bytes as
  dense) the op runs at 34-37 GB/s vs 29-32 GB/s for the positional layout, because a page's rows
  are contiguous while the positional cache interleaves 8 heads per position (2 KB stride).
- **Fixed cost in cluster mode is ~50 us per op** (vs 26 us dense: header/directory/chunk reads
  with invalidates, list copies), visible at 6% density where 8 blocks per head take 121 us instead
  of the ~93 the dense model predicts. Above ~12% density time tracks bytes.

Regression: the dense path is the same code with `cl_on == false`; `test-backend-ops -o FLASH_ATTN_EXT
-b HTP0` stays at 2175/2196 (the 21 known sinks=1 flappers), and with `GGML_HEXAGON_CLUSTER_ATTN=0,256`
set on the model (nodes tagged, shadow empty) llama-bench at d4096 reads 20.0-20.2 t/s against
20.1-20.2 untagged and greedy `llama-completion` text is identical.

Attention-op speedup at 25% density: **3.2x at 4k, 4.1x at 16k**, at 28-34 GB/s effective. Against
the token model of Section 4i (attention 34% / 62% of the token at 4k / 16k) that is ~1.3x / ~1.8x
on the token before the selection cost of Stage 4 (~35 us per layer estimated).

## Stage 2 -- GPU chunk-local k-means (next)

## Stage 3 -- clustering sidecar (later)

## Stage 4 -- on-device selection (later)

## Stage 5 -- evaluation (later)
