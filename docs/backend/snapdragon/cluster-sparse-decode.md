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

## Stage 2 -- GPU chunk-local k-means (built, measured 2026-09-06)

`examples/cluster-kmeans-bench` (`llama-cluster-kmeans-bench`): OpenCL Lloyd iteration in two kernels
(`km_assign`: one work-item per key, centroids tiled through local memory, `|mu|^2 - 2 k.mu`,
change counter for early stop; `km_update`: one work-group per (centroid, head), lane = dim, empty
clusters re-seeded from the farthest key), CPU k-means++ init on a 256-key subsample, CPU counting
sort by (cluster, distance), then `km_gather` (one work-item per 256 B row into the page) and
`km_page_mean` (f16 page descriptors). Keys in the real cache layout ([pos][head][D]).

| problem per layer (8 KV heads) | assign | update | CPU sort | gather | page mean | total | per 28-layer chunk |
|---|--:|--:|--:|--:|--:|--:|--:|
| chunk-local N=1024, C=16, blobs, k-means++ (5 iters to converge) | 2.2 ms | 1.4 ms | 0.3 ms | 0.6 ms | 0.03 ms | **4.5 ms** | **126 ms** |
| chunk-local, stride init | 2.1 | 1.5 | 0.1 | 0.6 | 0.03 | 4.5 | 125 |
| chunk-local, uniform random keys (8 iters, no convergence) | 3.4 | 2.5 | 0.5 | 0.6 | 0.03 | 7.1 | 199 |
| whole layer N=4096, C=64 | 21.6 | 22.5 | 2.5 | 3.3 | 0.14 | 50 | 1402 |
| whole layer N=16384, C=256 | 402 | 2350 | 1.8 | 15.9 | 0.5 | 2771 | 77 581 |

The hide budget is the HTP's prefill time per ub=1024 ubatch, ~20 ms per layer, ~560 ms per chunk
of 28 layers: chunk-local clustering fits 4.4x over (2.8x on the worst-case data), so the sidecar
running one ubatch behind is comfortably hidden. Whole-layer clustering does not fit at any depth
that matters (the update kernel scans every key per centroid group: 2.3 s per layer at 16k), which
is the measured justification for chunk-local clusters. Assignment quality: the GPU inertia equals a
double-precision CPU Lloyd run from the same init (ratio 1.000 at N=1024 and 4096; 0.92 at 16k, the
GPU's re-seeding finding a better optimum). The gather check confirms every page row is the permuted
key. The assignment kernel runs at 77 GFLOP/s at N=1024 (launch-bound) and 170 GFLOP/s at 16k.

Interference, both directions (unit eb49fb9d): HTP prefill `-p 4096 -ub 1024` alone 1798 t/s, with
the chunk-local bench looping on the GPU 1810 t/s (unchanged; the GPU-activity effect of
heterogeneous-npu-gpu.md 4i if anything); the bench alone 4.5 ms per layer, while the HTP prefills
5.6 ms (+24%, the gather and assignment kernels pay for the shared DRAM), i.e. 157 ms per chunk,
still 3.6x inside the budget. The sidecar can run under prefill without touching it.



## Stage 3 -- clustering sidecar in the backend (built, measured 2026-09-06)

A host thread with its own OpenCL queue (`ggml_hexagon_cluster_sidecar`, pattern of the hetero
relay). The backend tracks the `SET_ROWS` nodes that write `cache_k_l%d` / `cache_v_l%d`, reads the
row-index leaf, and after each batch completes posts a job {positions written, n_tokens}. Per layer
the sidecar keeps `written_end` / `covered_end`; whenever a 1024-position chunk is pending (and, at
the end of a prompt, a tail of at least two pages) it copies the chunk's K rows to the GPU, runs the
Stage 2 k-means (k-means++ init on a 256-key subsample, Lloyd with early stop, 8 iterations max),
reads the assignment back, sorts keys by (cluster, distance) on the CPU, gathers K and V rows into
the shadow pages in that order, computes the f16 page descriptors, and publishes: chunk entry and
page count first, `covered_end` last, each followed by a cache clean. Pages already published are
never rewritten; positions written from 0 again reset the layer; out-of-range or non-contiguous
indices skip the batch or mark the layer stale (dense). `GGML_HEXAGON_CLUSTER_VERIFY=1` re-checks
every published chunk against the cache (pos_map a permutation, page rows bit-exact, descriptors
within 1e-2); `GGML_HEXAGON_CLUSTER_POSITIONAL=1` is the Quest-style baseline (pages of 64
consecutive positions, no k-means).

Correctness on the model (Qwen3-1.7B, 4.6k-token prompt): every chunk verified; greedy text
identical to dense at density 1000 and at 250; decode-mode perplexity at density 1000 equals
dense (9.1330 vs 9.1332, ctx 4096).

### Three things that cost a debugging round each

1. **The SET_ROWS index leaf must be read before compute, not after.** It is a transient graph
   input whose buffer the scheduler recycles for activations during the graph, so reading it after
   completion returns float garbage; every layer went stale and the kernel silently ran dense
   (no token speedup, no error). The positions are now captured at the top of `graph_compute`,
   after llama's `set_inputs` and before the ops run; the job is posted after completion, when the
   DSP has written the rows.
2. **The shadow doubles the KV footprint, and the cDSP has one ~3.2 GB virtual space.** Weights,
   cache and shadow are all mapped into it (`vmem 3355443200` at init, 3200 MB reported by the
   session). Qwen3-1.7B at 16k is 1.9 GB of cache + 1.3 GB of weights: a full shadow cannot be
   mapped, the fastrpc mapping is delayed, and the DSP faults on first touch -- with the fault
   surfacing even when the cluster kernel path was bypassed, which is what identified it. At 8k the
   total sits exactly at the budget (3.24 GB), which was the intermittent segfault. Init now sizes
   the shadow to what is left after the cache and a weights margin (`GGML_HEXAGON_CLUSTER_MEM_MB`
   overrides) and clusters only that many leading positions:

   | context | KV cache | shadow | outcome |
   |--:|--:|--:|---|
   | 4k  | 0.50 GB | 484 MB, all 4352 positions | full |
   | 8k  | 0.97 GB | 733 MB, first 6592 of 8448 positions | capped |
   | 12k | 1.43 GB | 178 MB, first 1600 positions | mostly dense |
   | 16k | 1.91 GB | none fits | dense, with the reason logged |

   This is plan risk R7 realised and it is a property of the shadow-copy design on this device,
   not of the kernel: the production form has to drop the positional copy for clustered ranges
   (in-place permutation) or shrink the pages (Q8 descriptors and pages halve it).
3. **llama frees the KV cache before the backend session is released**, while the sidecar may
   still be clustering queued chunks from it: a segfault at exit after correct results. The hexagon
   buffer free now quiesces the sidecar (drops queued jobs, waits for a chunk in flight, forgets the
   roots) and stop drops the queue.

Sidecar cost on the model: ~10 ms of GPU k-means and ~35 ms of CPU work per layer per chunk (the
CPU does the k-means++ init, the sort, the gather into the shadow and the page means), i.e. ~1.3 s
per 28-layer chunk against a ~0.56 s prefill ubatch, so at 4k the sidecar finishes ~2 s after the
prompt and the first decode tokens see a partly covered shadow (correct, just less sparse). The
Stage 2 bench showed the gather and page means take 0.6 ms on the GPU; moving them there is the
obvious next step and would bring the sidecar under the prefill time.


## Stage 4 -- on-device selection inside the FA op (built, measured 2026-09-06)

`flash_attn_ext_f16_select_thread` runs before the decode pass when the density tag is non-zero:
one unit per KV head stages the head's page descriptors (chunks of <= 128 rows per descriptor) and
its G query rows in VTCM, scores 32 descriptors per `hvx_dot_f16_f16_aa_rx32_tree` call per query
row, folds the group by max (flag: sum), pads the last group with -inf, and takes the top-B pages by
repeated masked arg-max (B = ceil(n_cand x density / 1000), clamped by the min-pages flag). The list
is in score order; the Stage 1 list path runs unchanged behind it. Tool: `--select-device`
(density becomes the permille budget; the CPU recomputes the f32 top-B and the reference attends the
echoed device list).

| kv | budget | device list vs host f32 top-B | FA op (device selection) | FA op (host lists, Stage 1) | dense |
|--:|--:|---|--:|--:|--:|
| 4096  | 15 of 60   | identical, all 8 heads | 186 us | 178 us | 564 us |
| 16384 | 63 of 252  | identical, all 8 heads | 551 us | 513 us | 2093 us |
| 4096  | 60 of 60   | identical (full set)   | 507 us | 491 us | 568 us |

Selection costs ~8 us per op at 4k (60 candidates) and ~38 us at 16k (252 candidates: 64 KB of
descriptors per head plus 63 arg-max passes), in line with the estimate. The outputs match the CPU
reference over the echoed lists to 2e-5; with a full budget the result equals the dense set.


## Stage 5 -- evaluation (later)
