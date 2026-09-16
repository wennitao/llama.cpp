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


## Stage 5 -- evaluation on the model (2026-09-06, unit 55b03820, Qwen3-1.7B Q4_0, `-dev HTP0`)

### Speed (`llama-bench -fa 1 -p 0 -n 64 -r 2`, arms alternated, warm-up discarded)

| depth | dense | cluster, 25% budget | cluster, 12.5% budget | shadow |
|--:|--:|--:|--:|---|
| d4096  | 20.1-20.4 t/s | 25.3-25.5 (**1.25x**) | 25.9-26.1 (**1.28x**) | full (4352 positions, 484 MB) |
| d8192  | 15.4-15.5 t/s | 21.2-21.3 (**1.38x**) | 21.8-21.9 (**1.41x**) | capped to the first 6592 positions (733 MB) |
| d16384 | 10.5 t/s      | 10.5-10.6 (1.00x)     | --                    | none fits the cDSP budget; dense with the reason logged |

The gains are the attention phase shrinking as Stage 1 predicted (op 3.2x faster at 25%), diluted
by the token's GEMVs and by the part of the context the shadow does not cover (at 8k a quarter of
the cache stays in the dense tail). The first repetition of each cluster arm runs while the sidecar
is still catching up, hence the wider error bars; steady state is the upper number. Every run
exits cleanly at every depth.

### Prefill: does the sidecar cost the HTP anything? (2026-09-07, `llama-bench -p N -n 0 -r 2`, arms alternated)

| prompt | sidecar off | sidecar on (clustering during the prompt) | HTP-only + GPU keep-alive (control) |
|--:|--:|--:|--:|
| pp4096 | 1767 t/s | 1780-1792 t/s | 1829 t/s |
| pp8192 | 1341-1347 t/s | 1337-1363 t/s | 1342 t/s |

No overhead within noise (+-1-2%; the small positive drift is the GPU-activity fabric effect of
heterogeneous-npu-gpu.md 4i, at most +3.5% on a compute-bound prefill). Three reasons: prefill
attention nodes are never tagged, so the DSP's prefill graphs are unchanged; the GPU's k-means
traffic does not touch a compute-bound HTP (Stage 2 interference test); and the sidecar's CPU work
runs on another core. What the sidecar does cost is readiness: at the end of a 4k prompt the last
layer publishes ~4.1 s after its batch (35 ms of CPU gather per layer per chunk x 28 layers x 4
chunks), so the first ~4 s of decode (~80-100 tokens) run with a partly covered shadow -- correct,
just less sparse, which is the wider error bar on the first repetition of every cluster arm above.
Time-to-first-token is unaffected (the sidecar is asynchronous). Moving the gather and page means
to the GPU (0.6 ms per layer in the Stage 2 bench) would bring the whole chunk under the prefill
time and remove the lag.

### Quality (`llama-perplexity -c 4096 --chunks 2 -b 1 -ub 1`: decode mode, since only decode-shaped attention nodes are tagged)

| arm | PPL |
|---|--:|
| dense | 9.133 +/- 0.46 |
| cluster pages, 100% budget | 9.123 +/- 0.46 (equal: the whole shadow path is exact) |
| cluster pages, 50% | 15.82 +/- 1.00 |
| cluster pages, 25%, W 256 | 31.51 +/- 2.48 |
| cluster pages, 25%, W 1024 | 26.93 +/- 2.00 |
| cluster pages, 12.5% | 73.55 +/- 6.91 |
| positional pages (Quest-style baseline), 50% | 15.09 +/- 0.94 |
| positional pages, 25% | 25.84 +/- 1.91 |

**This looked like a negative result on the selection policy; it was the missing sink page**
(see "Sink pages" below, 2026-09-07: with page 0 always selected every arm here returns to dense
perplexity). The analysis that follows is kept as written on 2026-09-06. The plumbing is
exact (100% equals dense), and the kernel and sidecar deliver the speed, but choosing pages by the
dot product of the query with the page's *mean key*, under a fixed per-head budget, loses far too
much: 1.7x the perplexity at 50%, 3.5x at 25%. And the k-means pages are no better than plain
positional pages at equal density (15.8 vs 15.1 at 50%, 31.5 vs 25.8 at 25%): the cluster order
buys bandwidth (Stage 1: +16-20% from contiguous pages) but not selection quality with this
descriptor. Two mechanisms are the likely cause, both testable with the machinery as it stands:

- **No sink pages.** Qwen3, like most decoders, puts a large share of every row's softmax mass on
  the first tokens; a page whose mean key scores low for a given query is dropped even when its
  keys carry that mass. Positional pages keep the sinks in page 0, which is probably why they edge
  ahead; cluster packing scatters position 0 into whichever cluster its key falls in. The fix is a
  rule, not a kernel: always select the page(s) holding the first positions (a per-page
  "contains position < 64" bit in the chunk table, or simply page 0 of chunk 0 in positional mode).
- **The mean is the wrong descriptor.** Quest-style selection scores a page by an *upper bound*
  of q.k over the page (per-dimension min/max), which never under-estimates a page that contains a
  high-scoring key; the mean under-estimates exactly the pages whose keys are spread, and k-means
  makes pages tighter but not tight enough at 64 keys per page. Min/max descriptors cost two rows
  per page instead of one (the select pass scores 32 descriptors per call either way).

A score-threshold budget (the prefill kernel's union-threshold policy) instead of a fixed page count
is the third lever, since the fixed budget spends the same pages on every head regardless of how
peaked its attention is. None of these changes the shadow, the kernel's list path or the sidecar;
they change what the select pass scores and what it must always include.

### Page size: 64 vs 32 vs 16 keys (2026-09-07, `--page-keys`, host lists, scattered pages, equal bytes)

The page size is now a shadow parameter (`GGML_HEXAGON_CLUSTER_PAGE=16|32|64`, header field
`page_keys`); the sidecar clusters into N/page_keys centroids per chunk and the dense tail keeps
its 64-key blocks. At the same number of bytes fetched:

| kv | bytes | page 64 | page 32 | page 16 |
|--:|--:|--:|--:|--:|
| 4096  | 100% (16 MB)    | 492 us, 34.1 GB/s | 597 us, 28.1 | 962 us, 17.4 GB/s |
| 4096  | 50% (8.5 MB)    | 283 us | 339 us | 527 us |
| 4096  | 25% (4.75 MB)   | 177 us | 210 us | 306 us |
| 16384 | 100% (64 MB)    | 1823 us, 36.8 GB/s | 2221 us, 30.2 | 3739 us, 17.9 GB/s |
| 16384 | 25% (16.75 MB)  | 513 us | 623 us | 1005 us |

Per page the cost is ~2.3 us fixed (the 32-row dot, the 64-lane softmax vector work, the DMA pop
and bookkeeping) plus ~60 ns per key: a 16-key page costs half a 64-key page while carrying a
quarter of the keys, so 16-key pages halve the op's effective bandwidth (17 vs 34 GB/s) and a
16-key list at 50% density costs as much as a 64-key list at 100%. This is the compute-bound
crossover predicted earlier (~40 keys per block) measured directly. Finer pages therefore have to
buy more than 2x in selection quality per byte to pay off on this kernel; a kernel that packed
several small pages into one 64-lane softmax would remove most of the fixed part. Device
selection stays exact at 16 keys (lists equal the host top-B; 2 near-tie pages differ at 16k).

On the model (unit 55b03820, same protocol as Stage 5):

| arm | 64-key pages | 16-key pages |
|---|--:|--:|
| decode d4096 tg64, 25% budget | 25.2-25.7 t/s (1.25x) | 21.9-22.0 t/s (1.09x) |
| PPL, cluster pages, 50%        | 15.82 | 12.41 |
| PPL, cluster pages, 25%        | 31.51 | 21.11 |
| PPL, cluster pages, 12.5%      | 73.55 | 38.47 |
| PPL, positional pages, 50%     | 15.09 | **10.81** |
| PPL, positional pages, 25%     | 25.84 | 15.04 |
| dense | 9.13 | 9.13 |

Finer pages help quality a great deal (positional 25% at 16 keys equals positional 50% at 64
keys), and positional pages beat k-means pages at every page size and density -- the cluster
order buys bandwidth, not selection accuracy, with the mean descriptor. But per unit of TIME the
finer page does not pay on this kernel: 16-key positional at 25% (PPL 15.0, 306 us per layer,
1.09x on the token) is matched by 64-key positional at 50% (PPL 15.1, 283 us, ~1.1x). The
quality-per-time frontier is set by the ~2.3 us fixed cost per page, so the kernel change that
packs several 16-key pages into one 64-lane softmax is what would let fine granularity pay, and
the descriptor (min/max bounds) and the sink pages remain the levers for accuracy at a given
budget. The best point measured so far is 16-key positional pages at 50%: PPL 10.8 (dense 9.1)
at about the speed of dense attention.

### Sink pages (2026-09-07, unit 87b3a4aa): the quality loss was the missing sink page

`GGML_HEXAGON_CLUSTER_SINK=4`: the sidecar sorts positions 0..3 of chunk 0 ahead of every cluster,
so they start page 0 of the layer (positional pages hold them in page 0 anyway), and the select pass
scores page 0 at +inf so it is always taken, inside the budget (`HTP_FA_CLUSTER_FLAG_FORCE`, bits
16-23 of op_params[15]; one page per head, no measurable cost). Decode-mode perplexity as above
(ctx 4096, 2 chunks, dense 9.13 +/- 0.46). "Attended" is the average fraction of the context the
scored tokens actually read, counting the dense tail: the tail snaps to the clustering chunk
boundary, so with W 256 / chunk 1024 it averages 768 keys and the "25%" budget attends 45% of the
keys; with W 64 / chunk 256 it averages 192 keys and the 25% budget attends 30%.

| arm (16-key pages unless noted) | window / chunk | attended | PPL without sink | PPL with sink page |
|---|---|--:|--:|--:|
| cluster 25%                 | 256 / 1024 | 45% | 21.11 (control re-run: 20.99) | **9.13** |
| positional 25%              | 256 / 1024 | 45% | 15.04 | **9.07** |
| cluster 25%                 | 64 / 256   | 30% | -- | **9.02** |
| positional 25%              | 64 / 256   | 30% | -- | **9.05** |
| cluster 25%, 64-key pages   | 256 / 1024 | 45% | 31.51 | **9.09** |
| positional 50%              | 256 / 1024 | 63% | 10.81 | **9.03** |
| cluster 50%                 | 256 / 1024 | 63% | 12.41 | **9.09** |

Every arm is at dense perplexity (9.02-9.13 against 9.13 +/- 0.46), including the 64-key k-means
pages at 25% that lost 3.5x without the sink, and the arms that read 30% of the context. So the
losses in the two tables above were never the mean descriptor, the fixed budget or the page size:
they were the first four tokens missing. Qwen3 puts a large share of every row's softmax mass on
them (the StreamingLLM observation), and when they are absent that mass spreads over whatever was
selected, whether the selection was otherwise right or not. K-means scattered position 0 into
whichever cluster its key fell in; positional pages kept it in page 0 but selected that page only
when its diluted mean scored, which is why positional pages looked better.

Consequences for the plan:

- **Page size and descriptor are no longer quality levers at these budgets.** 64-key pages, the
  fast ones, are at dense perplexity at 25%; the min/max descriptor and the threshold budget stay
  on the list only for pushing the budget lower (runs below).
- **K-means versus positional is now a bandwidth question, not a quality one**, and positional
  pages need no shadow: a page is 16 or 64 consecutive rows of llama's cache (the dense path's
  2D descriptor already fetches exactly that), and the only extra data is one mean per page
  (256 B per 16 keys, 1/32 of the KV). That removes the shadow copy and with it the 16k memory
  problem (Stage 3, lesson 2), the CPU gather and the readiness lag. The cluster order's remaining
  argument is the 16-20% streaming gain from contiguous pages (Stage 1) plus one descriptor per
  cluster when whole clusters are selected.
- **The window can become the 64 keys asked for.** Today the tail between the last clustered chunk
  and n - W is dense because it has no pages; with positional tail pages (means computed as rows
  arrive) only W keys need to stay dense, and the clustering lag stops costing bandwidth.

#### Speed with the sink page, and how low the budget can go (2026-09-07, unit 87b3a4aa)

`llama-bench -fa 1 -d 4096 -p 0`, arms alternated, 25% budget, sink page on; tg64 (`-r 2`) in two
rounds, then one tg256 round (`-r 1`) for the steady state once the sidecar has caught up with the
prompt (the shadow is rebuilt on every repetition because llama-bench clears the cache):

| arm | tg64 round 1 | tg64 round 2 | tg256 |
|---|--:|--:|--:|
| dense | 20.0 | 20.3 | 20.2 |
| k-means 64-key, W 256 / chunk 1024 | 25.3 | 23.8 | 24.7 |
| k-means 64-key, W 64 / chunk 256 | 24.9 | 23.0 | 24.4 |
| positional 64-key, W 64 / chunk 256 | **26.4** | **26.3** | **26.0** |
| k-means 16-key, W 64 / chunk 256 | 20.6 | 20.6 | 21.6 |
| positional 16-key, W 64 / chunk 256 | 23.3 | 23.1 | 22.3 |

The sink page costs nothing (25.3 vs 25.2-25.7 without it). Positional pages are faster than k-means
pages at every page size, by 1.3-3.3 t/s, and 16-key k-means pages give no speedup at all in tg64.
The shadow layout is the same in both modes, so the difference is most likely the sidecar itself:
the k-means backlog (16 chunks x 28 layers after a 4k prompt) runs inside the decode window, delaying
coverage and competing with the HTP's dispatch thread, while positional mode only gathers. At tg512
the gap shrinks but does not close (k-means 64-key 25%: 24.9 / 24.5; positional: 25.8 / 26.1;
dense 20.1), which fits a ~4 s backlog covering the first 15-20% of the window plus a burst of
28 k-means runs every 256 tokens, but is not proof; the remedy is the same either way (a cheaper
sidecar, or no sidecar: positional pages need none).

Lower budgets, decode-mode PPL, W 64 / chunk 256, sink page on (dense 9.13 +/- 0.46). "Read" is the
average fraction of the context the scored tokens attend, dense tail included:

| budget | read | k-means 16-key | positional 16-key | k-means 64-key | positional 64-key |
|--:|--:|--:|--:|--:|--:|
| 25%   | 30% | 9.02 | 9.05 | 9.08 | 9.14 |
| 12.5% | 18% | 9.05 | 9.11 | 9.25 | 9.36 |
| 6.2%  | 12% | 9.26 | 9.18 | 9.58 | -- |
| 3.1%  |  9% | 9.50 | -- | -- | -- |

With the sinks in place, 16-key pages stay at dense level down to a 12.5% budget (9.05-9.11,
reading 18% of the context) and hold to within 1-1.4% at 6.2% (reading 12%); at 3.1% (reading 9%)
k-means 16-key pages lose 4%. 64-key pages are at dense level at 25% and lose 1.3-2.4% at 12.5% and
5% at 6.2%. So fine pages help exactly where the budget gets small, as expected, and page order
(k-means vs positional) makes no consistent quality difference at any budget: the descriptor and
the budget rule were never the problem, the sink page was.

Speed at those budgets (d4096 tg64 `-r 2`, W 64 / chunk 256, sink page; round 2 shown, round 1
agreed except for its first two arms, a DVFS outlier after idle), with the perplexity from the
table above:

| point | t/s | speedup | PPL | context read |
|---|--:|--:|--:|--:|
| dense | 20.3 | 1.00x | 9.13 | 100% |
| positional 64-key, 25% | 26.4 | 1.30x | 9.14 | 30% |
| positional 64-key, 12.5% | 27.3 | 1.34x | 9.36 | 18% |
| positional 16-key, 12.5% | 24.6 | 1.21x | 9.11 | 18% |
| positional 16-key, 6.2% | 26.8 | 1.32x | 9.18 | 12% |
| k-means 16-key, 6.2% | 22.0 | 1.09x | 9.26 | 12% |
| k-means 64-key, 25% (tg512) | 24.5-24.9 | 1.22x | 9.08 | 30% |

Two points define the frontier at 4k: positional 64-key pages at 25% (1.30x at dense perplexity)
and positional 16-key pages at 6.2% (1.32x at +0.5%). Beyond that the token is no longer limited by
attention bytes: at 6.2% the attention op is mostly its fixed costs (26 us floor, select pass, one
descriptor pair per page) and the rest of the token is the weight stream, which sparse attention
cannot touch.

## Stage 6 -- in-place positional pages: no shadow (built 2026-09-07)

`GGML_HEXAGON_CLUSTER_INPLACE=1`. Page p of a layer is rows [p*PK, p*PK+PK) of llama's own cache
(PK = `GGML_HEXAGON_CLUSTER_PAGE`, 16 or 64). The shadow buffer keeps only the header, the
per-layer directory and the page descriptors (one f16 mean per page per KV head: 256 B per page,
1 MB for a 1k context, 14 MB for 16k with 64-key pages, 56 MB with 16-key pages); there are no K/V
pages, pos_map or n_valid (`HTP_FA_CLUSTER_HDR_INPLACE` in `header.flags`; `htp_fa_cluster_layout_ex`
lays the regions out with zero size). What changes:

- **Sidecar.** A CPU thread and nothing else: no OpenCL, no k-means, no gather. After every graph
  it computes the mean of each newly completed page from the cache rows the DSP just wrote (a cache
  clean/invalidate first, as before), writes the descriptors, cleans them, then publishes
  `n_pages_pub` and `covered_end = n_pages_pub * PK`. A page of 16 keys is 16 rows x 2 KB read and
  16k f16 conversions, tens of microseconds; a 4k prompt costs ~0.1 s of sidecar CPU in total,
  hidden behind the next ubatch, and one new page every PK decode tokens. The readiness lag that
  cost the shadow design the first ~100 decode tokens after a prompt is gone.
- **Rewinds.** Positional pages make invalidation trivial: rows written below `covered_end` shrink
  the published prefix to the page boundary below the lowest rewritten position and the pages are
  recomputed as rows arrive; nothing goes `stale`. Non-contiguous writes shrink to the lowest
  position and stop publishing until contiguous writes resume from there. A write from position 0
  resets the layer as before.
- **Kernel.** `hvx_fa_cl_setup` takes `dense_start = min(covered_end, n_kv - W)` rounded down to a
  64-key block and `n_cand = dense_start / PK`; `hvx_fa_dec_blk_src` addresses a selected page as
  rows `page*PK ..` of the positional K/V with the dense block's strided descriptor (`nb[1]`
  stride), `bsz = PK`, no mask (every candidate row lies below `dense_start <= n_kv`). The select
  pass, the lists, the staging ring and the merge are untouched. The dense tail is therefore
  exactly W rounded up to the next 64-key block: with W 64, 64..127 keys, no longer snapped to a
  1024-key clustering chunk. With `GGML_HEXAGON_CLUSTER_SINK=4` forcing page 0, the attended set is
  the sink page + W..W+63 recent keys + the top pages by descriptor score: the 4 + 64 + selected
  design, on the cache as it stands.
- **Limitations.** Single sequence; full attention only (the pages skip the mask, so a sliding-window
  layer would attend outside its window: not tagged for Qwen3, which has none); the window is a
  whole 64-key block.

### Who computes the descriptors: a coherence lesson (2026-09-07)

The first in-place build computed the page means on the host, as the shadow sidecar had always
done, and was not exact: 100% of the pages gave PPL 9.39 against 9.13 dense, greedy text diverged
after a dozen tokens, and the sparse arms degraded much faster than the shadow mode had
(6.2%: 14.7 vs 9.18). Isolating it:

- the kernel's strided page fetch is exact at the op level (host lists, device selection, 16- and
  64-key pages, `max|err|` 2e-5, same time as shadow pages: 16k / 16-key / 6% = 404 vs 401 us);
- the host thread's clean-and-invalidate does not corrupt rows (sidecar active with the kernel
  forced dense: PPL 9.1332, identical to dense);
- but the **shadow** path itself breaks the same way when its chunk shrinks from 256 to 128 keys
  (PPL 20.85 / 21.07 vs 9.128 at 256), even though a delayed re-read of every published page found
  no row that differed from the cache.

Three more measurements found it. Holding publication back by 64 or 256 rows changed nothing
(in-place 9.41 / 9.41; 128-key chunks 20.99), so it was not timing. The kernel with a 128-key
chunk table (32 entries) is exact in the tool. And a **single**-context perplexity run is exact for
every mode (dense 9.357; 128-key shadow chunks 9.374; in-place CPU descriptors 9.331): the damage
appears only in the second perplexity context, right after `llama_memory_clear`. That is the
mechanism: the sidecar retires the old coverage *asynchronously*, so the first graph of the new
context still sees the previous directory, and its query -- position 0, the attention sink --
attends pages of the previous context (stale shadow pages, or in-place cache rows 1..191 that
have not been rewritten yet). One corrupted token, but the sink's K/V poison every later token of
the context: PPL 47 on that context with the shadow (21 over both), +6% in-place. With 256- or
1024-key chunks no chunk fits below the 192-key horizon of a first graph, the kernel falls back to
dense, and nothing happens -- the shadow design had been correct by accident.

Two fixes, both kept. **Synchronous clamp** (`HTP_FA_CLUSTER_HDR_HOST_ROWS`, every mode): the dispatch
thread writes `hdir.rows_valid` = the lowest row this graph will write, on its own 128-B line,
before the graph is submitted; the kernel never uses a page or chunk beyond it, so a reset or
rewind takes effect on the very next graph while the sidecar catches up. **DSP-owned descriptors**
(`HTP_FA_CLUSTER_HDR_DSP_DESC`, the default for in-place mode) go further and remove the host from
the data path entirely: the FA op computes the descriptors itself from `rows_valid`. The host writes one word per layer before a graph
is submitted (`hdir.rows_valid`: rows [0, n) complete and in the sequence, taken from the SET_ROWS
index leaf), on its own 128-B line; the op reads it, describes any complete page not yet described
(one 2D descriptor per page brings all heads' rows into VTCM, HVX f16->f32 sums, f16 mean, flushed
before the select pass DMAs it), shrinks after a rewind, and owns the directory. No sidecar thread,
no OpenCL, no host read of DSP-written memory at all. The first decode token after a prompt pays a
one-time catch-up (all pages of the prompt: ~0.4 ms per layer per 4k of context).

With DSP descriptors the in-place mode is exact across the reset: two-context decode-mode PPL at
100% is **9.1185** (dense 9.1332, shadow 9.123-9.128). Greedy text at 100% still differs from dense
after a dozen tokens, identically for CPU and DSP descriptors, while the single-context perplexity
matches to 0.3%; with a complete attended set and an exact op-level check, that is the score-ordered
accumulation flipping a near-tie token, not a defect.

Final measurements, DSP descriptors, W 64, sink page on (unit 87b3a4aa, 2026-09-08). Quality is the
two-context decode-mode perplexity at ctx 4096 (dense 9.133 +/- 0.46):

| budget | 64-key pages | 16-key pages | shadow positional (for reference) |
|--:|--:|--:|--:|
| 100%  | 9.1185 | -- | 9.128 |
| 25%   | 9.20 | -- | 9.14 |
| 12.5% | 9.51 | 9.09 | 9.36 / 9.11 |
| 6.2%  | -- | 9.27 | 9.18 |

The same pages and the same descriptors (up to f16 rounding of the mean) give the same quality as
the shadow-positional arms. With the synchronous clamp the two paths that broke across a context
reset are exact as well: shadow positional pages with 128-key chunks 9.1183 (was 20.9), in-place
with CPU descriptors 9.1185 (was 9.39).

Speed, `llama-bench -fa 1 -p 0`, arms alternated, two rounds:

| d (context) | dense | in-place 64-key, 25% | in-place 16-key, 6.2% | in-place 16-key, 12.5% |
|--:|--:|--:|--:|--:|
| 4096 (tg64)  | 20.1-20.3 t/s | 26.1-26.2 (1.30x) | 26.5-26.8 (1.32x) | 24.7-24.8 (1.23x) |
| 8192 (tg64)  | 15.4-15.5 | 23.6-23.7 (1.53x) | 24.2-24.7 (1.58x) | -- |
| 16384 (tg32) | 10.5-10.8 | 19.6-19.7 (**1.85x**) | 20.3-21.1 (**1.95x**) | 18.2-18.4 (1.72x) |

16k is the first context length at which this device has ever decoded sparse: the shadow design's
memory budget went to zero there and disabled the mode; the in-place buffer is 15 MB. The
first decode token after a 4k prompt pays the descriptor catch-up for 64 pages x 28 layers: tg1
after pp4096 is 29.4-31.1 t/s against 31.1-31.4 dense, i.e. 0.4-1.9 ms once, and pp4096 itself is
unchanged (1751-1767 vs 1754-1762 t/s). In-place pages cost the same as shadow pages on the token
(d4096, 64-key 25%: 26.1-26.2 vs 25.6-26.0).

At 16k, decode-mode perplexity over one 16384-token context of `corpus16k.txt` (a concatenation, so
the absolute value is low): dense 3.781, 16-key pages at 6.2% 3.794 (+0.3%, reading ~7% of the
context, decoding 1.95x faster), 64-key pages at 25% 3.812 (+0.8%). The budget that holds at 4k
holds at 16k.

## Stage 7 -- retrieval quality: positional vs k-means pages on RULER at 4k (GPU emulation, 2026-09-08)

Consolidated results record: [ruler-page-vs-cluster.md](ruler-page-vs-cluster.md). The sections below are the working notes.

Perplexity cannot separate the two page orders (both sit at dense level with the sink page), and it
is dominated by local context. RULER's synthetic tasks (needle-in-a-haystack in eight variants,
variable tracking, common/frequent word extraction, two QA tasks) test exactly what page selection
can break: a few tokens far back that the query must find. The HTP decode attention was emulated in
PyTorch on an H100 (`examples/sparse-attn-sim/ruler_sparse.py`, registered as a custom attention
function for the HF Qwen3-1.7B bf16 model): dense prefill; at decode, dense tail = the last
W..W+63 keys, candidate pages over the rest, page descriptor = f16 mean, score = max over the GQA
group of q . descriptor, page 0 (the sinks) always taken, top-B pages with B = ceil(density x
n_cand), softmax over the selected keys only. Positional pages are PK consecutive keys; cluster
pages come from k-means over the prompt's keys per layer and KV head (C = n/PK clusters, 8 Lloyd
iterations, keys sorted by (cluster, distance), sinks first), i.e. the shadow design's clustering
with the whole prompt as one chunk -- its best case. RULER data: the official generators, 100
samples per task, 4096 tokens, Qwen3 tokenizer, base template; scoring with RULER's own
string-match metrics (`ruler_summarize.py`). Same budgets as the device frontier: 64-key pages at
25%, 16-key pages at 12.5% and 6.2%.

Scores (RULER string match, 100 samples per task; "attended" is the mean fraction of keys read at
decode, sink page and dense tail included):

| task | dense | positional 64-key 25% | k-means 64-key 25% | positional 16-key 12.5% | k-means 16-key 12.5% | positional 16-key 6.2% | k-means 16-key 6.2% |
|---|--:|--:|--:|--:|--:|--:|--:|
| attended | 100% | 27.5% | 27.5% | 14.8% | 14.8% | 8.8% | 8.8% |
| niah_single_1 | 100 | 100 | 93 | 100 | 94 | 100 | 87 |
| niah_single_2 | 100 | 98 | 89 | 100 | 94 | 100 | 87 |
| niah_single_3 | 98 | 96 | 82 | 97 | 91 | 97 | 77 |
| niah_multikey_1 | 100 | 100 | 98 | 100 | 100 | 100 | 97 |
| niah_multikey_2 | 99 | 98 | 96 | 99 | 98 | 93 | 93 |
| niah_multikey_3 | 98 | 75 | 78 | 63 | 61 | 34 | 30 |
| niah_multivalue | 99.2 | 96.5 | 96.0 | 97.2 | 95.8 | 96.8 | 84.0 |
| niah_multiquery | 100 | 98.2 | 97.8 | 99.8 | 98.8 | 99.2 | 96.5 |
| vt | 93.4 | 92.4 | 93.2 | 90.8 | 91.4 | 90.6 | 87.2 |
| cwe | 96.0 | 94.3 | 97.0 | 87.3 | 96.4 | 82.6 | 94.8 |
| fwe | 63.7 | 60.7 | 68.0 | 58.7 | 63.7 | 54.3 | 61.7 |
| qa_1 | 49 | 50 | 48 | 50 | 46 | 49 | 46 |
| qa_2 | 37 | 36 | 33 | 35 | 35 | 38 | 36 |
| **average** | **87.2** | **84.2** | **82.2** | **82.9** | **81.9** | **79.6** | **75.2** |

What the table says:

- **Positional pages beat k-means pages at every budget on the average** (84.2 vs 82.2, 82.9 vs
  81.9, 79.6 vs 75.2), and the gap widens as the budget shrinks. The mechanism is visible in the
  single-needle rows: a needle is several tokens ("the special magic number for X is 1234567"),
  positional pages keep them together, k-means scatters them into different clusters and pages, and
  a partially selected needle yields a partially right answer (the model then writes "54372. Wait,
  but the text also mentions...").
- **K-means pages win the two aggregation tasks** (`cwe` 97 vs 94 and 96 vs 87; `fwe` 68 vs 61, above
  dense). Repeated words cluster together, so one selected page carries all occurrences: exactly
  the case where semantic grouping is the right unit. It is a real but narrow advantage.
- **Positional pages at 25% are within two points of dense on 11 of 13 tasks**, at 12.5% on 10
  of 13, and even at 6.2% (8.8% of the keys read) retrieval of a single needle is still 97-100.
  The budget-limited tasks are the ones that need many scattered facts at once (`cwe`, `fwe`).
- **The mean descriptor fails on `niah_multikey_3`** (UUID keys and UUID values: 98 dense, 75 / 63 /
  34 positional). A page mean cannot represent a 30-token random string, so the page holding the
  right UUID does not score high enough for a 25% budget. This is the Quest observation, and the
  reason the bound-descriptor arms below exist.

**Bound descriptors** (Quest-style: per-dimension min and max of the page's keys, score =
sum_d max(q_d min_d, q_d max_d), an upper bound on any key's dot product; `--desc minmax`), same
budgets, same runs otherwise:

| task | dense | positional 64-key 25%, mean | positional 64-key 25%, bounds | k-means 64-key 25%, mean | k-means 64-key 25%, bounds | positional 16-key 12.5%, mean | positional 16-key 12.5%, bounds |
|---|--:|--:|--:|--:|--:|--:|--:|
| niah_single_1 | 100 | 100 | 100 | 93 | 100 | 100 | 100 |
| niah_single_2 | 100 | 98 | 100 | 89 | 96 | 100 | 100 |
| niah_single_3 | 98 | 96 | 99 | 82 | 87 | 97 | 100 |
| niah_multikey_1 | 100 | 100 | 100 | 98 | 99 | 100 | 100 |
| niah_multikey_2 | 99 | 98 | 93 | 96 | 96 | 99 | 98 |
| niah_multikey_3 | 98 | 75 | 63 | 78 | 74 | 63 | **81** |
| niah_multivalue | 99.2 | 96.5 | 99.2 | 96.0 | 97.5 | 97.2 | 99.2 |
| niah_multiquery | 100 | 98.2 | 99.2 | 97.8 | 98.0 | 99.8 | 99.8 |
| vt | 93.4 | 92.4 | 92.6 | 93.2 | 93.6 | 90.8 | 94.4 |
| cwe | 96.0 | 94.3 | 91.2 | 97.0 | 87.9 | 87.3 | 87.9 |
| fwe | 63.7 | 60.7 | 56.7 | 68.0 | 53.7 | 58.7 | 55.7 |
| qa_1 | 49 | 50 | 47 | 48 | 46 | 50 | 55 |
| qa_2 | 37 | 36 | 38 | 33 | 34 | 35 | 40 |
| **average** | **87.2** | **84.2** | **83.0** | **82.2** | **81.7** | **82.9** | **85.5** |

Bounds are a page-size question. Over 16-key pages they are the best sparse configuration measured:
85.5 against 87.2 dense while reading 14.8% of the keys, UUID retrieval 63 -> 81, every other
needle task 98-100, variable tracking above dense. Over 64-key pages they are worse than the mean
(83.0 vs 84.2): the per-dimension envelope of 64 keys is so wide that most pages look possible and
the ranking stops discriminating (UUID retrieval 75 -> 63). This is why Quest uses 16-key pages.
K-means pages lose their one advantage under bounds (cwe 97 -> 88, fwe 68 -> 54), since that
advantage came from the mean of a cluster of repeated words matching the query.

**What this means for the device.** Two configurations, both positional, both in-place:
- quality first: 16-key pages, min/max descriptors, 12.5% budget -- RULER 85.5 (98% of dense),
  decode about 1.2x at 4k and 1.7x at 16k from the device measurements of the 12.5% arm, minus the
  cost of the bound pass;
- speed first: 16-key pages, mean descriptor, 6.2% budget -- RULER 79.6 (91% of dense; single
  needles still 97-100, aggregation tasks and UUIDs pay), decode 1.32x at 4k and 1.95x at 16k.

The bound pass on the HTP is two descriptor rows per page (min and max, computed by the same DSP
descriptor pass with vmin/vmax instead of a sum) and an elementwise multiply-max-reduce per page
and query head instead of the rx32 dot: roughly 2-3x the select pass, i.e. a few percent of the
attention op at 4k and 15-20% of it at 16k unless the descriptors are made hierarchical.

### Whole-cluster selection: the k-means result above was an artifact of the page cut

The "k-means" arms above are the device variant: keys sorted by cluster and then **cut into fixed
16- or 64-key DMA pages**, each page scored by its own mean, the sink keys inside the k-means. That
cutting destroys what clustering is for. `--mode cluster_var` is the policy the clustering papers
actually use (ClusterKV, Squeezed Attention): k-means over the **middle keys only** (positions 4 ..
ds; the sinks and the recent window are never clustered), variable-size clusters (average 32 keys,
20 Lloyd iterations), score = max over the GQA group of q . centroid, clusters taken **whole** in
score order until the key budget is filled, sinks and window always attended. Same budgets,
"attended" measured the same way:

| task | dense | positional 64-key 25% | **whole clusters 25%** | positional 16-key 12.5% | positional 16-key 12.5% + bounds | **whole clusters 12.5%** | positional 16-key 6.2% | **whole clusters 6.2%** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| attended | 100% | 27.5% | 27.0% | 14.8% | 14.8% | 14.9% | 8.8% | 8.9% |
| niah_single_1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_2 | 100 | 98 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_3 | 98 | 96 | 100 | 97 | 100 | 100 | 97 | 100 |
| niah_multikey_1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_2 | 99 | 98 | 99 | 99 | 98 | 99 | 93 | 98 |
| niah_multikey_3 | 98 | 75 | **98** | 63 | 81 | **98** | 34 | **97** |
| niah_multivalue | 99.2 | 96.5 | 99.0 | 97.2 | 99.2 | 99.2 | 96.8 | 100 |
| niah_multiquery | 100 | 98.2 | 100 | 99.8 | 99.8 | 99.8 | 99.2 | 99.8 |
| vt | 93.4 | 92.4 | 93.2 | 90.8 | 94.4 | 93.0 | 90.6 | 94.6 |
| cwe | 96.0 | 94.3 | 97.1 | 87.3 | 87.9 | 96.1 | 82.6 | 93.5 |
| fwe | 63.7 | 60.7 | 65.7 | 58.7 | 55.7 | 62.7 | 54.3 | 57.7 |
| qa_1 | 49 | 50 | 49 | 50 | 55 | 49 | 49 | 46 |
| qa_2 | 37 | 36 | 38 | 35 | 40 | 38 | 38 | 37 |
| **average** | **87.2** | **84.2** | **87.6** | **82.9** | **85.5** | **87.3** | **79.6** | **86.4** |

Whole-cluster selection is at dense level at every budget: 87.6 / 87.3 / 86.4 against 87.2 while
reading 27% / 15% / 9% of the keys, with the UUID needles at 97-98 where positional pages fall to
75 / 63 / 34 and bound descriptors only reach 81. The mechanism: all the digits (or all the
UUID fragments) of a context land in the same clusters, so selecting a cluster fetches every
candidate value at once and the exact attention picks the right one; a positional page has to be
found through a mean diluted by the sentence around the needle. The aggregation tasks stay at
dense level too (cwe 96-97, fwe 63-66 at 25%/12.5%), for the reason the page-cut variant already
showed. So the ranking of page orders on this benchmark is: whole clusters > positional pages with
bounds (16-key) > positional pages with means > page-cut clusters, and the earlier conclusion that
"positional pages are good enough" holds for perplexity and for retrieval at 25%, but not for
retrieval at the budgets where the speedup lives.

Cluster size and descriptor for whole clusters, all at the 12.5% budget (14.9% of keys read):

| task | dense | avg cluster 16 | avg cluster 32 | avg cluster 64 | avg cluster 32, min/max bounds |
|---|--:|--:|--:|--:|--:|
| niah_single_1 / 2 / 3 | 100 / 100 / 98 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 99 / 100 | 100 / 91 / 53 |
| niah_multikey_1 / 2 / 3 | 100 / 99 / 98 | 100 / 99 / 97 | 100 / 99 / 98 | 100 / 96 / 96 | 89 / 97 / 78 |
| niah_multivalue / multiquery | 99.2 / 100 | 99.2 / 100 | 99.2 / 99.8 | 99.2 / 99.5 | 75.5 / 79.8 |
| vt | 93.4 | 94.0 | 93.0 | 95.0 | 92.2 |
| cwe / fwe | 96.0 / 63.7 | 96.3 / 63.3 | 96.1 / 62.7 | 94.2 / 62.3 | 71.8 / 54.0 |
| qa_1 / qa_2 | 49 / 37 | 49 / 37 | 49 / 38 | 47 / 36 | 44 / 32 |
| **average** | **87.2** | **87.3** | **87.3** | **86.5** | **73.6** |

Cluster sizes 16 and 32 are equivalent and at dense level; 64 costs about a point (a few clusters
too coarse for the budget). Per-cluster min/max bounds are the wrong descriptor for semantic
clusters (73.6): a cluster's envelope is tight in the dimensions that define it and wide in the rest,
so the bound over-estimates almost every cluster and the ranking degrades. The centroid is the
descriptor for clusters, the bound is the descriptor for positional pages.

**What this means for the device.** Whole-cluster selection needs the keys of a cluster to be
fetchable as one unit, and that is exactly what the cluster-ordered **shadow** layout provides: a
cluster is a contiguous run of rows in cluster order, one variable-length 2D descriptor fetches it,
and the per-descriptor fixed cost (the thing that made 16-key pages cost twice per byte) is
amortized over the whole cluster. The in-place layout cannot do this: a 32-key cluster scattered
over the positional cache is 32 descriptors of 256 B. So the design returns to the shadow copy,
with what was learned since: sinks and window outside the clustering, whole clusters instead of a
page cut (the kernel's list of (offset, length) runs instead of page indices; the select pass
scores centroids and accumulates sizes to the budget), descriptors and directory owned by the DSP
or clamped synchronously, and the memory problem (a second copy of the K/V of the clustered range)
to be paid for with Q8 pages or a partial shadow of the oldest context. The GPU sidecar that
clusters during prefill is the piece that already exists.

**Update (2026-09-15): the "32 descriptors" cost was measured** (heterogeneous-npu-gpu.md 4l,
`--cluster --runs --scatter`): the HTP fetching a run's rows from the positional cache one linked
descriptor per row runs at 8.4-9.6 GB/s, half the shadow's rate and independent of the address
pattern (about 0.1 us per descriptor). The Adreno gathering the same scattered rows through index
lists runs at 33-45 GB/s, as fast as the HTP streams contiguous pages. So the memory problem has a
third answer besides Q8 pages and a partial shadow: no shadow at all, with the GPU gathering the
selected clusters' rows from llama's cache and the HTP merging its partials.

## Stage 8 -- whole-cluster selection on the HTP (built 2026-09-08)

`GGML_HEXAGON_CLUSTER_RUNS=1` (with `GGML_HEXAGON_CLUSTER_AVG=32`, `_ATTN=<permille>,64`, sinks default
4): the device implementation of the policy that won on RULER. Three things are fixed by
construction: the sink keys (positions 0-3) are always attended, the recent window (64..127 keys
plus the not-yet-clustered tail) is attended densely from llama's cache, and only the keys in
between are ever clustered or skipped.

- **Layout** (`HTP_FA_CLUSTER_HDR_RUNS`, header v2). The shadow keeps each head's K and V rows in
  cluster order, one row per position of the covered range (rows [pos_begin, pos_end) of a chunk),
  plus a **run table** per head (`struct htp_fa_cluster_run {row_first, n_rows, flags}`, one entry
  per cluster, in cluster order) and one f16 **centroid per run**. The chunk table's unit fields
  count runs. No fixed pages: a cluster is a contiguous run of rows and is fetched as such.
- **Sidecar** (`cluster_run_chunk_runs`). Per prefill chunk of 1024 keys and per KV head: the sink
  keys of chunk 0 are copied first as a *forced* run (`HTP_FA_CLUSTER_RUN_FORCED`) and never enter
  the k-means; the remaining keys are clustered on the GPU (k-means++ init on a subsample, 8 Lloyd
  iterations, C = keys / 32 clusters), sorted by (cluster, distance), gathered into the rows in
  that order; run entries and centroids (mean of each run's keys) follow; publish order is rows,
  pos_map, runs, centroids, chunk entry, unit count, covered_end. Positional mode (`_POSITIONAL=1`)
  gives runs of 32 consecutive keys for A/B.
- **Kernel.** The setup accepts chunks whose runs are published and clamped by `rows_valid`;
  the budget is `density x (candidate rows - forced rows)`. The select pass stages the head's
  centroids and its run table, scores centroids as before (rx32 dot, max over the GQA group), then
  runs a greedy fill: forced runs score +inf and are free, empty runs never score, runs are taken
  in score order while they fit the row budget (the first non-forced run always does), and each
  taken run is expanded into 64-row DMA blocks `(row, bsz)`. The dec pass walks the block list
  (contiguous shadow rows, no mask) followed by the dense tail; the staging ring and merge are
  unchanged. Echo writes the block list for the tool.
- **Op-level exactness** (`llama-hetero-decode-attn --cluster --runs`, synthetic variable-size runs
  over a random permutation, sink run forced): max|err| 2.5e-5 .. 4.7e-5 for avg 16/32, budgets
  6-100%, kv 4k and 16k; the kernel's greedy run set equals the host's f32 greedy set exactly.
  FA op at kv 4096: 178 us at 12% (535 rows/head attended), 136 us at 6% (295 rows), 274 us at 25%,
  against 571 us dense; at kv 16384, 6%: 381 us against 2108 us. The 100% arm costs 800 us (every
  run a separate partial block: the price of variable units when nothing is skipped).

**On the model** (unit b4bd0901, Qwen3-1.7B Q4_0, `-dev HTP0`, sinks 4, W 64, avg cluster 32):

Quality, two-context decode-mode PPL at ctx 4096 (dense 9.133):

| arm | PPL |
|---|--:|
| runs 100% (exactness) | 9.121 |
| runs 12.5%, chunk 1024 | 9.063 |
| runs 6.2%, chunk 1024 | 9.087 |
| runs 12.5%, avg cluster 16 | 8.977 |
| runs 12.5%, chunk 256 | 9.058 |
| runs 6.2%, chunk 256 | 9.175 |

Speed (`llama-bench -fa 1 -p 0`, arms alternated). The first build's sidecar was serial over the
28 layers (~35 ms per layer and chunk) and finished ~4 s after a 4k prompt, so a tg64 run was mostly
dense (1.10x at 4k, 1.16x at 8k, noisy). Parallelizing the sidecar across layers
(`GGML_HEXAGON_CLUSTER_THREADS`, default 4; thread-local scratch, the OpenCL k-means serialized)
made it keep up with prefill:

| d (context) | dense | runs 12.5% | runs 6.2% | runs 6.2%, chunk 256 | in-place positional 16-key 6.2% |
|--:|--:|--:|--:|--:|--:|
| 4096 tg64 | 20.0-20.3 t/s | 25.3-25.5 (1.26x) | 26.0-26.3 (1.30x) | 26.5-26.8 (1.32x) | 26.7 |
| 8192 tg64 | 15.5-15.6 | 19.2-19.3 (1.24x) | -- | 20.2-20.7 (1.32x) | -- |
| 4096 tg512 (serial sidecar) | 19.7-20.1 | 24.1-24.3 | 25.2-25.3 | -- | 26.7-27.0 |
| 8192 tg512 (serial sidecar) | 15.4 | 18.8 | 19.7 | -- | -- |

**Retrieval on the device and the chunk question.** The device clusters per prefill chunk, the
RULER emulation clustered the whole prompt. Re-running the emulator with chunk-local clustering
(`--cchunk`, 12.5% budget, avg 32):

| task | dense | whole prompt | chunk 1024 | chunk 256 | positional 16-key |
|---|--:|--:|--:|--:|--:|
| niah_multikey_3 (UUIDs) | 98 | 98 | 97 | 93 | 63 |
| niah_multikey_2 | 99 | 99 | 99 | 98 | 99 |
| vt / cwe / fwe | 93.4 / 96.0 / 63.7 | 93.0 / 96.1 / 62.7 | 94.4 / 97.6 / 63.0 | 95.0 / 97.0 / 60.0 | 90.8 / 87.3 / 58.7 |
| **13-task average** | **87.2** | **87.3** | **87.3** | **86.6** | **82.9** |

1024-key chunks lose nothing; 256-key chunks cost 0.7 on the average and 5 points on the UUID task,
because chunk-local clustering multiplies the number of "digit clusters" by the number of chunks and
a fixed budget then cannot take them all. On the device (20 prompts, 64 generated tokens, hot unit)
the same pattern with a smaller budget: `niah_single_3` 100 for dense and every runs arm;
`niah_multikey_3` dense 95, runs 12.5% chunk 1024 **85**, runs 6.2% chunk 256 **15**. The 40-prompt sweep over budgets and chunk sizes (unit eb49fb9d, 64 generated tokens, arms run
back to back on a cool device) separates the two effects:

| arm | score | exact answers |
|---|--:|--:|
| dense | 97.5 | 39/40 |
| whole clusters, 25%, chunk 1024 | **97.5** | 39/40 |
| whole clusters, 12.5%, chunk 1024 | **97.5** | 39/40 |
| whole clusters, 6.2%, chunk 1024 | 80.0 | 32/40 |
| whole clusters, 12.5%, chunk 256 | 70.0 | 28/40 |
| whole clusters, 6.2%, chunk 256 | 27.5 | 11/40 |
| in-place positional 16-key pages, 12.5% | 45.0 | 18/40 |

This is the RULER result reproduced on the phone: at a 12.5% budget with 1024-key chunks the device
retrieves UUIDs exactly as well as dense attention (97.5, 39 of 40 answers exact), while positional
pages at the same budget reach 45. Both knobs cost quality when tightened, and they compound: 6.2%
costs 17.5 points, 256-key chunks cost 27.5 points, and together they collapse to 27.5. The dense
arm is reproducible run to run (95.0 on the same 20 prompts in two sessions on different units);
the sparse arms move by about 10 points on 20 prompts between sessions because how much of the
context the sidecar has clustered by the time decoding starts depends on timing, which is why these
numbers are reported over 40 prompts.

**Recommendation for the device**: whole clusters of 32 keys, centroid scoring, sinks and window
outside the clustering, 1024-key chunks with the end-of-prefill tail clustered, 12.5% budget, the
parallel sidecar. That is dense perplexity (9.06 vs 9.13), dense UUID retrieval (97.5 vs 97.5) and
1.26x at 4k / 1.24x at 8k on tg64. Neither knob is worth tightening: 6.2% buys 4-6% more speed and
costs 17.5 points of retrieval, and 256-key chunks buy nothing at all (the end-of-prefill tail is
clustered either way) while costing 27.5 points.

### Status

Stage 8: whole-cluster selection runs on the HTP (`GGML_HEXAGON_CLUSTER_RUNS=1`): op-level exact,
dense-level perplexity, 1.26x / 1.30x at 4k and 1.24x / 1.32x at 8k (12.5% / 6.2%) once the sidecar
was parallelized across layers; 1024-key chunk-local clustering matches whole-prompt clustering on
RULER (87.3), 256-key chunks cost 0.7; on the device the UUID retrieval task is 85 at 12.5% against
95 dense with 20 prompts.

Built and measured end to end: page-list HVX decode kernel (Stage 1), GPU chunk-local k-means
(Stage 2), the clustering sidecar with the cDSP memory budget respected (Stage 3), on-device
selection (Stage 4), page sizes 16/32/64. Exact at full density; 1.25x / 1.4x decode at 4k / 8k
at a 25% budget with 64-key pages; the sidecar costs prefill nothing; the shadow-copy design
cannot fit a 16k context on this SoC (Stage 3, lesson 2). Selection accuracy is solved by one
rule: always select the page holding the sink tokens (`GGML_HEXAGON_CLUSTER_SINK=4`). With it,
16-key pages are at dense perplexity down to a 12.5% budget and within 1.4% at 6.2%, 64-key pages
at dense perplexity at 25%; without it the same arms lost 1.7-3.5x. Frontier at 4k: positional
64-key pages at 25% = 1.30x at dense PPL; positional 16-key pages at 6.2% = 1.32x at +0.5%, reading
12% of the context. K-means pages are no better in quality and slower on the token (sidecar
backlog). In-place positional pages with
DSP-computed descriptors (Stage 6) replace the shadow: exact across context resets, 15 MB at 16k,
quality at dense level down to 12.5% (16-key pages), decode 1.30x / 1.55x / 1.9x at 4k / 8k / 16k.
On RULER at 4k, positional pages beat *page-cut* k-means pages (84.2 / 82.9 / 79.6 vs 82.2 /
81.9 / 75.2 against 87.2 dense), but **whole-cluster selection with the sinks and window excluded
from clustering is at dense level at every budget (87.6 / 87.3 / 86.4 while reading 27% / 15% / 9%
of the keys)**, so the cluster-ordered shadow layout, fetching whole clusters as single runs, is
the design to return to for retrieval-heavy workloads. Next, in order: whole-cluster selection on
the device (shadow layout, run lists, DSP-owned metadata), (in-place pages of llama's cache + one mean per page: removes the memory
problem, the gather and the lag), positional tail pages so the dense window is really 64 keys, and
the fixed per-page cost (K|V in one descriptor, several 16-key pages per 64-lane softmax).

## Stage 9 -- the runs kernel's own fixed costs (2026-09-15, unit f3b4a4c5)

With 16k out of scope the decode direction is the NPU alone (heterogeneous-npu-gpu.md 4l). At 4k and
12.5% the runs-mode op took 151 us for 532 rows per head, 16 GB/s against 36 GB/s for the dense
stream, and a density sweep put ~68 us of it in a fixed cost. Four qtimer stamps inside the op
(echoed to the tool through the echo header, `phases of the last op`) split it: setup 4, select 49,
decode loop 88, merge 4 us. The select pass then got its own four (DMA issue 1, DMA wait 1, scoring
8, greedy fill 28 us on the slowest thread), and cycle counters inside the greedy phase found the
cost in one place: the loop that marks forced runs +inf and empty runs -inf before the greedy fill
took 44 thousand cycles per op, about 170 cycles per run, for one scalar VTCM load and one scalar
VTCM store each. The greedy fill itself was 6 thousand cycles for its vector half and 6 for its
scalar half; its two vector-to-scalar extractions per taken run, my first suspect, made no
difference when replaced by `vextract`.

**What changed.**
- The marking is vectorised: the run table in VTCM is 16 runs per vector with words alternating
  `row_first` and `(n_rows | flags << 16)`; a word deal of two vectors (`Q6_W_vdeal_VVR(hi, lo, -4)`,
  odd words in the HIGH half) lines the packed words of runs 32i..32i+31 up with score vector i, and
  two compares and two muxes do the marking. Lesson that cost four device rounds: the lane-validity
  mask must use `Q6_Q_vsetq2_R`; `Q6_Q_vsetq_R(128)` wraps to zero lanes, so full vectors were never
  marked and the sink run went unforced (the scoring code already guarded that case).
- Two short runs share one 64-row DMA block: the select pass pairs consecutive taken segments whose
  rows fit in 64 (`htp_fa_cluster_blk` carries a second `row2 / bsz2`), the stage pushes two
  descriptors per tensor into one staging slot and the consumer pops both, and the per-block fixed
  cost (descriptors, full-width softmax, accumulator scale) is paid once for both runs. Not in the
  scatter fetch, whose row list is per segment. The echo stride grew with the 16-byte record.
- `hvx_vec_get_i32/f32` use `vextract` instead of a vector store and scalar reload (no measurable
  change here, kept as the cheaper form).

**Op level** (`llama-hetero-decode-attn --cluster --runs`, window 64, sinks 4, chunk-local clusters,
20 iterations, FA op median us; every arm exact to 4e-5 and its run set identical to the host's greedy
selection):

| config | before | after | select before -> after | decode loop before -> after |
|---|--:|--:|--:|--:|
| 4k, 12.5%, avg 32 | 151 | 121 | 49 -> 29 | 88 -> 77 |
| 4k, 25%, avg 32 | 231 | 188 | 57 -> 40 | 159 -> 133 |
| 4k, 1%, avg 32 | 83 | 60 | 39 -> 21 | 30 -> 26 |
| 4k, 12.5%, avg 16 | 224 | 159 | 87 -> 50 | 124 -> 94 |
| 4k, 12.5%, avg 64 | 124 | 113 | 33 -> 23 | 78 -> 75 |
| 8k, 12.5%, avg 32 | 262 | 201 | 90 -> 54 | 158 -> 133 |

At 4k and 12.5% two thirds of the blocks now carry two runs (54 of 83 per head). In-place 64-key
positional pages over the same rows take 97 us, so 24 us of gap remain: the select pass (a
positional page needs none) and the blocks a run pairing cannot fill.

**Model level** (Qwen3-1.7B Q4_0, `GGML_HEXAGON_CLUSTER_RUNS=1 AVG=32 CHUNK=1024 ATTN=125,64
THREADS=4`, old and new DSP library alternated in one session): decode-mode perplexity over two
4096 contexts 19.6375 old, 19.6178 new (dense 20.3694; the sparse arms of the prefill work also
scored 2-3% under dense on this test); tg64 24.4-25.8 old vs 25.2-25.6 new at d4096, 19.8-20.3 vs
20.3 at d8192. The op saves 30 us per layer at 4k and 61 at 8k, 2% and 3.5% of the token, under
tg64's noise with the sidecar's timing-dependent coverage; the op-level numbers are the evidence.

**Where the rest is.** Decode loop 77 us at 4k: ~14 blocks per thread at ~5.5 us, about 2.3 us of
which is per-block fixed cost, the rest rows at ~28 GB/s aggregate. Select 29 us: scoring 7, greedy
7, DMA 3, the remainder thread dispatch and, in the tool only, the echo write. Two scalar VTCM reads
per block remain in the decode loop's block source (~0.1 us each). Larger blocks (128 rows) would
halve the fixed cost again but change the kernel's 64-lane softmax.
