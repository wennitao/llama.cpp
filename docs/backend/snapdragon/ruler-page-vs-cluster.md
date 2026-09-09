# Sparse decode attention on RULER at 4k: positional pages vs clusters (results record, 2026-09-08)

Question: with the sink tokens and a recent window always attended, and the rest of the context
read only through a per-head budget of selected units, which selection unit keeps retrieval
quality -- fixed positional pages (Quest-style), clusters cut into fixed pages (the first device
design), or whole variable-size clusters (ClusterKV-style)?  Perplexity could not tell them apart
(all at dense level with the sink page); RULER can.

## Setup

- Model: Qwen/Qwen3-1.7B, bf16, HF transformers 4.57, one H100. The Snapdragon HTP decode
  attention is emulated as a custom attention function (`examples/sparse-attn-sim/ruler_sparse.py`):
  dense prefill; at decode the last W..W+63 keys (W = 64, boundary snapped to the kernel's 64-key
  block) are attended densely, page/cluster selection applies to the keys before that boundary,
  softmax runs over the selected keys only.
- Benchmark: official NVIDIA/RULER generators (commit c3f5e3b), all 13 synthetic tasks, 4096-token
  sequences with the Qwen3 tokenizer, `base` template, 100 samples per task (RULER default is 500;
  per-task standard error is ~3-5 points on the needle tasks, the 13-task average ~1.5). Scoring
  with RULER's metrics (`ruler_summarize.py`, verified identical to `eval/evaluate.py`). Greedy
  decoding stopped at the end of the first answer line (RULER scores by containment; the same for
  every arm; `--full_gen` restores the full token budget).
- Budget: fraction of the candidate keys (those before the window). "attended" below is the
  measured mean fraction of all keys read per decode step, sink unit and window included.

Selection policies (arm names):

| arm | unit | descriptor | selection | sinks |
|---|---|---|---|---|
| `posPK-D` | PK consecutive keys (16 or 64) | mean key | top-B pages, B = ceil(D x pages) | page 0 always selected |
| `posPK-D-mm` | same | per-dim min/max bounds, score = sum_d max(q_d min_d, q_d max_d) | same | same |
| `cluPK-D` | k-means over all keys, sorted by (cluster, distance), **cut into PK-key pages** | mean of the page | top-B pages | sinks sorted first into page 0 |
| `cvarS-D` | **whole clusters**, k-means over the middle keys only (sinks and window excluded), average size S, 20 Lloyd iterations | centroid | clusters taken whole in score order until D x middle keys | positions 0-3 always attended |
| `cvarS-D-mm` | same | per-cluster min/max bounds | same | same |

Score = max over the GQA group (2 query heads per KV head) of the descriptor score. Dense = 87.2.

## Results

### 25% budget (~27% of keys read)

| task | dense | pos64-25 | pos64-25-mm | clu64-25 | clu64-25-mm | **cvar32-25** |
|---|--:|--:|--:|--:|--:|--:|
| attended | 100 | 27.5 | 27.5 | 27.5 | 27.5 | 27.0 |
| niah_single_1 | 100 | 100 | 100 | 93 | 100 | 100 |
| niah_single_2 | 100 | 98 | 100 | 89 | 96 | 100 |
| niah_single_3 | 98 | 96 | 99 | 82 | 87 | 100 |
| niah_multikey_1 | 100 | 100 | 100 | 98 | 99 | 100 |
| niah_multikey_2 | 99 | 98 | 93 | 96 | 96 | 99 |
| niah_multikey_3 | 98 | 75 | 63 | 78 | 74 | 98 |
| niah_multivalue | 99.2 | 96.5 | 99.2 | 96.0 | 97.5 | 99.0 |
| niah_multiquery | 100 | 98.2 | 99.2 | 97.8 | 98.0 | 100 |
| vt | 93.4 | 92.4 | 92.6 | 93.2 | 93.6 | 93.2 |
| cwe | 96.0 | 94.3 | 91.2 | 97.0 | 87.9 | 97.1 |
| fwe | 63.7 | 60.7 | 56.7 | 68.0 | 53.7 | 65.7 |
| qa_1 | 49 | 50 | 47 | 48 | 46 | 49 |
| qa_2 | 37 | 36 | 38 | 33 | 34 | 38 |
| **average** | **87.2** | **84.2** | **83.0** | **82.2** | **81.7** | **87.6** |

### 12.5% budget (~15% of keys read)

| task | dense | pos16-12 | pos16-12-mm | clu16-12 | **cvar16-12** | **cvar32-12** | cvar64-12 | cvar32-12-mm |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| attended | 100 | 14.8 | 14.8 | 14.8 | 15.2 | 14.9 | 14.5 | 14.9 |
| niah_single_1 | 100 | 100 | 100 | 94 | 100 | 100 | 100 | 100 |
| niah_single_2 | 100 | 100 | 100 | 94 | 100 | 100 | 99 | 91 |
| niah_single_3 | 98 | 97 | 100 | 91 | 100 | 100 | 100 | 53 |
| niah_multikey_1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 89 |
| niah_multikey_2 | 99 | 99 | 98 | 98 | 99 | 99 | 96 | 97 |
| niah_multikey_3 | 98 | 63 | 81 | 61 | 97 | 98 | 96 | 78 |
| niah_multivalue | 99.2 | 97.2 | 99.2 | 95.8 | 99.2 | 99.2 | 99.2 | 75.5 |
| niah_multiquery | 100 | 99.8 | 99.8 | 98.8 | 100 | 99.8 | 99.5 | 79.8 |
| vt | 93.4 | 90.8 | 94.4 | 91.4 | 94.0 | 93.0 | 95.0 | 92.2 |
| cwe | 96.0 | 87.3 | 87.9 | 96.4 | 96.3 | 96.1 | 94.2 | 71.8 |
| fwe | 63.7 | 58.7 | 55.7 | 63.7 | 63.3 | 62.7 | 62.3 | 53.7 |
| qa_1 | 49 | 50 | 55 | 46 | 49 | 49 | 47 | 44 |
| qa_2 | 37 | 35 | 40 | 35 | 37 | 38 | 36 | 32 |
| **average** | **87.2** | **82.9** | **85.5** | **81.9** | **87.3** | **87.3** | **86.5** | **73.6** |

### 6.2% budget (~9% of keys read)

| task | dense | pos16-6 | pos16-6-mm | clu16-6 | **cvar32-6** |
|---|--:|--:|--:|--:|--:|
| attended | 100 | 8.8 | 8.8 | 8.8 | 8.9 |
| niah_single_1 | 100 | 100 | 100 | 87 | 100 |
| niah_single_2 | 100 | 100 | 100 | 87 | 100 |
| niah_single_3 | 98 | 97 | 97 | 77 | 100 |
| niah_multikey_1 | 100 | 100 | 100 | 97 | 100 |
| niah_multikey_2 | 99 | 93 | 95 | 93 | 98 |
| niah_multikey_3 | 98 | 34 | 24 | 30 | 97 |
| niah_multivalue | 99.2 | 96.8 | 98.0 | 84.0 | 100 |
| niah_multiquery | 100 | 99.2 | 98.8 | 96.5 | 99.8 |
| vt | 93.4 | 90.6 | 93.6 | 87.2 | 94.6 |
| cwe | 96.0 | 82.6 | 82.6 | 94.8 | 93.5 |
| fwe | 63.7 | 54.3 | 51.3 | 61.7 | 57.7 |
| qa_1 | 49 | 49 | 50 | 46 | 46 |
| qa_2 | 37 | 38 | 33 | 36 | 37 |
| **average** | **87.2** | **79.6** | **78.7** | **75.2** | **86.4** |

### Chunk-local clustering (what the device can do incrementally), 12.5% budget, avg 32

| task | dense | whole prompt | chunk 1024 | chunk 256 |
|---|--:|--:|--:|--:|
| niah_multikey_3 | 98 | 98 | 97 | 93 |
| vt / cwe / fwe | 93.4 / 96.0 / 63.7 | 93.0 / 96.1 / 62.7 | 94.4 / 97.6 / 63.0 | 95.0 / 97.0 / 60.0 |
| **average** | **87.2** | **87.3** | **87.3** | **86.6** |

## Findings

1. **Whole-cluster selection is at dense level at every budget**: 87.6 / 87.3 / 86.4 against 87.2
   while reading 27% / 15% / 9% of the keys. Every needle task is within a point of dense,
   including the UUID task (`niah_multikey_3`: 98 / 98 / 97). The aggregation tasks (`cwe`,
   `fwe`) and variable tracking are at dense level too. Cluster sizes 16 and 32 are equivalent;
   64 costs about a point.
2. **Positional pages degrade with the budget** (84.2 / 82.9 / 79.6). Single needles stay at
   97-100 even at 6.2%, but the UUID task collapses (75 / 63 / 34) and the aggregation tasks lose
   9-13 points at 12.5% and below. The mean descriptor of a page is diluted by the sentence around
   a needle; a 30-token random string never scores high enough.
3. **Bound descriptors (Quest) help positional 16-key pages** (82.9 -> 85.5 at 12.5%, UUIDs
   63 -> 81) **but not 64-key pages** (84.2 -> 83.0: the envelope of 64 keys is too wide) and not
   at 6.2% (79.6 -> 78.7). They are the wrong descriptor for clusters (87.3 -> 73.6): a cluster's
   envelope is tight in the dimensions that define it and wide in the rest, so the bound
   over-estimates almost every cluster. Centroid for clusters, bounds for positional pages.
4. **Clusters cut into fixed pages are the worst policy** (82.2 / 81.9 / 75.2). Each page is scored
   by its own mean and a cluster's tokens are split across pages, so a needle is fetched in part;
   the only thing the cut preserved was the aggregation advantage of clustering. This is the
   variant the first device design implemented, and the reason the earlier "positional pages are
   good enough" conclusion was wrong for retrieval.
5. Chunk-local clustering with 1024-key chunks (the device's incremental scheme) matches whole-prompt
   clustering; 256-key chunks cost 0.7 on the average and 5 points on the UUID task, since every chunk
   contributes its own digit clusters and a fixed budget cannot take them all.
6. Mechanism: all the digits (or all the UUID fragments) of a context land in the same few
   clusters; selecting a cluster fetches every candidate value at once and the exact attention over
   the fetched keys picks the right one. The unit of selection must therefore be the whole cluster,
   and the sinks must be kept out of the clustering (their norms pull centroids).

## Implications for the HTP design

- The selection unit is the whole cluster (16-32 keys average), scored by its centroid, with the
  sinks (4 keys) and the recent window (64-127 keys) outside the clustering and always attended.
- That unit maps onto the **cluster-ordered shadow layout**: a cluster is a contiguous run of rows
  in cluster order, one variable-length 2D descriptor fetches it, and the ~1.8 us per-descriptor
  fixed cost (which made 16-key pages cost twice per byte) is amortized over the run. The in-place
  positional layout cannot serve it: a 32-key cluster scattered over the cache is 32 descriptors.
- Kernel changes from the page design: the per-head list becomes (offset, length) runs; the select
  pass scores centroids and accumulates cluster sizes to the key budget instead of taking a fixed
  page count; the K/V of a cluster are contiguous so the existing 2D DMA path applies unchanged.
- Everything else learned stays: DSP-owned descriptors and directory, the synchronous rows_valid
  clamp across context resets, the GPU sidecar that clusters during prefill, and the memory cost
  of the shadow (a second copy of the clustered range) to be paid with Q8 pages or a partial
  shadow of the oldest context.

## On the device

The policy was then implemented on the HTP (`GGML_HEXAGON_CLUSTER_RUNS=1`; see
[cluster-sparse-decode.md](cluster-sparse-decode.md) Stage 8) and the hardest task of this study was
re-run on the phone: `niah_multikey_3`, 40 prompts, 64 generated tokens, Qwen3-1.7B Q4_0, greedy.

| arm | score | exact answers |
|---|--:|--:|
| dense | 97.5 | 39/40 |
| whole clusters, 12.5% budget, 1024-key chunks | **97.5** | 39/40 |
| whole clusters, 6.2% budget, 1024-key chunks | 80.0 | 32/40 |
| whole clusters, 12.5% budget, 256-key chunks | 70.0 | 28/40 |
| in-place positional 16-key pages, 12.5% budget | 45.0 | 18/40 |

The emulation's central claim holds on the hardware: whole-cluster selection retrieves UUIDs as well
as dense attention at an eighth of the key budget, where positional pages of the same budget lose
more than half the answers.

## Caveats

One model (1.7B, 8 KV heads), one length (4k), 100 samples per task, an emulation rather than
the device (bf16 vs Q4_0/f16, exact f32 scoring vs the HTP's f16), and generation stopped after
the first answer line. The comparison between policies is controlled (same prompts, same budget
accounting, same window and sinks); the absolute scores are not comparable to published RULER
numbers for larger models.

## Reproduction

```
# data (official generators, 4k, Qwen3 tokenizer, 100 samples)
python RULER/scripts/data/prepare.py --save_dir ruler_data4k --benchmark synthetic --task <task> \
  --tokenizer_path Qwen/Qwen3-1.7B --tokenizer_type hf --max_seq_length 4096 --num_samples 100 --model_template_type base
# arms
python examples/sparse-attn-sim/ruler_sparse.py --data_dir ruler_data4k --out_dir pred/cvar32-12 --mode cluster_var --csize 32 --density 0.125
python examples/sparse-attn-sim/ruler_sparse.py --data_dir ruler_data4k --out_dir pred/pos16-12 --mode positional --page 16 --density 0.125 [--desc minmax]
python examples/sparse-attn-sim/ruler_sparse.py --data_dir ruler_data4k --out_dir pred/clu64-25 --mode cluster --page 64 --density 0.25
# table
python examples/sparse-attn-sim/ruler_summarize.py pred dense pos16-12 cvar32-12 ...
```
