# ggml-hexagon: flash attention returns wrong results with a per-head mask

**Status:** found, root-caused and fixed in this tree. Pre-existing in upstream
llama.cpp (reproduced against pristine `origin/master` @ `3f545becc`), independent
of any block-sparse work. Filable upstream as-is.

## Summary

`GGML_OP_FLASH_ATTN_EXT` on the hexagon HTP backend produces **wrong results
whenever the attention mask is not head-broadcast** (`mask->ne[2] > 1`). Errors are
~0.3 against a 5e-4 tolerance -- not a precision issue, a data race.

It survived because **upstream's own FA test never builds such a mask**:
`test_flash_attn_ext::build_graph` hardcodes

```c
m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, nr23[1]);   // ne[2] == 1
```

so `fa_push_mask_dma_gqa()` -- the non-broadcast path -- is exercised by nothing.

## Reproduction

`FLASH_ATTN_EXT_PERMASK` in `tests/test-backend-ops.cpp` is a minimal, sparsity-free
repro: dense flash attention, one mask per query head, nothing else unusual.

```sh
D=HTP0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops test -o FLASH_ATTN_EXT_PERMASK -b HTP0
```

Against **pristine upstream** libraries:

| case | `mask->ne[2]` | result |
|---|--:|---|
| `hs=128, nh=1, nr=1, kv=1024, nb=64` | 1 | **OK** (control) |
| `hs=128, nh=2, nr=1, kv=1024, nb=64` | 2 | FAIL, ERR 0.335 |
| `hs=128, nh=4, nr=1, kv=1024, nb=64` | 4 | FAIL, ERR 0.333 |
| `hs=128, nh=2, nr=2, kv=1024, nb=64` | 4 | FAIL, ERR 0.305 |

Independent of `nr`, and independent of sparsity.

## Root cause

The non-broadcast mask DMA writes to a **single fixed VTCM buffer** with no double
buffering:

```c
uint8_t * ms_dst = (uint8_t *) factx->vtcm_mask_buf + g * m_line_bytes;
```

The pipelined loop prefetches one KV block ahead, so the mask DMA for block N+1
lands on the buffer the softmax is still reading for block N. The broadcast path
escapes this only because it goes through `dma_cache_push()`, which rotates over
`HMX_FA_DMA_CACHE_SIZE` slots.

**Confirming experiment** (no code change): `NHVX=1` forces `n_threads < 2`, which
sets `hmx.pipeline = 0` (`ggml-hexagon.cpp:2052-2056`). With pipelining disabled all
four cases pass:

```sh
NHVX=1 D=HTP0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops test -o FLASH_ATTN_EXT_PERMASK -b HTP0
# 4/4 tests passed
```

A second latent defect sits alongside it: the mask slot is sized
`align_up(Br * m_line_size, 256)`, i.e. for **one** line per query row, while the
per-head path writes **G** lines per query row. For `G > 1` it already overflows the
slot, and past `G > HMX_FA_DMA_CACHE_SIZE` it would run off the end of the mask
region entirely.

## Fix

The per-head path needs `2 * G` slots to double-buffer; broadcast needs
`HMX_FA_DMA_CACHE_SIZE`. Sizing for both would tax the common case, so
`mask_per_head` is threaded into the layout:

- `flash-attn-ops.h` — `m_buf_slots = mask_per_head ? (2 * gqa_factor) : HMX_FA_DMA_CACHE_SIZE`,
  plumbed through `hmx_fa_vtcm_layout_build()`, `hmx_fa_compute_vtcm_usage()` and
  `hmx_fa_find_chunk_size()`.
- `flash-attn-ops.c` — `fa_push_mask_dma_gqa()` takes a buffer index; the pipelined
  path alternates on `buf_idx` exactly as K/V do, and the consumer reads the
  matching buffer.
- `ggml-hexagon.cpp` — host computes `mask_per_head = (mask != nullptr && mask->ne[2] != 1)`
  for the VTCM budget check.

### A trap worth recording

The first attempt used `!factx.mask_broadcast` as the DSP-side predicate. But the
host sets

```c
kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
```

which is **0 both for a per-head mask and for no mask at all**. Maskless FA therefore
got a different layout on the DSP than the host had budgeted, and the offsets
diverged: **30 failures against a 22 baseline**, i.e. 8 fresh regressions. The
correct predicate is `(mask != NULL) && !factx.mask_broadcast`.

Host and DSP must derive this flag identically or the VTCM layout silently differs
between the budget check and the actual build.

## Verification

| | before | after |
|---|---|---|
| `FLASH_ATTN_EXT_PERMASK` | 0/4 | **4/4** |
| sparse + per-head + gather + `GET_ROWS` | — | **49/49** |
| dense `FLASH_ATTN_EXT` | 2173/2195 (pristine) | **2174/2195**, stable over 2 runs |

Beyond sampling, the no-regression claim is structural: when `mask_per_head` is
false, `m_buf_slots` evaluates to `HMX_FA_DMA_CACHE_SIZE` -- the exact previous
expression -- so the layout is bit-identical for every broadcast and maskless case.
Only the per-head paths changed behaviour.

The dense suite has real run-to-run variance (several pre-existing failures sit
within ~10% of the 5e-4 tolerance), so single readings are not meaningful; the
numbers above are repeated runs.

## Remaining risk

For `G > 2` the per-head mask region is now larger than before (`2G` slots vs 4),
which could push a high-GQA configuration over the VTCM budget and make
`supports_op` decline FA -- falling back to CPU rather than computing wrong answers.
That is the right failure direction, but it is **untested at `G >= 4` with a
per-head mask**, since no such configuration exists in the suite. Worth adding
before upstreaming.

## Pre-existing failures this did not address

~21 dense `FLASH_ATTN_EXT` cases still fail on HTP, nearly all with `sinks=1` and
errors just above tolerance, plus a few large ones (`nr23=[32,1]`, `hsk=320`,
ERR ~= 1.0) that look like a separate upstream bug deserving its own investigation.
