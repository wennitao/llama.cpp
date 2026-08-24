# GEMM Latency on Hexagon HTP: llama.cpp ggml-hexagon, Qwen3-1.7B Shapes

A baseline of single-MatMul latency on the llama.cpp Hexagon backend,
covering every GEMM shape that shows up in Qwen3-1.7B prefill and decode.
Companion to the QNN HTP reference — same hardware, different software
stack, so the two can be compared head-to-head.

The benchmark is reproducible — see the F16 / Q4_0 sweeps added to
[../../../tests/test-backend-ops.cpp](../../../tests/test-backend-ops.cpp),
filterable with `-p 'type_a=f16,type_b=f16'` or `-p 'type_a=q4_0,type_b=f32'`.

---

## What was measured

A single `MUL_MAT` op per case, dispatched through `ggml_backend_graph_compute`
on the `HTP0` device. The test harness:

1. Runs one untimed warmup.
2. Replicates the op into a single graph until the total work hits a
   target flop count (≥1 s wall time).
3. Reports the average per-op wall time.

This amortizes per-call dispatch overhead, so the numbers below should
be read as **steady-state op cost**, not single `graphExecute` latency.
The QNN reference doc times 10 individual `graphExecute` calls — its
numbers carry per-call launch overhead that ours don't.

**Test setup:**

- Hardware: Snapdragon 8 Elite (SM8750) — V79 HTP
- Backend: `libggml-htp-v79.so` from llama.cpp commit `4c1c3ac09` + local edits
- Driver: `libcdsprpc.so` system default
- `GGML_HEXAGON_HOSTBUF=0` (default — required so Q4_0 weights land in
  the repack buffer; harmless for F16)

---

## LLM linear projections (Qwen3-1.7B)

Geometry: hidden=2048, intermediate=6144, Hq=16, Hkv=8, head_dim=128.
Per-head widths: q/o_proj=2048, k/v_proj=1024, gate/up=6144, down=2048.

### Decode — M=1

| op | shape | Hexagon Q4_0 | Hexagon F16 | QNN fp16 ref |
|---|---|---:|---:|---:|
| q_proj  | (2048, 1, 2048) | 43 µs · 196 GF/s | 134 µs · 63 GF/s | 460 µs |
| kv_proj | (1024, 1, 2048) | 24 µs · 177 GF/s | 68 µs  · 62 GF/s | 170 µs |
| o_proj  | (2048, 1, 2048) | 43 µs · 196 GF/s | 134 µs · 63 GF/s | 450 µs |
| gate/up | (6144, 1, 2048) | 120 µs · 209 GF/s | 395 µs · 64 GF/s | 3010 µs |
| down    | (2048, 1, 6144) | 120 µs · 210 GF/s | 400 µs · 63 GF/s | 2150 µs |

**Per-layer linear-only decode budget (Q4_0):** 1·q + 2·kv + 1·o + 2·(gate/up) +
1·down ≈ **0.59 ms/layer**. For 28 Qwen3-1.7B layers that's **~16 ms/token**.
This is the headline number — well below the 100 ms/token comfort budget,
and the same workload at fp16 costs ~3× more (~50 ms/token), confirming
that decode is DDR-bound and Q4_0's 4× smaller weights buy a ~3× speedup.

F16 decode plateaus at ~63 GF/s across all shapes — that's ~60 GB/s
effective DDR read bandwidth (memory-bound). Q4_0 hits ~200 GF/s with
the same wall-clock-per-byte budget (~60 GB/s of compressed weight
reads), confirming the bottleneck is weight bandwidth, not dequant.

### Prefill — M=512

| op | shape | Hexagon Q4_0 | QNN fp16 ref |
|---|---|---:|---:|
| q_proj  | (2048, 512, 2048) | 698 µs · 6.15 TF/s | 930 µs · 4.6 TF/s |
| kv_proj | (1024, 512, 2048) | 442 µs · 4.85 TF/s | 380 µs · 5.7 TF/s |
| o_proj  | (2048, 512, 2048) | 698 µs · 6.15 TF/s | 940 µs · 4.6 TF/s |
| gate/up | (6144, 512, 2048) | 1721 µs · 7.49 TF/s | 3280 µs · 3.9 TF/s |
| down    | (2048, 512, 6144) | 2534 µs · 5.08 TF/s | 4650 µs · 2.8 TF/s |

### Prefill — M=1024

| op | shape | Hexagon Q4_0 | QNN fp16 ref |
|---|---|---:|---:|
| q_proj  | (2048, 1024, 2048) | 1426 µs · 6.02 TF/s | 1380 µs · 6.2 TF/s |
| kv_proj | (1024, 1024, 2048) | 910 µs · 4.72 TF/s | 600 µs · 7.2 TF/s |
| o_proj  | (2048, 1024, 2048) | 1419 µs · 6.05 TF/s | 1380 µs · 6.2 TF/s |
| gate/up | (6144, 1024, 2048) | 3460 µs · 7.45 TF/s | 9920 µs · 2.6 TF/s |
| down    | (2048, 1024, 6144) | 4987 µs · 5.17 TF/s | 7970 µs · 3.2 TF/s |

**Notes:**

- **Peak rate seen: 7.5 TF/s** at M=512 gate/up and M=1024 gate/up.
  Hexagon Q4_0 holds a tighter range across shapes than QNN fp16 — no
  equivalent of the "M=2048 gate cliff" the QNN doc reports.
- **gate/up wins biggest vs QNN** (2.9× at M=1024, 1.9× at M=512). down
  also benefits significantly (1.8× at M=1024). The QNN compiler appears
  to land on a bad schedule for the wide-M K=2048 case that ggml-hexagon
  avoids.
- **kv_proj is the one shape where QNN beats Hexagon** (1.5× at M=1024,
  1.2× at M=512). Small output dim (1024) — possibly underutilizing the
  Hexagon backend's tile schedule.

### Prefill — M ≥ 2048

**Not supported.** Hexagon backend rejects MUL_MAT with `nrows(src1) > 1024`
([ggml-hexagon.cpp:2341](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2341)).
To get prefill latency for M=2048 or M=4096 you have to split the
sequence into ≤1024-token chunks; pasted naively, 2× M=1024 q_proj is
~2.85 ms vs. QNN's single-call 2.59 ms at M=2048 (≈10% overhead from
chunking, no compiler cliff to avoid).

---

## Attention matmuls (Hq=16, D=128)

After GQA expansion to Hq=16 heads. Q·K^T uses transposed B.

### Decode — Sq=1 (F16, 16 heads)

| op | shape | Hexagon F16 | QNN fp16 ref |
|---|---|---:|---:|
| Q·K^T Skv=512  | (512, 1, 128, bs=16)  | 242 µs · 8.6 GF/s | 250 µs |
| Q·K^T Skv=2048 | (2048, 1, 128, bs=16) | 1551 µs · 5.4 GF/s | 420 µs |
| Q·K^T Skv=4096 | (4096, 1, 128, bs=16) | 3092 µs · 5.4 GF/s | 760 µs |
| A·V   Skv=512  | (128, 1, 512, bs=16)  | 103 µs · 20.4 GF/s | 90 µs |
| A·V   Skv=2048 | (128, 1, 2048, bs=16) | 394 µs · 21.3 GF/s | 200 µs |
| A·V   Skv=4096 | (128, 1, 4096, bs=16) | 734 µs · 22.9 GF/s | 380 µs |

**Q·K^T regression at large Skv is the bad case.** Parity with QNN at
Skv=512 but 4× slower by Skv=4096 — the batched-GEMV path looks like
it's not tiling across the 16 heads well. A·V degrades more gracefully
(2× slower at Skv=4096) but is also off-pace.

Aggregate attention decode at Skv=4096: QK + AV = 3.8 ms vs. QNN's 1.1 ms.
**Attention becomes the dominant cost in Hexagon decode for long context**,
flipping the QNN finding ("attention is not the bottleneck at decode").
If you're targeting long-context decode on this backend, attention needs
investigation before linear-projection optimizations are worth doing.

### Prefill — Sq ≥ 64

**Not supported.** With bs=16 heads, `nrows(src1) = Sq × 16`, so any
Sq ≥ 64 trips the 1024-row limit. Real prefill attention has to be
either: split per head (drops bs to 1 and lets each head run as a
plain MUL_MAT), or fall back to CPU/GPU for the attention block.

---

## Square roofline

Pure square M=N=K, F16 inputs unavailable (see Caveats), so Q4_0 only.

| size | Hexagon Q4_0 | QNN fp16 ref |
|---:|---:|---:|
| 256³  | 46 µs · 0.73 TF/s | 71 µs · 0.47 TF/s |
| 512³  | 126 µs · 2.13 TF/s | 114 µs · 2.35 TF/s |
| 1024³ | 481 µs · 4.47 TF/s | 305 µs · 7.03 TF/s |
| 2048³ | not supported | 2530 µs · 6.80 TF/s |
| 4096³ | not supported | 30 ms · 4.52 TF/s |

QNN's fp16 roofline beats Hexagon Q4_0 at 1024³ (7.03 vs 4.47 TF/s) —
the only regime where pure HMX rate matters more than weight bandwidth.
The Hexagon backend doesn't hit fp16 HMX peak on the square workload;
either the schedule is worse or HMX usage is gated behind a flag I
haven't found.

---

## Caveats / Known issues

- **F16 MUL_MAT with n > 1 aborts the DSP.** Any F16 prefill case
  triggers
  ```
  ggml-hexagon.cpp:1895: ggml-hex: dspqueue_read failed: 0x0000002e
  Aborted (core dumped)
  ```
  Reproduces on 256³ F16 (smallest case attempted). The support check
  passes — this is a runtime path, not a planned restriction. Hexagon
  F16 prefill is effectively unusable in this build; use Q4_0 for any
  prefill measurement. Worth filing upstream.

- **`nrows(src1) ≤ 1024` is the hard supported envelope** for both F16
  and Q4_0. It rules out M ≥ 2048 single-call prefill and any batched
  attention with Sq ≥ 64.

- **Test-harness amortization vs single-call timing.** The numbers here
  divide total wall time by (n_runs · ops_in_graph), so any per-call
  dispatch overhead is absorbed. The QNN reference times 10 separate
  `graphExecute` calls and reports that. For decode-class shapes
  (sub-ms ops) this is the dominant reason Hexagon numbers look much
  faster than QNN's — both backends are bandwidth-bound and the actual
  device-side work is similar.

- **Dynamic vs static inputs.** Both A and B are treated as dynamic
  inputs here (filled with random data each test); the QNN doc was
  also dynamic-B for apples-to-apples. A real `Linear` layer with a
  baked-in weight tensor would let either backend pre-tile it at
  finalize and likely be 10–20% faster than the prefill numbers.

- **No quantized attention.** Attention matmuls in this sweep are F16
  only — Hexagon's Q4_0 path rejects `src1.ne[2] != 1`, so batched
  matmuls (multi-head attention) cannot use the REPACK fast path.

---

## How to reproduce

The added cases live in [`make_test_cases_perf()`](../../../tests/test-backend-ops.cpp)
right after the qwen3 mul_mat_id sweep. Build inside the snapdragon
toolchain container:

```bash
docker run -it -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.3
# inside container:
cd /workspace
cmake --build build-snapdragon --target test-backend-ops -j
cp build-snapdragon/bin/test-backend-ops pkg-snapdragon/llama.cpp/bin/
exit
```

Push and run from the host:

```bash
adb push pkg-snapdragon/llama.cpp /data/local/tmp/

# Q4_0 sweep (linear + squares)
HB=0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops perf -b HTP0 \
  -o MUL_MAT -p 'type_a=q4_0,type_b=f32'

# F16 decode-only sweep (any n>1 case will abort — keep filter narrow)
HB=0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops perf -b HTP0 \
  -o MUL_MAT -p 'type_a=f16,type_b=f16,.*n=1,'
```

Numbers will drift across llama.cpp commits — the Hexagon backend is
marked experimental and the F16 prefill abort is exactly the kind of
thing that may move with a backend rev. Re-run the sweep after pulling.
