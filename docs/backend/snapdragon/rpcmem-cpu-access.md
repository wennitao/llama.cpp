# CPU access to hexagon rpcmem: there is no penalty

This settles a claim made twice in this work and wrong both times: that CPU computation over
ggml-hexagon's rpcmem buffers is expensive, and that it therefore rules out CPU/DSP
heterogeneous splits. It does not.

`examples/rpcmem-probe` allocates eight buffer arms that differ in exactly one variable each, so
the previously reported 7.5x can be attributed to a specific link rather than to "rpcmem":

```
M-def --(A: page size)--> M-4k --(B: ION)--> R-raw --(C: fastrpc_mmap)--> R-virgin --(D: DSP)--> R-dsp
```

Link C had never been measured and no hypothesis covered it: `ggml_hexagon_shared_buffer::alloc()`
calls `mmap()` unconditionally (`ggml-hexagon.cpp:402`), so every buffer previously described as
"rpcmem the DSP has not touched" had *already* been registered with FastRPC.

## The persistence timeline kills the DSP hypothesis outright

antidiag g=512, N=4 MiB, 1 thread, pinned to the prime cores. Tail p50 µs of the DSP-touched arm:

| state | | median |
|:--|:--|--:|
| S0 virgin | 188, 186 | **188** |
| S1 after a 1-element DSP op | 186, 201, 199, 193, (302) | **199** |
| S7 after a 16 MiB DSP op | (300), 191, 191 | **191** |
| S9 after forced `drop_mmap` | 189, 190, 189 | **189** |
| S13 after re-touching | 204, 202, 200 | **202** |

**No step at any transition.** The DSP writing 16 MiB through that buffer does not change what CPU
access to it costs. (The two bracketed values are isolated outliers in a run whose A/A gate
already reported ~20% round-to-round drift on this device.)

## And rpcmem is not slower than malloc

Same pattern, same size, same state:

| arm | tail p50 |
|:--|--:|
| M-def (malloc, default THP) | 220 µs |
| M-huge (THP forced) | 224 µs |
| R-virgin (rpcmem + fastrpc_mmap) | 201 µs |
| **R-dsp (rpcmem, DSP-written)** | **190 µs** |

rpcmem that the DSP has written is *faster* than malloc here, not slower.

## Where the original 7.5x came from

Two artifacts, neither of them a property of the memory:

1. **CPU thread placement.** SM8750 has 6 cores at 2.78 GHz and 2 prime cores at 4.09 GHz on the
   `walt` governor, observed idling at 1.0-2.4 GHz. An unpinned 8-thread CPU backend spreads onto
   the little cores. Pinning to the two prime cores -- with 2 threads instead of 8 -- cut the same
   work from 797 to 366 µs. See the correction banner in `xattn-scoring.md`.
2. **Comparing two different measurement contexts.** The remaining gap was between an R1 gate
   reading and a per-Lk table reading taken later in the same process, on a device whose own A/A
   gate shows ~20% drift. That is drift, not a DSP effect.

## Consequences

- **CPU/DSP splits are not blocked by memory.** The relocation experiment in `xattn-split`
  measured 0.97x (break-even) once pinned; with the async submit path already landed
  (`GGML_HEXAGON_ASYNC`) real overlap is the remaining question, not memory cost.
- **Pin CPU threads, or measure the scheduler.** Every CPU figure in this work taken without
  `--affinity` is partly a measurement of core placement. The probe and `xattn-split` both take an
  affinity mask; use it.
- **This device is noisy.** The probe's A/A gate -- two identical measurement rounds with nothing
  between them -- reported up to 21% difference. Any single-sample comparison below ~1.3x on this
  hardware is indistinguishable from drift.

## What the probe still cannot answer

Its sentinel check fails, so the tool self-reports INVALID for the go/no-go cell. The sentinel
does a raw pointer read of a location the DSP just wrote, after the CPU dirtied that line itself;
there is no host-side cache maintenance anywhere in the hexagon backend for tensor data
(`ggml_backend_hexagon_buffer_get_tensor` is a plain memcpy for F32), so a stale read is possible.
Whether that indicates a genuine coherency hazard or merely a broken sentinel is **unresolved**,
and it is worth resolving: the persistence numbers above stand on their own, but a real coherency
gap would matter for any pipeline that has the CPU read DSP output.
