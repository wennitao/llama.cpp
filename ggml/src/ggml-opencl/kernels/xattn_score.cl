#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

//------------------------------------------------------------------------------
// XAttention block-selection scoring, fused.
//
// Replaces the MUL_MAT -> SOFT_MAX -> RESHAPE -> SUM_ROWS chain that computes,
// per KV head h and query representative n:
//
//   S[j]   = scale * dot(K[:,j,h], Q[:,n,h])       j in 0..kv-1
//   P      = softmax(S)  over j
//   out[b] = sum of P[j] over the keys of selection block b
//
// The [kv, nq, nh] score tensor never exists: it is produced, softmaxed and
// block-summed entirely inside registers. Only K, Q and the small
// [nblk, nq, nh] result cross DRAM.
//
// Compile-time parameters:
//   XA_HS  head size (128)
//   XA_BS  selection block size (64, == the wave width, see below)
//   XA_KT  key rows per lane   (1 or 2)
//   XA_QT  query reps per lane (4, 8 or 16)
//
// Wave width is hard-coded to 64: XA_BS == 64 lanes means one iteration of the
// t loop covers exactly one selection block, so a single sub_group_reduce_add
// over the wave IS that block's sum. The host only dispatches this on Adreno,
// where the backend already assumes a 64-wide wave everywhere.
//------------------------------------------------------------------------------

#define XA_SG      64
#define XA_HS_VEC  (XA_HS/4)
#define XA_W       (XA_SG*XA_KT)   // keys consumed by one workgroup

// The kernels are built with -cl-finite-math-only. On some Adreno GPUs an
// infinite operand causes undefined behavior / miscompilation for exp, so the
// running max starts from a large negative value instead (same sentinel and
// merge guard as flash_attn_f32_f16.cl).
#define XA_M_INIT (-3.0e38f)

//------------------------------------------------------------------------------
// Pass 1: score + softmax + block sum for one (KV head, query tile, kv split).
//
// Decomposition: one lane owns XA_KT whole key rows and walks them along the
// head-size axis. K is [hs, kv, nh] contiguous, so that walk is along the
// contiguous axis -- no transpose, no repack, and the dot product needs no
// cross-lane reduction at all. The generic mul_mv_f16_f32_l4 path instead
// spreads one key row across the wave, which at hs=128 leaves 32 of 64 lanes
// idle and then pays a 6-step subgroup reduce per query row.
//
// Reduction cost here is XA_QT*(1 + XA_KT) subgroup reduces against
// XA_KT*XA_QT*XA_HS FMAs -- ~7% at KT=2/QT=8.
//
// gws = { 64, nh*nqt, nsplits }   lws = { 64, 1, 1 }
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_xattn_score_partial(
        global const char * k_void,
        ulong               k_offset,
        global const char * q_void,
        ulong               q_offset,
        global float *      bsum,
        global float *      ml,
        int                 kv,
        int                 nq,
        int                 nqt,
        int                 nblk,
        int                 nsplits,
        float               scale
) {
    const int lane  = get_local_id(0);
    const int h     = get_group_id(1) / nqt;
    const int qt0   = (get_group_id(1) - h*nqt) * XA_QT;
    const int split = get_group_id(2);

    // Q tile in local memory: XA_QT*512 B (4 KB at QT=8). K deliberately stays
    // in global -- each lane streams its own private rows, so staging them would
    // only add an LDS round trip. This keeps the LDS budget at a quarter of the
    // 16 KB/WG that the FA tile kernel is known to run at on this device, so
    // occupancy is governed by registers rather than by local memory.
    __local float4 q_lds[XA_QT][XA_HS_VEC];

    // 64 lanes sweep the XA_QT*XA_HS_VEC float4 of the tile linearly, so
    // adjacent lanes read adjacent float4 of Q.
    {
        const global float4 * q4 = (const global float4 *)(q_void + q_offset);
        for (int i = lane; i < XA_QT*XA_HS_VEC; i += XA_SG) {
            const int qi = i / XA_HS_VEC;
            const int dv = i - qi*XA_HS_VEC;
            const int qg = qt0 + qi;
            q_lds[qi][dv] = qg < nq ? q4[((ulong)h*nq + qg)*XA_HS_VEC + dv] : (float4)(0.0f);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // key row owned by this lane for tile t: split*XA_W + t*64 + lane, so for
    // each t the wave covers exactly one XA_BS-wide selection block.
    const global half4 * krow[XA_KT];
    {
        const global half4 * k4 = (const global half4 *)(k_void + k_offset);
        #pragma unroll
        for (int t = 0; t < XA_KT; ++t) {
            const int j = split*XA_W + t*XA_SG + lane;
            krow[t] = k4 + ((ulong)h*kv + j)*XA_HS_VEC;
        }
    }

    // Named, statically indexed accumulators: a dynamically indexed private
    // array gets spilled to per-thread scratch on Adreno and collapses the rate.
    float acc[XA_KT][XA_QT];
    #pragma unroll
    for (int t = 0; t < XA_KT; ++t) {
        #pragma unroll
        for (int qi = 0; qi < XA_QT; ++qi) {
            acc[t][qi] = 0.0f;
        }
    }

    // half4 for K (8 B, hardware-widened) against a float4 LDS broadcast for Q:
    // one 128-bit local read feeds 4*XA_KT FMAs, i.e. 8:1 at KT=2. The FA tile
    // kernel measures ~4:1 as LDS-issue-bound, so stay above it.
    // Accumulate in fp32 -- the scores feed exp().
    #pragma unroll
    for (int dv = 0; dv < XA_HS_VEC; ++dv) {
        half4 kvec[XA_KT];
        #pragma unroll
        for (int t = 0; t < XA_KT; ++t) {
            kvec[t] = krow[t][dv];
        }
        #pragma unroll
        for (int qi = 0; qi < XA_QT; ++qi) {
            const float4 qv = q_lds[qi][dv];
            #pragma unroll
            for (int t = 0; t < XA_KT; ++t) {
                acc[t][qi] += dot(convert_float4(kvec[t]), qv);
            }
        }
    }

    // Epilogue. Every score this workgroup produced is live in registers, so the
    // max and the sum are two passes over registers -- no online rescale, no
    // extra barrier, and nothing spills. Only the cross-split merge is online.
    const ulong bsum_base = (ulong)h*nq*nblk;
    const ulong ml_base   = (ulong)h*nq*nsplits;
    const int   blk0      = split*(XA_W/XA_BS);

    #pragma unroll
    for (int qi = 0; qi < XA_QT; ++qi) {
        const int qg = qt0 + qi;
        // uniform across the wave, so the subgroup reduces below stay convergent
        if (qg < nq) {
            float mx = XA_M_INIT;
            #pragma unroll
            for (int t = 0; t < XA_KT; ++t) {
                mx = fmax(mx, scale*acc[t][qi]);
            }
            mx = sub_group_reduce_max(mx);

            float l = 0.0f;
            #pragma unroll
            for (int t = 0; t < XA_KT; ++t) {
                const float p  = native_exp(scale*acc[t][qi] - mx);
                const float bt = sub_group_reduce_add(p);
                l += bt;
                // blocks nest inside splits (XA_W % XA_BS == 0), so this block is
                // written by exactly one workgroup and needs no atomics
                if (lane == 0) {
                    bsum[bsum_base + (ulong)qg*nblk + (blk0 + t)] = bt;
                }
            }
            if (lane == 0) {
                ml[(ml_base + (ulong)qg*nsplits + split)*2 + 0] = mx;
                ml[(ml_base + (ulong)qg*nsplits + split)*2 + 1] = l;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Pass 2: merge the per-split (m, l) records and normalize the block sums.
//
// Exact, not approximate: m = max_s m_s, l = sum_s l_s*exp(m_s - m), and because
// selection blocks nest inside kv splits every block needs exactly ONE rescale.
//
// dst is the SUM_ROWS output [1, nblk, nq, nh] contiguous, so the linear index
// is ((h*nq + q)*nblk + b) and adjacent lanes write adjacent floats.
//
// gws = { 64, nh, nq }   lws = { 64, 1, 1 }
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_xattn_score_merge(
        global const float * bsum,
        global const float * ml,
        global char *        dst_void,
        ulong                dst_offset,
        int                  nq,
        int                  nblk,
        int                  nsplits,
        int                  blk_per_split
) {
    const int lane = get_local_id(0);
    const int h    = get_global_id(1);
    const int qg   = get_global_id(2);

    const ulong base    = ((ulong)h*nq + qg)*nblk;
    const ulong ml_base = ((ulong)h*nq + qg)*nsplits*2;

    // nsplits is at most kv/64, so every lane recomputing (m, l) from the same
    // handful of floats is cheaper than an LDS reduction plus its barrier; the
    // records are wave-uniform reads that land in cache.
    float m = XA_M_INIT;
    for (int s = 0; s < nsplits; ++s) {
        m = fmax(m, ml[ml_base + (ulong)s*2 + 0]);
    }
    float l = 0.0f;
    for (int s = 0; s < nsplits; ++s) {
        const float m_s = ml[ml_base + (ulong)s*2 + 0];
        const float l_s = ml[ml_base + (ulong)s*2 + 1];
        if (m_s > XA_M_INIT) {
            l += l_s * native_exp(m_s - m);
        }
    }
    const float l_inv = l > 0.0f ? 1.0f/l : 0.0f;

    global float * dst = (global float *)(dst_void + dst_offset);
    for (int b = lane; b < nblk; b += XA_SG) {
        const float m_s = ml[ml_base + (ulong)(b/blk_per_split)*2 + 0];
        dst[base + b] = bsum[base + b] * native_exp(m_s - m) * l_inv;
    }
}
