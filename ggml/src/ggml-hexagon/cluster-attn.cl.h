#pragma once

static const char * ggml_hexagon_cluster_attn_cl = R"CL(
// Cluster runs on the GPU (heterogeneous decode): work-group (KV head, split) attends its share of
// the rows of the blocks the DSP handed over in the control region (read with atomic loads, so a
// pre-launched kernel never reads a stale line), each shadow row mapped to its position in llama's
// cache through pos_map. Q is read with atomic loads when qmode != 0 for the same reason. Partials in
// the HVX split-KV format; the last group to finish (counter == want * n_groups) publishes done = want.
#define SEL_MAX 64
__kernel void fa_dec_runs(__global uchar * base,
#if FA_SEPARATE_BUFFERS
                          __global uchar * qbase, __global uchar * kbase, __global uchar * v_storage,
#endif
                          uint q_off, uint k_off, uint v_off, uint nbq2, uint nbk1, uint nbk2, uint nbv1, uint nbv2,
                          uint sel_off, uint sel_stride, uint posmap_off, uint posmap_hstride,
                          uint p_off, uint part_stride, uint nsplit, float scale,
                          uint ready_off, uint cnt_off, uint done_off, uint want, uint max_spin, uint qmode) {
#if !FA_SEPARATE_BUFFERS
    __global uchar * qbase = base, * kbase = base, * v_storage = base;
#endif
    const int  t   = get_local_id(0);
    const int  kvh = get_group_id(1);
    const int  sp  = get_group_id(2);
    const uint n_groups = get_num_groups(1) * get_num_groups(2);

    __local float Q_l[FA_G][FA_D];
    __local float S_l[FA_G][FA_D];
    __local float red[FA_G][FA_D];
    __local uint  I_l[FA_D];
    __local uint  b_row[RUNS_MAX], b_row2[RUNS_MAX], b_bsz[RUNS_MAX], b_bsz2[RUNS_MAX], b_pre[RUNS_MAX + 1];
    __local uint  n_blk_l;
    __local uint  indexed_l;

    // qmode bits: 1 = Q via atomic loads, 2 = skip the epilogue fences (timing only), 4 = local barrier after
    // the spin, 8 = skip Q and records (timing only), 16 = rec[3] selects block or index records
    if (t == 0) {
        const uint wait_off = (qmode & 32) ? sel_off + kvh * sel_stride + 16 : ready_off;
        uint it = 0;
        while (atomic_load_explicit((volatile __global atomic_uint *)(base + wait_off), memory_order_acquire, memory_scope_all_svm_devices) != want && it < max_spin) { it++; }
    }
    if (qmode & 4) barrier(CLK_LOCAL_MEM_FENCE); else barrier(CLK_GLOBAL_MEM_FENCE);

    __global const uint * rec = (__global const uint *)(base + sel_off + kvh * sel_stride);
    if (!(qmode & 8)) {
        for (int g = 0; g < FA_G; ++g) {
            __global const uint * qw = (__global const uint *)(qbase + q_off + (kvh * FA_G + g) * nbq2);
            const uint bits = (qmode & 1) ? atomic_load_explicit((volatile __global atomic_uint *)(qw + t), memory_order_relaxed, memory_scope_all_svm_devices) : qw[t];
            Q_l[g][t] = as_float(bits) * scale;
        }
        if (t == 0) {
            const uint n = atomic_load_explicit((volatile __global atomic_uint *) rec, memory_order_acquire, memory_scope_all_svm_devices);
            n_blk_l = n;
            indexed_l = (qmode & 16) ? atomic_load_explicit((volatile __global atomic_uint *)(rec + 3), memory_order_relaxed, memory_scope_all_svm_devices) : 0;
        }
    } else if (t == 0) {
        n_blk_l = 0;
        indexed_l = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const bool indexed = indexed_l != 0;
    const uint n_blk = indexed ? 0 : min(n_blk_l, (uint) RUNS_MAX);
    for (uint i = t; i < n_blk; i += FA_D) {
        __global const uint * e = rec + 32 + i * 4;   // row, row2, bsz | bsz2 << 16, pad
        b_row[i]  = atomic_load_explicit((volatile __global atomic_uint *)(e + 0), memory_order_relaxed, memory_scope_all_svm_devices);
        b_row2[i] = atomic_load_explicit((volatile __global atomic_uint *)(e + 1), memory_order_relaxed, memory_scope_all_svm_devices);
        const uint w = atomic_load_explicit((volatile __global atomic_uint *)(e + 2), memory_order_relaxed, memory_scope_all_svm_devices);
        b_bsz[i]  = w & 0xffffu;
        b_bsz2[i] = w >> 16;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        uint pre = 0;
        for (uint i = 0; i < n_blk; ++i) { b_pre[i] = pre; pre += b_bsz[i] + b_bsz2[i]; }
        b_pre[n_blk] = pre;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint tot = indexed ? n_blk_l : b_pre[n_blk];
    const uint per = (tot + nsplit - 1) / nsplit;
    const uint r0  = min(tot, (uint) sp * per);
    const uint r1  = min(tot, r0 + per);
    __global const uint * pmap = (__global const uint *)(base + posmap_off + kvh * posmap_hstride);

    // the running state is uniform over the group: every lane keeps a copy
    float m_run[FA_G], l_run[FA_G], o_acc[FA_G];
    for (int g = 0; g < FA_G; ++g) { m_run[g] = -INFINITY; l_run[g] = 0.0f; o_acc[g] = 0.0f; }

    for (uint bs = r0; bs < r1; bs += FA_D) {
        const int blk_n = (int) min((uint) FA_D, r1 - bs);
        {
            // row index -> block -> shadow row -> cache position; lanes past blk_n repeat a valid row (their P is 0)
            const uint ri = bs + min((uint) t, (uint) (blk_n - 1));
            if (indexed) {
                I_l[t] = atomic_load_explicit((volatile __global atomic_uint *)(rec + 32 + ri), memory_order_relaxed, memory_scope_all_svm_devices);
            } else {
                uint j = 0;
                if (n_blk > 32) {
                    uint end = n_blk;
                    while (j + 1 < end) {
                        const uint mid = (j + end) / 2;
                        if (b_pre[mid] <= ri) j = mid; else end = mid;
                    }
                } else {
                    while (j + 1 < n_blk && b_pre[j + 1] <= ri) { j++; }
                }
                const uint off  = ri - b_pre[j];
                const uint srow = off < b_bsz[j] ? b_row[j] + off : b_row2[j] + (off - b_bsz[j]);
#if GPU_SHADOW
                I_l[t] = srow;
#else
                I_l[t] = pmap[srow];
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // scores: lane t = row t of the chunk
        float s[FA_G];
        for (int g = 0; g < FA_G; ++g) s[g] = -INFINITY;
        if (t < blk_n) {
            __global const half * krow = (__global const half *)(kbase + k_off + I_l[t] * nbk1 + kvh * nbk2);
            float8 acc[FA_G];
            for (int g = 0; g < FA_G; ++g) acc[g] = (float8)(0.0f);
            #pragma unroll
            for (int d8 = 0; d8 < FA_D / 8; ++d8) {
                const float8 k8 = convert_float8(vload8(d8, krow));
                for (int g = 0; g < FA_G; ++g) acc[g] += k8 * vload8(d8, Q_l[g]);
            }
            for (int g = 0; g < FA_G; ++g) {
                const float8 a8 = acc[g];
                s[g] = (a8.s0 + a8.s1 + a8.s2 + a8.s3 + a8.s4 + a8.s5 + a8.s6 + a8.s7);
            }
        }
        // block max over the 128 lanes: tree reduction
        for (int g = 0; g < FA_G; ++g) red[g][t] = s[g];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int o = FA_D / 2; o > 0; o >>= 1) {
            if (t < o) { for (int g = 0; g < FA_G; ++g) red[g][t] = fmax(red[g][t], red[g][t + o]); }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float m_new[FA_G], a[FA_G], p[FA_G];
        for (int g = 0; g < FA_G; ++g) {
            m_new[g] = fmax(m_run[g], red[g][0]);
            a[g]     = (m_run[g] == -INFINITY) ? 0.0f : native_exp(m_run[g] - m_new[g]);
            p[g]     = (t < blk_n && !isinf(s[g]) && m_new[g] != -INFINITY) ? native_exp(s[g] - m_new[g]) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // everyone has read red[g][0]
        for (int g = 0; g < FA_G; ++g) { S_l[g][t] = p[g]; red[g][t] = p[g]; }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int o = FA_D / 2; o > 0; o >>= 1) {
            if (t < o) { for (int g = 0; g < FA_G; ++g) red[g][t] += red[g][t + o]; }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int g = 0; g < FA_G; ++g) { l_run[g] = a[g] * l_run[g] + red[g][0]; m_run[g] = m_new[g]; }

        // P.V: lane t = dimension t; 32 scattered row loads in flight per step
        {
            float pv[FA_G];
            for (int g = 0; g < FA_G; ++g) pv[g] = 0.0f;
            __global const uchar * vbase = v_storage + v_off + kvh * nbv2 + t * 2;
            for (int c = 0; c < blk_n; c += 32) {
                const uint8 oa = vload8((c >> 3) + 0, I_l), ob = vload8((c >> 3) + 1, I_l), oc = vload8((c >> 3) + 2, I_l), od = vload8((c >> 3) + 3, I_l);
                float8 va, vb, vc, vd;
#define LDV(dst, o8) \
                dst.s0 = vload_half(0, (__global const half *)(vbase + o8.s0 * nbv1)); dst.s1 = vload_half(0, (__global const half *)(vbase + o8.s1 * nbv1)); \
                dst.s2 = vload_half(0, (__global const half *)(vbase + o8.s2 * nbv1)); dst.s3 = vload_half(0, (__global const half *)(vbase + o8.s3 * nbv1)); \
                dst.s4 = vload_half(0, (__global const half *)(vbase + o8.s4 * nbv1)); dst.s5 = vload_half(0, (__global const half *)(vbase + o8.s5 * nbv1)); \
                dst.s6 = vload_half(0, (__global const half *)(vbase + o8.s6 * nbv1)); dst.s7 = vload_half(0, (__global const half *)(vbase + o8.s7 * nbv1));
                LDV(va, oa) LDV(vb, ob) LDV(vc, oc) LDV(vd, od)
#undef LDV
                for (int g = 0; g < FA_G; ++g) {
                    const float8 pa = vload8((c >> 3) + 0, S_l[g]) * va + vload8((c >> 3) + 1, S_l[g]) * vb
                                    + vload8((c >> 3) + 2, S_l[g]) * vc + vload8((c >> 3) + 3, S_l[g]) * vd;
                    pv[g] += pa.s0 + pa.s1 + pa.s2 + pa.s3 + pa.s4 + pa.s5 + pa.s6 + pa.s7;
                }
            }
            for (int g = 0; g < FA_G; ++g) o_acc[g] = o_acc[g] * a[g] + pv[g];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int g = 0; g < FA_G; ++g) {
        const int row = kvh * FA_G + g;
        __global float * part = (__global float *)(base + p_off + ((ulong) row * nsplit + sp) * part_stride);
        if (t == 0) { part[0] = (r0 < r1) ? m_run[g] : HTP_M_INIT; part[1] = l_run[g]; }
        part[32 + t] = o_acc[g];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (t == 0 && !(qmode & 128)) {   // 128 = measurement re-run: leave the counter and done alone
        if (!(qmode & 2)) atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
        const uint old = atomic_fetch_add_explicit((volatile __global atomic_uint *)(base + cnt_off), 1u, memory_order_acq_rel, memory_scope_device);
        if (old + 1 == want * n_groups) {
            if (!(qmode & 2)) atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_all_svm_devices);
            atomic_store_explicit((volatile __global atomic_uint *)(base + done_off), want, memory_order_seq_cst, memory_scope_all_svm_devices);
        }
    }
}
)CL";
