#ifndef HVX_FA_KERNELS_H
#define HVX_FA_KERNELS_H

#include <assert.h>
#include <math.h>
#include "hvx-utils.h"

// Little inner kernels for HVX

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
static __attribute__((unused)) __attribute__((noinline)) void hvx_vec_f32_to_f16_a(void *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx4(const void * restrict y,
                                                const uint8_t * restrict x,
                                                const size_t stride_x,
                                                const size_t nvec,
                                                const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;                   // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);      // fp16
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);  // fp16
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);  // fp16
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;                   // fp16

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];
        HVX_Vector x2_hf = vx2[i];
        HVX_Vector x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_Vector     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_Vector rsum0 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_Vector rsum1 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_Vector rsum2 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_Vector rsum3 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    return hvx_vec_reduce_sum_f32x4(rsum0123);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx32(const void * restrict y,
                                                 const uint8_t * restrict x,
                                                 const size_t stride_x,
                                                 const size_t n,
                                                 float        s) {

    const size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    const size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector   sums = Q6_V_vzero();
    const size_t stride_x_4 = stride_x * 4;
    for (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// DK = 128 specialisation of the tree dot: 8 keys per group (8 independent FMA chains of 2),
// fully unrolled so the scheduler can overlap a group's reduction with the next group's MACs.
static inline HVX_Vector hvx_dot_tree8_d128(const uint8_t * restrict xb, const size_t stride_x, HVX_Vector y0, HVX_Vector y1) {
    HVX_VectorPair p[8];
    #pragma unroll(8)
    for (uint32_t k = 0; k < 8; ++k) {
        const HVX_Vector * restrict vx = (const HVX_Vector * restrict) (xb + k * stride_x);
        HVX_VectorPair a = hvx_vec_mpyacc_f32_f16(Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero()), vx[0], y0);
        p[k] = hvx_vec_mpyacc_f32_f16(a, vx[1], y1);
    }
    HVX_Vector v[8];
    #pragma unroll(8)
    for (uint32_t k = 0; k < 8; ++k) {
        v[k] = HVX_OP_ADD_F32(Q6_V_lo_W(p[k]), Q6_V_hi_W(p[k]));
    }
    // level 1 (4 B)
    const HVX_VectorPair w01 = Q6_W_vshuff_VVR(v[1], v[0], 4);
    const HVX_VectorPair w23 = Q6_W_vshuff_VVR(v[3], v[2], 4);
    const HVX_VectorPair w45 = Q6_W_vshuff_VVR(v[5], v[4], 4);
    const HVX_VectorPair w67 = Q6_W_vshuff_VVR(v[7], v[6], 4);
    const HVX_Vector a01 = HVX_OP_ADD_F32(Q6_V_lo_W(w01), Q6_V_hi_W(w01));
    const HVX_Vector a23 = HVX_OP_ADD_F32(Q6_V_lo_W(w23), Q6_V_hi_W(w23));
    const HVX_Vector a45 = HVX_OP_ADD_F32(Q6_V_lo_W(w45), Q6_V_hi_W(w45));
    const HVX_Vector a67 = HVX_OP_ADD_F32(Q6_V_lo_W(w67), Q6_V_hi_W(w67));
    // level 2 (8 B)
    const HVX_VectorPair wb0 = Q6_W_vshuff_VVR(a23, a01, 8);
    const HVX_VectorPair wb1 = Q6_W_vshuff_VVR(a67, a45, 8);
    const HVX_Vector b0 = HVX_OP_ADD_F32(Q6_V_lo_W(wb0), Q6_V_hi_W(wb0));
    const HVX_Vector b1 = HVX_OP_ADD_F32(Q6_V_lo_W(wb1), Q6_V_hi_W(wb1));
    // level 3 (16 B): [k0..k7] x 4
    const HVX_VectorPair wc = Q6_W_vshuff_VVR(b1, b0, 16);
    return HVX_OP_ADD_F32(Q6_V_lo_W(wc), Q6_V_hi_W(wc));
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx32_tree_d128(const void * restrict y, const uint8_t * restrict x, const size_t stride_x, float s) {
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y;
    const HVX_Vector y0 = vy[0], y1 = vy[1];
    const HVX_Vector h0 = hvx_dot_tree8_d128(x,                stride_x, y0, y1);
    const HVX_Vector h1 = hvx_dot_tree8_d128(x +  8 * stride_x, stride_x, y0, y1);
    const HVX_Vector h2 = hvx_dot_tree8_d128(x + 16 * stride_x, stride_x, y0, y1);
    const HVX_Vector h3 = hvx_dot_tree8_d128(x + 24 * stride_x, stride_x, y0, y1);
    // level 4 (32 B): [k0..k15] x 2
    const HVX_VectorPair we0 = Q6_W_vshuff_VVR(h1, h0, 32);
    const HVX_VectorPair we1 = Q6_W_vshuff_VVR(h3, h2, 32);
    const HVX_Vector e0 = HVX_OP_ADD_F32(Q6_V_lo_W(we0), Q6_V_hi_W(we0));
    const HVX_Vector e1 = HVX_OP_ADD_F32(Q6_V_lo_W(we1), Q6_V_hi_W(we1));
    // level 5 (64 B): [k0..k31]
    const HVX_VectorPair wf = Q6_W_vshuff_VVR(e1, e0, 64);
    const HVX_Vector sums = HVX_OP_ADD_F32(Q6_V_lo_W(wf), Q6_V_hi_W(wf));
    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// 32-key dot with a transpose-add reduction tree. Same contract as hvx_dot_f16_f16_aa_rx32:
// lane k of the result is s * dot(y, x + k*stride_x) for k = 0..31, in key order.
//
// rx32 reduces 4 keys at a time and finishes each group with a serial rotate-add chain, so
// most of its cycles are dependency stalls. Here every key's products are summed into one
// 32-lane partial vector, and the 32 partials are reduced together by single-level vshuff
// exchanges (Rt = 4, 8, 16, 32, 64 bytes) + adds: 31 shuffles, 31 adds, 16-wide independent
// at the first level. The exchange sequence lands the sums in key order without a permute.
static inline HVX_Vector hvx_dot_f16_f16_aa_rx32_tree(const void * restrict y,
                                                      const uint8_t * restrict x,
                                                      const size_t stride_x,
                                                      const size_t n,
                                                      float        s) {
    const size_t nvec = n / VLEN_FP16;
    const size_t nloe = n % VLEN_FP16;
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y;
    const HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);

    HVX_Vector g[8];   // after levels 1-2: g[j] holds keys 4j..4j+3 as [k0 k1 k2 k3] x 8 partials
    for (uint32_t grp = 0; grp < 8; ++grp) {
        const uint8_t * xb = x + (size_t) grp * 4 * stride_x;
        const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) xb;
        const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (xb + stride_x);
        const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (xb + stride_x * 2);
        const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (xb + stride_x * 3);

        HVX_VectorPair p0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        HVX_VectorPair p1 = p0, p2 = p0, p3 = p0;
        uint32_t i = 0;
        for (i = 0; i < nvec; ++i) {
            const HVX_Vector yv = vy[i];
            p0 = hvx_vec_mpyacc_f32_f16(p0, vx0[i], yv);
            p1 = hvx_vec_mpyacc_f32_f16(p1, vx1[i], yv);
            p2 = hvx_vec_mpyacc_f32_f16(p2, vx2[i], yv);
            p3 = hvx_vec_mpyacc_f32_f16(p3, vx3[i], yv);
        }
        if (nloe) {
            const HVX_Vector yv = Q6_V_vand_QV(bmask, vy[i]);
            p0 = hvx_vec_mpyacc_f32_f16(p0, Q6_V_vand_QV(bmask, vx0[i]), yv);
            p1 = hvx_vec_mpyacc_f32_f16(p1, Q6_V_vand_QV(bmask, vx1[i]), yv);
            p2 = hvx_vec_mpyacc_f32_f16(p2, Q6_V_vand_QV(bmask, vx2[i]), yv);
            p3 = hvx_vec_mpyacc_f32_f16(p3, Q6_V_vand_QV(bmask, vx3[i]), yv);
        }
        const HVX_Vector v0 = HVX_OP_ADD_F32(Q6_V_lo_W(p0), Q6_V_hi_W(p0));
        const HVX_Vector v1 = HVX_OP_ADD_F32(Q6_V_lo_W(p1), Q6_V_hi_W(p1));
        const HVX_Vector v2 = HVX_OP_ADD_F32(Q6_V_lo_W(p2), Q6_V_hi_W(p2));
        const HVX_Vector v3 = HVX_OP_ADD_F32(Q6_V_lo_W(p3), Q6_V_hi_W(p3));

        // level 1 (4 B): even lanes key a, odd lanes key b, 16 partials each
        const HVX_VectorPair w01 = Q6_W_vshuff_VVR(v1, v0, 4);
        const HVX_VectorPair w23 = Q6_W_vshuff_VVR(v3, v2, 4);
        const HVX_Vector a01 = HVX_OP_ADD_F32(Q6_V_lo_W(w01), Q6_V_hi_W(w01));
        const HVX_Vector a23 = HVX_OP_ADD_F32(Q6_V_lo_W(w23), Q6_V_hi_W(w23));
        // level 2 (8 B): [k0 k1 k2 k3] x 8
        const HVX_VectorPair w = Q6_W_vshuff_VVR(a23, a01, 8);
        g[grp] = HVX_OP_ADD_F32(Q6_V_lo_W(w), Q6_V_hi_W(w));
    }

    // level 3 (16 B): [k0..k7] x 4
    HVX_Vector h[4];
    for (uint32_t j = 0; j < 4; ++j) {
        const HVX_VectorPair w = Q6_W_vshuff_VVR(g[2 * j + 1], g[2 * j], 16);
        h[j] = HVX_OP_ADD_F32(Q6_V_lo_W(w), Q6_V_hi_W(w));
    }
    // level 4 (32 B): [k0..k15] x 2
    const HVX_VectorPair we0 = Q6_W_vshuff_VVR(h[1], h[0], 32);
    const HVX_VectorPair we1 = Q6_W_vshuff_VVR(h[3], h[2], 32);
    const HVX_Vector e0 = HVX_OP_ADD_F32(Q6_V_lo_W(we0), Q6_V_hi_W(we0));
    const HVX_Vector e1 = HVX_OP_ADD_F32(Q6_V_lo_W(we1), Q6_V_hi_W(we1));
    // level 5 (64 B): [k0..k31]
    const HVX_VectorPair wf = Q6_W_vshuff_VVR(e1, e0, 64);
    const HVX_Vector sums = HVX_OP_ADD_F32(Q6_V_lo_W(wf), Q6_V_hi_W(wf));

    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// P*V accumulate WITHOUT the per-vector vshuff of x. The widening multiply puts the even
// lanes of x in the low vector and the odd lanes in the high one, so accumulator pair i holds
// (even dims, odd dims) of x lanes [64i, 64i+64). Whole padded vectors are accumulated (no
// tail masking); hvx_unshuff_copy_f32_aa restores natural order once, at the end.
static inline void hvx_mad_f32_f16_aa_vec_noshuff(float * restrict y, const void * restrict x, HVX_Vector S0, uint32_t nvec_x) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec_x; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], vx0[i], S0);
    }
}

static inline void hvx_mad_f32_f16_aa_rx2_vec_noshuff(float * restrict y, const void * restrict x0, const void * restrict x1,
                                                      HVX_Vector S0, HVX_Vector S1, uint32_t nvec_x) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec_x; ++i) {
        HVX_VectorPair a = hvx_vec_mpyacc_f32_f16(vy_p[i], vx0[i], S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(a, vx1[i], S1);
    }
}

// P*V for a whole block with the accumulators in registers: 4 independent FMA chains (keys j mod 4)
// instead of one 64-long load-FMA-store chain through VTCM. y (pairs, even/odd layout, see
// hvx_mad_f32_f16_aa_vec_noshuff) is read once and written once. p16 holds the block's P
// weights as f16 scalars (a VTCM copy of the P vector); each is broadcast with a scalar splat.
// nvec_x must be 1 or 2 (DV <= 128).
static inline void hvx_pv_block_f32_f16_noshuff(float * restrict y, const uint8_t * restrict v_base, const size_t stride_v,
                                                const uint16_t * restrict p16, uint32_t n_keys, uint32_t nvec_x) {
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    const HVX_VectorPair z = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    HVX_VectorPair c0_0 = vy_p[0], c1_0 = z, c2_0 = z, c3_0 = z;
    HVX_VectorPair c0_1 = z, c1_1 = z, c2_1 = z, c3_1 = z;
    if (nvec_x == 2) c0_1 = vy_p[1];

    uint32_t j = 0;
    for (; j + 4 <= n_keys; j += 4) {
        const HVX_Vector s0 = Q6_Vh_vsplat_R(p16[j + 0]);
        const HVX_Vector s1 = Q6_Vh_vsplat_R(p16[j + 1]);
        const HVX_Vector s2 = Q6_Vh_vsplat_R(p16[j + 2]);
        const HVX_Vector s3 = Q6_Vh_vsplat_R(p16[j + 3]);
        const HVX_Vector * restrict v0 = (const HVX_Vector * restrict) (v_base + (j + 0) * stride_v);
        const HVX_Vector * restrict v1 = (const HVX_Vector * restrict) (v_base + (j + 1) * stride_v);
        const HVX_Vector * restrict v2 = (const HVX_Vector * restrict) (v_base + (j + 2) * stride_v);
        const HVX_Vector * restrict v3 = (const HVX_Vector * restrict) (v_base + (j + 3) * stride_v);
        c0_0 = hvx_vec_mpyacc_f32_f16(c0_0, v0[0], s0);
        c1_0 = hvx_vec_mpyacc_f32_f16(c1_0, v1[0], s1);
        c2_0 = hvx_vec_mpyacc_f32_f16(c2_0, v2[0], s2);
        c3_0 = hvx_vec_mpyacc_f32_f16(c3_0, v3[0], s3);
        if (nvec_x == 2) {
            c0_1 = hvx_vec_mpyacc_f32_f16(c0_1, v0[1], s0);
            c1_1 = hvx_vec_mpyacc_f32_f16(c1_1, v1[1], s1);
            c2_1 = hvx_vec_mpyacc_f32_f16(c2_1, v2[1], s2);
            c3_1 = hvx_vec_mpyacc_f32_f16(c3_1, v3[1], s3);
        }
    }
    for (; j < n_keys; ++j) {
        const HVX_Vector s0 = Q6_Vh_vsplat_R(p16[j]);
        const HVX_Vector * restrict v0 = (const HVX_Vector * restrict) (v_base + j * stride_v);
        c0_0 = hvx_vec_mpyacc_f32_f16(c0_0, v0[0], s0);
        if (nvec_x == 2) c0_1 = hvx_vec_mpyacc_f32_f16(c0_1, v0[1], s0);
    }
    // fold the chains
    HVX_Vector lo = HVX_OP_ADD_F32(HVX_OP_ADD_F32(Q6_V_lo_W(c0_0), Q6_V_lo_W(c1_0)), HVX_OP_ADD_F32(Q6_V_lo_W(c2_0), Q6_V_lo_W(c3_0)));
    HVX_Vector hi = HVX_OP_ADD_F32(HVX_OP_ADD_F32(Q6_V_hi_W(c0_0), Q6_V_hi_W(c1_0)), HVX_OP_ADD_F32(Q6_V_hi_W(c2_0), Q6_V_hi_W(c3_0)));
    vy_p[0] = Q6_W_vcombine_VV(hi, lo);
    if (nvec_x == 2) {
        lo = HVX_OP_ADD_F32(HVX_OP_ADD_F32(Q6_V_lo_W(c0_1), Q6_V_lo_W(c1_1)), HVX_OP_ADD_F32(Q6_V_lo_W(c2_1), Q6_V_lo_W(c3_1)));
        hi = HVX_OP_ADD_F32(HVX_OP_ADD_F32(Q6_V_hi_W(c0_1), Q6_V_hi_W(c1_1)), HVX_OP_ADD_F32(Q6_V_hi_W(c2_1), Q6_V_hi_W(c3_1)));
        vy_p[1] = Q6_W_vcombine_VV(hi, lo);
    }
}

// (even dims, odd dims) pairs -> natural order. Pair i of src becomes vectors 2i (dims 64i..64i+31)
// and 2i+1 (dims 64i+32..64i+63) of dst. vshuff with -4 is the full 4-byte interleave.
static inline void hvx_unshuff_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t nvec_x) {
    const HVX_VectorPair * restrict sp = (const HVX_VectorPair *) src;
    HVX_VectorPair * restrict dp = (HVX_VectorPair *) dst;
    for (uint32_t i = 0; i < nvec_x; ++i) {
        const HVX_VectorPair p = sp[i];
        dp[i] = Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4);
    }
}

// MAD: y (F32) += x (F16) * s (F16)
static inline void hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x, const __fp16 * restrict s, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
static inline void hvx_mad_f32_f16_aa_rx2(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          const __fp16 * restrict s0, const __fp16 * restrict s1, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
static inline void hvx_mad_f32_f16_aa_vec(float * restrict y, const void * restrict x, HVX_Vector S0, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

static inline void hvx_mad_f32_f16_aa_rx2_vec(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          HVX_Vector S0, HVX_Vector S1, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

static inline void hvx_scale_vec_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t n, HVX_Vector vs) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);

    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#endif /* HVX_FA_KERNELS_H */
