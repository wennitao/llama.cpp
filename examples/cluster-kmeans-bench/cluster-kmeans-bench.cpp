// cluster-kmeans-bench: the GPU side of cluster-selected decode attention (docs/backend/snapdragon/
// cluster-sparse-decode.md, Stage 2). Chunk-local k-means over one prefill chunk of keys per KV head
// (N keys x D f16, C = N/64 centroids), then packing the keys into 64-key pages in (cluster,
// distance) order and computing each page's f16 mean. Measures per-kernel time on the Adreno so the
// sidecar's cost can be compared with the HTP's prefill time per ubatch (~20 ms per layer at
// ub=1024), and checks the assignment quality against a CPU Lloyd reference with the same init.
//
// The kernels are kept in one string so the backend sidecar (Stage 3) can lift them verbatim.

#include "ggml.h"

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const char * cl_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef KM_D
#define KM_D 128
#endif
#ifndef KM_TILE
#define KM_TILE 16
#endif
#ifndef KM_NMAX
#define KM_NMAX 16384
#endif

// K rows live in the real cache layout: row n of head h at K + n*nbk_row + h*nbk_head (bytes).
inline __global const half * km_row(__global const uchar * K, uint n, uint h, uint nbk_row, uint nbk_head) {
    return (__global const half *) (K + (ulong) n * nbk_row + (ulong) h * nbk_head);
}

// Assignment: one work-item per key, centroids tiled through local memory. dist = |mu|^2 - 2 k.mu.
__kernel void km_assign(__global const uchar * K, uint nbk_row, uint nbk_head,
                        __global const float * mu, __global const float * mu2,
                        __global uchar * assign, __global float * dist, __global uint * n_changed,
                        uint N, uint C) {
    const uint n = get_global_id(0), h = get_global_id(1), lid = get_local_id(0), lsz = get_local_size(0);
    __local float lmu[KM_TILE * KM_D];
    __local float lmu2[KM_TILE];
    float8 k[KM_D / 8];
    if (n < N) {
        __global const half * kr = km_row(K, n, h, nbk_row, nbk_head);
        for (uint i = 0; i < KM_D / 8; ++i) k[i] = convert_float8(vload8(i, kr));
    }
    float best = INFINITY; uint besti = 0;
    for (uint c0 = 0; c0 < C; c0 += KM_TILE) {
        const uint tc = min((uint) KM_TILE, C - c0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = lid; i < tc * KM_D; i += lsz) lmu[i] = mu[((ulong) h * C + c0) * KM_D + i];
        for (uint i = lid; i < tc; i += lsz) lmu2[i] = mu2[h * C + c0 + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (n < N) {
            for (uint c = 0; c < tc; ++c) {
                float8 acc = (float8)(0.0f);
                for (uint i = 0; i < KM_D / 8; ++i) acc += k[i] * vload8(i, lmu + c * KM_D);
                const float dt = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;
                const float dd = lmu2[c] - 2.0f * dt;
                if (dd < best) { best = dd; besti = c0 + c; }
            }
        }
    }
    if (n < N) {
        const uint idx = h * N + n;
        if (assign[idx] != (uchar) besti) atomic_inc(n_changed);
        assign[idx] = (uchar) besti;
        dist[idx]   = best;
    }
}

// Update: one work-group per (centroid, head), lane = dim. Empty cluster -> the farthest key.
__kernel void km_update(__global const uchar * K, uint nbk_row, uint nbk_head,
                        __global const uchar * assign, __global const float * dist,
                        __global float * mu, __global float * mu2, __global uint * counts, uint N, uint C) {
    const uint d = get_local_id(0), c = get_group_id(1), h = get_group_id(2);
    __local uchar la[KM_NMAX];
    __local float red[KM_D];
    __local uint  lfar;
    for (uint i = d; i < N; i += KM_D) la[i] = assign[h * N + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f; uint cnt = 0;
    for (uint n = 0; n < N; ++n) {
        if (la[n] == (uchar) c) { sum += vload_half(d, km_row(K, n, h, nbk_row, nbk_head)); cnt++; }
    }
    float m;
    if (cnt > 0) {
        m = sum / (float) cnt;
    } else {
        if (d == 0) {   // re-seed: the key farthest from its centroid
            float bd = -INFINITY; uint bi = 0;
            for (uint n = 0; n < N; ++n) { const float v = dist[h * N + n]; if (v > bd) { bd = v; bi = n; } }
            lfar = bi;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        m = vload_half(d, km_row(K, lfar, h, nbk_row, nbk_head));
    }
    mu[((ulong) h * C + c) * KM_D + d] = m;
    red[d] = m * m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = KM_D / 2; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (d == 0) { mu2[h * C + c] = red[0]; counts[h * C + c] = cnt; }
}

// Gather: one work-item per (page slot, head) copies one 256 B K row and one V row into the page.
// perm[p*64 + i] = key index of page p slot i (shared by heads).
__kernel void km_gather(__global const uchar * K, __global const uchar * V, uint nbk_row, uint nbk_head,
                        __global const uint * perm, __global uchar * pages_k, __global uchar * pages_v,
                        uint n_pages, ulong head_stride) {
    const uint s = get_global_id(0), h = get_global_id(1);   // s = p*64 + i
    if (s >= n_pages * 64) return;
    const uint n = perm[s];
    __global const uchar16 * sk = (__global const uchar16 *) (K + (ulong) n * nbk_row + (ulong) h * nbk_head);
    __global const uchar16 * sv = (__global const uchar16 *) (V + (ulong) n * nbk_row + (ulong) h * nbk_head);
    __global uchar16 * dk = (__global uchar16 *) (pages_k + (ulong) h * head_stride + (ulong) s * (KM_D * 2));
    __global uchar16 * dv = (__global uchar16 *) (pages_v + (ulong) h * head_stride + (ulong) s * (KM_D * 2));
    for (uint i = 0; i < (KM_D * 2) / 16; ++i) { dk[i] = sk[i]; dv[i] = sv[i]; }
}

// Page descriptor: one work-group per (page, head), lane = dim, f16 mean of the 64 rows.
__kernel void km_page_mean(__global const uchar * pages_k, __global half * cent, uint n_pages, ulong head_stride, uint cent_stride_h) {
    const uint d = get_local_id(0), p = get_group_id(1), h = get_group_id(2);
    __global const half * rows = (__global const half *) (pages_k + (ulong) h * head_stride + (ulong) p * 64 * (KM_D * 2));
    float s = 0.0f;
    for (uint i = 0; i < 64; ++i) s += vload_half(i * KM_D + d, rows);
    vstore_half(s * (1.0f / 64.0f), (ulong) h * cent_stride_h + (ulong) p * KM_D + d, cent);
}
)CL";

struct opts {
    int n = 1024, heads = 8, d = 128, c = 0, km_iters = 8, layers = 28, reps = 11, seed = 1;
    std::string init = "kpp", data = "blobs";
    int layout_real = 1;    // 1 = positional cache layout [pos][heads][D], 0 = [heads][pos][D]
    int verbose = 0;
};

static bool cl_ok(cl_int e, const char * w) { if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL %s failed (%d)\n", w, e); return false; } return true; }
static double ev_ms(cl_event ev) { cl_ulong a = 0, b = 0; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr); clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr); return (double) (b - a) * 1e-6; }
static double med(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto ni = [&](int & v) { if (i + 1 < argc) v = atoi(argv[++i]); };
        if (a == "--n") ni(o.n); else if (a == "--heads") ni(o.heads); else if (a == "--c") ni(o.c); else if (a == "--km-iters") ni(o.km_iters);
        else if (a == "--layers") ni(o.layers); else if (a == "--reps") ni(o.reps); else if (a == "--seed") ni(o.seed);
        else if (a == "--init") { if (i + 1 < argc) o.init = argv[++i]; } else if (a == "--data") { if (i + 1 < argc) o.data = argv[++i]; }
        else if (a == "--layout") { if (i + 1 < argc) o.layout_real = std::string(argv[++i]) == "real"; }
        else if (a == "-v") o.verbose = 1;
        else { printf("usage: llama-cluster-kmeans-bench [--n N] [--heads H] [--c C] [--km-iters T] [--layers L] [--reps R] [--init kpp|stride] [--data blobs|random] [--layout real|planar] [--seed S] [-v]\n"); return 1; }
    }
    if (o.c == 0) o.c = std::max(1, o.n / 64);
    if (o.d != 128 || o.n % 64 || o.c > 256 || o.n > 16384) { fprintf(stderr, "d=128, n a multiple of 64 <= 16384, c <= 256\n"); return 1; }
    const int N = o.n, H = o.heads, D = o.d, C = o.c, P = N / 64;
    const uint32_t nbk_head = D * 2, nbk_row = o.layout_real ? (uint32_t) H * D * 2 : (uint32_t) D * 2;
    const size_t kv_bytes = (size_t) N * H * D * 2;

    // ---- synthetic keys: C blobs (random centres + noise) or uniform noise
    std::mt19937 rng(o.seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<uint16_t> K(kv_bytes / 2), V(kv_bytes / 2);
    std::vector<int> truth((size_t) N * H);
    {
        std::vector<float> centres((size_t) H * C * D);
        for (auto & x : centres) x = g(rng) * 2.0f;
        std::uniform_int_distribution<int> pick(0, C - 1);
        for (int h = 0; h < H; ++h) for (int n = 0; n < N; ++n) {
            const int t = pick(rng); truth[(size_t) h * N + n] = t;
            const size_t off = o.layout_real ? ((size_t) n * H + h) * D : ((size_t) h * N + n) * D;
            for (int d = 0; d < D; ++d) {
                const float kx = o.data == "blobs" ? centres[((size_t) h * C + t) * D + d] + 0.5f * g(rng) : g(rng);
                K[off + d] = ggml_fp32_to_fp16(kx);
                V[off + d] = ggml_fp32_to_fp16(g(rng));
            }
        }
    }
    auto key = [&](int h, int n, int d) { return ggml_fp16_to_fp32(K[(o.layout_real ? ((size_t) n * H + h) * D : ((size_t) h * N + n) * D) + d]); };

    // ---- init centroids (CPU): k-means++ on a 256-key subsample, or stride
    std::vector<float> mu0((size_t) H * C * D);
    for (int h = 0; h < H; ++h) {
        std::vector<int> chosen;
        if (o.init == "stride") {
            for (int c = 0; c < C; ++c) chosen.push_back((int) (((long) c * N + N / 2) / C));
        } else {
            std::vector<int> sub; const int ns = std::min(N, 256);
            for (int i = 0; i < ns; ++i) sub.push_back((int) (((long) i * N) / ns));
            std::uniform_int_distribution<int> first(0, ns - 1);
            chosen.push_back(sub[first(rng)]);
            std::vector<double> dmin(ns, 1e30);
            while ((int) chosen.size() < C) {
                const int last = chosen.back(); double tot = 0;
                for (int i = 0; i < ns; ++i) { double s = 0; for (int d = 0; d < D; ++d) { const double t = key(h, sub[i], d) - key(h, last, d); s += t * t; } dmin[i] = std::min(dmin[i], s); tot += dmin[i]; }
                std::uniform_real_distribution<double> u(0.0, tot); double r = u(rng); int pickd = ns - 1;
                for (int i = 0; i < ns; ++i) { r -= dmin[i]; if (r <= 0) { pickd = i; break; } }
                chosen.push_back(sub[pickd]);
            }
        }
        for (int c = 0; c < C; ++c) for (int d = 0; d < D; ++d) mu0[((size_t) h * C + c) * D + d] = key(h, chosen[c], d);
    }

    // ---- OpenCL
    cl_int err; cl_uint nn = 0; cl_platform_id plat; cl_device_id dev;
    if (!cl_ok(clGetPlatformIDs(1, &plat, &nn), "platform") || !cl_ok(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nn), "device")) return 1;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    char opt[128]; snprintf(opt, sizeof(opt), "-cl-std=CL2.0 -DKM_D=%d -DKM_TILE=16 -DKM_NMAX=%d", D, N);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_src, nullptr, &err);
    if (clBuildProgram(prog, 1, &dev, opt, nullptr, nullptr) != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln); std::string log(ln, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ln, log.data(), nullptr); fprintf(stderr, "build failed:\n%s\n", log.c_str()); return 1;
    }
    cl_kernel k_assign = clCreateKernel(prog, "km_assign", &err), k_update = clCreateKernel(prog, "km_update", &err);
    cl_kernel k_gather = clCreateKernel(prog, "km_gather", &err), k_mean = clCreateKernel(prog, "km_page_mean", &err);
    if (!k_assign || !k_update || !k_gather || !k_mean) { fprintf(stderr, "kernel create failed\n"); return 1; }

    const size_t head_stride = (size_t) P * 64 * D * 2;
    cl_mem bK = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kv_bytes, K.data(), &err);
    cl_mem bV = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kv_bytes, V.data(), &err);
    cl_mem bMu = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * C * D * 4, nullptr, &err);
    cl_mem bMu2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * C * 4, nullptr, &err);
    cl_mem bCnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * C * 4, nullptr, &err);
    cl_mem bAs = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * N, nullptr, &err);
    cl_mem bDist = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * N * 4, nullptr, &err);
    cl_mem bChg = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, nullptr, &err);
    cl_mem bPerm = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t) N * 4, nullptr, &err);
    cl_mem bPK = clCreateBuffer(ctx, CL_MEM_READ_WRITE, head_stride * H, nullptr, &err);
    cl_mem bPV = clCreateBuffer(ctx, CL_MEM_READ_WRITE, head_stride * H, nullptr, &err);
    cl_mem bCent = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) H * P * D * 2, nullptr, &err);
    if (!bK || !bV || !bMu || !bMu2 || !bCnt || !bAs || !bDist || !bChg || !bPerm || !bPK || !bPV || !bCent) { fprintf(stderr, "buffer alloc failed\n"); return 1; }

    std::vector<float> mu2_0((size_t) H * C);
    for (size_t i = 0; i < mu2_0.size(); ++i) { double s = 0; for (int d = 0; d < D; ++d) s += (double) mu0[i * D + d] * mu0[i * D + d]; mu2_0[i] = (float) s; }
    const cl_uint uN = N, uC = C, uNb = nbk_row, uNh = nbk_head, uP = P, uCs = (cl_uint) P * D; const cl_ulong uHs = head_stride;
    std::vector<uint8_t> assign_h((size_t) H * N), zeros_as((size_t) H * N, 0xff);
    std::vector<float> dist_h((size_t) H * N);
    std::vector<uint32_t> perm_h(N);

    std::vector<double> t_assign, t_update, t_gather, t_mean, t_sort, t_total, iters_used;
    double inertia_gpu = 0, inertia_cpu = 0;
    for (int rep = -1; rep < o.reps; ++rep) {
        // reset centroids and assignments
        clEnqueueWriteBuffer(q, bMu, CL_TRUE, 0, mu0.size() * 4, mu0.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(q, bMu2, CL_TRUE, 0, mu2_0.size() * 4, mu2_0.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(q, bAs, CL_TRUE, 0, zeros_as.size(), zeros_as.data(), 0, nullptr, nullptr);
        double ta = 0, tu = 0; int it_used = 0;
        for (int it = 0; it < o.km_iters; ++it) {
            const cl_uint zero = 0; clEnqueueWriteBuffer(q, bChg, CL_FALSE, 0, 4, &zero, 0, nullptr, nullptr);
            int a = 0;
            clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bK); clSetKernelArg(k_assign, a++, 4, &uNb); clSetKernelArg(k_assign, a++, 4, &uNh);
            clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bMu); clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bMu2);
            clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bAs); clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bDist); clSetKernelArg(k_assign, a++, sizeof(cl_mem), &bChg);
            clSetKernelArg(k_assign, a++, 4, &uN); clSetKernelArg(k_assign, a++, 4, &uC);
            size_t g1[2] = { (size_t) N, (size_t) H }, l1[2] = { 64, 1 }; cl_event e1, e2;
            if (!cl_ok(clEnqueueNDRangeKernel(q, k_assign, 2, nullptr, g1, l1, 0, nullptr, &e1), "assign")) return 1;
            a = 0;
            clSetKernelArg(k_update, a++, sizeof(cl_mem), &bK); clSetKernelArg(k_update, a++, 4, &uNb); clSetKernelArg(k_update, a++, 4, &uNh);
            clSetKernelArg(k_update, a++, sizeof(cl_mem), &bAs); clSetKernelArg(k_update, a++, sizeof(cl_mem), &bDist);
            clSetKernelArg(k_update, a++, sizeof(cl_mem), &bMu); clSetKernelArg(k_update, a++, sizeof(cl_mem), &bMu2); clSetKernelArg(k_update, a++, sizeof(cl_mem), &bCnt);
            clSetKernelArg(k_update, a++, 4, &uN); clSetKernelArg(k_update, a++, 4, &uC);
            size_t g2[3] = { (size_t) D, (size_t) C, (size_t) H }, l2[3] = { (size_t) D, 1, 1 };
            if (!cl_ok(clEnqueueNDRangeKernel(q, k_update, 3, nullptr, g2, l2, 0, nullptr, &e2), "update")) return 1;
            cl_uint changed = 0; clEnqueueReadBuffer(q, bChg, CL_TRUE, 0, 4, &changed, 0, nullptr, nullptr);
            ta += ev_ms(e1); tu += ev_ms(e2); clReleaseEvent(e1); clReleaseEvent(e2); it_used++;
            if (changed == 0) break;
        }
        // assignments + distances back; pack (CPU counting sort by (assign, dist)); this is per head but
        // the perm is shared across heads in the shadow layout -> use head 0's order for the pages of
        // every head in this bench (the sidecar packs per head; see the note in the doc).
        clEnqueueReadBuffer(q, bAs, CL_TRUE, 0, assign_h.size(), assign_h.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, bDist, CL_TRUE, 0, dist_h.size() * 4, dist_h.data(), 0, nullptr, nullptr);
        const auto ts0 = std::chrono::steady_clock::now();
        {
            std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) {
                if (assign_h[x] != assign_h[y]) return assign_h[x] < assign_h[y];
                return dist_h[x] < dist_h[y]; });
            for (int i = 0; i < N; ++i) perm_h[i] = (uint32_t) idx[i];
        }
        const double tsort = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
        clEnqueueWriteBuffer(q, bPerm, CL_TRUE, 0, perm_h.size() * 4, perm_h.data(), 0, nullptr, nullptr);
        int a = 0; cl_event e3, e4;
        clSetKernelArg(k_gather, a++, sizeof(cl_mem), &bK); clSetKernelArg(k_gather, a++, sizeof(cl_mem), &bV); clSetKernelArg(k_gather, a++, 4, &uNb); clSetKernelArg(k_gather, a++, 4, &uNh);
        clSetKernelArg(k_gather, a++, sizeof(cl_mem), &bPerm); clSetKernelArg(k_gather, a++, sizeof(cl_mem), &bPK); clSetKernelArg(k_gather, a++, sizeof(cl_mem), &bPV);
        clSetKernelArg(k_gather, a++, 4, &uP); clSetKernelArg(k_gather, a++, 8, &uHs);
        size_t g3[2] = { (size_t) P * 64, (size_t) H }, l3[2] = { 64, 1 };
        if (!cl_ok(clEnqueueNDRangeKernel(q, k_gather, 2, nullptr, g3, l3, 0, nullptr, &e3), "gather")) return 1;
        a = 0;
        clSetKernelArg(k_mean, a++, sizeof(cl_mem), &bPK); clSetKernelArg(k_mean, a++, sizeof(cl_mem), &bCent); clSetKernelArg(k_mean, a++, 4, &uP); clSetKernelArg(k_mean, a++, 8, &uHs); clSetKernelArg(k_mean, a++, 4, &uCs);
        size_t g4[3] = { (size_t) D, (size_t) P, (size_t) H }, l4[3] = { (size_t) D, 1, 1 };
        if (!cl_ok(clEnqueueNDRangeKernel(q, k_mean, 3, nullptr, g4, l4, 0, nullptr, &e4), "page_mean")) return 1;
        clFinish(q);
        if (rep < 0) {
            // quality: inertia of the GPU assignment vs a CPU Lloyd run from the same init (head 0)
            std::vector<float> mu_h((size_t) H * C * D); clEnqueueReadBuffer(q, bMu, CL_TRUE, 0, mu_h.size() * 4, mu_h.data(), 0, nullptr, nullptr);
            for (int n = 0; n < N; ++n) { const int c = assign_h[n]; double s = 0; for (int d = 0; d < D; ++d) { const double t = key(0, n, d) - mu_h[(size_t) c * D + d]; s += t * t; } inertia_gpu += s; }
            std::vector<double> cm(C * (size_t) D); for (size_t i = 0; i < cm.size(); ++i) cm[i] = mu0[i];
            std::vector<int> as(N, -1);
            for (int it = 0; it < o.km_iters; ++it) {
                bool ch = false;
                for (int n = 0; n < N; ++n) { double bd = 1e300; int bi = 0; for (int c = 0; c < C; ++c) { double s = 0; for (int d = 0; d < D; ++d) { const double t = key(0, n, d) - cm[(size_t) c * D + d]; s += t * t; } if (s < bd) { bd = s; bi = c; } } if (as[n] != bi) { as[n] = bi; ch = true; } }
                if (!ch) break;
                std::vector<double> sum(C * (size_t) D, 0.0); std::vector<int> cnt(C, 0);
                for (int n = 0; n < N; ++n) { cnt[as[n]]++; for (int d = 0; d < D; ++d) sum[(size_t) as[n] * D + d] += key(0, n, d); }
                for (int c = 0; c < C; ++c) if (cnt[c]) for (int d = 0; d < D; ++d) cm[(size_t) c * D + d] = sum[(size_t) c * D + d] / cnt[c];
            }
            for (int n = 0; n < N; ++n) { double s = 0; for (int d = 0; d < D; ++d) { const double t = key(0, n, d) - cm[(size_t) as[n] * D + d]; s += t * t; } inertia_cpu += s; }
            // gather check: page rows are the permuted keys
            std::vector<uint8_t> pk(head_stride); clEnqueueReadBuffer(q, bPK, CL_TRUE, 0, head_stride, pk.data(), 0, nullptr, nullptr);
            int bad = 0; for (int s = 0; s < N; ++s) { const uint32_t n = perm_h[s]; if (memcmp(pk.data() + (size_t) s * D * 2, K.data() + (o.layout_real ? ((size_t) n * H) * D : (size_t) n * D), D * 2)) bad++; }
            if (bad) printf("gather check: %d of %d rows wrong\n", bad, N);
            clReleaseEvent(e3); clReleaseEvent(e4);
            continue;
        }
        t_assign.push_back(ta); t_update.push_back(tu); t_sort.push_back(tsort); t_gather.push_back(ev_ms(e3)); t_mean.push_back(ev_ms(e4)); iters_used.push_back(it_used);
        t_total.push_back(ta + tu + tsort + ev_ms(e3) + ev_ms(e4));
        clReleaseEvent(e3); clReleaseEvent(e4);
    }
    const double per_layer = med(t_total);
    const double flops = 2.0 * N * C * D * med(iters_used) * H;
    printf("k-means bench: N %d keys x D %d, C %d centroids, %d heads (one layer's chunk), init %s, data %s, layout %s, %d reps\n",
           N, D, C, H, o.init.c_str(), o.data.c_str(), o.layout_real ? "real [pos][head][D]" : "planar", o.reps);
    printf("  per layer (median): assign %.3f ms (%.0f iters), update %.3f ms, CPU sort %.3f ms, gather %.3f ms, page mean %.3f ms -> total %.3f ms (%.1f GFLOP/s on the assignment)\n",
           med(t_assign), med(iters_used), med(t_update), med(t_sort), med(t_gather), med(t_mean), per_layer, flops / (med(t_assign) * 1e-3) / 1e9);
    printf("  per chunk of %d layers: %.1f ms  (hide budget: HTP prefill ~20 ms per layer per ub=1024 ubatch => %.0f ms per chunk)\n", o.layers, per_layer * o.layers, 20.0 * o.layers);
    printf("  quality (head 0): inertia GPU %.4g vs CPU Lloyd from the same init %.4g (ratio %.3f)\n", inertia_gpu, inertia_cpu, inertia_cpu > 0 ? inertia_gpu / inertia_cpu : 0.0);
    return 0;
}
