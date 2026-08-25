// Measure the real overlap between adjacent query blocks' XAttention selections.
//
// Everything else about block-sparse attention on HTP is now characterised, but the
// deployment choice between an exact per-scorer-block selection (Bq = Bl) and a cheaper
// unioned one (Bq = R*Bl) turns on ONE unmeasured quantity: how much do adjacent query
// blocks' selections actually have in common? The cost surface says Bq=128 wins at
// f >= 0.5 and Bq=256 at f >= 0.75
// (docs/backend/snapdragon/sparse-attn-inkernel-vs-gather.md).
//
// It cannot be measured from random tensors -- random Q/K give near-uniform softmax
// scores and essentially random top-k, which understates overlap badly. So this runs a
// real prefill, captures the post-RoPE Qcur/Kcur of every layer through the eval
// callback, and runs the real XAttention estimator over them.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct layer_qk {
    std::vector<float> q;   // [d, Hq, T]
    std::vector<float> k;   // [d, Hkv, T]
    int d = 0, hq = 0, hkv = 0, t = 0;
};

static std::map<int,double> g_union_sum;
static std::map<int,long>   g_union_n;
static int                  keep_g = 0;

struct capture {
    std::map<int, layer_qk> layers;
};

// Qcur-<L> / Kcur-<L> after ROPE are the tensors the scorer would see.
static bool cb_capture(struct ggml_tensor * t, bool ask, void * user_data) {
    capture * c = (capture *) user_data;
    const char * nm = t->name;
    const bool isq = strncmp(nm, "Qcur-", 5) == 0;
    const bool isk = strncmp(nm, "Kcur-", 5) == 0;
    // the ROPE output is the last write of that name with 3 live dims
    if (!(isq || isk) || t->op != GGML_OP_ROPE) {
        return ask ? false : true;
    }
    if (ask) {
        return true;   // yes, we want this one's data
    }
    const int layer = atoi(nm + 5);
    const int d = t->ne[0], h = t->ne[1], T = t->ne[2];
    std::vector<float> buf((size_t) d * h * T);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    layer_qk & L = c->layers[layer];
    L.d = d; L.t = T;
    if (isq) { L.hq  = h; L.q = std::move(buf); }
    else     { L.hkv = h; L.k = std::move(buf); }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 0;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    // Bl and S are read from the environment so a sweep needs no rebuild.
    const int Bl = getenv("XA_BL")   ? atoi(getenv("XA_BL"))   : 64;
    const int S  = getenv("XA_S")    ? atoi(getenv("XA_S"))    : 16;
    const int DEN= getenv("XA_DEN")  ? atoi(getenv("XA_DEN"))  : 4;   // keep nblk/DEN blocks

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    capture cap;
    params.cb_eval = cb_capture;
    params.cb_eval_user_data = &cap;
    params.warmup = false;

    auto init = common_init_from_params(params);
    if (!init->context()) { LOG_ERR("failed to load model\n"); return 1; }
    llama_context * ctx = init->context();

    auto tokens = common_tokenize(ctx, params.prompt, true, true);
    if ((int) tokens.size() < 4 * Bl) {
        LOG_ERR("prompt too short: %zu tokens, need >= %d\n", tokens.size(), 4 * Bl);
        return 1;
    }
    LOG_INF("prefill of %zu tokens\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("decode failed\n"); return 1;
    }

    // ---- the XAttention estimator, per (layer, kv head) ----
    // Reference: reshaped_key packs S consecutive keys into the feature dim; reshaped_query
    // does the same with the S positions REVERSED, so the offsets always sum to S-1 --
    // the antidiagonal of the S x S sub-tile.
    double sum_f = 0.0; long n_f = 0;
    std::vector<double> per_layer_f;

    for (auto & kv : cap.layers) {
        layer_qk & L = kv.second;
        if (L.q.empty() || L.k.empty()) continue;
        const int d = L.d, T = L.t, Hq = L.hq, Hkv = L.hkv, G = Hq / Hkv;
        const int N   = T / S;          // reduced rows
        const int P   = Bl / S;         // reduced rows per block
        const int NB  = T / Bl;
        if (N < 2 || P < 1 || NB < 4) continue;
        const int keep = std::max(1, NB / DEN);

        double layer_f = 0.0; long layer_n = 0;
        for (int hk = 0; hk < Hkv; ++hk) {
            // sum the G query heads sharing this KV head -- sel is per KV head
            std::vector<double> attn_sum((size_t) NB * NB, 0.0);
            for (int g = 0; g < G; ++g) {
                const int hq = hk * G + g;
                std::vector<double> red((size_t) N * N, 0.0);
                for (int rq = 0; rq < N; ++rq) {
                    for (int rk = 0; rk <= rq; ++rk) {          // causal on the reduced grid
                        double acc = 0.0;
                        for (int s = 0; s < S; ++s) {
                            const int qt = rq * S + (S - 1 - s);
                            const int kt = rk * S + s;
                            const float * qp = &L.q[((size_t) qt * Hq + hq) * d];
                            const float * kp = &L.k[((size_t) kt * Hkv + hk) * d];
                            for (int e = 0; e < d; ++e) acc += (double) qp[e] * kp[e];
                        }
                        red[(size_t) rq * N + rk] = acc / (std::sqrt((double) d) * S);
                    }
                }
                for (int rq = 0; rq < N; ++rq) {               // softmax over rk <= rq
                    double mx = -1e300;
                    for (int rk = 0; rk <= rq; ++rk) mx = std::max(mx, red[(size_t) rq*N+rk]);
                    double z = 0.0;
                    for (int rk = 0; rk <= rq; ++rk) { double e = std::exp(red[(size_t)rq*N+rk]-mx); red[(size_t)rq*N+rk]=e; z+=e; }
                    for (int rk = 0; rk <= rq; ++rk) red[(size_t)rq*N+rk] /= z;
                    for (int rk = rq+1; rk < N; ++rk) red[(size_t)rq*N+rk] = 0.0;
                }
                for (int bq = 0; bq < NB; ++bq)                 // pool P x P into blocks
                    for (int bk = 0; bk < NB; ++bk) {
                        double a = 0.0;
                        for (int i = 0; i < P; ++i)
                            for (int j = 0; j < P; ++j)
                                a += red[(size_t)(bq*P+i)*N + (bk*P+j)];
                        attn_sum[(size_t) bq*NB + bk] += a;
                    }
            }
            // top-k per query block, then overlap between ADJACENT query blocks
            std::vector<std::vector<int>> sel(NB);
            for (int bq = 0; bq < NB; ++bq) {
                std::vector<int> idx;
                for (int bk = 0; bk <= bq; ++bk) idx.push_back(bk);   // causal candidates
                std::stable_sort(idx.begin(), idx.end(),
                    [&](int a, int b){ return attn_sum[(size_t)bq*NB+a] > attn_sum[(size_t)bq*NB+b]; });
                const int kk = std::min((int) idx.size(), keep);
                idx.resize(kk);
                std::sort(idx.begin(), idx.end());
                sel[bq] = idx;
            }
            // Direct union sizes: for an attention block spanning R scorer blocks, how
            // many DISTINCT blocks does it need? This is what plugs into the cost table --
            // inferring it from adjacent overlap assumes a linear decay that real
            // selections need not follow (a forced sink block, for instance, is shared by
            // ALL query blocks regardless of distance).
            for (int R : {2, 4, 8}) {
                for (int b0 = 0; b0 + R <= NB; b0 += R) {
                    bool full = true;
                    std::vector<int> uni;
                    for (int j = 0; j < R; ++j) {
                        if ((int) sel[b0+j].size() < keep) { full = false; break; }
                        uni.insert(uni.end(), sel[b0+j].begin(), sel[b0+j].end());
                    }
                    if (!full) continue;
                    std::sort(uni.begin(), uni.end());
                    uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
                    g_union_sum[R] += (double) uni.size();
                    g_union_n[R]   += 1;
                }
            }
            for (int bq = 1; bq < NB; ++bq) {
                // only meaningful where both blocks actually had `keep` candidates,
                // otherwise top-k degenerates to "everything" and overlap is trivially 1
                if ((int) sel[bq].size() < keep || (int) sel[bq-1].size() < keep) continue;
                std::vector<int> inter;
                std::set_intersection(sel[bq].begin(), sel[bq].end(),
                                      sel[bq-1].begin(), sel[bq-1].end(), std::back_inserter(inter));
                layer_f += (double) inter.size() / keep; layer_n++;
            }
        }
        if (layer_n) {
            per_layer_f.push_back(layer_f / layer_n);
            sum_f += layer_f; n_f += layer_n;
            LOG_INF("layer %2d: mean adjacent-block overlap f = %.3f  (%ld pairs)\n",
                    kv.first, layer_f / layer_n, layer_n);
        }
    }

    if (n_f) {
        std::sort(per_layer_f.begin(), per_layer_f.end());
        LOG_INF("\n==== Bl=%d S=%d keep=nblk/%d ====\n", Bl, S, DEN);
        LOG_INF("OVERALL mean adjacent-block overlap f = %.3f over %ld pairs\n", sum_f / n_f, n_f);
        LOG_INF("  per-layer min %.3f  median %.3f  max %.3f\n",
                per_layer_f.front(), per_layer_f[per_layer_f.size()/2], per_layer_f.back());
        LOG_INF("  crossover: Bq=128 needs f >= 0.50, Bq=256 needs f >= 0.75\n\n");
        LOG_INF("  MEASURED UNION SIZES (what actually plugs into the cost table):\n");
        for (int R : {2, 4, 8}) {
            if (!g_union_n[R]) continue;
            const double mu = g_union_sum[R] / g_union_n[R];
            LOG_INF("    R=%d (Bq=%d): mean |union| = %.2f of a possible %d  (%ld samples)\n",
                    R, R * Bl, mu, R * keep_g, g_union_n[R]);

        }
    } else {
        LOG_ERR("no usable layers -- prompt too short for Bl=%d?\n", Bl);
    }
    return 0;
}
