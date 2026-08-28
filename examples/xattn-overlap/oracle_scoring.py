#!/usr/bin/env python3
"""How close can a cheap block scorer get to the oracle?

Block-sparse attention selects u KV blocks of `bs` tokens per (query block, KV head).
The ORACLE ranks those blocks by the attention mass they actually carry -- it is the
best any selector could do at that budget, and it is not implementable (it needs the
attention it is trying to avoid computing). Everything else is an approximation to it.

This measures the gap. For each scorer we report RECALL: the fraction of true attention
mass that lands inside the selected blocks, averaged over query rows and heads. Recall
is the right metric because it is what block-sparse attention actually loses -- a row
whose mass is 95% inside the selection computes a softmax over 95% of its own
distribution.

Captures post-RoPE Q/K from a real prefill by patching apply_rotary_pos_emb, so the
scorers see exactly what the kernel would see.
"""
import argparse, json, math, os, sys, time
import torch

# --------------------------------------------------------------------------- capture

ROPE_MODULE = {
    'qwen3': 'transformers.models.qwen3.modeling_qwen3',
    'qwen2': 'transformers.models.qwen2.modeling_qwen2',
    'llama': 'transformers.models.llama.modeling_llama',
    'mistral': 'transformers.models.mistral.modeling_mistral',
}

class Capturer:
    """Holds a model once and yields post-RoPE Q/K for each prompt.

    The hook is on apply_rotary_pos_emb rather than on the attention module, because
    that is the exact tensor the scorer would see in a runtime: after q_norm/k_norm and
    after RoPE, before the attention interface. Which module owns that symbol depends on
    the architecture, hence the table above.
    """
    def __init__(self, model_id, device='cuda', dtype=torch.bfloat16):
        import importlib
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        mt = cfg.model_type
        if mt not in ROPE_MODULE:
            raise SystemExit(f'no RoPE hook registered for model_type={mt}; add one to ROPE_MODULE')
        self.M = importlib.import_module(ROPE_MODULE[mt])
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device, attn_implementation='sdpa').eval()
        self.device = device
        self.name = model_id
        self.geom = (cfg.num_attention_heads, cfg.num_key_value_heads,
                     getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads),
                     cfg.num_hidden_layers)

    def __call__(self, text, n_tokens, layers=None):
        ids = self.tok(text, return_tensors='pt').input_ids[:, :n_tokens].to(self.device)
        T = ids.shape[1]
        grabbed, orig = [], self.M.apply_rotary_pos_emb
        def patched(q, k, cos, sin, *a, **kw):
            qr, kr = orig(q, k, cos, sin, *a, **kw)
            grabbed.append((qr.detach()[0].clone(), kr.detach()[0].clone()))
            return qr, kr
        self.M.apply_rotary_pos_emb = patched
        try:
            with torch.no_grad():
                self.model(ids, use_cache=False)
        finally:
            self.M.apply_rotary_pos_emb = orig
        if layers is not None:
            grabbed = [grabbed[i] for i in layers]
        return grabbed, T


def load_texts(spec, n, min_chars=20000):
    """spec is `ruler:<task>` or `longbench:<subset>`; returns up to n prompt strings.

    RULER instances are already length-calibrated, so context+question is used verbatim.
    LongBench documents vary wildly, so short ones are skipped -- a document that cannot
    fill the context would silently make every policy look identical.
    """
    from datasets import load_dataset
    kind, name = spec.split(':', 1)
    if kind == 'ruler':
        d = load_dataset('simonjegou/ruler', '8192', split='test')
        d = d.filter(lambda r: r['task'] == name)
        return [r['context'] + '\n' + r['question'] for r in d.select(range(min(n, len(d))))]
    if kind == 'longbench':
        d = load_dataset('THUDM/LongBench', name, split='test')
        out = []
        for r in d:
            if len(r['context']) >= min_chars:
                out.append(r['context'] + '\n' + r.get('input', ''))
            if len(out) >= n:
                break
        return out
    raise SystemExit(f'unknown data spec {spec}')


# ------------------------------------------------------------------- oracle + scorers
# Every scorer returns a score tensor [NBq, Hkv, NBk]; higher = more wanted.
# `q` is [Hq,T,d], `k` is [Hkv,T,d], both post-RoPE. G = Hq // Hkv.

def _blocks(T, bs):
    return (T + bs - 1) // bs

def oracle_and_mass(q, k, bs, dtype=torch.float32, qchunk=8):
    """Exact block attention mass.

    Returns mass [NBq, Hkv, NBk] -- for each (query block, kv head, kv block) the total
    probability mass the block receives, summed over the G query heads sharing that KV
    head and over the rows of the query block, with each row's distribution normalised
    to 1 first. Also returns row_total [NBq, Hkv] = number of (row, head) pairs, so
    recall can be computed as captured / row_total.
    """
    Hq, T, d = q.shape
    Hkv = k.shape[0]
    G, NBq, NBk = Hq // Hkv, _blocks(T, bs), _blocks(T, bs)
    scale = 1.0 / math.sqrt(d)
    mass = torch.zeros(NBq, Hkv, NBk, device=q.device, dtype=dtype)
    rows = torch.zeros(NBq, Hkv, device=q.device, dtype=dtype)

    qv = q.view(Hkv, G, T, d)                      # query head h = kv_head*G + j
    for b0 in range(0, NBq, qchunk):
        b1 = min(b0 + qchunk, NBq)
        s, e = b0 * bs, min(b1 * bs, T)
        qs = qv[:, :, s:e, :]                                       # [Hkv,G,n,d]
        logits = torch.einsum('hgnd,hmd->hgnm', qs.to(dtype), k.to(dtype)) * scale
        pos = torch.arange(s, e, device=q.device).view(1, 1, -1, 1)
        key = torch.arange(T, device=q.device).view(1, 1, 1, -1)
        logits = logits.masked_fill(key > pos, float('-inf'))
        p = torch.softmax(logits, dim=-1)                           # [Hkv,G,n,T]
        # pool keys into blocks
        pad = NBk * bs - T
        if pad:
            p = torch.nn.functional.pad(p, (0, pad))
        pb = p.view(Hkv, G, e - s, NBk, bs).sum(-1)                 # [Hkv,G,n,NBk]
        # pool query rows into their blocks
        for bi in range(b0, b1):
            rs, re = bi * bs - s, min((bi + 1) * bs, T) - s
            if re <= rs:
                continue
            mass[bi] = pb[:, :, rs:re, :].sum(dim=(1, 2))           # [Hkv,NBk]
            rows[bi] = float((re - rs) * G)
        del logits, p, pb
    return mass, rows


def sc_oracle(mass, **kw):
    return mass.clone()

def sc_xattn(q, k, bs, S=16, **kw):
    """XAttention: antidiagonal packing at stride S, softmax, block-pool."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv
    NB, P = _blocks(T, bs), bs // S
    Tp = NB * bs
    qi = torch.arange(Tp, device=q.device)
    rev = (qi // S) * S + (S - 1 - qi % S)                       # antidiagonal on Q
    qz = torch.zeros(Hq, Tp, d, device=q.device, dtype=q.dtype); qz[:, :T] = q
    kz = torch.zeros(Hkv, Tp, d, device=k.device, dtype=k.dtype); kz[:, :T] = k
    qp = qz[:, rev, :].reshape(Hkv, G, Tp // S, S * d).float()   # [Hkv,G,Nq,S*d]
    kp = kz.reshape(Hkv, Tp // S, S * d).float()                 # [Hkv,Nk,S*d]
    # Score PER QUERY HEAD -- XAttention does not pre-aggregate over the GQA group;
    # that is QUOKA's idea, and averaging unnormalised q across heads before the
    # matmul costs this scorer most of its accuracy.
    sc = torch.einsum('hgnc,hmc->hgnm', qp, kp) / (math.sqrt(d) * S)
    ii = torch.arange(sc.shape[2], device=q.device).view(-1, 1)
    jj = torch.arange(sc.shape[3], device=q.device).view(1, -1)
    sc = sc.masked_fill(jj > ii, float('-inf'))
    sc = torch.softmax(sc, dim=-1)
    sc = sc.view(Hkv, G, NB, P, NB, P).sum((3, 5))               # [Hkv,G,NBq,NBk]
    return sc.amax(1).permute(1, 0, 2).contiguous()              # max over the G heads

def sc_quoka(q, k, bs, NQ=16, **kw):
    """QUOKA: keep the NQ lowest-cos-sim-to-mean queries, GQA pre-aggregate, cosine, max."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv; NB = _blocks(T, bs)
    kn = torch.nn.functional.normalize(k.float(), dim=-1)
    out = torch.empty(NB, Hkv, NB, device=q.device, dtype=torch.float32)
    for bi in range(NB):
        s, e = bi * bs, min((bi + 1) * bs, T)
        qs = q[:, s:e, :].float()                                # [Hq,n,d]
        mq = qs.mean(1, keepdim=True)
        cs = torch.nn.functional.cosine_similarity(qs, mq, dim=-1)   # [Hq,n]
        n = qs.shape[1]
        idx = (-cs).topk(min(NQ, n), dim=-1).indices                 # lowest cos-sim
        sel = torch.gather(qs, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        sel = torch.nn.functional.normalize(sel, dim=-1)
        qbar = sel.view(Hkv, G, -1, d).mean(1)                       # [Hkv,nq,d]
        sc = torch.einsum('hnd,hmd->hnm', qbar, kn)                  # [Hkv,nq,T]
        pad = NB * bs - T
        if pad: sc = torch.nn.functional.pad(sc, (0, pad), value=-1e30)
        out[bi] = sc.view(Hkv, -1, NB, bs).amax(-1).amax(1)          # max over q, max in block
    return out

def sc_meanpool(q, k, bs, **kw):
    """Cheapest sane baseline: block-mean of Q dotted with block-mean of K."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv; NB = _blocks(T, bs)
    pad = NB * bs - T
    qz = torch.nn.functional.pad(q.float(), (0, 0, 0, pad))
    kz = torch.nn.functional.pad(k.float(), (0, 0, 0, pad))
    qb = qz.view(Hkv, G, NB, bs, d).mean(3).mean(1)               # [Hkv,NB,d]
    kb = kz.view(Hkv, NB, bs, d).mean(2)                          # [Hkv,NB,d]
    return torch.einsum('hnd,hmd->hnm', qb, kb).permute(1, 0, 2).contiguous() / math.sqrt(d)

def sc_maxpool(q, k, bs, **kw):
    """Block-max of the raw logits -- an upper envelope rather than an average."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv; NB = _blocks(T, bs)
    out = torch.empty(NB, Hkv, NB, device=q.device, dtype=torch.float32)
    for bi in range(NB):
        s, e = bi * bs, min((bi + 1) * bs, T)
        qs = q[:, s:e, :].float().view(Hkv, G, -1, d)
        sc = torch.einsum('hgnd,hmd->hgnm', qs, k.float()) / math.sqrt(d)
        pad = NB * bs - T
        if pad: sc = torch.nn.functional.pad(sc, (0, pad), value=-1e30)
        out[bi] = sc.view(Hkv, G, -1, NB, bs).amax(-1).amax(2).amax(1)
    return out

def sc_recent(q, k, bs, **kw):
    """Position-only: sink block + most recent blocks. No data dependence at all."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; NB = _blocks(T, bs)
    j = torch.arange(NB, device=q.device).float()
    sc = j.view(1, 1, -1).expand(NB, Hkv, NB).clone()
    sc[:, :, 0] = 1e6                                             # sink always wins
    return sc

def sc_random(q, k, bs, **kw):
    """Random fill. With sink+diagonal forced on top, this isolates exactly what those
    two blocks buy -- everything above this line is what SCORING is worth."""
    Hq, T, d = q.shape; Hkv = k.shape[0]; NB = _blocks(T, bs)
    g = torch.Generator(device=q.device); g.manual_seed(0)
    return torch.rand(NB, Hkv, NB, device=q.device, generator=g)

def sc_quoka_strided(q, k, bs, NQ=16, **kw):
    """QUOKA with the cosine query ranking replaced by an evenly spaced sample.

    This is the configuration that wins on the NPU (690 us against meanpool's 2444 and
    XAttention's 7477 at Lq=2048/Lk=4096), because it never reads most of Q. Its cost is
    independent of both the chunk and the cache. Its QUALITY has never been measured --
    that is what this scorer exists for. It keeps QUOKA's l2 normalisation, GQA
    pre-aggregation and max-over-queries, and discards only the claim that low
    cosine-similarity-to-mean queries are the informative ones.
    """
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv; NB = _blocks(T, bs)
    kn = torch.nn.functional.normalize(k.float(), dim=-1)
    out = torch.empty(NB, Hkv, NB, device=q.device, dtype=torch.float32)
    step = max(1, bs // NQ)
    for bi in range(NB):
        s, e = bi * bs, min((bi + 1) * bs, T)
        sel = q[:, s:e:step, :].float()                              # [Hq,<=NQ,d]
        sel = torch.nn.functional.normalize(sel, dim=-1)
        qbar = sel.view(Hkv, G, -1, d).mean(1)
        sc = torch.einsum('hnd,hmd->hnm', qbar, kn)
        pad = NB * bs - T
        if pad: sc = torch.nn.functional.pad(sc, (0, pad), value=-1e30)
        out[bi] = sc.view(Hkv, -1, NB, bs).amax(-1).amax(1)
    return out

def sc_meanpool_s4(q, k, bs, **kw):
    return sc_meanpool_sub(q, k, bs, QSUB=4)

def sc_meanpool_s2(q, k, bs, **kw):
    return sc_meanpool_sub(q, k, bs, QSUB=2)

def sc_meanpool_sub(q, k, bs, QSUB=8, **kw):
    """meanpool over a subsample: mean of QSUB evenly spaced rows per block, not all bs.

    The synthesis the NPU numbers point at -- meanpool's cost is entirely reading Q, so
    reading 1/8 of it is an 8x cut in the only term that costs anything.
    """
    Hq, T, d = q.shape; Hkv = k.shape[0]; G = Hq // Hkv; NB = _blocks(T, bs)
    pad = NB * bs - T
    qz = torch.nn.functional.pad(q.float(), (0, 0, 0, pad))
    kz = torch.nn.functional.pad(k.float(), (0, 0, 0, pad))
    step = max(1, bs // QSUB)
    qb = qz.view(Hkv, G, NB, bs, d)[:, :, :, ::step, :].mean(3).mean(1)   # [Hkv,NB,d]
    kb = kz.view(Hkv, NB, bs, d).mean(2)
    return torch.einsum('hnd,hmd->hnm', qb, kb).permute(1, 0, 2).contiguous() / math.sqrt(d)

SCORERS = {'xattn': sc_xattn, 'quoka_str': sc_quoka_strided, 'meanpool_s8': sc_meanpool_sub, 'meanpool_s4': sc_meanpool_s4, 'meanpool_s2': sc_meanpool_s2, 'quoka': sc_quoka, 'meanpool': sc_meanpool,
           'maxpool': sc_maxpool, 'recent': sc_recent, 'random': sc_random}


def force_sink_diag(scores):
    """Force block 0 (attention sink) and block a (the diagonal) into every selection.

    Not a thumb on the scale -- it is what every deployment does. XAttention's
    find_blocks force-includes both unconditionally, and the llama.cpp pipeline carries
    them in its bias leaf, because a query block whose selection excludes its own
    diagonal has no causally valid keys at all in its newest rows. Applying it uniformly
    to every scorer is what makes them comparable; the `sinkdiag` scorer isolates how
    much recall it buys on its own.
    """
    NBq, Hkv, NBk = scores.shape
    out = scores.clone()
    big = 1e9
    out[:, :, 0] = big
    idx = torch.arange(NBq, device=scores.device)
    out[idx, :, idx] = big + 1.0                                  # diagonal outranks sink
    return out


# ------------------------------------------------------------------------- evaluation

def merge_rows(scores, k, how='amax'):
    """Collapse k adjacent query blocks onto one shared list, as a chunk-wide selection does.

    The merged row must serve every block in the group, so its causal reach is the LAST
    block's -- earlier queries simply mask the extra blocks off, which is what the kernel
    does anyway. Recall is still scored per original query block against that block's own
    mass, so this measures exactly what coarsening costs.
    """
    if k <= 1:
        return scores
    NBq, Hkv, NBk = scores.shape
    pad = (-NBq) % k
    if pad:
        scores = torch.cat([scores, scores[-1:].expand(pad, -1, -1)], 0)
    g = scores.view(-1, k, Hkv, NBk)
    m = g.amax(1) if how == 'amax' else g.sum(1)
    return m.repeat_interleave(k, dim=0)[:NBq]


def recall_at(scores, mass, rows, u, bs, slack=2.0):
    """Fraction of true mass captured by the top-u blocks of `scores`.

    Causally impossible blocks are excluded from both the ranking and the denominator:
    a query block a can only reach blocks 0..a, so n_avail = a+1, and rows where
    n_avail <= u have no choice to make and are skipped (every policy is identical
    there, so including them would compress all differences toward 1).
    """
    NBq, Hkv, NBk = mass.shape
    num = den = 0.0
    for a in range(NBq):
        avail = a + 1
        # `slack` blocks of genuine choice, not just avail > u: at avail = u+1 every
        # policy picks u of u+1 and scores ~1.0, and those rows would otherwise
        # dominate the average and compress every difference toward the oracle.
        if avail < slack * u:
            continue
        sc = scores[a, :, :avail]
        idx = sc.topk(u, dim=-1).indices                          # [Hkv,u]
        got = torch.gather(mass[a, :, :avail], 1, idx).sum()
        num += got.item()
        den += rows[a].sum().item()
    return num / den if den else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--data', nargs='+', default=['longbench:narrativeqa'],
                    help='ruler:<task> or longbench:<subset>')
    ap.add_argument('--docs', type=int, default=3, help='instances per dataset')
    ap.add_argument('--tokens', type=int, default=8192)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--layers', type=int, nargs='*', default=None)
    ap.add_argument('--densities', type=float, nargs='*', default=[0.0625, 0.125, 0.25, 0.5])
    ap.add_argument('--force', type=int, default=1)
    ap.add_argument('--slack', type=float, default=2.0)
    ap.add_argument('--merge', type=int, nargs='*', default=[1],
                    help='k adjacent query blocks sharing one list (1 = per-block)')
    ap.add_argument('--out', default='oracle_scoring.json')
    a = ap.parse_args()

    cap = Capturer(a.model)
    Hq, Hkv, d, L = cap.geom
    print(f'{a.model}: Hq={Hq} Hkv={Hkv} d={d} layers={L}', flush=True)

    acc = {}          # (data, density, scorer) -> [ratios]
    absacc = {}       # (data, density, scorer) -> [recalls]
    for spec in a.data:
        texts = load_texts(spec, a.docs)
        print(f'\n### {spec}: {len(texts)} instances', flush=True)
        for di, text in enumerate(texts):
            qk, T = cap(text, a.tokens, a.layers)
            NB = _blocks(T, a.bs)
            if NB < 8:
                print(f'  doc {di}: only T={T}, skipped', flush=True); continue
            for li, (q, k) in enumerate(qk):
                mass, rows = oracle_and_mass(q, k, a.bs)
                sc = {'oracle': sc_oracle(mass)}
                for n, f in SCORERS.items():
                    sc[n] = f(q, k, a.bs)
                if a.force:
                    sc = {n: (v if n == 'oracle' else force_sink_diag(v)) for n, v in sc.items()}
                for mk in a.merge:
                    # The oracle is merged too, by SUM of mass -- the best list a group of
                    # k blocks could share. Comparing a merged scorer against the per-block
                    # oracle would charge it for coarsening twice.
                    scm = {n: merge_rows(v, mk, 'sum' if n == 'oracle' else 'amax')
                           for n, v in sc.items()}
                    for dens in a.densities:
                        u = max(1, int(round(dens * NB)))
                        r = {n: recall_at(v, mass, rows, u, a.bs, a.slack) for n, v in scm.items()}
                        if r['oracle'] != r['oracle'] or r['oracle'] <= 0:
                            continue
                        key = (spec, dens, mk)
                        for n, v in r.items():
                            acc.setdefault(key + (n,), []).append(v / r['oracle'])
                            absacc.setdefault(key + (n,), []).append(v)
                del mass, rows, sc
                torch.cuda.empty_cache()
            print(f'  doc {di} (T={T}) done', flush=True)

    names = ['oracle'] + list(SCORERS)
    print(f'\n=== {a.model}, bs={a.bs}, T={a.tokens}, mean over layers x instances ===')
    hdr = f"{'dataset':<22}{'merge':>5}{'dens':>6}{'oracle':>8}" + ''.join(f'{n:>12}' for n in names[1:])
    print(hdr); print('-' * len(hdr))
    for spec in a.data:
      for mk in a.merge:
        for dens in a.densities:
            k0 = (spec, dens, mk, 'oracle')
            if k0 not in absacc: continue
            o = sum(absacc[k0]) / len(absacc[k0])
            cells = ''.join(f'{100*sum(acc[(spec,dens,mk,n)])/len(acc[(spec,dens,mk,n)]):9.1f}%'
                            for n in names[1:])
            print(f'{spec:<22}{"k="+str(mk):>5}{dens:>6}{o:>8.3f}{cells}')
    out = {'model': a.model, 'T': a.tokens, 'bs': a.bs, 'force': a.force,
           'ratio': {f'{s}|{d}|{m}|{n}': sum(v)/len(v) for (s,d,m,n), v in acc.items()},
           'recall': {f'{s}|{d}|{m}|{n}': sum(v)/len(v) for (s,d,m,n), v in absacc.items()},
           'n': {f'{s}|{d}|{m}|{n}': len(v) for (s,d,m,n), v in acc.items()}}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}')

if __name__ == '__main__':
    main()
