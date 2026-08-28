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

def capture_qk(model_id, text, n_tokens, device, dtype, layers=None):
    """Run a prefill and return post-RoPE q,k per layer: q[L] = [Hq,T,d], k[L] = [Hkv,T,d]."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3 as M

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device, attn_implementation="sdpa")
    model.eval()

    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)
    T = ids.shape[1]

    grabbed, orig = [], M.apply_rotary_pos_emb
    def patched(q, k, cos, sin, *a, **kw):
        qr, kr = orig(q, k, cos, sin, *a, **kw)
        grabbed.append((qr.detach()[0].clone(), kr.detach()[0].clone()))   # [H,T,d]
        return qr, kr
    M.apply_rotary_pos_emb = patched
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        M.apply_rotary_pos_emb = orig

    del model
    torch.cuda.empty_cache()
    if layers is not None:
        grabbed = [grabbed[i] for i in layers]
    return grabbed, T


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

SCORERS = {'xattn': sc_xattn, 'quoka': sc_quoka, 'meanpool': sc_meanpool,
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
    ap.add_argument('--tokens', type=int, default=8192)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--layers', type=int, nargs='*', default=None)
    ap.add_argument('--densities', type=float, nargs='*', default=[0.125, 0.25, 0.5])
    ap.add_argument('--force', type=int, default=1,
                    help='1 = force sink+diagonal into every selection (what deployments do)')
    ap.add_argument('--slack', type=float, default=2.0,
                    help='only score query blocks with avail >= slack*u available blocks')
    ap.add_argument('--out', default='oracle_scoring.json')
    ap.add_argument('--text-file', default=None)
    a = ap.parse_args()

    dev = 'cuda'
    if a.text_file:
        text = open(a.text_file).read()
    else:
        from datasets import load_dataset
        ds = load_dataset('THUDM/LongBench', 'narrativeqa', split='test')
        text = ds[0]['context']
    t0 = time.time()
    qk, T = capture_qk(a.model, text, a.tokens, dev, torch.bfloat16, a.layers)
    NB = _blocks(T, a.bs)
    print(f'captured {len(qk)} layers, T={T}, NB={NB}, {time.time()-t0:.1f}s', flush=True)

    res = {}
    for li, (q, k) in enumerate(qk):
        lname = a.layers[li] if a.layers else li
        mass, rows = oracle_and_mass(q, k, a.bs)
        sc = {'oracle': sc_oracle(mass)}
        for n, f in SCORERS.items():
            sc[n] = f(q, k, a.bs)
        if a.force:
            sc = {n: (v if n == 'oracle' else force_sink_diag(v)) for n, v in sc.items()}
        for dens in a.densities:
            u = max(1, int(round(dens * NB)))
            for n, s in sc.items():
                r = recall_at(s, mass, rows, u, a.bs, slack=a.slack)
                res.setdefault(f'{dens}', {}).setdefault(n, []).append(r)
            print(f'  layer {lname} d={dens} u={u}  ' +
                  '  '.join(f'{n}={res[str(dens)][n][-1]:.4f}' for n in sc), flush=True)
        del mass, rows, sc
        torch.cuda.empty_cache()

    summary = {d: {n: sum(v)/len(v) for n, v in m.items()} for d, m in res.items()}
    print('\n=== mean recall over layers ===')
    for d, m in sorted(summary.items(), key=lambda t: float(t[0])):
        o = m['oracle']
        print(f'density {d}:  ' + '  '.join(
            f'{n}={v:.4f}' + (f' ({v/o*100:.1f}% of oracle)' if n != 'oracle' else '')
            for n, v in sorted(m.items(), key=lambda t: -t[1])))
    json.dump({'T': T, 'bs': a.bs, 'per_layer': res, 'summary': summary},
              open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}')

if __name__ == '__main__':
    main()
