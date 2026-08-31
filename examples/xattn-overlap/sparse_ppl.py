#!/usr/bin/env python3
"""Perplexity under block-sparse attention, with the ALGORITHM as the only variable.

The device numbers say uniform top-25% at bs=64 with k=4 coarsening costs 26% perplexity
at 4k. XAttention reports near-zero drop on RULER/LongBench. Those are not comparable:
different block size, different selection rule (threshold vs fixed top-k), different query
granularity, different context length, different workload. This runs all of them through
one attention hook on one corpus so the comparison means something.

Every policy here is exact about what it selects; nothing is approximated by a proxy
metric. The number reported is the model's own loss.
"""
import argparse, json, math, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import sc_xattn, sc_meanpool, sc_meanpool_sub, _blocks   # noqa: E402

CFG = {}          # set per run


def _sel_mask(q, k, bs, bq, u_frac, scorer, mode, force=True, cfg_mult=1.0, cfg_sizes=None, rec=None, cfg_head=None):
    """Return a bool [NBq_fine, Hkv, NBk] mask of KEPT blocks. NBq_fine = ceil(T/bs)."""
    Hq, T, d = q.shape
    Hkv = k.shape[0]
    NB = _blocks(T, bs)
    sc = scorer(q, k, bs).float()                      # [NB, Hkv, NB]

    # reachability: block b is usable by query block a only if b <= a
    ar = torch.arange(NB, device=q.device)
    reach = ar.view(-1, 1) >= ar.view(1, -1)           # [NB, NB]
    sc = sc.masked_fill(~reach.unsqueeze(1), -float('inf'))

    R = bq // bs                                       # fine blocks per selection group
    if R > 1 and mode not in ('union', 'union_fixed'):
        # COARSEN: one score row per group, from the group's mean. This is what the NPU
        # graph does, and it is NOT a union -- a fine block gets the group's choice, not
        # its own, so it can lose blocks it would have kept.
        pad = (-NB) % R
        if pad:
            sc = torch.cat([sc, sc[-1:].expand(pad, -1, -1)], 0)
        g = sc.view(-1, R, Hkv, sc.shape[-1])
        sc = g.mean(1).repeat_interleave(R, 0)[:NB]
        sc = sc.masked_fill(~reach.unsqueeze(1), -float('inf'))

    keep = torch.zeros(NB, Hkv, NB, dtype=torch.bool, device=q.device)
    for a in range(NB):
        avail = a + 1
        row = sc[a, :, :avail]                          # [Hkv, avail]
        if mode in ('perlayer', 'perlayerhead'):
            # Fixed u, but taken from the per-layer (or per-layer-per-head) budget the
            # threshold actually spent. Per-layer is expressible on the HTP -- u is a
            # graph-build-time constant per op. Per-head is NOT: n_sel is sel->ne[0],
            # shared across the head axis. So the gap between these two is exactly the
            # part of the threshold's value the kernel cannot currently capture.
            fr = u_frac if mode == 'perlayer' else None
            for h in range(Hkv):
                f = fr if fr is not None else float(cfg_head[h])
                uu = max(1, min(avail, int(round(f * avail))))
                keep[a, h, row[h].topk(uu).indices] = True
        elif mode == 'thresh':
            # XAttention's rule: softmax the row, keep blocks in descending order until
            # the kept mass reaches `u_frac`. Density adapts per (layer, head, query block).
            p = torch.softmax(row, dim=-1)
            srt, idx = p.sort(dim=-1, descending=True)
            csum = srt.cumsum(-1)
            n_keep = (csum < u_frac).sum(-1) + 1        # at least one
            for h in range(Hkv):
                keep[a, h, idx[h, :n_keep[h]]] = True
        else:
            uu = max(1, min(avail, int(round(u_frac * avail))))
            idx = row.topk(uu, dim=-1).indices
            keep[a].scatter_(1, idx, True)
        if force:
            keep[a, :, 0] = True                        # sink
            keep[a, :, a] = True                        # own diagonal
    if R > 1 and mode == 'union_fixed':
        # THE DEPLOYABLE UNION. The kernel needs n_sel fixed across query blocks
        # (ggml_hexagon_supported_fa_sparse: chunk count, pipelining and VTCM all derive
        # from sel->ne[0]), so a variable-size union cannot be expressed. Fix the size at
        # u_fixed = mult * u instead and rank by (is in the union) first, score second:
        #   - every union member wins a slot while there is room, so the result stays a
        #     superset of each fine block's own picks whenever u_fixed >= |union|;
        #   - spare slots go to the next best blocks rather than to masked padding, which
        #     costs the kernel exactly the same and keeps more mass;
        #   - an over-large union is truncated by score, which is the only lossy case.
        pad = (-NB) % R
        kk = keep
        scc = sc
        if pad:
            kk = torch.cat([kk, torch.zeros(pad, Hkv, NB, dtype=torch.bool, device=q.device)], 0)
            scc = torch.cat([scc, scc[-1:].expand(pad, -1, -1)], 0)
        uni = kk.view(-1, R, Hkv, NB).any(1)                     # [NG, Hkv, NB]
        cfg_sizes.append(uni.sum(-1).float().mean().item())
        grp = scc.view(-1, R, Hkv, NB).amax(1)                   # group score, for ranking
        keep2 = torch.zeros_like(uni)
        NG = uni.shape[0]
        for g in range(NG):
            a_last = min((g + 1) * R - 1, NB - 1)
            avail = a_last + 1
            uu = max(1, min(avail, int(round(u_frac * avail * cfg_mult))))
            rank = grp[g, :, :avail] + uni[g, :, :avail].float() * 1e9
            idx = rank.topk(uu, dim=-1).indices
            keep2[g].scatter_(1, idx, True)
        keep = keep2.repeat_interleave(R, 0)[:NB]
        keep &= reach.unsqueeze(1)
        for a in range(NB):
            keep[a, :, 0] = True
            keep[a, :, a] = True
        return keep

    if rec is not None:
        # Realized density as a FRACTION OF AVAILABLE blocks, per KV head. This is what a
        # threshold actually spends, and it is the table a fixed-u policy would have to
        # reproduce to match it.
        fr = torch.zeros(Hkv, device=q.device)
        n = 0
        for a in range(NB):
            avail = a + 1
            if avail < 4:
                continue
            fr += keep[a, :, :avail].float().sum(-1) / avail
            n += 1
        rec.append((fr / max(n, 1)).cpu())

    if R > 1 and mode == 'union':
        # UNION: every fine block keeps its own picks, and the group takes the superset.
        # Lossless by construction; the price is density, not accuracy.
        pad = (-NB) % R
        kk = keep
        if pad:
            kk = torch.cat([kk, torch.zeros(pad, Hkv, NB, dtype=torch.bool, device=q.device)], 0)
        g = kk.view(-1, R, Hkv, NB).any(1)
        keep = g.repeat_interleave(R, 0)[:NB]
        keep &= reach.unsqueeze(1)
    return keep


def sparse_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    cfg = CFG
    B, Hq, T, d = query.shape
    Hkv = key.shape[1]
    G = Hq // Hkv

    if cfg['mode'] != 'dense' and T > cfg['bq']:
        il = getattr(module, 'layer_idx', 0)
        uf = cfg['u_frac']
        if cfg['mode'] == 'perlayer':
            uf = cfg['tab'][il].mean().item()
        keep = _sel_mask(query[0], key[0], cfg['bs'], cfg['bq'], uf,
                         cfg['scorer'], cfg['mode'], cfg_mult=cfg.get('mult', 1.0),
                         cfg_sizes=cfg.setdefault('usize', []),
                         rec=cfg.setdefault('rec', {}).setdefault(il, []) if cfg.get('record') else None,
                         cfg_head=cfg['tab'][il] if cfg['mode'] == 'perlayerhead' else None)
        cfg['density'].append(keep.float().mean().item() * 2)   # /2 for causal half
        NB = keep.shape[0]
        m = keep.repeat_interleave(cfg['bs'], 0)[:T]            # [T, Hkv, NB]
        m = m.repeat_interleave(cfg['bs'], 2)[:, :, :T]         # [T, Hkv, T]
        block = ~m.permute(1, 0, 2)                             # [Hkv, T, T] True = drop
    else:
        block = None

    kr = key.repeat_interleave(G, 1)
    vr = value.repeat_interleave(G, 1)
    S = kr.shape[-2]

    # Chunk over query positions. The full [B, Hq, T, T] score matrix is the only large
    # allocation here and it is not needed all at once; the GPUs this runs on are shared.
    QC = cfg.get('qchunk', 512)
    out = torch.empty(B, Hq, T, value.shape[-1], dtype=query.dtype, device=query.device)
    for s0 in range(0, T, QC):
        s1 = min(s0 + QC, T)
        w = torch.matmul(query[:, :, s0:s1], kr.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, s0:s1, :S]
        else:
            # Registering a custom attention implementation means transformers hands us
            # attention_mask=None and expects US to be causal. Getting this wrong is not
            # subtle in the sparse rows -- the block mask already carries block-level
            # causality -- but it makes the DENSE baseline attend to the future, which is
            # how a 597 perplexity baseline sat next to 7-point sparse rows.
            qp = torch.arange(s0, s1, device=w.device).view(-1, 1)
            kp = torch.arange(S, device=w.device).view(1, -1)
            w = w.masked_fill((kp > qp).view(1, 1, s1 - s0, S), torch.finfo(w.dtype).min)
        if block is not None:
            w = w.view(B, Hkv, G, s1 - s0, S).masked_fill(
                block[:, s0:s1, :].view(1, Hkv, 1, s1 - s0, S),
                torch.finfo(w.dtype).min).view(B, Hq, s1 - s0, S)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, :, s0:s1] = torch.matmul(w, vr)
        del w
    return out.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--file', default='wikitext-2-raw/wiki.test.raw')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--chunks', type=int, default=8)
    ap.add_argument('--out', default='sparse_ppl.json')
    a = ap.parse_args()

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS['sparse_exp'] = sparse_attention

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation='sparse_exp').cuda().eval()
    ids = tok(open(a.file).read(), return_tensors='pt').input_ids[0]
    print(f'{a.model}: {ids.numel()} tokens, ctx={a.ctx}, {a.chunks} chunks', flush=True)

    POLICIES = [
        ('dense',                  dict(mode='dense',  bs=64,  bq=64,  u_frac=1.0,  scorer=sc_meanpool)),
        ('xattn bs128 k1 top25',   dict(mode='topk',   bs=128, bq=128, u_frac=0.25, scorer=sc_xattn)),
        ('xattn bs128 k1 thr0.9',  dict(mode='thresh', bs=128, bq=128, u_frac=0.90, scorer=sc_xattn)),
        ('xattn bs128 k1 thr0.95', dict(mode='thresh', bs=128, bq=128, u_frac=0.95, scorer=sc_xattn)),
        ('meanpool bs64 k1 top25', dict(mode='topk',   bs=64,  bq=64,  u_frac=0.25, scorer=sc_meanpool)),
        ('meanpool bs64 k4 top25', dict(mode='topk',   bs=64,  bq=256, u_frac=0.25, scorer=sc_meanpool_sub)),
        ('meanpool bs64 k4 UNION', dict(mode='union',  bs=64,  bq=256, u_frac=0.25, scorer=sc_meanpool_sub)),
        ('meanpool bs64 k1 thr0.9',dict(mode='thresh', bs=64,  bq=64,  u_frac=0.90, scorer=sc_meanpool)),
        ('k4 union FIXED x1.00',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=1.00, scorer=sc_meanpool)),
        ('k4 union FIXED x1.25',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=1.25, scorer=sc_meanpool)),
        ('k4 union FIXED x1.50',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=1.50, scorer=sc_meanpool)),
        ('k4 union FIXED x1.75',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=1.75, scorer=sc_meanpool)),
        ('k4 union FIXED x2.00',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=2.00, scorer=sc_meanpool)),
        ('k4 union FIXED x2.25',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=2.25, scorer=sc_meanpool)),
        ('k4 union FIXED x2.50',   dict(mode='union_fixed', bs=64, bq=256, u_frac=0.25, mult=2.50, scorer=sc_meanpool)),
    ]

    def run(cfg, tag):
        CFG.clear(); CFG.update(cfg); CFG['density'] = []
        nll, ntok = 0.0, 0
        for c in range(a.chunks):
            x = ids[c*a.ctx:(c+1)*a.ctx].unsqueeze(0).cuda()
            if x.numel() < a.ctx:
                break
            with torch.no_grad():
                lg = model(x).logits[0]
                for p0 in range(0, lg.shape[0] - 1, 256):
                    p1 = min(p0 + 256, lg.shape[0] - 1)
                    nll += torch.nn.functional.cross_entropy(
                        lg[p0:p1].float(), x[0, p0 + 1:p1 + 1], reduction='sum').item()
                ntok += lg.shape[0] - 1
                del lg
            torch.cuda.empty_cache()
        ppl = math.exp(nll / ntok)
        dd = CFG['density']
        print(f'  {tag:<26} PPL {ppl:8.3f}   density '
              f'{(sum(dd)/len(dd) if dd else 1.0):5.1%}', flush=True)
        return ppl, dict(CFG)

    # ---- how much of an adaptive threshold survives being flattened? ----
    print('\n=== adaptivity decomposition (threshold -> per-layer -> global) ===', flush=True)
    base = dict(mode='thresh', bs=64, bq=64, u_frac=0.90, scorer=sc_meanpool, record=True)
    ppl_thr, out = run(base, 'thr0.9 (full adaptive)')
    tab = {il: torch.stack(v).mean(0) for il, v in out['rec'].items()}
    gmean = torch.stack([t.mean() for t in tab.values()]).mean().item()
    print(f'    per-layer budget: min {min(t.mean().item() for t in tab.values()):.3f} '
          f'max {max(t.mean().item() for t in tab.values()):.3f} mean {gmean:.3f}')
    run(dict(mode='perlayerhead', bs=64, bq=64, u_frac=0.0, scorer=sc_meanpool, tab=tab),
        'per-layer + per-head u')
    run(dict(mode='perlayer', bs=64, bq=64, u_frac=0.0, scorer=sc_meanpool, tab=tab),
        'per-layer u (HTP-legal)')
    run(dict(mode='topk', bs=64, bq=64, u_frac=gmean, scorer=sc_meanpool),
        f'global fixed u ({gmean:.0%})')
    print('=== end decomposition ===\n', flush=True)

    res = {}
    for name, cfg in POLICIES:
        CFG.clear(); CFG.update(cfg); CFG['density'] = []
        nll, ntok = 0.0, 0
        for c in range(a.chunks):
            x = ids[c*a.ctx:(c+1)*a.ctx].unsqueeze(0).cuda()
            if x.numel() < a.ctx:
                break
            with torch.no_grad():
                lg = model(x).logits[0]                    # [T, V], left in bf16
                # Cross-entropy in slices: logits.float() over the whole chunk is 2.3 GB
                # at ctx=4096 and these GPUs are shared.
                for p0 in range(0, lg.shape[0] - 1, 256):
                    p1 = min(p0 + 256, lg.shape[0] - 1)
                    l = torch.nn.functional.cross_entropy(
                        lg[p0:p1].float(), x[0, p0 + 1:p1 + 1], reduction='sum')
                    nll += l.item()
                ntok += lg.shape[0] - 1
                del lg
            torch.cuda.empty_cache()
        ppl = math.exp(nll / ntok)
        dens = sum(CFG['density'])/len(CFG['density']) if CFG['density'] else 1.0
        us = CFG.get('usize') or []
        umean = sum(us)/len(us) if us else 0.0
        res[name] = {'ppl': ppl, 'density': dens, 'union_mean': umean}
        extra = f'   |union| mean {umean:5.1f}' if umean else ''
        print(f'  {name:<26} PPL {ppl:8.3f}   density {dens:5.1%}{extra}', flush=True)
    json.dump({'model': a.model, 'ctx': a.ctx, 'chunks': a.chunks, 'res': res},
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
