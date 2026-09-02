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


def _sel_mask(q, k, bs, bq, u_frac, scorer, mode, force=True, cfg_mult=1.0, cfg_sizes=None, rec=None, cfg_head=None, cfg_shared=False):
    """Return a bool [NBq_fine, Hkv, NBk] mask of KEPT blocks. NBq_fine = ceil(T/bs)."""
    Hq, T, d = q.shape
    Hkv = k.shape[0]
    NB = _blocks(T, bs)
    sc = scorer(q, k, bs).float()                      # [NB, Hkv, NB]
    if cfg_shared:
        # One list shared by every KV head -- what the deployed whole-row scorer
        # produces (sel carries no head axis). Score = head mean.
        sc = sc.mean(1, keepdim=True).expand(-1, Hkv, -1).contiguous()

    # reachability: block b is usable by query block a only if b <= a
    ar = torch.arange(NB, device=q.device)
    reach = ar.view(-1, 1) >= ar.view(1, -1)           # [NB, NB]

    R = bq // bs                                       # fine blocks per selection group
    if R > 1 and mode not in ('union', 'union_fixed', 'union_thr', 'union_elem'):
        # DEVICE-FAITHFUL COARSEN. The deployed scorer pools the whole 256-query group
        # with NO per-row causal masking, forces sink + group-last, cuts one list at the
        # GROUP-LAST row's reach, and leaves per-row causality to the kernel's mask.
        # Averaging -inf-masked rows instead (the old code here) makes every KV block
        # recent to the group -inf for the whole group -- blocks a-1..a-(R-1) become
        # unselectable even for rows that CAN reach them. That artifact is why this
        # harness said bq256 costs +13.6%% PPL while the device measured +2.4%% at the
        # same geometry and density.
        pad = (-NB) % R
        scp = torch.cat([sc, sc[-1:].expand(pad, -1, -1)], 0) if pad else sc
        gsc = scp.view(-1, R, Hkv, NB).mean(1)          # [NG, Hkv, NB], raw mean
        NG = gsc.shape[0]
        keep_g = torch.zeros(NG, Hkv, NB, dtype=torch.bool, device=q.device)
        for gi in range(NG):
            a_last = min((gi + 1) * R - 1, NB - 1)
            avail = a_last + 1                          # group-last row's reach
            row = gsc[gi, :, :avail]
            if mode == 'thresh':
                p = torch.softmax(row, dim=-1)
                srt, idx = p.sort(dim=-1, descending=True)
                csum = srt.cumsum(-1)
                n_keep = (csum < u_frac).sum(-1) + 1
                for h in range(Hkv):
                    keep_g[gi, h, idx[h, :n_keep[h]]] = True
            else:
                uu = max(1, min(avail, int(round(u_frac * avail))))
                idx = row.topk(uu, dim=-1).indices
                keep_g[gi].scatter_(1, idx, True)
            if force:
                keep_g[gi, :, 0] = True                 # sink (bias-forced on device)
                keep_g[gi, :, a_last] = True            # group-last (bias-forced)
        keep = keep_g.repeat_interleave(R, 0)[:NB]
        keep &= reach.unsqueeze(1)                      # the kernel's per-row mask
        if rec is not None:
            fr = torch.zeros(Hkv, device=q.device)
            n = 0
            for a in range(NB):
                avail = a + 1
                if avail < 4:
                    continue
                fr += keep[a, :, :avail].float().sum(-1) / avail
                n += 1
            rec.append((fr / max(n, 1)).cpu())
        return keep

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
        elif mode == 'union_elem':
            # Per-ELEMENT softmax threshold: keep block j iff p_j * avail > c. Monotone
            # in the row's own score, so it needs no sort/cumsum on device -- membership
            # is step(p*avail - c), the union is step over the group mean, and the packed
            # list is the argsort the scorer already runs. u_frac carries c.
            pp = torch.softmax(row, dim=-1)
            keep[a, :, :avail] = pp * avail > u_frac
        elif mode in ('thresh', 'union_thr'):
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

    if R > 1 and mode in ('union', 'union_thr', 'union_elem'):
        # UNION: every fine block keeps its own picks, and the group takes the superset.
        # Lossless by construction; the price is density, not accuracy. union_thr sizes
        # each fine block by the threshold, so the union length varies per tile -- the
        # thing only a dynamic-n_sel kernel can serve. The kernel would give every row
        # the whole union (no per-row selection mask), so keep is g & causal, exactly.
        pad = (-NB) % R
        kk = keep
        if pad:
            kk = torch.cat([kk, torch.zeros(pad, Hkv, NB, dtype=torch.bool, device=q.device)], 0)
        g = kk.view(-1, R, Hkv, NB).any(1)
        if cfg_sizes is not None:
            # The KERNEL's cost driver: the union list length, as a fraction of the
            # blocks available at the group-last row.
            NG = g.shape[0]
            fr = 0.0
            for gi in range(NG):
                a_last = min((gi + 1) * R - 1, NB - 1)
                fr += (g[gi, :, :a_last + 1].float().sum(-1) / (a_last + 1)).mean().item()
            cfg_sizes.append(fr / NG)
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
                         cfg_head=cfg['tab'][il] if cfg['mode'] == 'perlayerhead' else None,
                         cfg_shared=cfg.get('shared', False))
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
    ap.add_argument('--suite', default='full', choices=['full', 'dynval', 'dynval2', 'devanchor', 'unionthr', 'unionelem'])
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

    if a.suite == 'unionelem':
        # The per-element rule against the cumulative rule at matched cost. c=1 keeps
        # blocks that beat the uniform mass; smaller c keeps more.
        run(dict(mode='dense', bs=64, bq=64, u_frac=1.0, scorer=sc_meanpool), 'dense')
        for c in (0.3, 0.5, 1.0):
            _, o = run(dict(mode='union_elem', bs=64, bq=256, u_frac=c, scorer=sc_meanpool,
                            shared=False), f'bq256 per-head union-elem c={c}')
            us = o.get('usize') or []
            print(f'    union mean {sum(us)/len(us):.1%} of avail' if us else '    (no usize)', flush=True)
        return

    if a.suite == 'unionthr':
        # Candidate B: dynamic n_sel at TODAY'S geometry (Br=256 kept). Each 64-query
        # block thresholds its own head-shared list; the tile serves the UNION, sized
        # per tile. Paired with fixed-u at the union's own mean length, so the pair
        # isolates what the variable length buys at equal kernel cost.
        run(dict(mode='dense', bs=64, bq=64, u_frac=1.0, scorer=sc_meanpool), 'dense')
        for tau in (0.90, 0.80):
            _, o = run(dict(mode='union_thr', bs=64, bq=256, u_frac=tau, scorer=sc_meanpool,
                            shared=True), f'bq256 SHARED union-thr{tau}')
            us = o.get('usize') or []
            uf = sum(us) / len(us) if us else 0.5
            run(dict(mode='topk', bs=64, bq=256, u_frac=uf, scorer=sc_meanpool, shared=True),
                f'bq256 SHARED fixed ({uf:.0%})')
        return

    if a.suite == 'devanchor':
        # Anchor against the DEVICE quality curve (llama-perplexity, Q4_0, bq256 shared
        # coarsened fixed-u): dense 17.67, 75% 17.77 (+0.6%), 50% 18.33 (+3.7%),
        # 25% 22.28 (+26.1%). If this harness is faithful, the same geometry at the same
        # fraction must land near the same RELATIVE increment.
        run(dict(mode='dense', bs=64, bq=64, u_frac=1.0, scorer=sc_meanpool), 'dense')
        for f in (0.75, 0.625, 0.50, 0.25):
            run(dict(mode='topk', bs=64, bq=256, u_frac=f, scorer=sc_meanpool, shared=True),
                f'bq256 SHARED fixed {f:.0%}')
        return

    if a.suite in ('dynval', 'dynval2'):
        # Adaptive (threshold) vs fixed-u, at each selection geometry the kernel could
        # serve. bq64/per-head is the geometry thr0.9 was originally measured at (NOT
        # deployable -- sel has no head axis and one list serves a 256-query tile);
        # bq256/SHARED is what a dynamic-n_sel kernel would actually get. Density is
        # matched within each pair, so each pair isolates adaptive-vs-fixed alone.
        print('\n=== dynamic n_sel validation at deployment geometry ===', flush=True)
        run(dict(mode='dense', bs=64, bq=64, u_frac=1.0, scorer=sc_meanpool), 'dense')
        arms = [('bq64 per-head',  dict(bs=64, bq=64,  shared=False)),
                ('bq256 per-head', dict(bs=64, bq=256, shared=False)),
                ('bq256 SHARED',   dict(bs=64, bq=256, shared=True))]
        if a.suite == 'dynval2':
            # The cells that separate GRANULARITY from HEAD-SHARING. bq64 is reachable
            # with Br=64 and today's shared-list sel format; per-head lists are the
            # part the on-device scorer cannot currently produce (permuted-operand rule).
            arms = [('bq64 SHARED',    dict(bs=64, bq=64,  shared=True)),
                    ('bq128 SHARED',   dict(bs=64, bq=128, shared=True)),
                    ('bq128 per-head', dict(bs=64, bq=128, shared=False))]
        for name, g in arms:
            _, o = run(dict(mode='thresh', u_frac=0.90, scorer=sc_meanpool, record=True, **g),
                       f'thr0.9 {name}')
            tab = {il: torch.stack(v).mean(0) for il, v in o['rec'].items()}
            gmean = torch.stack([t.mean() for t in tab.values()]).mean().item()
            run(dict(mode='topk', u_frac=gmean, scorer=sc_meanpool, **g),
                f'fixed {name} ({gmean:.0%})')
        return

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
