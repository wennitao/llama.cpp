#!/usr/bin/env python3
"""Simulate the prefix-length kernel against the measured tail distribution.

The waste from a shared list is always a SUFFIX of an index-sorted selection, so a query
block needs a prefix, not an arbitrary subset. But HMX processes fixed Br x Bc tiles, so a
prefix can only be truncated at CHUNK granularity: member p runs ceil(L_p / m) chunks
instead of the group's ceil(u / m). If the unusable tail sits inside the final chunk, that
saves nothing.

This measures how often the tail actually crosses a chunk boundary, and prices three
kernels against the fitted cost model  358*chunks + 93*u  (us, per op at nb=1024):
  now     every member runs the full chunk count over the full list
  prefix  member p runs ceil(L_p/m) chunks  (implementable; the FIFO reorder)
  ideal   full chunks but MACs only on usable blocks (not implementable on a tile engine)
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import Capturer, load_texts, sc_meanpool, _blocks   # noqa: E402

A, B = 358.0, 93.0


def m_of(u, bs=64, M=8):
    kv = u * bs
    lim = max(bs, min(((kv - 1) // 2) // bs * bs, min(u, M) * bs) // bs * bs)
    return max(d for d in range(1, lim // bs + 1) if u % d == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--data', nargs='+', default=['longbench:narrativeqa', 'ruler:niah_multikey_2'])
    ap.add_argument('--docs', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--R', type=int, default=4)
    ap.add_argument('--ufix', type=int, default=40)
    ap.add_argument('--dens', type=float, default=0.25)
    ap.add_argument('--out', default='prefix_sim.json')
    a = ap.parse_args()

    cap = Capturer(a.model)
    m = m_of(a.ufix)
    CH = -(-a.ufix // m)
    print(f'u_fixed={a.ufix}  m={m}  chunks={CH}  R={a.R}', flush=True)

    Lrel, saved_ch, n = [], [], 0
    for spec in a.data:
        for text in load_texts(spec, a.docs):
            qk, T = cap(text, a.tokens, None)
            NB = _blocks(T, a.bs)
            for q, k in qk:
                sc = sc_meanpool(q, k, a.bs).float()
                Hkv = sc.shape[1]
                ar = torch.arange(NB, device=sc.device)
                sc = sc.masked_fill(~(ar.view(-1, 1) >= ar.view(1, -1)).unsqueeze(1), -float('inf'))
                for g in range(NB // a.R):
                    lastb = (g + 1) * a.R - 1
                    avail = lastb + 1
                    if avail < a.ufix + a.R:            # only where u_fixed is a real choice
                        continue
                    uni = torch.zeros(Hkv, NB, dtype=torch.bool, device=sc.device)
                    for j in range(a.R):
                        aa = g * a.R + j
                        u0 = max(1, int(round(a.dens * (aa + 1))))
                        uni |= torch.zeros_like(uni).scatter_(
                            1, sc[aa, :, :aa+1].topk(min(u0, aa+1), -1).indices, True)
                    # deployed list: union members first, then best-scoring fill, to u_fixed
                    grp = sc[g*a.R:(g+1)*a.R, :, :avail].amax(0)
                    rank = grp + uni[:, :avail].float() * 1e9
                    sel = rank.topk(a.ufix, dim=-1).indices                 # [Hkv, ufix]
                    sel, _ = sel.sort(dim=-1)                               # INDEX-sorted
                    for j in range(a.R):
                        aa = g * a.R + j
                        L = (sel <= aa).sum(-1).float()                     # prefix length
                        Lrel.append(L.mean().item())
                        saved_ch.append((CH - torch.ceil(L / m)).clamp(min=0).mean().item())
                        n += 1
            del qk
            torch.cuda.empty_cache()
        print(f'  {spec} done', flush=True)

    L = sum(Lrel)/len(Lrel)
    sc_ = sum(saved_ch)/len(saved_ch)
    now    = A*CH + B*a.ufix
    prefix = A*(CH - sc_) + B*(m*(CH - sc_))
    ideal  = A*CH + B*L
    print(f'\nmean usable prefix L = {L:.1f} of u={a.ufix}   ({L/a.ufix:.1%} usable, '
          f'{1-L/a.ufix:.1%} wasted)')
    print(f'mean whole chunks saved by a prefix kernel = {sc_:.3f} of {CH}\n')
    print(f'  {"kernel":<8}{"model us":>10}{"vs now":>9}   what it needs')
    print(f'  {"now":<8}{now:>10.0f}{1.0:>8.2f}x   nothing')
    print(f'  {"prefix":<8}{prefix:>10.0f}{now/prefix:>8.2f}x   sorted sel + per-qb prefix + chunk-outer loop')
    print(f'  {"ideal":<8}{ideal:>10.0f}{now/ideal:>8.2f}x   per-block MAC skip (not on a tile engine)')
    json.dump({'L': L, 'ufix': a.ufix, 'm': m, 'chunks': CH, 'saved_chunks': sc_,
               'now': now, 'prefix': prefix, 'ideal': ideal, 'n': n}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
