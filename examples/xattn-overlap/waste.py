#!/usr/bin/env python3
"""How many of a shared list's blocks can each query block not legally read?

A group of R query blocks shares one selection. The group's reach is its LAST member's, so
every earlier member carries entries past its own diagonal. Those are staged and computed
and then killed by the per-row mask -- pure waste, and it is the price of sharing a list.

Measures the waste exactly, and its structure: if the wasted entries are always the TAIL of
an index-sorted list, a query block only ever needs a PREFIX of the list, which is a far
cheaper thing for a kernel to support than an arbitrary per-block u.
"""
import argparse, json, os, sys, collections
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import Capturer, load_texts, sc_meanpool, _blocks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--data', nargs='+', default=['longbench:narrativeqa', 'ruler:niah_multikey_2'])
    ap.add_argument('--docs', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--dens', type=float, default=0.25)
    ap.add_argument('--out', default='waste.json')
    a = ap.parse_args()

    cap = Capturer(a.model)
    res = {}
    for R in (2, 4, 8, 32):
        res[R] = {'waste': [], 'u': [], 'tail': 0, 'tot': 0}

    for spec in a.data:
        for text in load_texts(spec, a.docs):
            qk, T = cap(text, a.tokens, None)
            NB = _blocks(T, a.bs)
            for q, k in qk:
                sc = sc_meanpool(q, k, a.bs).float()
                Hkv = sc.shape[1]
                ar = torch.arange(NB, device=sc.device)
                reach = ar.view(-1, 1) >= ar.view(1, -1)
                sc = sc.masked_fill(~reach.unsqueeze(1), -float('inf'))
                for R in (2, 4, 8, 32):
                    for g in range(NB // R):
                        lastb = (g + 1) * R - 1
                        if lastb + 1 < 4 * R:
                            continue
                        keep = torch.zeros(Hkv, NB, dtype=torch.bool, device=sc.device)
                        for j in range(R):
                            aa = g * R + j
                            u0 = max(1, int(round(a.dens * (aa + 1))))
                            keep |= torch.zeros_like(keep).scatter_(
                                1, sc[aa, :, :aa+1].topk(min(u0, aa+1), -1).indices, True)
                        usz = keep.sum(-1)                                  # [Hkv]
                        for j in range(R):
                            aa = g * R + j
                            usable = keep[:, :aa+1].sum(-1)
                            res[R]['waste'].append((usz - usable).float().mean().item())
                            res[R]['u'].append(usz.float().mean().item())
                            # is the waste always the index-sorted TAIL? i.e. is every
                            # unusable entry > every usable one? true iff unusable indices
                            # all exceed aa, which they do by construction -- verify.
                            bad = keep[:, :aa+1].sum(-1) + (keep[:, aa+1:].sum(-1))
                            res[R]['tot'] += 1
                            res[R]['tail'] += int((bad == usz).all().item())
            del qk
            torch.cuda.empty_cache()
        print(f'  {spec} done', flush=True)

    print(f"\n{'R':>4}{'mean u':>9}{'wasted':>9}{'% of u':>9}{'tail-structured':>17}")
    out = {}
    for R in (2, 4, 8, 32):
        w = res[R]['waste']; u = res[R]['u']
        if not w:
            continue
        mw, mu = sum(w)/len(w), sum(u)/len(u)
        out[R] = {'u': mu, 'waste': mw, 'frac': mw/mu, 'tail': res[R]['tail']/res[R]['tot']}
        print(f"{R:>4}{mu:>9.1f}{mw:>9.2f}{mw/mu:>8.1%}{res[R]['tail']/res[R]['tot']:>16.1%}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
