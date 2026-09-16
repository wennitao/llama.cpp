#!/usr/bin/env python3
"""What u does the real scorer actually ask for, per bq block?

The kernel needs ONE u for the whole op, and its latency is
    cost(u) ~ chunk_cost * chunks(u) + 93us * u,       chunks(u) = u / m(u)
with m(u) the largest divisor of u under the m<=8 and pipeline caps. So the cost of the
fixed-u constraint is entirely determined by the SPREAD of per-block demand: a fixed u must
cover the tail, and every block below the tail pays for blocks it did not want.

This measures that spread with meanpool on real activations, for the two candidate
policies:
  union  -- each 64-token block takes its own top-u0, the bq group takes the union
  thresh -- keep blocks in descending score until `tau` of the softmax mass is covered
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import Capturer, load_texts, sc_meanpool, _blocks   # noqa: E402


def chunks_of(u, M=8, bs=64):
    kv = u * bs
    lim = max(bs, min(((kv - 1) // 2) // bs * bs, min(u, M) * bs) // bs * bs)
    m = max(d for d in range(1, lim // bs + 1) if u % d == 0)
    return (u + m - 1) // m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--data', nargs='+', default=['longbench:narrativeqa', 'ruler:niah_multikey_2'])
    ap.add_argument('--docs', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--bq', type=int, default=256)
    ap.add_argument('--dens', type=float, default=0.25)
    ap.add_argument('--tau', type=float, default=0.90)
    ap.add_argument('--out', default='u_demand.json')
    a = ap.parse_args()

    cap = Capturer(a.model)
    R = a.bq // a.bs
    uni_all, thr_all = [], []          # (avail_fine, u_needed)

    for spec in a.data:
        for text in load_texts(spec, a.docs):
            qk, T = cap(text, a.tokens, None)
            NB = _blocks(T, a.bs)
            for q, k in qk:
                sc = sc_meanpool(q, k, a.bs).float()          # [NB, Hkv, NB]
                Hkv = sc.shape[1]
                ar = torch.arange(NB, device=sc.device)
                sc = sc.masked_fill(~(ar.view(-1, 1) >= ar.view(1, -1)).unsqueeze(1), -float('inf'))
                for g in range(NB // R):
                    last = (g + 1) * R - 1
                    avail = last + 1
                    if avail < 4 * R:
                        continue
                    # union of the R fine blocks' own top-u0
                    keep = torch.zeros(Hkv, NB, dtype=torch.bool, device=sc.device)
                    for j in range(R):
                        aa = g * R + j
                        av = aa + 1
                        u0 = max(1, int(round(a.dens * av)))
                        keep |= torch.zeros_like(keep).scatter_(
                            1, sc[aa, :, :av].topk(min(u0, av), -1).indices, True)
                    uni_all.append((avail, keep.sum(-1).float().mean().item(),
                                    keep.sum(-1).max().item()))
                    # threshold on the group's own row (max over the R fine rows)
                    row = sc[g*R:(g+1)*R, :, :avail].amax(0)
                    p = torch.softmax(row, -1)
                    srt, _ = p.sort(-1, descending=True)
                    n_keep = (srt.cumsum(-1) < a.tau).sum(-1) + 1
                    thr_all.append((avail, n_keep.float().mean().item(), n_keep.max().item()))
            del qk
            torch.cuda.empty_cache()
        print(f'  {spec} done', flush=True)

    def report(name, rows):
        import statistics as st
        us = sorted(r[2] for r in rows)                        # per-op demand = max over heads
        n = len(us)
        pc = lambda p: us[min(n - 1, int(p * n))]
        print(f'\n=== {name}: u demanded per bq block (max over KV heads) ===')
        print(f'  n={n}  min {us[0]}  p50 {pc(.50)}  p90 {pc(.90)}  p99 {pc(.99)}  max {us[-1]}')
        print(f'  mean {sum(us)/n:.1f}   spread p99/p50 = {pc(.99)/max(1,pc(.50)):.2f}x')
        # Cost of each DEPLOYABLE fixed u. The raw percentiles are not deployable: p99=37
        # is prime, so m=1 and it costs 37 chunks. Only u with a large divisor is worth
        # using, so the menu is the frontier, not the quantiles.
        C, DENSE = 358.0, 8339.0
        print(f'  {"u":>4}{"chunks":>8}{"model us":>10}{"vs dense":>10}{"truncated":>11}  covers')
        for uf in (12, 18, 24, 32, 40, 48, 56, 64):
            ch = chunks_of(uf)
            cost = C * ch + 93 * uf
            miss = sum(1 for u in us if u > uf) / n
            covers = 100 * (1 - miss)
            print(f'  {uf:>4}{ch:>8}{cost:>10.0f}{DENSE/cost:>9.2f}x{miss:>10.1%}  p{covers:.0f}')
        return {'p50': pc(.50), 'p90': pc(.90), 'p99': pc(.99), 'max': us[-1],
                'mean': sum(us)/n, 'hist': us,
                'pairs': [(int(r[0]), int(r[2])) for r in rows]}

    out = {'union': report('UNION of 4 x top-25%', uni_all),
           'thresh': report(f'THRESHOLD tau={a.tau}', thr_all),
           'bq': a.bq, 'bs': a.bs, 'tokens': a.tokens}
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
