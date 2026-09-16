#!/usr/bin/env python3
"""Is per-head KV-block demand a COMPUTABLE rule, or only an oracle replay?

Replaying a per-head budget recovers PPL 17.408 against 17.608 for a global fixed u
(commit 4704a1af9) -- but that replay used each head's own measured demand, which the graph
cannot know. n_sel is one uint16 per op, shared across the head axis, so a deployable
per-head rule needs the head's budget to be PREDICTABLE: stable across query blocks, across
documents, and across datasets, so it can be calibrated once.

This measures that. It also fixes a real defect in u_demand.py, which reported the MAX over
KV heads and called it per-query-block demand -- the right statistic for a shared n_sel, but
not what the name says, and it hides exactly the per-head structure being asked about here.
"""
import argparse, json, os, sys, statistics as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import Capturer, load_texts, sc_meanpool, _blocks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--calib', nargs='+', default=['longbench:narrativeqa'])
    ap.add_argument('--eval', nargs='+', default=['ruler:niah_multikey_2', 'longbench:gov_report'])
    ap.add_argument('--docs', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--R', type=int, default=4)
    ap.add_argument('--dens', type=float, default=0.25)
    ap.add_argument('--out', default='head_demand.json')
    a = ap.parse_args()

    cap = Capturer(a.model)

    def measure(specs):
        # demand[(layer, head)] = list of union sizes, one per query block at full reach
        d = {}
        for spec in specs:
            for text in load_texts(spec, a.docs):
                qk, T = cap(text, a.tokens, None)
                NB = _blocks(T, a.bs)
                for li, (q, k) in enumerate(qk):
                    sc = sc_meanpool(q, k, a.bs).float()
                    Hkv = sc.shape[1]
                    ar = torch.arange(NB, device=sc.device)
                    sc = sc.masked_fill(~(ar.view(-1, 1) >= ar.view(1, -1)).unsqueeze(1), -float('inf'))
                    for g in range(NB // a.R):
                        lastb = (g + 1) * a.R - 1
                        if lastb + 1 < NB:            # full reach only, so avail is constant
                            continue
                        keep = torch.zeros(Hkv, NB, dtype=torch.bool, device=sc.device)
                        for j in range(a.R):
                            aa = g * a.R + j
                            u0 = max(1, int(round(a.dens * (aa + 1))))
                            keep |= torch.zeros_like(keep).scatter_(
                                1, sc[aa, :, :aa+1].topk(min(u0, aa+1), -1).indices, True)
                        for h in range(Hkv):
                            d.setdefault((li, h), []).append(int(keep[h].sum()))
                del qk
                torch.cuda.empty_cache()
            print(f'    {spec} done', flush=True)
        return d

    print('calibration set:', flush=True)
    cal = measure(a.calib)
    print('evaluation set:', flush=True)
    ev = measure(a.eval)

    keys = sorted(set(cal) & set(ev))
    cmax = {k: max(cal[k]) for k in keys}
    emax = {k: max(ev[k]) for k in keys}
    print(f'\n{len(keys)} (layer, head) pairs, {len(cal[keys[0]])} query blocks each\n')

    # 1. how much does demand vary BETWEEN heads vs WITHIN a head?
    within = st.mean(st.pstdev(cal[k]) for k in keys)
    between = st.pstdev([st.mean(cal[k]) for k in keys])
    print(f'  spread WITHIN a head (mean sd over query blocks): {within:5.2f} blocks')
    print(f'  spread BETWEEN heads (sd of per-head means):      {between:5.2f} blocks')
    print(f'  -> between/within = {between/within:.2f}x  ' +
          ('(head identity dominates: a rule is plausible)' if between > within
           else '(noise dominates: no stable per-head rule)'))

    # 2. does a budget calibrated on one dataset transfer to another?
    over = sum(1 for k in keys if emax[k] > cmax[k])
    slack = [emax[k] - cmax[k] for k in keys]
    print(f'\n  calibrate max on {a.calib}, evaluate on {a.eval}:')
    print(f'    heads whose eval demand EXCEEDS the calibrated budget: {over}/{len(keys)} '
          f'({over/len(keys):.0%})')
    print(f'    slack needed: mean {st.mean(slack):+.1f}, p90 {sorted(slack)[int(.9*len(slack))]:+d}, '
          f'max {max(slack):+d} blocks')

    # 3. what would it save? cost = 358*chunks(u) + 93*u, per op, u shared across heads
    def chunks(u, M=8, bs=64):
        kv = u*bs; lim = max(bs, min(((kv-1)//2)//bs*bs, min(u, M)*bs)//bs*bs)
        return -(-u // max(x for x in range(1, lim//bs+1) if u % x == 0))
    UF = 40
    per_layer = {}
    for (li, h) in keys:
        per_layer.setdefault(li, []).append(max(emax[(li, h)], 3))
    now = 358*chunks(UF) + 93*UF
    # per-head: the op cost is dominated by the max over heads (they run in one op today);
    # a true per-head kernel would pay the MEAN instead.
    ideal = st.mean(358*chunks(min(64, max(3, int(st.mean(v))))) + 93*st.mean(v) for v in per_layer.values())
    capped = st.mean(358*chunks(min(64, max(3, max(v)))) + 93*max(v) for v in per_layer.values())
    print(f'\n  cost model, per op:')
    print(f'    today, fixed u={UF}                     {now:7.0f} us')
    print(f'    per-LAYER u at that layer\'s max head   {capped:7.0f} us   {now/capped:.2f}x')
    print(f'    per-HEAD u (mean over heads)           {ideal:7.0f} us   {now/ideal:.2f}x')
    json.dump({'within': within, 'between': between, 'over': over, 'n': len(keys),
               'now': now, 'capped': capped, 'ideal': ideal}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
