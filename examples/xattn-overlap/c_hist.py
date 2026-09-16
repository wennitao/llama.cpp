#!/usr/bin/env python3
"""How many of the R fine sub-blocks want each KV block, under the DEPLOYED scorer?

The hetero split sends blocks wanted by >= c* of the R sub-blocks to the NPU and the rest to the
GPU, so its whole cost model is the histogram n_i = #{b : c(b) = i}. The published histogram was
solved from a top-k-at-25% table; the shipped pipeline runs a THRESHOLD rule instead
(llama-graph.cpp: keep b when softmax(sc)*n_avail > c, plus a forced sink and diagonal), and the
two need not agree. This measures both on the same activations.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_scoring import Capturer, load_texts, sc_meanpool, _blocks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--data', nargs='+', default=['longbench:narrativeqa'])
    ap.add_argument('--docs', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--bq', type=int, default=256)
    ap.add_argument('--dens', type=float, default=0.25)
    ap.add_argument('--thr', type=float, default=1.0)
    ap.add_argument('--min-reach', type=float, default=0.0,
                    help='only score query groups whose reach is >= this fraction of the context')
    ap.add_argument('--out', default='c_hist.json')
    a = ap.parse_args()

    cap = Capturer(a.model)
    R = a.bq // a.bs
    acc = {p: {'n': torch.zeros(R + 1, dtype=torch.float64), 'u': 0.0, 'g': 0}
           for p in ('thresh', 'topk')}

    for spec in a.data:
        for text in load_texts(spec, a.docs):
            qk, T = cap(text, a.tokens, None)
            NB = _blocks(T, a.bs)
            for q, k in qk:
                sc = sc_meanpool(q, k, a.bs).float()            # [NBq_fine, Hkv, NBk]
                Hkv = sc.shape[1]
                for g in range(NB // R):
                    avail = (g + 1) * R          # last fine block's reach, in blocks
                    if avail < 4 * R or avail < a.min_reach * NB:
                        continue
                    for pol in ('thresh', 'topk'):
                        keep = torch.zeros(R, Hkv, avail, dtype=torch.bool, device=sc.device)
                        for j in range(R):
                            aa = g * R + j
                            av = aa + 1                          # this fine row's own reach
                            row = sc[aa, :, :av]
                            if pol == 'thresh':
                                p = torch.softmax(row, -1) * av
                                sel = p > a.thr
                            else:
                                u0 = max(1, int(round(a.dens * av)))
                                sel = torch.zeros_like(row, dtype=torch.bool).scatter_(
                                    1, row.topk(min(u0, av), -1).indices, True)
                            sel[:, 0] = True                     # forced sink
                            sel[:, av - 1] = True                # forced diagonal
                            keep[j, :, :av] = sel
                        c = keep.sum(0)                          # [Hkv, avail], 0..R
                        for i in range(1, R + 1):
                            acc[pol]['n'][i] += (c == i).sum().item() / Hkv
                        acc[pol]['u'] += (c >= 1).sum().item() / Hkv
                        acc[pol]['g'] += 1

    out = {}
    for pol, d in acc.items():
        g = max(d['g'], 1)
        n = [float(d['n'][i]) / g for i in range(R + 1)]
        u = d['u'] / g
        row = {'n': n[1:], 'u': u}
        for cs in range(2, R + 1):
            shared = sum(n[i] for i in range(cs, R + 1))
            exc    = sum(n[i] * i for i in range(1, cs))
            row[f'c{cs}'] = {'shared': shared, 'exc_pairs': exc}
        out[pol] = row
        print(f"{pol:8s} u={u:6.2f}  n1..n{R}=" + " ".join(f"{x:6.2f}" for x in n[1:]))
        for cs in range(2, R + 1):
            print(f"           c*={cs}: shared {row[f'c{cs}']['shared']:6.2f}  "
                  f"exception pairs {row[f'c{cs}']['exc_pairs']:6.2f}")
    json.dump(out, open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
