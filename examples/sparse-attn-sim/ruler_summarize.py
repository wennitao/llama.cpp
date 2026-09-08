#!/usr/bin/env python3
"""Score RULER predictions per arm with RULER's own metrics and print a per-task table (arms as columns)."""
import json, os, re, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

def string_match_part(preds, refs):
    return 100.0 * sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref) for p, ref in zip(preds, refs)) / len(preds)

def string_match_all(preds, refs):
    return 100.0 * sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref) for p, ref in zip(preds, refs)) / len(preds)

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]

def score(pred_dir, task):
    f = os.path.join(pred_dir, f"{task}.jsonl")
    if not os.path.exists(f):
        return None, 0
    lines = [json.loads(l) for l in open(f)]
    if not lines:
        return None, 0
    np_pattern = re.compile(r"[\x00-\x1f]")
    preds = [np_pattern.sub("\n", l["pred"].strip()).strip() for l in lines]
    refs = [l["outputs"] for l in lines]
    fn = string_match_part if task.startswith("qa") else string_match_all
    return fn(preds, refs), len(lines)

def main():
    root = sys.argv[1]
    arms = sys.argv[2:] if len(sys.argv) > 2 else sorted(os.listdir(root))
    rows = []
    for t in TASKS:
        row = [t]
        for a in arms:
            s, n = score(os.path.join(root, a), t)
            row.append(f"{s:.1f}" + ("" if n >= 100 else f" ({n})") if s is not None else "--")
        rows.append(row)
    avgs = ["**avg**"]
    for a in arms:
        vals = [score(os.path.join(root, a), t)[0] for t in TASKS]
        vals = [v for v in vals if v is not None]
        avgs.append(f"**{sum(vals) / len(vals):.1f}**" + ("" if len(vals) == len(TASKS) else f" ({len(vals)}/{len(TASKS)})") if vals else "--")
    print("| task | " + " | ".join(arms) + " |")
    print("|---|" + "--:|" * len(arms))
    for r in rows + [avgs]:
        print("| " + " | ".join(r) + " |")

if __name__ == "__main__":
    main()
