#!/usr/bin/env python3
"""Emulate the Snapdragon HTP sparse decode attention in PyTorch and run RULER prompts through it.

Decode-time attention per layer / KV head (q_len == 1), as the HTP kernel does it:
  * dense recent tail: keys [ds, n), ds = floor((n - W) / 64) * 64  (W = window)
  * candidate pages over [0, ds): PK consecutive keys (positional) or PK keys of one k-means cluster
    (cluster: k-means over the prompt keys, keys sorted by (cluster, distance), cut into pages;
    the sink keys (positions < sink) are sorted first so page 0 holds them)
  * page descriptor = mean of its keys; score = max over the GQA group of q . descriptor
  * page 0 always selected (sink page); top-B pages by score, B = ceil(density * n_cand)
  * softmax over the selected pages' keys + the dense tail only
Prefill (q_len > 1) is dense.  Model: Qwen/Qwen3-1.7B (bf16).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

CFG = dict(mode="dense", page=64, density=0.25, window=64, sink=4, iters=8)
STATE = {}       # layer_idx -> dict(perm=[Hkv, n0] LongTensor or None, n0=int)
STATS = dict(steps=0, keys_attended=0, keys_total=0)


def repeat_kv(x, G):
    B, H, n, D = x.shape
    return x[:, :, None].expand(B, H, G, n, D).reshape(B, H * G, n, D)


@torch.no_grad()
def kmeans_order(K, PK, iters, sink, seed):
    """K: [Hkv, n0, D] (f32). Returns perm [Hkv, n0]: keys ordered by (cluster, distance), sinks first."""
    Hkv, n0, D = K.shape
    C = n0 // PK
    g = torch.Generator(device=K.device).manual_seed(seed)
    init = torch.stack([torch.randperm(n0, generator=g, device=K.device)[:C] for _ in range(Hkv)])   # [Hkv, C]
    mu = torch.gather(K, 1, init[..., None].expand(-1, -1, D)).clone()                                  # [Hkv, C, D]
    k2 = (K * K).sum(-1, keepdim=True)                                                                  # [Hkv, n0, 1]
    for _ in range(iters):
        d = k2 - 2 * K @ mu.transpose(1, 2) + (mu * mu).sum(-1)[:, None, :]                            # [Hkv, n0, C]
        assign = d.argmin(-1)                                                                          # [Hkv, n0]
        onehot = F.one_hot(assign, C).to(K.dtype)                                                       # [Hkv, n0, C]
        cnt = onehot.sum(1)                                                                            # [Hkv, C]
        new_mu = onehot.transpose(1, 2) @ K                                                            # [Hkv, C, D]
        keep = cnt > 0
        mu = torch.where(keep[..., None], new_mu / cnt.clamp(min=1)[..., None], mu)
    d = k2 - 2 * K @ mu.transpose(1, 2) + (mu * mu).sum(-1)[:, None, :]
    dist, assign = d.min(-1)
    key = assign.to(torch.float64) * 1e6 + (dist.to(torch.float64) - dist.min()).clamp(min=0) / (dist.max() - dist.min() + 1e-9) * 1e5
    key[:, :sink] = -1.0    # sinks first -> page 0
    return key.argsort(dim=1)


def sparse_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    B, H, q_len, D = query.shape
    Hkv, n = key.shape[1], key.shape[2]
    G = H // Hkv
    li = module.layer_idx
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(D)
    if q_len > 1 or CFG["mode"] == "dense":
        out = F.scaled_dot_product_attention(query, repeat_kv(key, G), repeat_kv(value, G), is_causal=(q_len > 1), scale=scaling)
        if q_len > 1 and CFG["mode"] == "cluster":
            W, PK = CFG["window"], CFG["page"]
            ds0 = ((n - W) // 64) * 64 if n > W else 0
            n0 = (ds0 // PK) * PK
            perm = kmeans_order(key[0, :, :n0].float(), PK, CFG["iters"], CFG["sink"], seed=1234 + li) if n0 >= 2 * PK else None
            STATE[li] = dict(perm=perm, n0=n0 if perm is not None else 0)
        elif q_len > 1:
            STATE[li] = dict(perm=None, n0=0)
        return out.transpose(1, 2).contiguous(), None

    # ---- decode step ----
    W, PK, dens = CFG["window"], CFG["page"], CFG["density"]
    ds = ((n - W) // 64) * 64 if n > W else 0
    n_cand = ds // PK
    if n_cand < 2:
        out = F.scaled_dot_product_attention(query, repeat_kv(key, G), repeat_kv(value, G), scale=scaling)
        STATS["steps"] += 1; STATS["keys_attended"] += n * Hkv; STATS["keys_total"] += n * Hkv
        return out.transpose(1, 2).contiguous(), None
    st = STATE.get(li, dict(perm=None, n0=0))
    k = key[0]                                                                         # [Hkv, n, D]
    if st["perm"] is not None:
        n0 = min(st["n0"], ds)
        idx = torch.cat([st["perm"][:, :n0], torch.arange(n0, ds, device=k.device)[None].expand(Hkv, -1)], dim=1)   # [Hkv, ds]
        kc = torch.gather(k[:, :ds], 1, idx[..., None].expand(-1, -1, D))
    else:
        idx = None
        kc = k[:, :ds]
    desc = kc.view(Hkv, n_cand, PK, D).float().mean(2)                                  # [Hkv, n_cand, D]
    q = query[0, :, 0].float().view(Hkv, G, D)                                           # [Hkv, G, D]
    s = torch.einsum("hgd,hpd->hgp", q, desc).amax(1)                                   # [Hkv, n_cand] GQA fold: max
    s[:, 0] = float("inf")                                                             # sink page always selected
    Bp = max(1, min(n_cand, math.ceil(dens * n_cand)))
    top = s.topk(Bp, dim=1).indices                                                     # [Hkv, Bp]
    page_keys = (top[..., None] * PK + torch.arange(PK, device=k.device)).view(Hkv, -1)  # ordinal keys within [0, ds)
    sel = torch.gather(idx, 1, page_keys) if idx is not None else page_keys              # actual key positions
    mask = torch.zeros(Hkv, n, dtype=torch.bool, device=k.device)
    mask[:, ds:] = True
    mask.scatter_(1, sel, True)
    STATS["steps"] += 1; STATS["keys_attended"] += int(mask.sum()); STATS["keys_total"] += n * Hkv
    am = mask[:, None, :].expand(Hkv, G, n).reshape(1, H, 1, n)
    out = F.scaled_dot_product_attention(query, repeat_kv(key, G), repeat_kv(value, G), attn_mask=am, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("sparse_sim", sparse_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("sparse_sim", sdpa_mask)

TOK_GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


class FirstLineStop(StoppingCriteria):
    """Stop once the first answer line is complete (a newline after at least two non-blank characters).
    RULER scores by containment and its answers sit on the first line; the model otherwise rambles to
    the token limit, which costs 10x the compute for nothing."""
    def __init__(self, tok, prompt_len):
        self.tok, self.prompt_len = tok, prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:]
        stop = False
        if gen.shape[0] >= 2:
            text = self.tok.decode(gen, skip_special_tokens=True).lstrip()
            stop = len(text) >= 2 and "\n" in text[2:]
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


def tokens_to_generate(task):
    for k, v in TOK_GEN.items():
        if task.startswith(k):
            return v
    return 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="RULER data dir: <data_dir>/<task>/validation.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2")
    ap.add_argument("--mode", default="dense", choices=["dense", "positional", "cluster"])
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--density", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="samples per task (0 = all)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template (thinking off)")
    ap.add_argument("--full_gen", action="store_true", help="generate the full token budget (RULER default) instead of stopping after the first answer line")
    a = ap.parse_args()
    CFG.update(mode=a.mode, page=a.page, density=a.density, window=a.window, sink=a.sink)
    os.makedirs(a.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sparse_sim").cuda().eval()
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    eos = sorted(set(e for e in eos if e is not None))
    print(f"mode {a.mode} page {a.page} density {a.density} window {a.window} sink {a.sink}; eos {eos}", flush=True)

    for task in a.tasks.split(","):
        path = os.path.join(a.data_dir, task, "validation.jsonl")
        if not os.path.exists(path):
            print(f"[{task}] missing {path}", flush=True); continue
        lines = [json.loads(l) for l in open(path)]
        if a.limit:
            lines = lines[: a.limit]
        out_path = os.path.join(a.out_dir, f"{task}.jsonl")
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(l)["index"] for l in open(out_path)}
        n_gen = tokens_to_generate(task)
        t0 = time.time(); STATS.update(steps=0, keys_attended=0, keys_total=0)
        with open(out_path, "a") as fo:
            for ln in lines:
                if ln["index"] in done:
                    continue
                prompt = ln["input"] + ln.get("answer_prefix", "")   # RULER keeps the answer prefix in its own field
                if a.chat:
                    prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(prompt, return_tensors="pt").input_ids.cuda()
                STATE.clear()
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=n_gen, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0],
                                         temperature=None, top_p=None, top_k=None,
                                         stopping_criteria=None if a.full_gen else StoppingCriteriaList([FirstLineStop(tok, ids.shape[1])]))
                pred = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                rec = dict(ln); rec["pred"] = pred; rec.setdefault("others", {})
                fo.write(json.dumps(rec) + "\n"); fo.flush()
        frac = STATS["keys_attended"] / max(1, STATS["keys_total"])
        print(f"[{task}] {len(lines)} samples, {time.time() - t0:.0f} s, attended {100 * frac:.1f}% of keys over {STATS['steps']} decode steps", flush=True)


if __name__ == "__main__":
    main()
