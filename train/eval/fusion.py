"""Fusion evaluation: BIO confidence × attention LoRA score, sweep α.

For each candidate c with bio_conf b and attn_score a (both rank-normalized to [0,1]):
    final_score(α) = α * b_norm + (1-α) * a_norm

Sweeps α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (and any user-provided values).
α=1.0 is BIO alone, α=0.0 is attention alone.

Adapter dirs are evaluated against thucnews_heldout (true held-out).
"""
import sys, json, time, random, argparse
sys.path.insert(0, "/root/Keyatten")
sys.path.insert(0, "/root/Keyatten/train")

from copy import copy
from pathlib import Path

import numpy as np
import torch
from keyatten import BIOExtractor
from benchmark.keyword_bench.data import Document, load_shencecup_labeled, load_multi_domain_jsonl
from benchmark.keyword_bench.metrics import evaluate_predictions

THUCNEWS_PATH = Path("/root/Keyatten/train/data/thucnews_annotated.jsonl")
THUCNEWS_TRAIN_LIMIT = 8000
DEFAULT_BIO_CKPT = "/root/Keyatten/train/remote_pull_resume16_epoch13/best_full_ckpt.pt"


def filter_extractive(docs):
    out = []
    for d in docs:
        ext = [k for k in d.keywords if k in d.text]
        if ext:
            d2 = copy(d)
            d2.keywords = ext
            out.append(d2)
    return out


def load_thucnews_heldout():
    docs = []
    with open(THUCNEWS_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[THUCNEWS_TRAIN_LIMIT:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj["text"].strip()
        kws = [k for k in obj.get("keywords", []) if k in text]
        if text and kws:
            docs.append(Document(
                doc_id=f"thucnews-heldout-{len(docs)}",
                text=text,
                keywords=kws,
                language="zh",
            ))
    return docs


def rank_normalize(scores):
    """Rank-normalize scores to [0,1]: best=1.0, worst=0.0. Ties get same rank."""
    if not scores:
        return []
    n = len(scores)
    if n == 1:
        return [1.0]
    # Use scipy-like rankdata: average rank for ties
    indexed = sorted(range(n), key=lambda i: -scores[i])  # desc
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[indexed[j + 1]] == scores[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2  # 0-indexed avg rank
        normalized = 1.0 - avg_rank / max(n - 1, 1)
        for k in range(i, j + 1):
            ranks[indexed[k]] = normalized
        i = j + 1
    return ranks


def score_candidates_fusion(doc, tokenizer, model, device, instruction_prefix,
                             max_length, layer_idx, bio_extractor,
                             bio_profile, attn_method, alphas):
    """Like score_candidates_with_attn but:
    - Returns dict {alpha: top_ranked_candidates}
    - Uses BOTH bio confidence AND attn score, fused by α
    """
    from train.jobs.attn_lora import find_candidate_occurrences

    model.eval()
    scored = bio_extractor.extract_spans_profile(doc.text, profile=bio_profile)
    if not scored:
        return {a: [] for a in alphas}
    candidates = [(c, float(b)) for c, b in scored]  # (text, bio_conf)
    cand_texts = [c for c, _ in candidates]
    cand_bio = [b for _, b in candidates]

    # Run attention forward
    full_text = instruction_prefix + doc.text
    prefix_len = len(instruction_prefix)
    enc = tokenizer(full_text, max_length=max_length, truncation=True, padding=False,
                    return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = enc["offset_mapping"][0].tolist()
    enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    valid_len = int(enc["attention_mask"][0].sum().item())
    attn_map = outputs.attentions[layer_idx].mean(dim=1)[0, :valid_len, :valid_len]

    if attn_method == "eos_attn":
        token_scores_t = attn_map[valid_len - 1, :]
    elif attn_method == "cls_attn":
        token_scores_t = attn_map[0, :]   # [CLS] -> all tokens
    elif attn_method == "received_attn":
        token_scores_t = attn_map.sum(dim=0)
    else:
        idx_t = torch.arange(1, valid_len + 1, device=device, dtype=attn_map.dtype) / valid_len
        token_scores_t = attn_map.sum(dim=0) * idx_t

    token_scores = token_scores_t.detach().cpu().float().numpy()

    # Map token scores to char offsets
    char_scores = {}
    for tok_idx, (ts, te) in enumerate(offset_mapping[:valid_len]):
        if ts == te or ts < prefix_len:
            continue
        cs, ce = ts - prefix_len, te - prefix_len
        sc = float(token_scores[tok_idx])
        for c in range(cs, ce):
            if c not in char_scores or sc > char_scores[c]:
                char_scores[c] = sc

    # Compute attention score per candidate (mean char score across spans)
    cand_attn = []
    for cand in cand_texts:
        spans = find_candidate_occurrences(doc.text, cand)
        best = None
        for cs, ce in spans:
            vals = [char_scores[c] for c in range(cs, ce) if c in char_scores]
            if vals:
                s = sum(vals) / len(vals)
                if best is None or s > best:
                    best = s
        cand_attn.append(best if best is not None else 0.0)

    # Rank-normalize
    bio_norm = rank_normalize(cand_bio)
    attn_norm = rank_normalize(cand_attn)

    # Fusion across alphas
    out = {}
    for alpha in alphas:
        scores = [alpha * bio_norm[i] + (1.0 - alpha) * attn_norm[i] for i in range(len(cand_texts))]
        order = sorted(range(len(cand_texts)), key=lambda i: -scores[i])
        out[alpha] = [cand_texts[i] for i in order]
    return out


def eval_attn_fusion(adapter_dir, args, datasets, bio):
    sys.path.insert(0, "/root/Keyatten/train")
    from train.jobs.attn_lora import INSTRUCTION_PREFIX
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    print(f"  [attn-fusion] loading base {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    base = AutoModel.from_pretrained(args.base_model, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16, attn_implementation="eager")
    if adapter_dir.lower() in ("base", "none", ""):
        print(f"  [attn-fusion] using BASE model (no LoRA adapter)")
        model = base.to(args.device)
    else:
        print(f"  [attn-fusion] loading adapter {adapter_dir}")
        model = PeftModel.from_pretrained(base, adapter_dir).to(args.device)
    model.eval()

    out = {"adapter_dir": adapter_dir}
    for name, docs in datasets.items():
        t0 = time.perf_counter()
        # preds[alpha] = list of pred_lists
        preds_per_alpha = {a: [] for a in args.alphas}
        golds = []
        for i, doc in enumerate(docs, 1):
            ranked = score_candidates_fusion(
                doc, tok, model, args.device, INSTRUCTION_PREFIX,
                args.max_length, args.layer, bio, args.bio_profile,
                args.attn_method, args.alphas,
            )
            for a in args.alphas:
                preds_per_alpha[a].append(ranked[a])
            golds.append(doc.keywords)
            if i % 100 == 0:
                print(f"    [attn-fusion/{name}] {i}/{len(docs)} ({time.perf_counter()-t0:.1f}s)")

        per_alpha_metrics = {}
        for a in args.alphas:
            m = evaluate_predictions(preds_per_alpha[a], golds)
            per_alpha_metrics[str(a)] = m
            line = (f"  [α={a:.2f}/{name}] F1@5={m.get('f1@5',0):.4f}  F1@10={m.get('f1@10',0):.4f}  "
                    f"P@10={m.get('p@10',0):.4f}  R@10={m.get('r@10',0):.4f}")
            print(line)
        out[name] = per_alpha_metrics

    del model, base
    torch.cuda.empty_cache()
    return out


def load_eval_datasets(args, rng):
    sets = {}
    if "thucnews_heldout" in args.datasets:
        d = load_thucnews_heldout()
        sets["thucnews_heldout"] = d
        print(f"[info] thucnews_heldout: {len(d)} docs")

    if "shence_test" in args.datasets:
        shence_root = Path("/root/Keyatten")
        shence_all = load_shencecup_labeled(shence_root)
        rng.shuffle(shence_all)
        pool = min(200, len(shence_all) // 5)
        test_pool = shence_all[:pool]
        s_test = test_pool[pool // 2:][:100]
        s_test = filter_extractive(s_test)
        sets["shence_test"] = s_test
        print(f"[info] shence_test: {len(s_test)} docs")

    return sets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["thucnews_heldout"],
                    choices=["thucnews_heldout", "shence_test"])
    ap.add_argument("--bio-ckpt", default=DEFAULT_BIO_CKPT)
    ap.add_argument("--bio-profile", default="balanced",
                    choices=["clean", "balanced", "high_recall"])
    ap.add_argument("--adapter-dirs", nargs="+", required=True,
                    help="Attn LoRA adapter dirs to evaluate")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="Fusion alphas (1.0=BIO alone, 0.0=attn alone)")
    ap.add_argument("--base-model", default="/root/Keyatten/models/gte-small-zh")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--attn-method", default="received_attn")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default="/root/Keyatten/outputs/eval_fusion.json")
    args = ap.parse_args(argv)

    rng = random.Random(args.split_seed)
    datasets = load_eval_datasets(args, rng)

    print(f"[info] BIO ckpt: {args.bio_ckpt}, profile: {args.bio_profile}")
    bio = BIOExtractor(args.bio_ckpt, device=args.device)

    results = {}
    for adapter_dir in args.adapter_dirs:
        print(f"\n=== Adapter: {adapter_dir} ===")
        results[adapter_dir] = eval_attn_fusion(adapter_dir, args, datasets, bio)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[info] Saved → {args.out}")

    # Pretty summary
    print("\n" + "=" * 110)
    print(f"{'adapter':<55} | {'dataset':<18} | " +
          " | ".join(f"α={a:.2f} F1@10" for a in args.alphas))
    print("-" * 110)
    for adapter_dir, res in results.items():
        for ds in args.datasets:
            cells = [f"{res.get(ds, {}).get(str(a), {}).get('f1@10', 0.0):.4f}"
                     for a in args.alphas]
            adapter_short = adapter_dir.split("/")[-2] if "/" in adapter_dir else adapter_dir
            print(f"{adapter_short:<55} | {ds:<18} | " + " | ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
