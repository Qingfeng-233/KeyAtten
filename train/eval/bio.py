"""BIO-only baseline: rank candidates by BIO confidence, no attention LoRA."""
import sys, json, time, random, argparse
sys.path.insert(0, "/root/Keyatten")

from copy import copy
from pathlib import Path

from keyatten import BIOExtractor
from benchmark.keyword_bench.data import load_shencecup_labeled, load_multi_domain_jsonl
from benchmark.keyword_bench.metrics import evaluate_predictions


def filter_extractive(docs):
    out = []
    for d in docs:
        ext = [k for k in d.keywords if k in d.text]
        if ext:
            d2 = copy(d)
            d2.keywords = ext
            out.append(d2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="clean", choices=["clean", "balanced", "high_recall"])
    ap.add_argument("--ckpt", default="/root/Keyatten/train/remote_pull_resume16_epoch13/best_full_ckpt.pt")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--shence-test-pool", type=int, default=200)
    ap.add_argument("--shence-test-limit", type=int, default=100)
    ap.add_argument("--md-test-limit", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[info] BIO profile={args.profile}, ckpt={args.ckpt}")
    bio = BIOExtractor(args.ckpt, device=args.device)

    rng = random.Random(args.split_seed)

    # ShenCe test (same split as train_attn_lora.py main):
    shence_root = None
    for cand in [Path("/root/Keyatten")]:
        raw = cand / "data" / "shencecup" / "raw"
        if (raw / "all_docs.txt").exists():
            shence_root = cand
            break
    shence_all = load_shencecup_labeled(shence_root)
    rng.shuffle(shence_all)
    shence_test_pool = shence_all[: args.shence_test_pool]
    shence_test = shence_test_pool[args.shence_test_pool // 2:]   # 后一半 = test
    shence_test = shence_test[: args.shence_test_limit]
    shence_test = filter_extractive(shence_test)
    print(f"[info] ShenCe test: {len(shence_test)} docs")

    # MD test
    md_path = Path("/root/Keyatten/data/multi_domain.jsonl")
    if not md_path.exists():
        md_path = Path("/root/Keyatten/data/multi_domain/annotated.jsonl")
    md_all = load_multi_domain_jsonl(md_path)
    rng.shuffle(md_all)
    md_test_size = min(1000, len(md_all) // 5)
    md_test = md_all[:md_test_size][: args.md_test_limit]
    md_test = filter_extractive(md_test)
    print(f"[info] MD test: {len(md_test)} docs")

    results = {"profile": args.profile}

    for name, docs in [("shence_test", shence_test), ("md_test", md_test)]:
        t0 = time.perf_counter()
        preds, golds = [], []
        for i, doc in enumerate(docs, 1):
            scored = bio.extract_spans_profile(doc.text, profile=args.profile)
            # Already sorted by confidence desc
            cands = [c for c, _ in scored]
            preds.append(cands)
            golds.append(doc.keywords)
            if i % 100 == 0:
                print(f"  [{name}] {i}/{len(docs)} ({time.perf_counter() - t0:.1f}s)")
        m = evaluate_predictions(preds, golds)
        elapsed = time.perf_counter() - t0
        f1_5 = m.get("f1@5", 0.0)
        f1_10 = m.get("f1@10", 0.0)
        p5 = m.get("p@5", 0.0)
        r5 = m.get("r@5", 0.0)
        p10 = m.get("p@10", 0.0)
        r10 = m.get("r@10", 0.0)
        line = (f"[result/{name}] F1@5={f1_5:.4f}  F1@10={f1_10:.4f}  "
                f"P@5={p5:.4f}  R@5={r5:.4f}  "
                f"P@10={p10:.4f}  R@10={r10:.4f}  "
                f"({len(docs)} docs, {elapsed:.1f}s)")
        print(line)
        results[name] = m

    out = Path(f"/root/Keyatten/outputs/eval_bio_{args.profile}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[info] Saved → {out}")


if __name__ == "__main__":
    main()
