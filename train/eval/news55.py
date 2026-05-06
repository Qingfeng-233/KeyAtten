#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import copy
from pathlib import Path

import torch

sys.path.insert(0, "/root/Keyatten")
sys.path.insert(0, "/root/Keyatten/benchmark")
sys.path.insert(0, "/root/Keyatten/train")

from benchmark.keyword_bench.data import Document
from benchmark.keyword_bench.metrics import evaluate_predictions
from keyatten import BIOExtractor
from train.eval.fusion import eval_attn_fusion

PROJECT_ROOT = Path("/root/Keyatten")
DEFAULT_DATA = PROJECT_ROOT / "data" / "news_annotated.jsonl"
DEFAULT_BIO_CKPT = PROJECT_ROOT / "train" / "remote_pull_resume16_epoch13" / "best_full_ckpt.pt"


def load_news55(path: Path) -> tuple[list[Document], dict[str, int]]:
    docs: list[Document] = []
    total_keywords = 0
    kept_keywords = 0
    dropped_docs = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            keywords = [str(kw).strip() for kw in obj.get("keywords", []) if str(kw).strip()]
            total_keywords += len(keywords)
            extractive = [kw for kw in keywords if kw in text]
            kept_keywords += len(extractive)
            if not text or not extractive:
                dropped_docs += 1
                continue
            doc_id = str(obj.get("id") or f"line-{line_no}")
            docs.append(Document(doc_id=doc_id, text=text, keywords=extractive, language="zh"))
    stats = {
        "docs": len(docs),
        "dropped_docs": dropped_docs,
        "total_keywords": total_keywords,
        "kept_extractive_keywords": kept_keywords,
    }
    return docs, stats


def metric_line(method: str, metrics: dict[str, float]) -> str:
    return (
        f"[{method}] F1@5={metrics.get('f1@5', 0):.4f} "
        f"F1@10={metrics.get('f1@10', 0):.4f} "
        f"P@10={metrics.get('p@10', 0):.4f} R@10={metrics.get('r@10', 0):.4f}"
    )


def save_progress(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[progress] saved {out_path}", flush=True)


def eval_bio_only(docs: list[Document], bio: BIOExtractor, profile: str) -> dict[str, float]:
    preds = []
    golds = []
    for idx, doc in enumerate(docs, 1):
        preds.append([cand for cand, _ in bio.extract_spans_profile(doc.text, profile=profile)])
        golds.append(doc.keywords)
        if idx % 10 == 0:
            print(f"[bio] {idx}/{len(docs)}", flush=True)
    metrics = evaluate_predictions(preds, golds)
    print(metric_line("BIO clean", metrics), flush=True)
    return metrics


def run_attn_adapters(args: argparse.Namespace, docs: list[Document], bio: BIOExtractor) -> dict:
    ns = copy(args)
    ns.datasets = ["news55"]
    ns.layer = args.attn_layer
    ns.attn_method = args.attn_method
    ns.adapter_dirs = args.adapter_dirs
    results = {}
    datasets = {"news55": docs}
    for adapter_dir in args.adapter_dirs:
        print(f"[attn] adapter={adapter_dir}", flush=True)
        results[adapter_dir] = eval_attn_fusion(adapter_dir, ns, datasets, bio)
    return {"method": "attention_lora_bio_fusion", "news55": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all available methods on clean news55 annotations.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default="/root/Keyatten/outputs/news55_all_methods_eval.json")
    parser.add_argument("--base-model", default="/root/Keyatten/models/gte-small-zh")
    parser.add_argument("--bio-ckpt", default=str(DEFAULT_BIO_CKPT))
    parser.add_argument("--bio-profile", choices=("clean", "balanced", "high_recall"), default="clean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9, 1.0])
    parser.add_argument("--methods", nargs="+", choices=("bio", "attn"), default=["bio", "attn"])
    parser.add_argument("--adapter-dirs", nargs="+", default=[
        "/root/Keyatten/models/exp1_soft_kl_1k/best_adapter",
        "/root/Keyatten/models/exp5_soft_kl_full_10k/best_adapter",
        "/root/Keyatten/models/exp_llm_qwen35_1k/best_adapter",
    ])
    parser.add_argument("--attn-layer", type=int, default=4)
    parser.add_argument("--attn-method", default="received_attn")
    args = parser.parse_args(argv)
    args.device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.out)
    docs, dataset_stats = load_news55(Path(args.data))
    print(f"[info] news55 docs={len(docs)} stats={dataset_stats}", flush=True)
    started = time.perf_counter()
    results = {
        "dataset": {
            "name": "news55_clean",
            "path": args.data,
            "stats": dataset_stats,
            "note": "User-provided clean annotated news set. Evaluation filters keywords to extractive keywords present in text.",
        },
        "config": vars(args),
        "results": {},
    }
    save_progress(out_path, results)

    bio = BIOExtractor(args.bio_ckpt, device=args.device)
    if "bio" in args.methods:
        results["results"]["bio_clean"] = {"method": "bio_clean", "news55": eval_bio_only(docs, bio, args.bio_profile)}
        save_progress(out_path, results)
    if "attn" in args.methods:
        results["results"]["attn"] = run_attn_adapters(args, docs, bio)
        save_progress(out_path, results)

    results["seconds"] = round(time.perf_counter() - started, 2)
    save_progress(out_path, results)
    print("[done] news55 all-method evaluation complete", flush=True)


if __name__ == "__main__":
    main()
