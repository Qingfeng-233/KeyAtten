from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
EVAL_DIR = BENCHMARK_ROOT / "eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from benchmark.scripts.eval_bio_qk_combo import (  # noqa: E402
    extract_bio_candidates,
    rank_candidates_with_fusion,
    score_candidates_with_qk_from_list,
)
from benchmark.scripts.run_shence_heldout_eval import build_shence_split  # noqa: E402
from keyword_bench.data import Document, load_multi_domain_jsonl  # noqa: E402
from keyword_bench.metrics import evaluate_predictions  # noqa: E402
from keyatten import BIOExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local BIO/QK evaluation matrix for ShenCe and multi-domain splits.")
    parser.add_argument("--bio-checkpoint", default=str(REPO_ROOT / "models" / "bio_ckipbert_extractive_ep13" / "bio_model_full.pt"))
    parser.add_argument("--qk-model", default=str(REPO_ROOT / "models" / "Qwen3-Embedding-0.6B"))
    parser.add_argument("--qk-adapter", action="append", default=[])
    parser.add_argument("--qk-layer", type=int, default=21)
    parser.add_argument("--data-root", default=str(REPO_ROOT))
    parser.add_argument("--md-jsonl", default=str(REPO_ROOT / "train" / "data" / "multi_domain.jsonl"))
    parser.add_argument("--datasets", nargs="+", choices=("shence", "md"), default=["shence", "md"])
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[50])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bio-mode", choices=("profile", "relaxed"), default="profile")
    parser.add_argument("--bio-profile", choices=("balanced", "clean", "high_recall"), default="clean")
    parser.add_argument("--b-threshold", type=float, default=0.15)
    parser.add_argument("--fusion-betas", type=float, nargs="*", default=[0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--skip-qk", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "outputs_bio_local_matrix"))
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def build_md_split(path: Path, split_seed: int, limit: int | None) -> List[Document]:
    docs = load_multi_domain_jsonl(path)
    rng = random.Random(split_seed)
    rng.shuffle(docs)
    test_size = min(1000, len(docs) // 5)
    docs = docs[:test_size]
    return docs[:limit] if limit is not None else docs


def build_datasets(args: argparse.Namespace) -> Dict[str, List[Document]]:
    datasets: Dict[str, List[Document]] = {}
    if "shence" in args.datasets:
        splits = build_shence_split(args.data_root)
        docs = splits["test_final"]
        datasets["shence"] = docs[: args.limit] if args.limit is not None else docs
    if "md" in args.datasets:
        datasets["md"] = build_md_split(Path(args.md_jsonl), args.split_seed, args.limit)
    return datasets


def evaluate_bio_only_with_progress(
    docs: Sequence[Document],
    bio: BIOExtractor,
    *,
    candidate_k: int,
    top_k: int,
    b_threshold: float,
    bio_mode: str,
    bio_profile: str,
    label: str,
) -> Dict[str, float]:
    candidate_predictions: List[List[str]] = []
    topk_predictions: List[List[str]] = []
    golds: List[List[str]] = []

    for index, doc in enumerate(docs, start=1):
        candidates = extract_bio_candidates(
            bio,
            doc.text,
            max_spans=candidate_k,
            b_threshold=b_threshold,
            bio_mode=bio_mode,
            bio_profile=bio_profile,
        )
        candidate_texts = [text for text, _ in candidates]
        candidate_predictions.append(candidate_texts)
        topk_predictions.append(candidate_texts[:top_k])
        golds.append(doc.keywords)
        if index % 50 == 0 or index == len(docs):
            print(f"[{label}] bio processed {index}/{len(docs)}")

    topk_metrics = evaluate_predictions(topk_predictions, golds)
    total_recall = 0.0
    for preds, gold in zip(candidate_predictions, golds):
        if gold:
            total_recall += sum(1 for keyword in gold if keyword in preds) / len(gold)

    return {
        "candidate_recall": total_recall / max(len(golds), 1),
        "p@5": topk_metrics.get("p@5", 0.0),
        "r@5": topk_metrics.get("r@5", 0.0),
        "f1@5": topk_metrics.get("f1@5", 0.0),
        "p@10": topk_metrics.get("p@10", 0.0),
        "r@10": topk_metrics.get("r@10", 0.0),
        "f1@10": topk_metrics.get("f1@10", 0.0),
    }


def evaluate_bio_qk_with_progress(
    docs: Sequence[Document],
    bio: BIOExtractor,
    qk_tokenizer,
    qk_model,
    *,
    device: str,
    qk_layer: int,
    candidate_k: int,
    top_k: int,
    b_threshold: float,
    bio_mode: str,
    bio_profile: str,
    fusion_betas: Sequence[float],
    label: str,
) -> Dict[str, Dict[str, float]]:
    qk_predictions: List[List[str]] = []
    fused_predictions: Dict[float, List[List[str]]] = {beta: [] for beta in fusion_betas}
    golds: List[List[str]] = []

    for index, doc in enumerate(docs, start=1):
        candidates = extract_bio_candidates(
            bio,
            doc.text,
            max_spans=candidate_k,
            b_threshold=b_threshold,
            bio_mode=bio_mode,
            bio_profile=bio_profile,
        )
        qk_ranked, bio_scores, qk_scores = score_candidates_with_qk_from_list(
            doc,
            candidates,
            qk_tokenizer,
            qk_model,
            device,
            qk_layer,
            top_k,
        )
        qk_predictions.append(qk_ranked)
        for beta in fusion_betas:
            fused_predictions[beta].append(
                rank_candidates_with_fusion(candidates, bio_scores, qk_scores, beta=beta, top_k=top_k)
            )
        golds.append(doc.keywords)
        if index % 20 == 0 or index == len(docs):
            print(f"[{label}] qk processed {index}/{len(docs)}")

    metrics: Dict[str, Dict[str, float]] = {"qk_only": evaluate_predictions(qk_predictions, golds)}
    for beta, predictions in fused_predictions.items():
        metrics[f"fusion_beta_{beta:g}"] = evaluate_predictions(predictions, golds)
    return metrics


def load_qk(adapter_path: str, qk_model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(qk_model_path, trust_remote_code=True, use_fast=True)
    base = AutoModel.from_pretrained(qk_model_path, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(args)
    print(f"[info] datasets={ {name: len(docs) for name, docs in datasets.items()} }")
    print(f"[info] loading BIO checkpoint: {args.bio_checkpoint}")
    bio = BIOExtractor(args.bio_checkpoint, device=args.device)

    results: Dict[str, Dict[str, float]] = {}
    for dataset_name, docs in datasets.items():
        for candidate_k in args.candidate_k:
            key = f"{dataset_name}/bio_only@{candidate_k}"
            metrics = evaluate_bio_only_with_progress(
                docs,
                bio,
                candidate_k=candidate_k,
                top_k=args.top_k,
                b_threshold=args.b_threshold,
                bio_mode=args.bio_mode,
                bio_profile=args.bio_profile,
                label=key,
            )
            results[key] = metrics
            print(json.dumps({key: metrics}, ensure_ascii=False))

    if not args.skip_qk:
        for adapter_path in args.qk_adapter:
            adapter_name = Path(adapter_path).parent.name if Path(adapter_path).name == "best_adapter" else Path(adapter_path).name
            print(f"[info] loading QK adapter: {adapter_path}")
            tokenizer, qk_model = load_qk(adapter_path, args.qk_model, args.device)
            for dataset_name, docs in datasets.items():
                for candidate_k in args.candidate_k:
                    label = f"{dataset_name}/{adapter_name}@{candidate_k}"
                    qk_metrics = evaluate_bio_qk_with_progress(
                        docs,
                        bio,
                        tokenizer,
                        qk_model,
                        device=args.device,
                        qk_layer=args.qk_layer,
                        candidate_k=candidate_k,
                        top_k=args.top_k,
                        b_threshold=args.b_threshold,
                        bio_mode=args.bio_mode,
                        bio_profile=args.bio_profile,
                        fusion_betas=args.fusion_betas,
                        label=label,
                    )
                    for method, metrics in qk_metrics.items():
                        key = f"{label}/{method}"
                        results[key] = metrics
                        print(json.dumps({key: metrics}, ensure_ascii=False))

    payload = {
        "bio_checkpoint": str(Path(args.bio_checkpoint).resolve()),
        "qk_model": str(Path(args.qk_model).resolve()),
        "qk_adapters": [str(Path(path).resolve()) for path in args.qk_adapter],
        "qk_layer": args.qk_layer,
        "datasets": {name: len(docs) for name, docs in datasets.items()},
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "bio_mode": args.bio_mode,
        "bio_profile": args.bio_profile,
        "b_threshold": args.b_threshold,
        "fusion_betas": args.fusion_betas,
        "split_seed": args.split_seed,
        "results": results,
    }
    output_path = output_dir / "bio_local_matrix_results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] saved to {output_path}")


if __name__ == "__main__":
    main()
