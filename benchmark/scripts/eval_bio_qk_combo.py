from __future__ import annotations

import argparse
import json
import math
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

from benchmark.scripts.run_shence_heldout_eval import (  # noqa: E402
    TOP_K,
    MAX_LENGTH,
    build_shence_split,
    compute_qk_scores,
)
from keyword_bench.metrics import evaluate_predictions  # noqa: E402
from keyatten import BIOExtractor  # noqa: E402
from keyatten.candidates.bio_mining import find_candidate_occurrences  # noqa: E402
from keyatten.scoring import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BIO-only, BIO+QK, and fused BIO/QK on ShenCe held-out split.")
    parser.add_argument(
        "--bio-checkpoint",
        default=str(REPO_ROOT / "models" / "bio_ckipbert_extractive_ep13" / "best_bio_head.pt"),
    )
    parser.add_argument(
        "--qk-model",
        default=str(REPO_ROOT / "models" / "Qwen3-Embedding-0.6B"),
    )
    parser.add_argument(
        "--qk-adapter",
        default=str(REPO_ROOT / "models" / "qk_qwen0.6B" / "best_adapter"),
    )
    parser.add_argument("--qk-layer", type=int, default=21)
    parser.add_argument("--data-root", default=str(REPO_ROOT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[50, 80, 100])
    parser.add_argument("--b-threshold", type=float, default=0.15)
    parser.add_argument(
        "--bio-mode",
        choices=("relaxed", "profile"),
        default="profile",
        help="candidate extraction mode for BIO",
    )
    parser.add_argument(
        "--bio-profile",
        default="clean",
        help="BIO profile used when --bio-mode=profile",
    )
    parser.add_argument(
        "--fusion-betas",
        type=float,
        nargs="*",
        default=[0.7, 0.8, 0.85, 0.9, 0.95],
        help="beta values for final_score = beta * bio + (1-beta) * qk after per-doc normalization",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "outputs_bio_qk_combo"))
    return parser.parse_args()


def extract_bio_candidates(
    bio: BIOExtractor,
    text: str,
    *,
    max_spans: int,
    b_threshold: float,
    bio_mode: str,
    bio_profile: str,
) -> List[Tuple[str, float]]:
    if bio_mode == "profile":
        return bio.extract_spans_profile(text, profile=bio_profile)[:max_spans]
    return bio.extract_spans_relaxed(
        text,
        max_spans=max_spans,
        b_threshold=b_threshold,
    )


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return {key: 1.0 for key in scores}
    scale = max_score - min_score
    return {key: (value - min_score) / scale for key, value in scores.items()}


def evaluate_bio_only(
    docs,
    bio: BIOExtractor,
    candidate_k: int,
    top_k: int,
    b_threshold: float,
    bio_mode: str,
    bio_profile: str,
) -> Dict[str, float]:
    candidate_predictions: List[List[str]] = []
    topk_predictions: List[List[str]] = []
    golds: List[List[str]] = []

    for doc in docs:
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

    topk_metrics = evaluate_predictions(topk_predictions, golds)

    total_recall = 0.0
    for preds, gold in zip(candidate_predictions, golds):
        if not gold:
            continue
        hit = sum(1 for keyword in gold if keyword in preds)
        total_recall += hit / len(gold)
    recall_at_k = total_recall / max(len(golds), 1)

    return {
        "candidate_recall": recall_at_k,
        "p@5": topk_metrics.get("p@5", 0.0),
        "r@5": topk_metrics.get("r@5", 0.0),
        "f1@5": topk_metrics.get("f1@5", 0.0),
        "p@10": topk_metrics.get("p@10", 0.0),
        "r@10": topk_metrics.get("r@10", 0.0),
        "f1@10": topk_metrics.get("f1@10", 0.0),
    }


def score_candidates_with_qk_from_list(
    doc,
    candidates: Sequence[Tuple[str, float]],
    tokenizer,
    model,
    device: str,
    layer_idx: int,
    top_k: int,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    if not candidates:
        return [], {}, {}

    full_text = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX + doc.text
    prefix_len = len(DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX)
    encoding = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.inference_mode():
        scores = compute_qk_scores(model, input_ids, attention_mask, layer_idx)[0].cpu().numpy()

    char_to_score: Dict[int, float] = {}
    for token_index, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end or tok_start < prefix_len:
            continue
        char_start = tok_start - prefix_len
        char_end = tok_end - prefix_len
        score = float(scores[token_index])
        for char_index in range(char_start, char_end):
            if char_index not in char_to_score or score > char_to_score[char_index]:
                char_to_score[char_index] = score

    bio_scores: Dict[str, float] = {}
    qk_scores: Dict[str, float] = {}
    for candidate_text, bio_score in candidates:
        occurrences = find_candidate_occurrences(doc.text, candidate_text)
        if not occurrences:
            continue
        best_score = None
        for start_char, end_char in occurrences:
            span_scores = [char_to_score[idx] for idx in range(start_char, end_char) if idx in char_to_score]
            score = float(sum(span_scores) / len(span_scores)) if span_scores else 0.0
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            bio_scores[candidate_text] = float(bio_score)
            qk_scores[candidate_text] = best_score

    ranked = sorted(qk_scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, _ in ranked[:top_k]], bio_scores, qk_scores


def rank_candidates_with_fusion(
    candidates: Sequence[Tuple[str, float]],
    bio_scores: Dict[str, float],
    qk_scores: Dict[str, float],
    *,
    beta: float,
    top_k: int,
) -> List[str]:
    if not candidates:
        return []
    normalized_bio = _min_max_normalize(bio_scores)
    normalized_qk = _min_max_normalize(qk_scores)
    final_scores: Dict[str, float] = {}
    for candidate_text, _ in candidates:
        if candidate_text not in normalized_bio or candidate_text not in normalized_qk:
            continue
        final_scores[candidate_text] = beta * normalized_bio[candidate_text] + (1.0 - beta) * normalized_qk[candidate_text]
    ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, _ in ranked[:top_k]]


def evaluate_bio_plus_qk(
    docs,
    bio: BIOExtractor,
    qk_tokenizer,
    qk_model,
    device: str,
    qk_layer: int,
    candidate_k: int,
    top_k: int,
    b_threshold: float,
    bio_mode: str,
    bio_profile: str,
    fusion_betas: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    qk_predictions: List[List[str]] = []
    fused_predictions: Dict[float, List[List[str]]] = {beta: [] for beta in fusion_betas}
    golds: List[List[str]] = []

    for doc in docs:
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
                rank_candidates_with_fusion(
                    candidates,
                    bio_scores,
                    qk_scores,
                    beta=beta,
                    top_k=top_k,
                )
            )
        golds.append(doc.keywords)

    metrics: Dict[str, Dict[str, float]] = {
        "qk_only": evaluate_predictions(qk_predictions, golds),
    }
    for beta, beta_predictions in fused_predictions.items():
        metrics[f"fusion_beta_{beta:g}"] = evaluate_predictions(beta_predictions, golds)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_shence_split(args.data_root)
    heldout_docs = splits["test_final"]

    bio = BIOExtractor(args.bio_checkpoint, device=args.device)

    print(f"[info] Loading QK base model: {args.qk_model}")
    qk_tokenizer = AutoTokenizer.from_pretrained(args.qk_model, trust_remote_code=True, use_fast=True)
    qk_base = AutoModel.from_pretrained(args.qk_model, trust_remote_code=True).to(args.device)
    qk_model = PeftModel.from_pretrained(qk_base, args.qk_adapter).to(args.device)
    qk_model.eval()

    results: Dict[str, Dict[str, float]] = {}
    for candidate_k in args.candidate_k:
        print(f"[info] Evaluating candidate_k={candidate_k}")
        bio_only = evaluate_bio_only(
            heldout_docs,
            bio,
            candidate_k=candidate_k,
            top_k=args.top_k,
            b_threshold=args.b_threshold,
            bio_mode=args.bio_mode,
            bio_profile=args.bio_profile,
        )
        fusion_metrics = evaluate_bio_plus_qk(
            heldout_docs,
            bio,
            qk_tokenizer,
            qk_model,
            args.device,
            args.qk_layer,
            candidate_k=candidate_k,
            top_k=args.top_k,
            b_threshold=args.b_threshold,
            bio_mode=args.bio_mode,
            bio_profile=args.bio_profile,
            fusion_betas=args.fusion_betas,
        )
        results[f"bio_only@{candidate_k}"] = bio_only
        print(json.dumps({f"bio_only@{candidate_k}": bio_only}, ensure_ascii=False))

        for key, value in fusion_metrics.items():
            results[f"{key}@{candidate_k}"] = value
            print(json.dumps({f"{key}@{candidate_k}": value}, ensure_ascii=False))

    payload = {
        "bio_checkpoint": str(Path(args.bio_checkpoint).resolve()),
        "qk_model": str(Path(args.qk_model).resolve()),
        "qk_adapter": str(Path(args.qk_adapter).resolve()),
        "qk_layer": args.qk_layer,
        "candidate_k": args.candidate_k,
        "b_threshold": args.b_threshold,
        "bio_mode": args.bio_mode,
        "bio_profile": args.bio_profile,
        "fusion_betas": args.fusion_betas,
        "heldout_docs": len(heldout_docs),
        "results": results,
    }
    (output_dir / "bio_qk_combo_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[info] Saved to {output_dir / 'bio_qk_combo_results.json'}")


if __name__ == "__main__":
    main()
