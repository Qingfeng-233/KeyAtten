from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ...candidates import locate_word_offsets, candidate_char_spans
from ...extractors.attention import KeyAttenExtractor
from ...extractors.qk_lora import QKLoRAExtractor, compute_qk_scores
from .features import build_feature_row
from .dataset import PairwiseExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pairwise reranker training data from existing candidate scores.")
    parser.add_argument("--dataset", choices=["shence"], default="shence")
    parser.add_argument("--root-dir", default=".", help="Repository root.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20, help="Top candidate pool size per document.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gte-model", default=None, help="Path to gte-small-zh. Default: <root>/models/gte-small-zh")
    parser.add_argument("--qwen-model", default=None, help="Path to Qwen3-Embedding-0.6B. Default: <root>/models/Qwen3-Embedding-0.6B")
    parser.add_argument("--qk-adapter", default=None, help="Path to QK LoRA adapter. Default: <root>/models/qk_qwen0.6B_hq_r16_nocsl8k_best_adapter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _normalize(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _align_gold(candidate_text: str, gold_keywords: list[str]) -> bool:
    normalized_candidate = candidate_text.strip().lower()
    for gold in gold_keywords:
        normalized_gold = gold.strip().lower()
        if not normalized_gold:
            continue
        if normalized_candidate == normalized_gold:
            return True
        if normalized_candidate in normalized_gold or normalized_gold in normalized_candidate:
            return True
    return False


def _load_docs(dataset: str, root_dir: Path, limit: int):
    if dataset != "shence":
        raise ValueError(f"Unsupported dataset: {dataset}")
    from benchmark.keyword_bench.data import load_shencecup_labeled

    return load_shencecup_labeled(root_dir, limit=limit)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gte_model = args.gte_model or str(root_dir / "models" / "gte-small-zh")
    qwen_model = args.qwen_model or str(root_dir / "models" / "Qwen3-Embedding-0.6B")
    qk_adapter = args.qk_adapter or str(root_dir / "models" / "qk_qwen0.6B_hq_r16_nocsl8k_best_adapter")

    docs = _load_docs(args.dataset, root_dir, args.limit)

    attn = KeyAttenExtractor(model=gte_model, language="zh", device=args.device)
    qk = QKLoRAExtractor(model=qwen_model, adapter_path=qk_adapter, language="zh", device=args.device, layer="auto")
    qk._ensure_loaded()

    examples: list[PairwiseExample] = []

    for index, doc in enumerate(docs, start=1):
        words, pos_tags, candidates, *_ = attn._prepare_document(doc.text)
        if not candidates:
            continue

        received_scores = attn._resolve_word_scores(
            words=words,
            pos_tags=pos_tags,
            method="received_attn",
            token_counts=dict(),
            idf_lookup=None,
        )
        fusion_scores = attn._resolve_word_scores(
            words=words,
            pos_tags=pos_tags,
            method="fusion_attn",
            token_counts=dict(),
            idf_lookup=None,
        )

        full_text = qk.instruction_prefix + doc.text
        prefix_len = len(qk.instruction_prefix)
        enc = qk._tokenizer(
            full_text,
            max_length=qk.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(qk.device)
        attention_mask = enc["attention_mask"].to(qk.device)
        offset_mapping = enc["offset_mapping"][0].tolist()
        with torch.no_grad():
            qk_token_scores = compute_qk_scores(qk._model, input_ids, attention_mask, qk._layer_idx)[0].cpu().numpy()

        char_to_score: dict[int, float] = {}
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end or tok_start < prefix_len:
                continue
            start = tok_start - prefix_len
            end = tok_end - prefix_len
            score = float(qk_token_scores[tok_idx])
            for char_idx in range(start, end):
                if char_idx not in char_to_score or score > char_to_score[char_idx]:
                    char_to_score[char_idx] = score

        word_offsets = locate_word_offsets(doc.text, words)
        char_spans = candidate_char_spans(candidates, word_offsets)

        candidate_rows = []
        for candidate, span in zip(candidates, char_spans):
            qk_vals = [char_to_score[pos] for pos in range(span[0], span[1]) if pos in char_to_score]
            qk_score = float(sum(qk_vals) / len(qk_vals)) if qk_vals else 0.0
            received_score = float(np.mean(received_scores[candidate.word_start : candidate.word_end]))
            fusion_score = float(np.mean(fusion_scores[candidate.word_start : candidate.word_end]))
            row = build_feature_row(
                document_text=doc.text,
                candidate_text=candidate.text,
                qk_score=qk_score,
                received_attn_score=received_score,
                fusion_attn_score=fusion_score,
                token_len=candidate.word_end - candidate.word_start,
                candidate_texts=[cand.text for cand in candidates],
            )
            candidate_rows.append(row)

        qk_norm = _normalize(row.qk_score for row in candidate_rows)
        received_norm = _normalize(row.received_attn_score for row in candidate_rows)
        fusion_norm = _normalize(row.fusion_attn_score for row in candidate_rows)

        merged_rows = []
        for idx, row in enumerate(candidate_rows):
            merged_rows.append(
                {
                    "candidate_text": row.candidate_text,
                    "qk_score": float(qk_norm[idx]),
                    "received_attn_score": float(received_norm[idx]),
                    "fusion_attn_score": float(fusion_norm[idx]),
                    "char_len": row.char_len,
                    "token_len": row.token_len,
                    "first_occurrence_ratio": row.first_occurrence_ratio,
                    "occurrence_count": row.occurrence_count,
                    "begins_with_weak_prefix": row.begins_with_weak_prefix,
                    "is_contained_by_other_candidate": row.is_contained_by_other_candidate,
                    "contains_other_candidate": row.contains_other_candidate,
                }
            )

        merged_rows.sort(
            key=lambda item: (
                item["qk_score"] * 0.5
                + item["received_attn_score"] * 0.3
                + item["fusion_attn_score"] * 0.2
            ),
            reverse=True,
        )

        positives = [row for row in merged_rows if _align_gold(row["candidate_text"], doc.keywords)]
        negatives = [row for row in merged_rows if not _align_gold(row["candidate_text"], doc.keywords)]
        top_negatives = negatives[: args.top_n]

        for positive in positives:
            for negative in top_negatives:
                if positive["candidate_text"] == negative["candidate_text"]:
                    continue
                examples.append(
                    PairwiseExample(
                        document_text=doc.text,
                        positive_candidate=positive["candidate_text"],
                        negative_candidate=negative["candidate_text"],
                        positive_features=positive,
                        negative_features=negative,
                    )
                )

        if index % 20 == 0 or index == len(docs):
            print(f"[progress] processed {index}/{len(docs)} docs, examples={len(examples)}")

    output_path.write_text(
        json.dumps([asdict(example) for example in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] exported {len(examples)} examples -> {output_path}")


if __name__ == "__main__":
    main()
