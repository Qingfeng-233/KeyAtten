from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from copy import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from keyword_bench.data import (
    build_csl_eval_sets,
    build_shencecup_eval_sets,
    load_multi_domain_jsonl,
)
from keyword_bench.bio_boundary_head import (
    BIOBoundaryHead,
    TokenizedBIOExample,
    IGNORE_LABEL,
    TAG_B,
    TAG_I,
    TAG_O,
    char_mask_to_bio_tags,
    collate_bio_examples,
    extract_keywords_relaxed_windowed,
    tokenize_with_bio_labels,
)
from keyword_bench.output_paths import resolve_output_dir
from keyatten.candidates.bio_mining import build_bio_positive_phrases
from keyatten.candidates import (
    build_candidates,
    candidate_char_spans,
    gravity_candidates,
    locate_word_offsets,
    segment_text,
)


CACHE_FORMAT_VERSION = 3
_AUX_BOUNDARY_FUNC_WORDS = {
    "的", "了", "和", "与", "并", "及", "在", "将", "对", "中", "上", "下", "内", "外",
}
_AUX_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BIO boundary head (frozen backbone + CRF) for keyword extraction."
    )
    parser.add_argument("--root-dir", default=".", help="Project root")
    parser.add_argument(
        "--output-dir",
        default="outputs_bio_boundary",
        help="Output dir under 测试沙箱/Outputs",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Backbone model name",
    )
    parser.add_argument("--train-dataset", default="csl_train_sample",
                        help="Only used for dev/test set resolution")
    parser.add_argument("--dev-dataset", default="shencecup_labeled",
                        help="Dev dataset name (shencecup_labeled recommended for extractive)")
    parser.add_argument("--train-limit", type=int, default=0,
                        help="CSL train limit (0 = skip CSL for training)")
    parser.add_argument("--dev-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--derived-limit", type=int, default=200)
    parser.add_argument("--shencecup-limit", type=int, default=None)
    parser.add_argument("--include-shencecup", action="store_true", default=True,
                        help="Include ShenCeCup labeled data as training data")
    parser.add_argument("--no-shencecup", dest="include_shencecup", action="store_false")
    parser.add_argument("--thucnews-jsonl", type=str, default=None,
                        help="Path to thucnews_annotated.jsonl (extractive, 100%)")
    parser.add_argument("--thucnews-limit", type=int, default=8000)
    parser.add_argument("--md-jsonl", type=str, default=None,
                        help="Path to multi-domain annotated JSONL for extra training data")
    parser.add_argument("--md-limit", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--layer-index", type=int, default=14)
    parser.add_argument("--aux-tag-loss-weight", type=float, default=0.0)
    parser.add_argument("--tag-weight-b", type=float, default=1.0)
    parser.add_argument("--tag-weight-i", type=float, default=1.0)
    parser.add_argument("--tag-weight-o", type=float, default=1.0)
    parser.add_argument(
        "--aux-supervision-source",
        choices=("none", "jieba", "attention", "hybrid"),
        default="none",
        help="Weak auxiliary supervision source for BIO clean-up",
    )
    parser.add_argument("--aux-positive-weight", type=float, default=0.35)
    parser.add_argument("--aux-negative-weight", type=float, default=0.6)
    parser.add_argument("--attention-aux-threshold-ratio", type=float, default=0.55)
    parser.add_argument("--attention-aux-min-length", type=int, default=2)
    parser.add_argument("--attention-aux-max-length", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--positive-label-source",
        choices=("gold", "gold_plus_phrases"),
        default="gold_plus_phrases",
        help="BIO supervision source",
    )
    parser.add_argument("--pseudo-phrase-limit", type=int, default=40)
    parser.add_argument("--eval-top-k", type=int, default=50)
    parser.add_argument("--eval-b-threshold", type=float, default=0.15)
    parser.add_argument("--eval-window-stride", type=int, default=128)
    parser.add_argument("--eval-window-strides", type=str, default="")
    parser.add_argument("--eval-threshold-schedule", type=str, default="")
    parser.add_argument("--eval-max-expand-steps", type=int, default=1)
    parser.add_argument("--eval-max-subspan-width", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "recall_at_k"),
        default="recall_at_k",
        help="Metric used for best-checkpoint selection",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full checkpoint (best_full_ckpt.pt) to resume training from")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for tokenized BIO examples",
    )
    return parser.parse_args()


def parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_cjk_single_char(token_text: str) -> bool:
    stripped = token_text.strip()
    return len(stripped) == 1 and "\u4e00" <= stripped <= "\u9fff"


def _build_jieba_aux_spans(text: str) -> list[tuple[int, int]]:
    words, pos_tags = segment_text(text, language="zh")
    if not words:
        return []
    word_offsets = locate_word_offsets(text, words)
    candidates = build_candidates(words, pos_tags, language="zh", max_ngram=4)
    spans = candidate_char_spans(candidates, word_offsets)
    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for (start, end), candidate in zip(spans, candidates, strict=False):
        phrase = candidate.text.strip()
        if len(phrase) < 2 or len(phrase) > 16:
            continue
        if _AUX_PUNCT_RE.search(phrase):
            continue
        if any(ch in phrase for ch in "，。！？、；：()（）[]【】"):
            continue
        span = (int(start), int(end))
        if span in seen:
            continue
        seen.add(span)
        unique.append(span)
    return unique


def _build_attention_aux_spans(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    *,
    threshold_ratio: float,
    min_length: int,
    max_span_length: int,
) -> list[tuple[int, int]]:
    if model is None or not text.strip():
        return []

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"][0].tolist()]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    if not attentions:
        return []
    last_attention = attentions[-1][0].float().mean(dim=0)
    valid_len = int(attention_mask[0].sum().item())
    attention_map = last_attention[:valid_len, :valid_len].detach().cpu().numpy()
    token_scores = attention_map.sum(axis=0).astype(np.float32, copy=False)
    token_offsets = offsets[:valid_len]
    for index, (start, end) in enumerate(token_offsets):
        if end <= start:
            token_scores[index] = 0.0
            continue
        token_text = text[start:end].strip()
        if not token_text or _AUX_PUNCT_RE.match(token_text):
            token_scores[index] = 0.0

    _, char_spans = gravity_candidates(
        text,
        token_scores,
        token_offsets,
        threshold_ratio=threshold_ratio,
        min_length=min_length,
        max_length=max_span_length,
    )
    return [(int(start), int(end)) for start, end in char_spans]


def _apply_char_span(
    labels: list[int],
    weights: list[float],
    span_start: int,
    span_end: int,
    *,
    begin_weight: float,
    inside_weight: float,
) -> None:
    if span_end <= span_start or span_start < 0:
        return
    if span_start < len(labels):
        if begin_weight >= weights[span_start]:
            labels[span_start] = TAG_B
            weights[span_start] = begin_weight
    for position in range(span_start + 1, min(span_end, len(labels))):
        if inside_weight >= weights[position]:
            labels[position] = TAG_I
            weights[position] = inside_weight


def _build_aux_char_targets(
    text: str,
    gold_keywords: Sequence[str],
    *,
    tokenizer,
    attention_model,
    device: torch.device,
    max_length: int,
    aux_supervision_source: str,
    aux_positive_weight: float,
    aux_negative_weight: float,
    attention_threshold_ratio: float,
    attention_min_length: int,
    attention_max_length: int,
) -> tuple[list[int], list[float]]:
    char_count = len(text)
    aux_labels = [IGNORE_LABEL] * char_count
    aux_weights = [0.0] * char_count
    if char_count == 0 or aux_supervision_source == "none":
        return aux_labels, aux_weights

    gold_tags = char_mask_to_bio_tags(text, gold_keywords)
    jieba_spans: list[tuple[int, int]] = []
    attention_spans: list[tuple[int, int]] = []

    if aux_supervision_source in {"jieba", "hybrid"}:
        jieba_spans = _build_jieba_aux_spans(text)
    if aux_supervision_source in {"attention", "hybrid"}:
        attention_spans = _build_attention_aux_spans(
            text,
            tokenizer,
            attention_model,
            device,
            max_length,
            threshold_ratio=attention_threshold_ratio,
            min_length=attention_min_length,
            max_span_length=attention_max_length,
        )

    positive_support = [0] * char_count
    for span_start, span_end in jieba_spans:
        for position in range(max(0, span_start), min(span_end, char_count)):
            positive_support[position] += 1
    for span_start, span_end in attention_spans:
        for position in range(max(0, span_start), min(span_end, char_count)):
            positive_support[position] += 1

    for span_start, span_end in jieba_spans + attention_spans:
        support = max(1, max(positive_support[span_start:span_end], default=1))
        positive_weight = aux_positive_weight * (1.0 if support == 1 else 1.35)
        _apply_char_span(
            aux_labels,
            aux_weights,
            span_start,
            span_end,
            begin_weight=positive_weight,
            inside_weight=positive_weight * 0.9,
        )

    for index, char in enumerate(text):
        if gold_tags[index] != TAG_O:
            continue
        if positive_support[index] > 0:
            continue
        if _AUX_PUNCT_RE.match(char) or char in _AUX_BOUNDARY_FUNC_WORDS or _is_cjk_single_char(char):
            if aux_negative_weight >= aux_weights[index]:
                aux_labels[index] = TAG_O
                aux_weights[index] = aux_negative_weight

    return aux_labels, aux_weights


def _char_aux_to_token_aux(
    offsets: Sequence[tuple[int, int]],
    aux_char_labels: Sequence[int],
    aux_char_weights: Sequence[float],
) -> tuple[list[int], list[float]]:
    token_labels: list[int] = []
    token_weights: list[float] = []
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end <= start:
            token_labels.append(IGNORE_LABEL)
            token_weights.append(0.0)
            continue
        clipped_end = min(end, len(aux_char_labels))
        span_labels = aux_char_labels[start:clipped_end]
        span_weights = aux_char_weights[start:clipped_end]
        best_label = IGNORE_LABEL
        best_weight = 0.0
        for label in (TAG_B, TAG_I, TAG_O):
            label_weights = [weight for current_label, weight in zip(span_labels, span_weights) if current_label == label]
            if not label_weights:
                continue
            current_weight = max(label_weights)
            if current_weight > best_weight:
                best_label = label
                best_weight = current_weight
        token_labels.append(best_label)
        token_weights.append(float(best_weight))
    return token_labels, token_weights


def build_bio_examples(
    docs,
    tokenizer,
    max_length: int,
    *,
    positive_label_source: str,
    pseudo_phrase_limit: int,
    aux_supervision_source: str,
    aux_positive_weight: float,
    aux_negative_weight: float,
    attention_model,
    attention_device: torch.device,
    attention_threshold_ratio: float,
    attention_min_length: int,
    attention_max_length: int,
) -> list[TokenizedBIOExample]:
    examples: list[TokenizedBIOExample] = []
    for doc in docs:
        if not doc.text.strip() or not doc.keywords:
            continue
        positives = build_bio_positive_phrases(
            doc.text,
            doc.keywords,
            include_recall_phrases=positive_label_source == "gold_plus_phrases",
            phrase_limit=pseudo_phrase_limit,
        )
        if not positives:
            continue
        example = tokenize_with_bio_labels(
            doc.text, positives, tokenizer=tokenizer, max_length=max_length
        )
        if aux_supervision_source != "none":
            encoded = tokenizer(
                doc.text,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
            aux_char_labels, aux_char_weights = _build_aux_char_targets(
                doc.text,
                doc.keywords,
                tokenizer=tokenizer,
                attention_model=attention_model,
                device=attention_device,
                max_length=max_length,
                aux_supervision_source=aux_supervision_source,
                aux_positive_weight=aux_positive_weight,
                aux_negative_weight=aux_negative_weight,
                attention_threshold_ratio=attention_threshold_ratio,
                attention_min_length=attention_min_length,
                attention_max_length=attention_max_length,
            )
            token_aux_labels, token_aux_weights = _char_aux_to_token_aux(
                offsets,
                aux_char_labels,
                aux_char_weights,
            )
            example.aux_labels = token_aux_labels
            example.aux_weights = token_aux_weights
        examples.append(example)
    return examples


def _docs_signature(docs) -> str:
    digest = hashlib.sha1()
    for doc in docs:
        digest.update(str(getattr(doc, "doc_id", "")).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(str(len(getattr(doc, "text", ""))).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(len(getattr(doc, "keywords", []))).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_path(
    cache_dir: Path,
    split_name: str,
    *,
    model_name: str,
    max_length: int,
    positive_label_source: str,
    pseudo_phrase_limit: int,
    aux_supervision_source: str,
    aux_positive_weight: float,
    aux_negative_weight: float,
    attention_threshold_ratio: float,
    attention_min_length: int,
    attention_max_length: int,
    docs,
) -> Path:
    key_payload = {
        "version": CACHE_FORMAT_VERSION,
        "split": split_name,
        "model_name": model_name,
        "max_length": int(max_length),
        "positive_label_source": positive_label_source,
        "pseudo_phrase_limit": int(pseudo_phrase_limit),
        "aux_supervision_source": aux_supervision_source,
        "aux_positive_weight": float(aux_positive_weight),
        "aux_negative_weight": float(aux_negative_weight),
        "attention_threshold_ratio": float(attention_threshold_ratio),
        "attention_min_length": int(attention_min_length),
        "attention_max_length": int(attention_max_length),
        "doc_count": len(docs),
        "docs_signature": _docs_signature(docs),
    }
    key = hashlib.sha1(
        json.dumps(key_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{split_name}_{key}.pt"


def build_bio_examples_cached(
    docs,
    tokenizer,
    max_length: int,
    *,
    split_name: str,
    cache_dir: Path | None,
    positive_label_source: str,
    pseudo_phrase_limit: int,
    aux_supervision_source: str,
    aux_positive_weight: float,
    aux_negative_weight: float,
    attention_model,
    attention_device: torch.device,
    attention_threshold_ratio: float,
    attention_min_length: int,
    attention_max_length: int,
) -> list[TokenizedBIOExample]:
    if cache_dir is None:
        return build_bio_examples(
            docs,
            tokenizer,
            max_length,
            positive_label_source=positive_label_source,
            pseudo_phrase_limit=pseudo_phrase_limit,
            aux_supervision_source=aux_supervision_source,
            aux_positive_weight=aux_positive_weight,
            aux_negative_weight=aux_negative_weight,
            attention_model=attention_model,
            attention_device=attention_device,
            attention_threshold_ratio=attention_threshold_ratio,
            attention_min_length=attention_min_length,
            attention_max_length=attention_max_length,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        cache_dir,
        split_name,
        model_name=tokenizer.name_or_path,
        max_length=max_length,
        positive_label_source=positive_label_source,
        pseudo_phrase_limit=pseudo_phrase_limit,
        aux_supervision_source=aux_supervision_source,
        aux_positive_weight=aux_positive_weight,
        aux_negative_weight=aux_negative_weight,
        attention_threshold_ratio=attention_threshold_ratio,
        attention_min_length=attention_min_length,
        attention_max_length=attention_max_length,
        docs=docs,
    )
    if cache_path.exists():
        print(f"[info] Loading cached {split_name} BIO examples from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    examples = build_bio_examples(
        docs,
        tokenizer,
        max_length,
        positive_label_source=positive_label_source,
        pseudo_phrase_limit=pseudo_phrase_limit,
        aux_supervision_source=aux_supervision_source,
        aux_positive_weight=aux_positive_weight,
        aux_negative_weight=aux_negative_weight,
        attention_model=attention_model,
        attention_device=attention_device,
        attention_threshold_ratio=attention_threshold_ratio,
        attention_min_length=attention_min_length,
        attention_max_length=attention_max_length,
    )
    torch.save(examples, cache_path)
    print(f"[info] Cached {split_name} BIO examples to {cache_path}")
    return examples


def _build_eval_candidates(
    model: BIOBoundaryHead,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    *,
    top_k: int,
    b_threshold: float,
    window_stride: int,
    window_strides: Sequence[int],
    threshold_schedule: Sequence[float],
    max_expand_steps: int,
    max_subspan_width: int,
) -> list[str]:
    relaxed = extract_keywords_relaxed_windowed(
        text,
        tokenizer,
        model,
        device,
        max_length,
        max_spans=top_k,
        b_threshold=b_threshold,
        window_stride=window_stride,
        window_strides=window_strides,
        threshold_schedule=threshold_schedule,
        max_expand_steps=max_expand_steps,
        max_subspan_width=max_subspan_width,
    )
    return [keyword for keyword, _ in relaxed]


def evaluate_candidate_metrics(
    model: BIOBoundaryHead,
    dataloader: DataLoader,
    tokenizer,
    docs,
    device: torch.device,
    *,
    max_length: int,
    top_k: int,
    b_threshold: float,
    window_stride: int,
    window_strides: Sequence[int],
    threshold_schedule: Sequence[float],
    max_expand_steps: int,
    max_subspan_width: int,
) -> dict[str, float]:
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gold_hits = 0
    total_gold_count = 0
    total_candidate_count = 0
    doc_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = int(input_ids.shape[0])
            for _ in range(batch_size):
                # Find the corresponding doc
                while doc_idx < len(docs) and (
                    not docs[doc_idx].text.strip() or not docs[doc_idx].keywords
                ):
                    doc_idx += 1
                if doc_idx >= len(docs):
                    break

                doc = docs[doc_idx]
                doc_idx += 1

                pred_keywords = _build_eval_candidates(
                    model,
                    tokenizer,
                    doc.text,
                    device,
                    max_length,
                    top_k=top_k,
                    b_threshold=b_threshold,
                    window_stride=window_stride,
                    window_strides=window_strides,
                    threshold_schedule=threshold_schedule,
                    max_expand_steps=max_expand_steps,
                    max_subspan_width=max_subspan_width,
                )

                # Gold keywords
                gold_keywords = [kw.strip() for kw in doc.keywords if kw.strip()]

                # Compute per-doc overlap
                pred_set = {kw.lower() for kw in pred_keywords}
                gold_set = {kw.lower() for kw in gold_keywords}
                total_tp += len(pred_set & gold_set)
                total_fp += len(pred_set - gold_set)
                total_fn += len(gold_set - pred_set)
                total_gold_hits += len(pred_set & gold_set)
                total_gold_count += len(gold_set)
                total_candidate_count += len(pred_keywords)

    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_tp + total_fn, 1e-8)
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    recall_at_k = total_gold_hits / max(total_gold_count, 1e-8)
    avg_candidates = total_candidate_count / max(len(docs), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_k": recall_at_k,
        "avg_candidates": avg_candidates,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root_dir = Path(args.root_dir).resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_window_strides = parse_int_csv(args.eval_window_strides)
    eval_threshold_schedule = parse_float_csv(args.eval_threshold_schedule)
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else root_dir / "train" / "cache" / "bio_examples"
    )

    # --- Load dev dataset ---
    requested = {args.dev_dataset}
    all_eval_sets: dict[str, list] = {}
    if any(name.startswith("csl_") for name in requested):
        all_eval_sets.update(
            build_csl_eval_sets(
                root_dir,
                train_limit=0, dev_limit=args.dev_limit,
                test_limit=args.test_limit, derived_limit=args.derived_limit,
            )
        )
    if any(name.startswith("shencecup_") for name in requested):
        all_eval_sets.update(
            build_shencecup_eval_sets(root_dir, shencecup_limit=args.shencecup_limit)
        )

    if args.dev_dataset not in all_eval_sets:
        raise ValueError(f"Unknown dev dataset: {args.dev_dataset}")

    # --- Load extractive training data ---
    # ShenCeCup (extractive, 98.6%)
    shencecup_docs: list = []
    if args.include_shencecup:
        shencecup_sets = build_shencecup_eval_sets(root_dir, shencecup_limit=args.shencecup_limit)
        if "shencecup_labeled" in shencecup_sets:
            shencecup_docs = shencecup_sets["shencecup_labeled"]
            print(f"[info] Loaded {len(shencecup_docs)} ShenCeCup docs for training")

    # THUCNews annotated (extractive, 100%)
    thucnews_docs: list = []
    if args.thucnews_jsonl:
        thucnews_path = Path(args.thucnews_jsonl)
        if not thucnews_path.is_absolute():
            thucnews_path = root_dir / thucnews_path
        thucnews_docs = load_multi_domain_jsonl(thucnews_path, limit=args.thucnews_limit)
        print(f"[info] Loaded {len(thucnews_docs)} THUCNews docs for training")

    # Multi-domain JSONL (extractive, 95.8%)
    md_docs: list = []
    if args.md_jsonl:
        md_path = Path(args.md_jsonl)
        if not md_path.is_absolute():
            md_path = root_dir / md_path
        md_docs = load_multi_domain_jsonl(md_path, limit=args.md_limit)
        print(f"[info] Loaded {len(md_docs)} multi-domain docs for training")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must provide a valid pad_token_id.")
    attention_device = torch.device(args.device)
    attention_model = None
    if args.aux_supervision_source in {"attention", "hybrid"}:
        print(f"[info] Loading auxiliary attention model from {args.model}")
        try:
            attention_model = AutoModel.from_pretrained(
                args.model,
                output_attentions=True,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except TypeError:
            attention_model = AutoModel.from_pretrained(
                args.model,
                output_attentions=True,
                trust_remote_code=True,
            )
        attention_model.to(attention_device)
        attention_model.eval()

    # --- Extractive filter: only keep keywords that appear in source text ---
    def _filter_extractive(docs: list, name: str) -> list:
        total_kw, kept_kw, dropped = 0, 0, 0
        filtered = []
        for doc in docs:
            extractive = [kw for kw in doc.keywords if kw in doc.text]
            total_kw += len(doc.keywords)
            kept_kw += len(extractive)
            if extractive:
                d = copy(doc)
                d.keywords = extractive
                filtered.append(d)
            else:
                dropped += 1
        ratio = kept_kw / max(total_kw, 1)
        print(f"[info] {name}: keywords {kept_kw}/{total_kw} kept ({ratio:.1%}), {dropped} docs dropped")
        return filtered

    shencecup_docs = _filter_extractive(shencecup_docs, "ShenCe train")
    thucnews_docs = _filter_extractive(thucnews_docs, "THUCNews train")
    md_docs = _filter_extractive(md_docs, "MD train")

    # --- Build examples ---
    train_docs = shencecup_docs + thucnews_docs + md_docs
    random.shuffle(train_docs)
    dev_docs = all_eval_sets[args.dev_dataset]
    print(f"[info] Total train docs: {len(train_docs)} (ShenCe={len(shencecup_docs)}, THUCNews={len(thucnews_docs)}, MD={len(md_docs)})")
    print(f"[info] Dev docs: {len(dev_docs)}")

    train_examples = build_bio_examples_cached(
        train_docs,
        tokenizer,
        args.max_length,
        split_name="train",
        cache_dir=cache_dir,
        positive_label_source=args.positive_label_source,
        pseudo_phrase_limit=args.pseudo_phrase_limit,
        aux_supervision_source=args.aux_supervision_source,
        aux_positive_weight=args.aux_positive_weight,
        aux_negative_weight=args.aux_negative_weight,
        attention_model=attention_model,
        attention_device=attention_device,
        attention_threshold_ratio=args.attention_aux_threshold_ratio,
        attention_min_length=args.attention_aux_min_length,
        attention_max_length=args.attention_aux_max_length,
    )
    dev_examples = build_bio_examples_cached(
        dev_docs,
        tokenizer,
        args.max_length,
        split_name=f"dev_{args.dev_dataset}",
        cache_dir=cache_dir,
        positive_label_source="gold",
        pseudo_phrase_limit=args.pseudo_phrase_limit,
        aux_supervision_source=args.aux_supervision_source,
        aux_positive_weight=args.aux_positive_weight,
        aux_negative_weight=args.aux_negative_weight,
        attention_model=attention_model,
        attention_device=attention_device,
        attention_threshold_ratio=args.attention_aux_threshold_ratio,
        attention_min_length=args.attention_aux_min_length,
        attention_max_length=args.attention_aux_max_length,
    )
    if attention_model is not None:
        del attention_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not train_examples:
        raise RuntimeError("No valid training examples were built.")

    collate_fn = lambda items: collate_bio_examples(
        items, pad_token_id=int(tokenizer.pad_token_id)
    )
    train_loader = DataLoader(
        train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- Model ---
    freeze_backbone = not args.unfreeze_backbone
    model = BIOBoundaryHead(
        args.model,
        layer_index=args.layer_index,
        freeze_backbone=freeze_backbone,
        trust_remote_code=True,
        aux_tag_loss_weight=args.aux_tag_loss_weight,
        tag_loss_weights=[
            args.tag_weight_b,
            args.tag_weight_i,
            args.tag_weight_o,
        ],
    )
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_metric = -1.0
    patience_counter = 0
    train_log: list[dict] = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[info] Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "best_metric" in ckpt:
                best_metric = ckpt["best_metric"]
            elif "best_f1" in ckpt:
                best_metric = ckpt["best_f1"]
            if "patience_counter" in ckpt:
                patience_counter = ckpt["patience_counter"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "train_log" in ckpt:
                train_log = ckpt["train_log"]
            print(f"[info] Resumed at epoch {start_epoch}, best_metric={best_metric:.4f}", flush=True)
        else:
            print(f"[warn] Resume path {resume_path} not found, starting from scratch")

    num_train_batches = len(train_loader)
    print(f"[info] Train batches per epoch: {num_train_batches}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            aux_labels = batch.get("aux_labels")
            aux_weights = batch.get("aux_weights")
            if aux_labels is not None:
                aux_labels = aux_labels.to(device)
            if aux_weights is not None:
                aux_weights = aux_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                aux_labels=aux_labels,
                aux_weights=aux_weights,
            )
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"  epoch {epoch} batch {batch_idx+1}/{num_train_batches} loss={loss.item():.2f} seq_len={input_ids.shape[1]}", flush=True)

        print(f"  epoch {epoch} training done, starting eval...", flush=True)
        # Span-level dev evaluation
        dev_metrics = evaluate_candidate_metrics(
            model,
            dev_loader,
            tokenizer,
            dev_docs,
            device=device,
            max_length=args.max_length,
            top_k=args.eval_top_k,
            b_threshold=args.eval_b_threshold,
            window_stride=args.eval_window_stride,
            window_strides=eval_window_strides,
            threshold_schedule=eval_threshold_schedule,
            max_expand_steps=args.eval_max_expand_steps,
            max_subspan_width=args.eval_max_subspan_width,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "dev_precision": float(dev_metrics["precision"]),
            "dev_recall": float(dev_metrics["recall"]),
            "dev_f1": float(dev_metrics["f1"]),
            f"dev_recall@{args.eval_top_k}": float(dev_metrics["recall_at_k"]),
            "dev_avg_candidates": float(dev_metrics["avg_candidates"]),
        }
        train_log.append(record)
        print(json.dumps(record, ensure_ascii=False))

        metric_value = float(
            dev_metrics["f1"]
            if args.selection_metric == "f1"
            else dev_metrics["recall_at_k"]
        )
        if metric_value > best_metric:
            best_metric = metric_value
            patience_counter = 0
            # Save head-only checkpoint (for inference)
            head_ckpt_path = output_dir / "best_bio_head.pt"
            torch.save(
                {
                    "classifier_state": model.classifier.state_dict(),
                    "crf_state": model.crf.state_dict(),
                    "model_name": args.model,
                    "layer_index": int(args.layer_index),
                    "freeze_backbone": freeze_backbone,
                    "max_length": int(args.max_length),
                    "tokenizer_name": args.model,
                    "trust_remote_code": True,
                },
                head_ckpt_path,
            )
            # Save full model checkpoint (for resume)
            full_ckpt_path = output_dir / "best_full_ckpt.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "patience_counter": patience_counter,
                    "train_log": train_log,
                    "model_name": args.model,
                    "layer_index": int(args.layer_index),
                    "freeze_backbone": freeze_backbone,
                    "max_length": int(args.max_length),
                    "trust_remote_code": True,
                    "selection_metric": args.selection_metric,
                },
                full_ckpt_path,
            )
            print(
                f"  Saved best checkpoint ({args.selection_metric}={best_metric:.4f})",
                flush=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
                break

    report = {
        "model": args.model,
        "train_dataset": args.train_dataset,
        "dev_dataset": args.dev_dataset,
        "md_jsonl": args.md_jsonl,
        "md_docs": len(md_docs),
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "freeze_backbone": freeze_backbone,
        "layer_index": int(args.layer_index),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "positive_label_source": args.positive_label_source,
        "selection_metric": args.selection_metric,
        "eval_top_k": int(args.eval_top_k),
        "eval_b_threshold": float(args.eval_b_threshold),
        "eval_window_stride": int(args.eval_window_stride),
        "eval_window_strides": eval_window_strides,
        "eval_threshold_schedule": eval_threshold_schedule,
        "eval_max_expand_steps": int(args.eval_max_expand_steps),
        "eval_max_subspan_width": int(args.eval_max_subspan_width),
        "aux_tag_loss_weight": float(args.aux_tag_loss_weight),
        "tag_loss_weights": {
            "B": float(args.tag_weight_b),
            "I": float(args.tag_weight_i),
            "O": float(args.tag_weight_o),
        },
        "aux_supervision_source": args.aux_supervision_source,
        "aux_positive_weight": float(args.aux_positive_weight),
        "aux_negative_weight": float(args.aux_negative_weight),
        "attention_aux_threshold_ratio": float(args.attention_aux_threshold_ratio),
        "attention_aux_min_length": int(args.attention_aux_min_length),
        "attention_aux_max_length": int(args.attention_aux_max_length),
        "best_metric": float(best_metric),
        "history": train_log,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
