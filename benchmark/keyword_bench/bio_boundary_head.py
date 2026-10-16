"""BIO head training/eval helpers: label alignment, batch collation, span F1.

The model class and span decoders live in keyatten.candidates.bio_head;
BIOBoundaryHead and extract_keywords_relaxed_windowed are re-exported here so
training scripts keep a single import site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from keyatten.candidates.bio_head import (
    IGNORE_LABEL,
    TAG_B,
    TAG_I,
    TAG_O,
    BIOBoundaryHead,
    extract_keywords_relaxed_windowed,
)


@dataclass(slots=True)
class TokenizedBIOExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]  # BIO tags: 0=B, 1=I, 2=O, -100=ignore
    aux_labels: list[int] | None = None
    aux_weights: list[float] | None = None


def char_mask_to_bio_tags(
    text: str, keywords: Sequence[str]
) -> np.ndarray:
    """Convert keyword list to per-character BIO tags.

    B marks the first character of each keyword span;
    I marks subsequent characters inside a keyword span;
    O marks everything else.

    Longest keywords are processed first so that overlapping
    shorter keywords inherit the outer span's tags.
    """
    tags = np.full(len(text), TAG_O, dtype=np.int32)
    if not text:
        return tags

    ranked = sorted(
        {kw.strip() for kw in keywords if kw and kw.strip()},
        key=len,
        reverse=True,
    )
    for keyword in ranked:
        start = 0
        while True:
            idx = text.find(keyword, start)
            if idx < 0:
                break
            end = idx + len(keyword)
            # Only mark B if this position is not already part of a span
            # that started earlier (i.e., already B or I).
            # For the first char: mark B only if it's currently O.
            if tags[idx] == TAG_O:
                tags[idx] = TAG_B
            # Mark remaining chars as I if they are O
            for pos in range(idx + 1, end):
                if tags[pos] == TAG_O:
                    tags[pos] = TAG_I
            start = idx + 1
    return tags


def tokenize_with_bio_labels(
    text: str,
    keywords: Sequence[str],
    tokenizer,
    max_length: int,
) -> TokenizedBIOExample:
    """Tokenize text and align BIO labels to subword tokens."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    char_bio = char_mask_to_bio_tags(text, keywords)

    labels: list[int] = []
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end <= start:
            # Special token ([CLS], [SEP])
            labels.append(IGNORE_LABEL)
            continue
        if start >= len(char_bio):
            labels.append(TAG_O)
            continue
        clipped_end = min(end, len(char_bio))
        span_tags = char_bio[start:clipped_end]
        # If any char in this subword is B, label B; else if any I, label I; else O
        if TAG_B in span_tags:
            labels.append(TAG_B)
        elif TAG_I in span_tags:
            labels.append(TAG_I)
        else:
            labels.append(TAG_O)

    return TokenizedBIOExample(
        input_ids=[int(t) for t in encoded["input_ids"]],
        attention_mask=[int(v) for v in encoded["attention_mask"]],
        labels=labels,
    )


def collate_bio_examples(
    batch: Sequence[TokenizedBIOExample], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in batch)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    aux_labels: list[list[int]] = []
    aux_weights: list[list[float]] = []
    has_aux_labels = any(item.aux_labels is not None for item in batch)
    has_aux_weights = any(item.aux_weights is not None for item in batch)
    for item in batch:
        pad_size = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [int(pad_token_id)] * pad_size)
        attention_masks.append(item.attention_mask + [0] * pad_size)
        labels.append(item.labels + [IGNORE_LABEL] * pad_size)
        if has_aux_labels:
            current_aux_labels = item.aux_labels or [IGNORE_LABEL] * len(item.input_ids)
            aux_labels.append(current_aux_labels + [IGNORE_LABEL] * pad_size)
        if has_aux_weights:
            current_aux_weights = item.aux_weights or [0.0] * len(item.input_ids)
            aux_weights.append(current_aux_weights + [0.0] * pad_size)
    batch_dict = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if has_aux_labels:
        batch_dict["aux_labels"] = torch.tensor(aux_labels, dtype=torch.long)
    if has_aux_weights:
        batch_dict["aux_weights"] = torch.tensor(aux_weights, dtype=torch.float32)
    return batch_dict

def span_level_f1(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
) -> dict[str, float]:
    """Compute span-level P/R/F1 (exact match on (start, end))."""
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def span_level_f1_text(
    pred_keywords: Sequence[str],
    gold_keywords: Sequence[str],
) -> dict[str, float]:
    """Compute keyword-level P/R/F1 (string match, case-insensitive)."""
    pred_set = {kw.strip().lower() for kw in pred_keywords if kw.strip()}
    gold_set = {kw.strip().lower() for kw in gold_keywords if kw.strip()}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}
