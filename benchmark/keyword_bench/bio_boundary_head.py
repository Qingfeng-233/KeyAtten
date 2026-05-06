from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

try:
    from torchcrf import CRF
except ImportError as exc:
    raise ImportError(
        "BIO boundary head requires `pytorch-crf` (pip install pytorch-crf)."
    ) from exc


IGNORE_LABEL = -100

# BIO tag ids: B=0, I=1, O=2
TAG_B = 0
TAG_I = 1
TAG_O = 2
NUM_TAGS = 3


@dataclass(slots=True)
class TokenizedBIOExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]  # BIO tags: 0=B, 1=I, 2=O, -100=ignore
    aux_labels: list[int] | None = None
    aux_weights: list[float] | None = None


class BIOBoundaryHead(nn.Module):
    """Frozen backbone + CRF head for BIO keyword boundary detection."""

    def __init__(
        self,
        model_name: str,
        layer_index: int = -1,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        aux_tag_loss_weight: float = 0.0,
        tag_loss_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer_index = int(layer_index)
        self.aux_tag_loss_weight = float(aux_tag_loss_weight)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, NUM_TAGS)
        self.crf = CRF(num_tags=NUM_TAGS, batch_first=True)
        if tag_loss_weights is not None:
            if len(tag_loss_weights) != NUM_TAGS:
                raise ValueError(
                    f"tag_loss_weights must have {NUM_TAGS} values, got {len(tag_loss_weights)}"
                )
            self.register_buffer(
                "tag_loss_weights",
                torch.tensor(tag_loss_weights, dtype=torch.float32),
            )
        else:
            self.tag_loss_weights = None
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        aux_labels: torch.Tensor | None = None,
        aux_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        hidden = hidden_states[self.layer_index]
        if hidden.dtype != self.classifier.weight.dtype:
            hidden = hidden.to(self.classifier.weight.dtype)
        emissions = self.classifier(hidden)  # (B, T, NUM_TAGS)

        crf_mask = attention_mask.bool()
        result: dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is not None:
            # Replace IGNORE_LABEL with TAG_O for CRF (mask handles padding)
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_LABEL] = TAG_O
            # CRF log-likelihood: higher is better
            log_likelihood = self.crf(
                emissions, clean_labels, mask=crf_mask
            )
            result["log_likelihood"] = log_likelihood
            batch_size = emissions.size(0)
            crf_loss = -log_likelihood / batch_size
            result["crf_loss"] = crf_loss
            total_loss = crf_loss
            if self.aux_tag_loss_weight > 0.0:
                aux_targets = labels if aux_labels is None else aux_labels
                aux_loss_vector = F.cross_entropy(
                    emissions.reshape(-1, NUM_TAGS),
                    aux_targets.reshape(-1),
                    weight=self.tag_loss_weights,
                    ignore_index=IGNORE_LABEL,
                    reduction="none",
                )
                valid_mask = aux_targets.reshape(-1) != IGNORE_LABEL
                if aux_weights is not None:
                    flat_weights = aux_weights.reshape(-1).to(aux_loss_vector.dtype)
                    valid_mask = valid_mask & (flat_weights > 0)
                    if torch.any(valid_mask):
                        aux_loss = (
                            aux_loss_vector[valid_mask] * flat_weights[valid_mask]
                        ).sum() / flat_weights[valid_mask].sum().clamp_min(1e-8)
                    else:
                        aux_loss = aux_loss_vector.new_zeros(())
                elif torch.any(valid_mask):
                    aux_loss = aux_loss_vector[valid_mask].mean()
                else:
                    aux_loss = aux_loss_vector.new_zeros(())
                result["aux_tag_loss"] = aux_loss
                total_loss = total_loss + self.aux_tag_loss_weight * aux_loss
            result["loss"] = total_loss
        return result

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi decode emissions → BIO tag sequences."""
        crf_mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=crf_mask)


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


def bio_tags_to_spans(tags: Sequence[int]) -> list[tuple[int, int]]:
    """Decode BIO tag sequence → list of (start, end) spans (half-open)."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag == TAG_B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == TAG_I:
            if start is None:
                # I without preceding B — treat as B
                start = i
        else:  # TAG_O or IGNORE_LABEL
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def spans_to_text(
    text: str, spans: list[tuple[int, int]], offsets: Sequence[tuple[int, int]]
) -> list[str]:
    """Convert token-level spans to text strings using offset_mapping.

    offsets: list of (char_start, char_end) from tokenizer, same length as tags.
    """
    keywords: list[str] = []
    for span_start, span_end in spans:
        if span_start >= len(offsets) or span_end > len(offsets):
            continue
        # Collect char offsets for tokens in this span
        char_start = offsets[span_start][0]
        char_end = offsets[span_end - 1][1]
        if char_end <= char_start:
            continue
        keyword = text[char_start:char_end].strip()
        if keyword:
            keywords.append(keyword)
    return keywords


def _span_to_keyword(
    text: str,
    offsets: Sequence[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> str:
    if span_start < 0 or span_end <= span_start or span_end > len(offsets):
        return ""
    char_start = offsets[span_start][0]
    char_end = offsets[span_end - 1][1]
    if char_end <= char_start:
        return ""
    return text[char_start:char_end].strip()


def _iter_span_variants(
    span_start: int,
    span_end: int,
    seq_len: int,
    *,
    max_subspan_width: int = 6,
    max_expand_steps: int = 1,
) -> list[tuple[tuple[int, int], float]]:
    base_width = span_end - span_start
    variants: list[tuple[tuple[int, int], float]] = [((span_start, span_end), 0.0)]

    # Single-step outward expansion catches off-by-one boundaries.
    for expand in range(1, max_expand_steps + 1):
        if span_start - expand >= 0:
            variants.append(((span_start - expand, span_end), 0.08 * expand))
        if span_end + expand <= seq_len:
            variants.append(((span_start, span_end + expand), 0.08 * expand))
        if span_start - expand >= 0 and span_end + expand <= seq_len:
            variants.append(((span_start - expand, span_end + expand), 0.12 * expand))

    # Subspan closure catches core phrases nested inside longer predicted spans.
    if max_subspan_width > 0:
        capped_width = min(base_width, max_subspan_width)
        for width in range(1, capped_width + 1):
            for start in range(span_start, span_end - width + 1):
                end = start + width
                if start == span_start and end == span_end:
                    continue
                trim_penalty = 0.04 * ((start - span_start) + (span_end - end))
                variants.append(((start, end), trim_penalty))

    # Deduplicate span variants, keep the smallest penalty.
    deduped: dict[tuple[int, int], float] = {}
    for span, penalty in variants:
        prev = deduped.get(span)
        if prev is None or penalty < prev:
            deduped[span] = penalty
    return sorted(deduped.items(), key=lambda item: (item[1], item[0][0], item[0][1]))


def extract_keywords_relaxed(
    text: str,
    offsets: Sequence[tuple[int, int]],
    emissions: torch.Tensor,
    decoded_tags: Sequence[int],
    *,
    max_spans: int = 50,
    b_threshold: float = 0.15,
    max_expand_steps: int = 1,
    max_subspan_width: int = 0,
) -> list[tuple[str, float]]:
    """Extract more recall-oriented keyword candidates from BIO emissions."""
    seq_len = min(len(offsets), int(emissions.size(0)), len(decoded_tags))
    if seq_len <= 0:
        return []

    probs = torch.softmax(emissions[:seq_len], dim=-1).detach().cpu()
    viterbi_spans = bio_tags_to_spans(decoded_tags[:seq_len])
    viterbi_set = set(viterbi_spans)

    relaxed_spans: list[tuple[tuple[int, int], float]] = []
    i = 0
    while i < seq_len:
        pb = float(probs[i, TAG_B])
        if pb > b_threshold:
            start = i
            scores = [pb]
            j = i + 1
            while j < seq_len and float(probs[j, TAG_I]) > float(probs[j, TAG_O]):
                scores.append(float(probs[j, TAG_I]))
                j += 1
            span = (start, j)
            conf = sum(scores) / len(scores)
            if span in viterbi_set:
                conf += 1.0
            relaxed_spans.append((span, conf))
            i = j
            continue
        i += 1

    relaxed_spans.sort(key=lambda item: -item[1])
    seen: set[str] = set()
    results: list[tuple[str, float]] = []
    for (span_start, span_end), conf in relaxed_spans:
        for (variant_start, variant_end), penalty in _iter_span_variants(
            span_start,
            span_end,
            seq_len,
            max_subspan_width=max_subspan_width,
            max_expand_steps=max_expand_steps,
        ):
            keyword = _span_to_keyword(text, offsets, variant_start, variant_end)
            if not keyword:
                continue
            lowered = keyword.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            results.append((keyword, conf - penalty))
            if len(results) >= max_spans:
                break
        if len(results) >= max_spans:
            break
    return results


def extract_keywords_relaxed_windowed(
    text: str,
    tokenizer,
    model: "BIOBoundaryHead",
    device: torch.device,
    max_length: int,
    *,
    max_spans: int = 100,
    b_threshold: float = 0.15,
    window_stride: int = 128,
    window_strides: Sequence[int] | None = None,
    threshold_schedule: Sequence[float] | None = None,
    max_expand_steps: int = 1,
    max_subspan_width: int = 0,
) -> list[tuple[str, float]]:
    """Extract high-recall candidates with sliding windows and threshold union."""
    if not text.strip():
        return []

    threshold_values = list(threshold_schedule or [])
    if not threshold_values:
        threshold_values = [
            b_threshold,
            max(0.0, b_threshold * 0.5),
            0.0,
        ]
    threshold_values = [max(0.0, float(value)) for value in threshold_values]
    stride_values = [max(0, int(window_stride))]
    if window_strides:
        stride_values = []
        for stride in window_strides:
            stride_int = max(0, int(stride))
            if stride_int not in stride_values:
                stride_values.append(stride_int)
    if not stride_values:
        stride_values = [max(0, int(window_stride))]

    merged: dict[str, tuple[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for stride_idx, stride in enumerate(stride_values):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                padding=False,
            )

            input_id_windows = encoded["input_ids"]
            attention_windows = encoded["attention_mask"]
            offset_windows = encoded["offset_mapping"]

            for input_ids_window, attention_window, offsets_window in zip(
                input_id_windows,
                attention_windows,
                offset_windows,
            ):
                input_ids = torch.tensor([input_ids_window], dtype=torch.long, device=device)
                attention_mask = torch.tensor([attention_window], dtype=torch.long, device=device)
                result = model(input_ids=input_ids, attention_mask=attention_mask)
                emissions = result["emissions"][0]
                decoded_tags = model.decode(result["emissions"], attention_mask)[0]
                seq_len = int(attention_mask[0].sum().item())
                offsets = offsets_window[:seq_len]
                emissions = emissions[:seq_len]
                decoded_tags = decoded_tags[:seq_len]

                # Always include the Viterbi spans first.
                for keyword in spans_to_text(text, bio_tags_to_spans(decoded_tags), offsets):
                    lowered = keyword.lower()
                    conf = 2.0 - stride_idx * 1e-4
                    prev = merged.get(lowered)
                    if prev is None or conf > prev[1]:
                        merged[lowered] = (keyword, conf)

                for schedule_idx, threshold in enumerate(threshold_values):
                    for keyword, conf in extract_keywords_relaxed(
                        text,
                        offsets,
                        emissions,
                        decoded_tags,
                        max_spans=max_spans,
                        b_threshold=threshold,
                        max_expand_steps=max_expand_steps,
                        max_subspan_width=max_subspan_width,
                    ):
                        lowered = keyword.lower()
                        adjusted_conf = float(conf) - schedule_idx * 1e-3 - stride_idx * 1e-4
                        prev = merged.get(lowered)
                        if prev is None or adjusted_conf > prev[1]:
                            merged[lowered] = (keyword, adjusted_conf)

    ranked = sorted(merged.values(), key=lambda item: -item[1])
    return ranked[:max_spans]


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
