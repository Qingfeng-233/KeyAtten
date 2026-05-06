from __future__ import annotations

"""Compatibility shim: re-exports BIO boundary head components for src/ use.

The canonical implementation lives in benchmark/keyword_bench/bio_boundary_head.py.
This module copies only the inference-critical pieces (model class, decode helpers)
so that src/ does not depend on benchmark/.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

try:
    from torchcrf import CRF
except ImportError as exc:
    raise ImportError(
        "BIO boundary head requires `pytorch-crf` (pip install pytorch-crf)."
    ) from exc

IGNORE_LABEL = -100
TAG_B = 0
TAG_I = 1
TAG_O = 2
NUM_TAGS = 3


class BIOBoundaryHead(nn.Module):
    """Frozen backbone + CRF head for BIO keyword boundary detection."""

    def __init__(
        self,
        model_name: str,
        layer_index: int = -1,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        init_from_config: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer_index = int(layer_index)
        if init_from_config:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            self.backbone = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, NUM_TAGS)
        self.crf = CRF(num_tags=NUM_TAGS, batch_first=True)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
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
        emissions = self.classifier(hidden)

        crf_mask = attention_mask.bool()
        result: dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_LABEL] = TAG_O
            log_likelihood = self.crf(emissions, clean_labels, mask=crf_mask)
            result["log_likelihood"] = log_likelihood
            batch_size = emissions.size(0)
            result["loss"] = -log_likelihood / batch_size
        return result

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        crf_mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=crf_mask)


def bio_tags_to_spans(tags: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag == TAG_B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == TAG_I:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def spans_to_text(
    text: str, spans: list[tuple[int, int]], offsets: Sequence[tuple[int, int]]
) -> list[str]:
    keywords: list[str] = []
    for span_start, span_end in spans:
        if span_start >= len(offsets) or span_end > len(offsets):
            continue
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
    max_subspan_width: int = 0,
    max_expand_steps: int = 1,
) -> list[tuple[tuple[int, int], float]]:
    variants: list[tuple[tuple[int, int], float]] = [((span_start, span_end), 0.0)]
    for expand in range(1, max_expand_steps + 1):
        if span_start - expand >= 0:
            variants.append(((span_start - expand, span_end), 0.08 * expand))
        if span_end + expand <= seq_len:
            variants.append(((span_start, span_end + expand), 0.08 * expand))
        if span_start - expand >= 0 and span_end + expand <= seq_len:
            variants.append(((span_start - expand, span_end + expand), 0.12 * expand))
    if max_subspan_width > 0:
        base_width = span_end - span_start
        capped_width = min(base_width, max_subspan_width)
        for width in range(1, capped_width + 1):
            for start in range(span_start, span_end - width + 1):
                end = start + width
                if start == span_start and end == span_end:
                    continue
                trim_penalty = 0.04 * ((start - span_start) + (span_end - end))
                variants.append(((start, end), trim_penalty))
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
    if not text.strip():
        return []
    threshold_values = list(threshold_schedule or [])
    if not threshold_values:
        threshold_values = [b_threshold, max(0.0, b_threshold * 0.5), 0.0]
    threshold_values = [max(0.0, float(value)) for value in threshold_values]
    stride_values = [max(0, int(window_stride))]
    if window_strides:
        stride_values = []
        for stride in window_strides:
            stride_int = max(0, int(stride))
            if stride_int not in stride_values:
                stride_values.append(stride_int)
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
            for input_ids_window, attention_window, offsets_window in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["offset_mapping"],
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
