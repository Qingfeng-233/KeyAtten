from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from transformers import AutoModel


IGNORE_LABEL = -100


@dataclass(slots=True)
class TokenizedLabelExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class HiddenStateKeywordHead(nn.Module):
    """Frozen backbone + linear token classifier head."""

    def __init__(
        self, model_name: str, layer_index: int = -1, freeze_backbone: bool = True
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer_index = int(layer_index)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
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
        logits = self.classifier(hidden).squeeze(-1)
        return logits


def find_keyword_char_mask(text: str, keywords: Sequence[str]) -> np.ndarray:
    mask = np.zeros(len(text), dtype=bool)
    if not text:
        return mask

    ranked_keywords = sorted(
        {keyword.strip() for keyword in keywords if keyword and keyword.strip()},
        key=len,
        reverse=True,
    )
    for keyword in ranked_keywords:
        start = 0
        while True:
            index = text.find(keyword, start)
            if index < 0:
                break
            end = index + len(keyword)
            mask[index:end] = True
            start = index + 1
    return mask


def tokenize_with_keyword_labels(
    text: str,
    keywords: Sequence[str],
    tokenizer,
    max_length: int,
) -> TokenizedLabelExample:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    char_mask = find_keyword_char_mask(text, keywords)

    labels: list[int] = []
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end <= start:
            labels.append(IGNORE_LABEL)
            continue
        if start >= len(char_mask):
            labels.append(0)
            continue
        clipped_end = min(end, len(char_mask))
        labels.append(1 if bool(char_mask[start:clipped_end].any()) else 0)

    return TokenizedLabelExample(
        input_ids=[int(token_id) for token_id in encoded["input_ids"]],
        attention_mask=[int(v) for v in encoded["attention_mask"]],
        labels=labels,
    )


def collate_token_examples(
    batch: Sequence[TokenizedLabelExample], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in batch)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for item in batch:
        pad_size = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [int(pad_token_id)] * pad_size)
        attention_masks.append(item.attention_mask + [0] * pad_size)
        labels.append(item.labels + [IGNORE_LABEL] * pad_size)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


def compute_pos_weight(examples: Iterable[TokenizedLabelExample]) -> float:
    positive = 0
    negative = 0
    for item in examples:
        for label in item.labels:
            if label == 1:
                positive += 1
            elif label == 0:
                negative += 1
    if positive <= 0:
        return 1.0
    return max(float(negative) / float(positive), 1.0)


def token_level_prf(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    valid_mask = labels >= 0
    if not bool(valid_mask.any()):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = labels[valid_mask]
    y_pred = (probs[valid_mask] >= threshold).float()
    tp = float(((y_pred == 1.0) & (y_true == 1.0)).sum().item())
    fp = float(((y_pred == 1.0) & (y_true == 0.0)).sum().item())
    fn = float(((y_pred == 0.0) & (y_true == 1.0)).sum().item())

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_mask = labels >= 0
    if not bool(valid_mask.any()):
        return logits.new_tensor(0.0)

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    raw_loss = criterion(logits, labels)
    return raw_loss[valid_mask].mean()


def aggregate_token_probs_to_words(
    word_ids: Sequence[int | None], token_probs: np.ndarray, word_count: int
) -> np.ndarray:
    if word_count <= 0:
        return np.zeros(0, dtype=np.float32)
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_probs[token_index])
        counts[word_id] += 1.0
    counts[counts <= 0.0] = 1.0
    return sums / counts
