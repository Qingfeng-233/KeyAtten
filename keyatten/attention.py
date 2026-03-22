from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ATTENTION_METHODS = ("cls_attn", "received_attn", "samrank", "fusion_attn")


def build_model_bundle(model_name: str, device: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _aggregate_subwords_to_words(
    word_ids: list[int | None],
    token_scores: np.ndarray,
    word_count: int,
) -> np.ndarray:
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_scores[token_index])
        counts[word_id] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def _resolve_layer_index(layer_index: int, layer_count: int) -> int:
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"Layer index {layer_index} is out of range for {layer_count} attention layers.")
    return resolved


def _scores_from_attention_map(
    word_ids: list[int | None],
    attention_map: np.ndarray,
    word_count: int,
) -> dict[str, np.ndarray]:
    cls_scores = attention_map[0]
    received_scores = attention_map.sum(axis=0)

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = normalize_array(cls_scores) * normalize_array(received_scores)

    return {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
    }


def _aggregate_layer_word_scores(
    per_layer_scores: Sequence[dict[str, np.ndarray]],
    layer_weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    if not per_layer_scores:
        return {}

    weights = np.asarray(layer_weights if layer_weights is not None else [1.0] * len(per_layer_scores), dtype=np.float32)
    if weights.size != len(per_layer_scores):
        raise ValueError("layer_weights must have the same length as per_layer_scores.")
    weight_sum = float(weights.sum())
    if math.isclose(weight_sum, 0.0):
        raise ValueError("layer_weights sum to zero.")

    normalized_weights = weights / weight_sum
    aggregated: dict[str, np.ndarray] = {}
    for method_name in per_layer_scores[0]:
        stacked = np.stack([layer_scores[method_name] for layer_scores in per_layer_scores], axis=0)
        aggregated[method_name] = np.average(stacked, axis=0, weights=normalized_weights)
    return aggregated


def batched_attention_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
) -> list[dict[int, dict[str, np.ndarray]]]:
    if not batch_words:
        return []

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    results: list[dict[int, dict[str, np.ndarray]]] = []
    for start in range(0, len(batch_words), batch_size):
        word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        attention_by_layer = {
            layer_index: outputs.attentions[resolved_index].mean(dim=1).detach().cpu().numpy()
            for layer_index, resolved_index in resolved_indices.items()
        }

        for item_index, words in enumerate(word_batch):
            per_item_scores: dict[int, dict[str, np.ndarray]] = {}
            valid_token_count = int(encoded["attention_mask"][item_index].sum().item())
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            for layer_index in layer_indices:
                attention_map = attention_by_layer[layer_index][item_index][:valid_token_count, :valid_token_count]
                per_item_scores[layer_index] = _scores_from_attention_map(valid_word_ids, attention_map, len(words))
            results.append(per_item_scores)

    return results


def attention_word_scores(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    batched_scores = batched_attention_word_scores(
        [list(words)],
        model_bundle,
        layer_indices=effective_layer_indices,
        batch_size=1,
    )
    if not batched_scores:
        return {}
    per_layer_scores = [batched_scores[0][index] for index in effective_layer_indices]
    return _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)


def attention_word_scores_with_raw(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int | None]]:
    """Like attention_word_scores but also returns the token-level attention map and word_ids.

    Returns:
        (scores_by_method, attention_map, word_ids)
        - scores_by_method: same as attention_word_scores
        - attention_map: (valid_tokens, valid_tokens) averaged over heads
        - word_ids: token-to-word mapping from the tokenizer
    """
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    encoded = tokenizer(
        [list(words)],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    layer_count = len(outputs.attentions)
    valid_token_count = int(encoded["attention_mask"][0].sum().item())
    valid_word_ids = word_ids[:valid_token_count]

    # Use last effective layer for the raw attention map
    resolved_last = _resolve_layer_index(effective_layer_indices[-1], layer_count)
    raw_attention_map = outputs.attentions[resolved_last].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()

    # Compute word scores per layer and aggregate
    per_layer_scores = []
    for li in effective_layer_indices:
        resolved = _resolve_layer_index(li, layer_count)
        attn_map = outputs.attentions[resolved].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()
        per_layer_scores.append(_scores_from_attention_map(valid_word_ids, attn_map, len(words)))

    scores_by_method = _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)
    return scores_by_method, raw_attention_map, valid_word_ids


def rescore_with_new_words(
    attention_map: np.ndarray,
    old_word_ids: list[int | None],
    old_words: Sequence[str],
    new_words: Sequence[str],
    merge_map: list[list[int]],
) -> dict[str, np.ndarray]:
    """Recompute word scores using a new word boundary mapping without re-running the model.

    Args:
        attention_map: (valid_tokens, valid_tokens) from the original forward pass
        old_word_ids: token-to-old-word mapping
        old_words: original word list
        new_words: merged word list
        merge_map: merge_map[new_idx] = [old_idx_1, old_idx_2, ...] mapping new words to old word indices
    """
    # Build new word_ids: remap old word indices to new word indices
    old_to_new = {}
    for new_idx, old_indices in enumerate(merge_map):
        for old_idx in old_indices:
            old_to_new[old_idx] = new_idx

    new_word_ids = []
    for wid in old_word_ids:
        if wid is None or wid not in old_to_new:
            new_word_ids.append(None)
        else:
            new_word_ids.append(old_to_new[wid])

    return _scores_from_attention_map(new_word_ids, attention_map, len(new_words))


__all__ = [
    "ATTENTION_METHODS",
    "normalize_array",
    "attention_word_scores",
    "attention_word_scores_with_raw",
    "rescore_with_new_words",
    "batched_attention_word_scores",
]
