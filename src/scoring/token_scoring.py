from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..candidates.word import is_valid_english_token, is_valid_token, segment_text
from .model_bundle import torch
from .utils import ATTENTION_METHODS, normalize_array, _is_text_piece

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


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


def _collect_text_token_offsets(
    offsets: Sequence[tuple[int, int]],
    prefix_char_count: int,
    text: str,
) -> tuple[list[int], list[tuple[int, int]], np.ndarray]:
    kept_indices: list[int] = []
    kept_offsets: list[tuple[int, int]] = []
    content_mask = np.zeros(len(offsets), dtype=bool)

    for token_index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < prefix_char_count:
            continue
        char_start = start - prefix_char_count
        char_end = end - prefix_char_count
        if char_end <= char_start:
            continue
        piece = text[char_start:char_end]
        if not _is_text_piece(piece):
            continue
        kept_indices.append(token_index)
        kept_offsets.append((char_start, char_end))
        content_mask[token_index] = True

    return kept_indices, kept_offsets, content_mask


def _token_method_scores_from_attention_map(
    attention_map: np.ndarray,
    token_indices: Sequence[int],
    *,
    is_causal: bool,
    content_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    if not token_indices:
        return {method_name: np.zeros(0, dtype=np.float32) for method_name in ATTENTION_METHODS}

    selected = np.asarray(token_indices, dtype=np.int32)
    resolved_content_mask = np.asarray(content_mask, dtype=bool)
    received_scores = attention_map.sum(axis=0)

    if is_causal:
        cls_scores = attention_map[-1].copy()
        cls_scores[~resolved_content_mask] = 0.0
        row_sum = float(cls_scores.sum())
        if row_sum > 1e-10:
            cls_scores /= row_sum

        masked_attn = attention_map.copy()
        masked_attn[:, ~resolved_content_mask] = 0.0
        row_sums = masked_attn.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
        masked_attn /= row_sums
        voted_scores = masked_attn.sum(axis=0)
    else:
        cls_scores = attention_map[0]
        voted_scores = received_scores.copy()

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = normalize_array(cls_scores) * normalize_array(received_scores)
    voter_count = np.arange(attention_map.shape[0], 0, -1, dtype=np.float32)
    voted_scores = voted_scores / voter_count
    excess_scores = _compute_excess_token_scores(
        attention_map,
        is_causal=is_causal,
        content_mask=resolved_content_mask,
    )

    return {
        "cls_attn": np.asarray(cls_scores[selected], dtype=np.float32),
        "received_attn": np.asarray(received_scores[selected], dtype=np.float32),
        "samrank": np.asarray(samrank_scores[selected], dtype=np.float32),
        "fusion_attn": np.asarray(fusion_scores[selected], dtype=np.float32),
        "voted_attn": np.asarray(voted_scores[selected], dtype=np.float32),
        "excess_attn": np.asarray(excess_scores[selected], dtype=np.float32),
    }


def _resolve_layer_index(layer_index: int, layer_count: int) -> int:
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"Layer index {layer_index} is out of range for {layer_count} attention layers.")
    return resolved


def _build_content_mask(
    word_ids: list[int | None],
    words: Sequence[str],
    pos_tags: Sequence[str],
    language: str,
) -> np.ndarray:
    """Build a boolean mask where True = content token, False = noise/special token."""
    n_tokens = len(word_ids)
    mask = np.zeros(n_tokens, dtype=bool)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= len(words):
            continue
        if language.startswith("en"):
            if is_valid_english_token(words[word_id]):
                mask[token_index] = True
        else:
            if is_valid_token(words[word_id], pos_tags[word_id]):
                mask[token_index] = True
    return mask


def _tokenize_instruction_prefix(
    instruction_prefix: str | None,
    language: str,
) -> tuple[list[str], list[str]]:
    prefix = (instruction_prefix or "").strip()
    if not prefix:
        return [], []
    return segment_text(prefix, language=language)


def _resolve_content_mask(
    word_ids: Sequence[int | None],
    content_mask: np.ndarray | None,
) -> np.ndarray:
    if content_mask is not None:
        return np.asarray(content_mask, dtype=bool)
    return np.asarray([word_id is not None for word_id in word_ids], dtype=bool)


def _compute_excess_token_scores(
    attention_map: np.ndarray,
    *,
    is_causal: bool,
    content_mask: np.ndarray,
) -> np.ndarray:
    """Compute sink-free excess attention over a row-wise uniform content baseline."""
    n_tokens = attention_map.shape[0]
    excess_scores = np.zeros(n_tokens, dtype=np.float32)

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            score_view = excess_scores[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            score_view = excess_scores

        visible_content_count = int(row_content_mask.sum())
        if visible_content_count <= 1:
            continue

        row_content = row_visible[row_content_mask].astype(np.float32, copy=False)
        content_mass = float(row_content.sum())
        if content_mass <= 1e-10:
            continue

        p = row_content / content_mass
        log_visible_content = math.log(visible_content_count)
        entropy = float(-(p * np.log(np.clip(p, 1e-10, None))).sum())
        certainty = max(0.0, 1.0 - (entropy / log_visible_content)) if log_visible_content > 1e-10 else 0.0

        baseline = 1.0 / visible_content_count
        contribution = (content_mass * certainty) * (p - baseline)
        score_view[row_content_mask] += contribution.astype(np.float32, copy=False)

    return excess_scores


def _scores_from_attention_map_causal(
    word_ids: list[int | None],
    attention_map: np.ndarray,
    word_count: int,
    content_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute attention scores adapted for decoder (causal) models.

    Only cls_attn is replaced with content-masked last-token attention.
    received_attn and samrank keep the original computation — the candidate
    filtering pipeline already handles noise in these signals.
    voted_attn is a new method: content-masked, row-renormalized, position-normalized.
    """
    n_tokens = attention_map.shape[0]
    resolved_content_mask = _resolve_content_mask(word_ids, content_mask)

    # --- cls_attn: content-masked last-token attention ---
    cls_scores = attention_map[-1].copy()
    cls_scores[~resolved_content_mask] = 0.0
    row_sum = cls_scores.sum()
    if row_sum > 1e-10:
        cls_scores /= row_sum

    # --- received_attn: original computation (no masking) ---
    received_scores = attention_map.sum(axis=0)

    # --- samrank: original computation on original received_attn ---
    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores

    # --- fusion_attn: new cls × original received ---
    fusion_scores = normalize_array(cls_scores) * normalize_array(received_scores)

    # --- voted_attn: content-masked, row-renormalized, position-normalized ---
    masked_attn = attention_map.copy()
    masked_attn[:, ~resolved_content_mask] = 0.0
    row_sums = masked_attn.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
    masked_attn /= row_sums
    voted_scores = masked_attn.sum(axis=0)
    voter_count = np.arange(n_tokens, 0, -1, dtype=np.float32)
    voted_scores /= voter_count
    excess_scores = _compute_excess_token_scores(
        attention_map,
        is_causal=True,
        content_mask=resolved_content_mask,
    )

    return {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
        "voted_attn": _aggregate_subwords_to_words(word_ids, voted_scores, word_count),
        "excess_attn": _aggregate_subwords_to_words(word_ids, excess_scores, word_count),
    }


def _scores_from_attention_map(
    word_ids: list[int | None],
    attention_map: np.ndarray,
    word_count: int,
    *,
    is_causal: bool = False,
    content_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    resolved_content_mask = _resolve_content_mask(word_ids, content_mask)
    if is_causal:
        return _scores_from_attention_map_causal(word_ids, attention_map, word_count, resolved_content_mask)

    cls_scores = attention_map[0]
    received_scores = attention_map.sum(axis=0)

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = normalize_array(cls_scores) * normalize_array(received_scores)

    # --- voted_attn: position-normalized received (encoder: no content mask needed) ---
    n_tokens = attention_map.shape[0]
    voter_count = np.arange(n_tokens, 0, -1, dtype=np.float32)
    voted_scores = received_scores / voter_count
    excess_scores = _compute_excess_token_scores(
        attention_map,
        is_causal=False,
        content_mask=resolved_content_mask,
    )

    return {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
        "voted_attn": _aggregate_subwords_to_words(word_ids, voted_scores, word_count),
        "excess_attn": _aggregate_subwords_to_words(word_ids, excess_scores, word_count),
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


def _encode_words_batch_onnx(
    batch_words: Sequence[Sequence[str]],
    tokenizer: Tokenizer,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, list[list[int | None]], list[int]]:
    encodings = tokenizer.encode_batch([list(words) for words in batch_words], is_pretokenized=True)
    trimmed_ids: list[np.ndarray] = []
    trimmed_masks: list[np.ndarray] = []
    word_ids_per_item: list[list[int | None]] = []
    valid_token_counts: list[int] = []

    for encoding in encodings:
        input_ids = np.asarray(encoding.ids[:max_length], dtype=np.int64)
        attention_mask = np.asarray(encoding.attention_mask[:max_length], dtype=np.int64)
        valid_token_count = int(attention_mask.sum())
        if valid_token_count <= 0:
            valid_token_count = min(len(input_ids), 1)
        trimmed_ids.append(input_ids[:valid_token_count])
        trimmed_masks.append(attention_mask[:valid_token_count])
        word_ids_per_item.append(list(encoding.word_ids[:valid_token_count]))
        valid_token_counts.append(valid_token_count)

    max_tokens = max(valid_token_counts, default=1)
    batch_input_ids = np.zeros((len(batch_words), max_tokens), dtype=np.int64)
    batch_attention_mask = np.zeros((len(batch_words), max_tokens), dtype=np.int64)

    for index, (input_ids, attention_mask) in enumerate(zip(trimmed_ids, trimmed_masks)):
        batch_input_ids[index, : input_ids.size] = input_ids
        batch_attention_mask[index, : attention_mask.size] = attention_mask

    return batch_input_ids, batch_attention_mask, word_ids_per_item, valid_token_counts


def _validate_onnx_layer_request(model_bundle: dict, layer_indices: Sequence[int]) -> None:
    exported_layer_index = int(model_bundle["layer_index"])
    if len(layer_indices) != 1 or int(layer_indices[0]) != exported_layer_index:
        raise ValueError(
            "ONNX backend supports only the exported single attention layer. "
            f"Requested {list(layer_indices)}, exported layer is {exported_layer_index}."
        )


def _batched_attention_word_scores_onnx(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
    *,
    batch_pos_tags: Sequence[Sequence[str]] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
) -> list[dict[int, dict[str, np.ndarray]]]:
    _validate_onnx_layer_request(model_bundle, layer_indices)
    tokenizer = model_bundle["tokenizer"]
    session = model_bundle["session"]
    max_length = int(model_bundle["max_length"])
    output_name = session.get_outputs()[0].name
    is_causal = model_bundle.get("is_causal", False)
    prefix_words, prefix_pos_tags = _tokenize_instruction_prefix(instruction_prefix, language)

    results: list[dict[int, dict[str, np.ndarray]]] = []
    for start in range(0, len(batch_words), batch_size):
        content_word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        pos_tag_batch = (
            [list(pt) for pt in batch_pos_tags[start : start + batch_size]]
            if batch_pos_tags is not None
            else None
        )
        word_batch = [prefix_words + words for words in content_word_batch]
        combined_pos_tag_batch = None
        if pos_tag_batch is not None:
            combined_pos_tag_batch = [prefix_pos_tags + pos_tags for pos_tags in pos_tag_batch]
        input_ids, attention_mask, word_ids_per_item, valid_token_counts = _encode_words_batch_onnx(
            word_batch,
            tokenizer,
            max_length=max_length,
        )
        outputs = session.run(
            [output_name],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        attention_batch = outputs[0]

        prefix_word_count = len(prefix_words)
        for item_index, words in enumerate(word_batch):
            valid_token_count = valid_token_counts[item_index]
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            attention_map = attention_batch[item_index][:valid_token_count, :valid_token_count].astype(np.float32, copy=False)
            content_mask = None
            if is_causal and combined_pos_tag_batch is not None:
                content_mask = _build_content_mask(valid_word_ids, words, combined_pos_tag_batch[item_index], language)
            combined_scores = _scores_from_attention_map(
                valid_word_ids, attention_map, len(words),
                is_causal=is_causal, content_mask=content_mask,
            )
            content_word_count = len(content_word_batch[item_index])
            results.append(
                {
                    layer_indices[0]: {
                        method_name: method_scores[prefix_word_count : prefix_word_count + content_word_count]
                        for method_name, method_scores in combined_scores.items()
                    },
                }
            )

    return results


def batched_attention_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
    *,
    batch_pos_tags: Sequence[Sequence[str]] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
) -> list[dict[int, dict[str, np.ndarray]]]:
    if not batch_words:
        return []

    if model_bundle.get("backend") == "onnx":
        return _batched_attention_word_scores_onnx(
            batch_words,
            model_bundle,
            layer_indices=layer_indices,
            batch_size=batch_size,
            batch_pos_tags=batch_pos_tags,
            language=language,
            instruction_prefix=instruction_prefix,
        )

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    is_causal = model_bundle.get("is_causal", False)
    max_length = int(model_bundle.get("max_length", 512))
    prefix_words, prefix_pos_tags = _tokenize_instruction_prefix(instruction_prefix, language)

    results: list[dict[int, dict[str, np.ndarray]]] = []
    for start in range(0, len(batch_words), batch_size):
        content_word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        pos_tag_batch = (
            [list(pt) for pt in batch_pos_tags[start : start + batch_size]]
            if batch_pos_tags is not None
            else None
        )
        word_batch = [prefix_words + words for words in content_word_batch]
        combined_pos_tag_batch = None
        if pos_tag_batch is not None:
            combined_pos_tag_batch = [prefix_pos_tags + pos_tags for pos_tags in pos_tag_batch]
        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_attentions=True)

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        attention_by_layer = {
            layer_index: outputs.attentions[resolved_index].mean(dim=1).detach().cpu().numpy()
            for layer_index, resolved_index in resolved_indices.items()
        }

        prefix_word_count = len(prefix_words)
        for item_index, words in enumerate(word_batch):
            per_item_scores: dict[int, dict[str, np.ndarray]] = {}
            valid_token_count = int(encoded["attention_mask"][item_index].sum().item())
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            content_mask = None
            if is_causal and combined_pos_tag_batch is not None:
                content_mask = _build_content_mask(valid_word_ids, words, combined_pos_tag_batch[item_index], language)
            content_word_count = len(content_word_batch[item_index])
            for layer_index in layer_indices:
                attention_map = attention_by_layer[layer_index][item_index][:valid_token_count, :valid_token_count]
                combined_scores = _scores_from_attention_map(
                    valid_word_ids, attention_map, len(words),
                    is_causal=is_causal, content_mask=content_mask,
                )
                per_item_scores[layer_index] = {
                    method_name: method_scores[prefix_word_count : prefix_word_count + content_word_count]
                    for method_name, method_scores in combined_scores.items()
                }
            results.append(per_item_scores)

    return results


def attention_word_scores(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
    *,
    pos_tags: Sequence[str] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    batched_scores = batched_attention_word_scores(
        [list(words)],
        model_bundle,
        layer_indices=effective_layer_indices,
        batch_size=1,
        batch_pos_tags=[list(pos_tags)] if pos_tags is not None else None,
        language=language,
        instruction_prefix=instruction_prefix,
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
    *,
    pos_tags: Sequence[str] | None = None,
    language: str = "zh",
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int | None]]:
    """Like attention_word_scores but also returns the token-level attention map and word_ids.

    Returns:
        (scores_by_method, attention_map, word_ids)
        - scores_by_method: same as attention_word_scores
        - attention_map: (valid_tokens, valid_tokens) averaged over heads
        - word_ids: token-to-word mapping from the tokenizer
    """
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    is_causal = model_bundle.get("is_causal", False)

    if model_bundle.get("backend") == "onnx":
        _validate_onnx_layer_request(model_bundle, effective_layer_indices)
        tokenizer = model_bundle["tokenizer"]
        session = model_bundle["session"]
        max_length = int(model_bundle["max_length"])
        output_name = session.get_outputs()[0].name

        input_ids, attention_mask, word_ids_per_item, valid_token_counts = _encode_words_batch_onnx(
            [list(words)],
            tokenizer,
            max_length=max_length,
        )
        outputs = session.run(
            [output_name],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        valid_token_count = valid_token_counts[0]
        valid_word_ids = word_ids_per_item[0][:valid_token_count]
        raw_attention_map = outputs[0][0][:valid_token_count, :valid_token_count].astype(np.float32, copy=False)
        content_mask = None
        if is_causal and pos_tags is not None:
            content_mask = _build_content_mask(valid_word_ids, words, pos_tags, language)
        scores_by_method = _scores_from_attention_map(
            valid_word_ids, raw_attention_map, len(words),
            is_causal=is_causal, content_mask=content_mask,
        )
        return scores_by_method, raw_attention_map, valid_word_ids

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    max_length = int(model_bundle.get("max_length", 512))

    encoded = tokenizer(
        [list(words)],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded, output_attentions=True)

    layer_count = len(outputs.attentions)
    valid_token_count = int(encoded["attention_mask"][0].sum().item())
    valid_word_ids = word_ids[:valid_token_count]

    content_mask = None
    if is_causal and pos_tags is not None:
        content_mask = _build_content_mask(valid_word_ids, words, pos_tags, language)

    # Use last effective layer for the raw attention map
    resolved_last = _resolve_layer_index(effective_layer_indices[-1], layer_count)
    raw_attention_map = outputs.attentions[resolved_last].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()

    # Compute word scores per layer and aggregate
    per_layer_scores = []
    for li in effective_layer_indices:
        resolved = _resolve_layer_index(li, layer_count)
        attn_map = outputs.attentions[resolved].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()
        per_layer_scores.append(_scores_from_attention_map(
            valid_word_ids, attn_map, len(words),
            is_causal=is_causal, content_mask=content_mask,
        ))

    scores_by_method = _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)
    return scores_by_method, raw_attention_map, valid_word_ids


def attention_token_scores(
    text: str,
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
    *,
    language: str = "zh",
    instruction_prefix: str | None = None,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    is_causal = model_bundle.get("is_causal", False)
    prefix = (instruction_prefix or "").strip()
    prefix_text = f"{prefix}\n" if prefix and is_causal and language.startswith("zh") else ""
    prefixed_text = f"{prefix_text}{text}"
    prefix_char_count = len(prefix_text)

    if model_bundle.get("backend") == "onnx":
        _validate_onnx_layer_request(model_bundle, effective_layer_indices)
        tokenizer = model_bundle["tokenizer"]
        session = model_bundle["session"]
        max_length = int(model_bundle["max_length"])
        output_name = session.get_outputs()[0].name

        encoding = tokenizer.encode(prefixed_text)
        input_ids = np.asarray(encoding.ids[:max_length], dtype=np.int64)
        attention_mask = np.asarray(encoding.attention_mask[:max_length], dtype=np.int64)
        valid_token_count = int(attention_mask.sum())
        if valid_token_count <= 0:
            return {method_name: np.zeros(0, dtype=np.float32) for method_name in ATTENTION_METHODS}, []

        offsets = [(int(start), int(end)) for start, end in encoding.offsets[:valid_token_count]]
        token_indices, token_offsets, content_mask = _collect_text_token_offsets(offsets, prefix_char_count, text)
        outputs = session.run(
            [output_name],
            {
                "input_ids": input_ids[:valid_token_count][None, :],
                "attention_mask": attention_mask[:valid_token_count][None, :],
            },
        )
        raw_attention_map = outputs[0][0][:valid_token_count, :valid_token_count].astype(np.float32, copy=False)
        scores_by_method = _token_method_scores_from_attention_map(
            raw_attention_map,
            token_indices,
            is_causal=is_causal,
            content_mask=content_mask[:valid_token_count],
        )
        return scores_by_method, token_offsets

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    max_length = int(model_bundle.get("max_length", 512))
    encoded = tokenizer(
        prefixed_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    raw_offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0].tolist()]
    token_indices, token_offsets, content_mask = _collect_text_token_offsets(raw_offsets, prefix_char_count, text)
    encoded.pop("offset_mapping")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded, output_attentions=True)

    valid_token_count = int(encoded["attention_mask"][0].sum().item())
    layer_count = len(outputs.attentions)
    per_layer_scores = []
    for layer_id in effective_layer_indices:
        resolved = _resolve_layer_index(layer_id, layer_count)
        attention_map = outputs.attentions[resolved].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()
        per_layer_scores.append(
            _token_method_scores_from_attention_map(
                attention_map,
                [index for index in token_indices if index < valid_token_count],
                is_causal=is_causal,
                content_mask=content_mask[:valid_token_count],
            )
        )
    return _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights), [
        offset for index, offset in zip(token_indices, token_offsets, strict=False) if index < valid_token_count
    ]


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
    "attention_word_scores",
    "attention_word_scores_with_raw",
    "attention_token_scores",
    "rescore_with_new_words",
    "batched_attention_word_scores",
]
