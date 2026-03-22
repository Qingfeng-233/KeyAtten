from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .attention import normalize_array
from .candidates import is_valid_english_token, is_valid_token
from .utils import normalize_phrase


def token_counter(words: Sequence[str], pos_tags: Sequence[str], language: str = "zh") -> Counter[str]:
    counts: Counter[str] = Counter()
    for word, pos_tag in zip(words, pos_tags):
        is_valid = is_valid_token(word, pos_tag) if language.startswith("zh") else is_valid_english_token(word)
        if not is_valid:
            continue
        normalized = normalize_phrase(word)
        if normalized:
            counts[normalized] += 1
    return counts


def inverse_document_frequency(token_sets: Iterable[Iterable[str]]) -> dict[str, float]:
    token_sets_list = [set(tokens) for tokens in token_sets]
    doc_count = max(len(token_sets_list), 1)
    document_freq: Counter[str] = Counter()
    for tokens in token_sets_list:
        document_freq.update(tokens)
    return {
        token: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
        for token, freq in document_freq.items()
    }


def word_scores_from_token_values(
    words: Sequence[str],
    pos_tags: Sequence[str],
    token_values: dict[str, float],
    language: str = "zh",
) -> np.ndarray:
    scores = np.zeros(len(words), dtype=np.float32)
    for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        is_valid = is_valid_token(word, pos_tag) if language.startswith("zh") else is_valid_english_token(word)
        if not is_valid:
            continue
        scores[index] = float(token_values.get(normalize_phrase(word), 0.0))
    return scores


def combine_word_scores(
    primary_scores: Sequence[float],
    secondary_scores: Sequence[float],
    mode: str = "product",
) -> np.ndarray:
    primary = np.asarray(primary_scores, dtype=np.float32)
    secondary = np.asarray(secondary_scores, dtype=np.float32)
    if primary.shape != secondary.shape:
        raise ValueError("primary_scores and secondary_scores must have the same shape.")

    primary_norm = normalize_array(primary)
    secondary_norm = normalize_array(secondary)
    if mode == "product":
        return primary_norm * secondary_norm
    if mode == "sum":
        return 0.5 * primary_norm + 0.5 * secondary_norm
    raise ValueError(f"Unsupported combine mode: {mode}")


__all__ = [
    "token_counter",
    "inverse_document_frequency",
    "word_scores_from_token_values",
    "combine_word_scores",
]
