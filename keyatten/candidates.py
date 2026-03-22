from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import jieba.posseg as pseg
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .utils import normalize_phrase


VALID_POS_PREFIXES = ("n", "nz", "eng", "v", "vn")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


@dataclass(slots=True)
class WordWeight:
    word: str
    index: int
    weight: float
    pos_tag: str


@dataclass(slots=True)
class Candidate:
    text: str
    word_start: int
    word_end: int


def is_valid_token(word: str, pos_tag: str) -> bool:
    if not word.strip():
        return False
    if PUNCT_RE.match(word):
        return False
    if not pos_tag.startswith(VALID_POS_PREFIXES):
        return False
    if len(word) == 1 and pos_tag != "eng":
        return False
    return True


def is_valid_english_token(word: str) -> bool:
    lowered = word.strip().lower()
    if len(lowered) <= 1:
        return False
    if lowered in ENGLISH_STOP_WORDS:
        return False
    return bool(re.search(r"[a-z]", lowered))


def segment_text(text: str, language: str = "zh") -> tuple[List[str], List[str]]:
    if language.startswith("en"):
        words = EN_TOKEN_RE.findall(text)
        return words, ["eng"] * len(words)

    words: List[str] = []
    pos_tags: List[str] = []
    for token in pseg.cut(text):
        word = token.word.strip()
        if not word:
            continue
        words.append(word)
        pos_tags.append(token.flag)
    return words, pos_tags


def build_candidates(
    words: Sequence[str],
    pos_tags: Sequence[str],
    language: str = "zh",
    max_ngram: int = 4,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    joiner = "" if language.startswith("zh") else " "

    for start in range(len(words)):
        is_valid = (
            is_valid_token(words[start], pos_tags[start])
            if language.startswith("zh")
            else is_valid_english_token(words[start])
        )
        if not is_valid:
            continue

        for end in range(start + 1, min(len(words), start + max_ngram) + 1):
            if language.startswith("zh"):
                if not all(is_valid_token(words[index], pos_tags[index]) for index in range(start, end)):
                    break
            else:
                if not all(is_valid_english_token(words[index]) for index in range(start, end)):
                    break
                if words[start].lower() in ENGLISH_STOP_WORDS or words[end - 1].lower() in ENGLISH_STOP_WORDS:
                    continue

            phrase = joiner.join(words[start:end]).strip()
            if language.startswith("en"):
                phrase = phrase.strip("- ")

            normalized = normalize_phrase(phrase)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            candidates.append(Candidate(text=phrase, word_start=start, word_end=end))
    return candidates


def _candidate_score(
    candidate: Candidate,
    word_scores: Sequence[float],
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
) -> float | None:
    span = np.asarray(word_scores[candidate.word_start : candidate.word_end], dtype=np.float32)
    if span.size == 0:
        return None

    if aggregation_mode == "mean":
        score = float(span.mean())
    elif aggregation_mode == "max":
        score = float(span.max())
    elif aggregation_mode == "top2_mean":
        score = float(np.sort(span)[-2:].mean())
    elif aggregation_mode == "sum_sqrt_len":
        score = float(span.sum() / math.sqrt(max(span.size, 1)))
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

    if repeat_boost > 0.0 and token_counts is not None and words is not None:
        normalized_counts = []
        for word in words[candidate.word_start : candidate.word_end]:
            normalized = normalize_phrase(word)
            if not normalized:
                continue
            normalized_counts.append(float(token_counts.get(normalized, 1.0)))
        if normalized_counts:
            score *= 1.0 + repeat_boost * math.log1p(sum(normalized_counts) / len(normalized_counts))
    return score


def candidate_score_values(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
    candidate_starts: np.ndarray | None = None,
    candidate_ends: np.ndarray | None = None,
) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=np.float32)

    if candidate_starts is None or candidate_ends is None:
        candidate_starts = np.fromiter((candidate.word_start for candidate in candidates), dtype=np.int32, count=len(candidates))
        candidate_ends = np.fromiter((candidate.word_end for candidate in candidates), dtype=np.int32, count=len(candidates))

    word_scores_array = np.asarray(word_scores, dtype=np.float32)

    if aggregation_mode == "mean" and repeat_boost <= 0.0:
        prefix = np.zeros(word_scores_array.size + 1, dtype=np.float32)
        np.cumsum(word_scores_array, out=prefix[1:])
        lengths = np.maximum(candidate_ends - candidate_starts, 1)
        return (prefix[candidate_ends] - prefix[candidate_starts]) / lengths.astype(np.float32)

    scores = np.zeros(len(candidates), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        score = _candidate_score(
            candidate,
            word_scores_array,
            token_counts=token_counts,
            words=words,
            aggregation_mode=aggregation_mode,
            repeat_boost=repeat_boost,
        )
        if score is not None:
            scores[index] = float(score)
    return scores


def rank_candidates_from_scores(
    candidates: Sequence[Candidate],
    candidate_scores: Sequence[float],
    top_k: int = 30,
) -> List[str]:
    if not candidates:
        return []

    scores_array = np.asarray(candidate_scores, dtype=np.float32)
    finite_indices = np.flatnonzero(np.isfinite(scores_array))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(scores_array[finite_indices])[::-1]]
    limit = min(top_k, sorted_indices.size)
    return [candidates[int(index)].text for index in sorted_indices[:limit]]


def candidate_rank_from_word_scores(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    top_k: int = 30,
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
    candidate_starts: np.ndarray | None = None,
    candidate_ends: np.ndarray | None = None,
) -> List[str]:
    candidate_scores = candidate_score_values(
        candidates,
        word_scores,
        token_counts=token_counts,
        words=words,
        aggregation_mode=aggregation_mode,
        repeat_boost=repeat_boost,
        candidate_starts=candidate_starts,
        candidate_ends=candidate_ends,
    )
    return rank_candidates_from_scores(candidates, candidate_scores, top_k=top_k)


__all__ = [
    "Candidate",
    "WordWeight",
    "VALID_POS_PREFIXES",
    "PUNCT_RE",
    "EN_TOKEN_RE",
    "is_valid_token",
    "is_valid_english_token",
    "segment_text",
    "build_candidates",
    "candidate_score_values",
    "rank_candidates_from_scores",
    "candidate_rank_from_word_scores",
]
