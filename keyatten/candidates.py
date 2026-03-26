from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .utils import normalize_phrase


VALID_POS_PREFIXES = ("n", "eng", "v")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_BRACKET_RE = re.compile(r"[《\u300a](.+?)[》\u300b]")
_JIEBA = None
_PSEG = None
_ENGLISH_STOP_WORDS = None
_LOADED_USER_DICT_PATHS: set[str] = set()
_REGISTERED_USER_TERMS: set[tuple[str, int | None, str | None]] = set()


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


def _require_jieba():
    global _JIEBA, _PSEG
    if _JIEBA is not None and _PSEG is not None:
        return _JIEBA, _PSEG

    try:
        import jieba
        import jieba.posseg as pseg
    except ImportError as exc:
        raise ImportError(
            "Chinese tokenization requires optional dependency `jieba>=0.42`. "
            "Install with `pip install \"keyatten[zh]\"`."
        ) from exc

    _JIEBA = jieba
    _PSEG = pseg
    return _JIEBA, _PSEG


def _english_stop_words() -> frozenset[str]:
    global _ENGLISH_STOP_WORDS
    if _ENGLISH_STOP_WORDS is not None:
        return _ENGLISH_STOP_WORDS

    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    except ImportError as exc:
        raise ImportError(
            "English token filtering requires optional dependency `scikit-learn>=1.0`. "
            "Install with `pip install \"keyatten[en]\"`."
        ) from exc

    _ENGLISH_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
    return _ENGLISH_STOP_WORDS


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
    if lowered in _english_stop_words():
        return False
    return bool(re.search(r"[a-z]", lowered))


def _register_bracketed_terms(text: str) -> None:
    """自动将书名号《》内的内容注册到 jieba 词典，词性标记为专有名词。"""
    jieba, _ = _require_jieba()
    for match in _BRACKET_RE.finditer(text):
        term = match.group(1).strip()
        if term:
            jieba.add_word(term, tag="nz")


def _register_user_term(term: str, freq: int | None = None, tag: str | None = "nz") -> None:
    normalized = term.strip()
    if not normalized:
        return
    jieba, _ = _require_jieba()
    key = (normalized, freq, tag)
    if key in _REGISTERED_USER_TERMS:
        return
    jieba.add_word(normalized, freq=freq, tag=tag)
    _REGISTERED_USER_TERMS.add(key)


def apply_user_dictionary(
    user_dict: str | Sequence[str] | Mapping[str, str | tuple[int | None, str | None]] | None,
) -> None:
    if user_dict is None:
        return

    if isinstance(user_dict, str):
        dictionary_path = str(Path(user_dict).resolve())
        if dictionary_path in _LOADED_USER_DICT_PATHS:
            return
        jieba, _ = _require_jieba()
        jieba.load_userdict(dictionary_path)
        _LOADED_USER_DICT_PATHS.add(dictionary_path)
        return

    if isinstance(user_dict, Mapping):
        for term, spec in user_dict.items():
            if isinstance(spec, tuple):
                freq, tag = spec
            else:
                freq, tag = None, spec
            _register_user_term(term, freq=freq, tag=tag or "nz")
        return

    for term in user_dict:
        _register_user_term(str(term), tag="nz")


def segment_text(
    text: str,
    language: str = "zh",
    user_dict: str | Sequence[str] | Mapping[str, str | tuple[int | None, str | None]] | None = None,
) -> tuple[list[str], list[str]]:
    if language.startswith("en"):
        words = EN_TOKEN_RE.findall(text)
        return words, ["eng"] * len(words)

    apply_user_dictionary(user_dict)
    _, pseg = _require_jieba()
    _register_bracketed_terms(text)
    words: list[str] = []
    pos_tags: list[str] = []
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
) -> list[Candidate]:
    candidates: list[Candidate] = []
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

            phrase = joiner.join(words[start:end]).strip()
            if language.startswith("en"):
                phrase = phrase.strip("- ")

            normalized = normalize_phrase(phrase)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            candidates.append(Candidate(text=phrase, word_start=start, word_end=end))
    return candidates


def _aggregate_span_scores(
    span: np.ndarray,
    aggregation_mode: str,
) -> float:
    if span.size == 0:
        return 0.0
    if aggregation_mode == "mean":
        return float(span.mean())
    if aggregation_mode == "max":
        return float(span.max())
    if aggregation_mode == "top2_mean":
        return float(np.sort(span)[-2:].mean())
    if aggregation_mode == "sum_sqrt_len":
        return float(span.sum() / math.sqrt(max(span.size, 1)))
    raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")


def locate_word_offsets(
    text: str,
    words: Sequence[str],
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = text.find(word, cursor)
        if start < 0:
            start = text.find(word)
            if start < 0:
                raise ValueError(f"Failed to align word {word!r} in text.")
        end = start + len(word)
        offsets.append((start, end))
        cursor = end
    return offsets


def candidate_char_spans(
    candidates: Sequence[Candidate],
    word_offsets: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for candidate in candidates:
        start = int(word_offsets[candidate.word_start][0])
        end = int(word_offsets[candidate.word_end - 1][1])
        spans.append((start, end))
    return spans


def token_values_from_word_values(
    token_offsets: Sequence[tuple[int, int]],
    word_offsets: Sequence[tuple[int, int]],
    word_values: Sequence[float],
) -> np.ndarray:
    token_values = np.zeros(len(token_offsets), dtype=np.float32)
    word_index = 0
    for token_index, (token_start, token_end) in enumerate(token_offsets):
        while word_index < len(word_offsets) and word_offsets[word_index][1] <= token_start:
            word_index += 1
        best_value = 0.0
        best_overlap = 0
        probe_index = word_index
        while probe_index < len(word_offsets):
            word_start, word_end = word_offsets[probe_index]
            if word_start >= token_end:
                break
            overlap = max(0, min(token_end, word_end) - max(token_start, word_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_value = float(word_values[probe_index])
            probe_index += 1
        token_values[token_index] = best_value
    return token_values


def _candidate_score(
    candidate: Candidate,
    word_scores: Sequence[float],
    token_counts: dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
) -> float | None:
    span = np.asarray(word_scores[candidate.word_start : candidate.word_end], dtype=np.float32)
    if span.size == 0:
        return None

    score = _aggregate_span_scores(span, aggregation_mode)

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
    token_counts: dict[str, float] | None = None,
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


def candidate_score_values_from_token_spans(
    candidate_spans: Sequence[tuple[int, int]],
    token_offsets: Sequence[tuple[int, int]],
    token_scores: Sequence[float],
    *,
    aggregation_mode: str = "mean",
) -> np.ndarray:
    if not candidate_spans:
        return np.zeros(0, dtype=np.float32)

    score_array = np.asarray(token_scores, dtype=np.float32)
    scores = np.zeros(len(candidate_spans), dtype=np.float32)
    for index, (char_start, char_end) in enumerate(candidate_spans):
        overlapped = [
            float(score_array[token_index])
            for token_index, (token_start, token_end) in enumerate(token_offsets)
            if token_end > char_start and token_start < char_end
        ]
        scores[index] = _aggregate_span_scores(np.asarray(overlapped, dtype=np.float32), aggregation_mode)
    return scores


def rank_candidates_from_scores(
    candidates: Sequence[Candidate],
    candidate_scores: Sequence[float],
    top_k: int = 30,
    dedup_nested: bool = False,
) -> list[str]:
    if not candidates:
        return []

    scores_array = np.asarray(candidate_scores, dtype=np.float32)
    finite_indices = np.flatnonzero(np.isfinite(scores_array))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(scores_array[finite_indices])[::-1]]
    limit = min(top_k, sorted_indices.size)
    if not dedup_nested:
        return [candidates[int(index)].text for index in sorted_indices[:limit]]

    ranked: list[str] = []
    selected_normalized: list[str] = []
    for index in sorted_indices:
        candidate = candidates[int(index)]
        normalized = normalize_phrase(candidate.text)
        if not normalized:
            continue

        is_nested = any(
            normalized in selected_text or selected_text in normalized
            for selected_text in selected_normalized
        )
        if is_nested:
            continue

        ranked.append(candidate.text)
        selected_normalized.append(normalized)
        if len(ranked) >= limit:
            break
    return ranked


def candidate_rank_from_word_scores(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    top_k: int = 30,
    dedup_nested: bool = False,
    token_counts: dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
    candidate_starts: np.ndarray | None = None,
    candidate_ends: np.ndarray | None = None,
) -> list[str]:
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
    return rank_candidates_from_scores(
        candidates,
        candidate_scores,
        top_k=top_k,
        dedup_nested=dedup_nested,
    )


def candidate_rank_from_token_scores(
    candidates: Sequence[Candidate],
    candidate_spans: Sequence[tuple[int, int]],
    token_offsets: Sequence[tuple[int, int]],
    token_scores: Sequence[float],
    top_k: int = 30,
    dedup_nested: bool = False,
    aggregation_mode: str = "mean",
) -> list[str]:
    candidate_scores = candidate_score_values_from_token_spans(
        candidate_spans,
        token_offsets,
        token_scores,
        aggregation_mode=aggregation_mode,
    )
    return rank_candidates_from_scores(
        candidates,
        candidate_scores,
        top_k=top_k,
        dedup_nested=dedup_nested,
    )


def merge_single_chars(
    words: list[str],
    pos_tags: list[str],
    attention_map: np.ndarray,
    word_ids: list[int | None],
    merge_threshold: float = 0.3,
) -> tuple[list[str], list[str], list[list[int]], bool]:
    """Merge consecutive single-character tokens guided by attention scores.

    Returns:
        (merged_words, merged_pos_tags, merge_map, changed)
        - merge_map[new_idx] = [old_idx_1, ...] for rescore_with_new_words
        - changed: True if any merging occurred
    """
    seq_len = attention_map.shape[0]

    # Precompute adjacency scores between consecutive tokens in model space
    adj_raw = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len - 1):
        adj_raw[i] = (attention_map[i, i + 1] + attention_map[i + 1, i]) / 2.0
    min_a, max_a = float(adj_raw.min()), float(adj_raw.max())
    if max_a > min_a:
        adj_norm = (adj_raw - min_a) / (max_a - min_a)
    else:
        adj_norm = np.zeros_like(adj_raw)

    # Build word_id -> token indices mapping
    word_to_tokens: dict[int, list[int]] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None and 0 <= wid < len(words):
            word_to_tokens.setdefault(wid, []).append(tok_idx)

    # Find merge score between adjacent words using their boundary tokens
    def _word_adjacency_score(w1: int, w2: int) -> float:
        toks1 = word_to_tokens.get(w1, [])
        toks2 = word_to_tokens.get(w2, [])
        if not toks1 or not toks2:
            return 0.0
        last_tok = toks1[-1]
        if last_tok < seq_len:
            return float(adj_norm[last_tok])
        return 0.0

    merged_words: list[str] = []
    merged_pos: list[str] = []
    merge_map: list[list[int]] = []
    changed = False

    i = 0
    while i < len(words):
        word = words[i]

        # Skip non-single-char or punctuation
        if len(word) != 1 or PUNCT_RE.match(word):
            merged_words.append(word)
            merged_pos.append(pos_tags[i])
            merge_map.append([i])
            i += 1
            continue

        # Collect consecutive single-char non-punctuation sequence
        single_start = i
        while i + 1 < len(words) and len(words[i + 1]) == 1 and not PUNCT_RE.match(words[i + 1]):
            i += 1

        if i == single_start:
            # Only one single char, keep as is
            merged_words.append(word)
            merged_pos.append(pos_tags[single_start])
            merge_map.append([single_start])
            i += 1
            continue

        # Multiple consecutive single chars: merge based on attention
        group = words[single_start]
        group_indices = [single_start]
        for j in range(single_start, i):
            score = _word_adjacency_score(j, j + 1)
            if score >= merge_threshold:
                group += words[j + 1]
                group_indices.append(j + 1)
            else:
                merged_words.append(group)
                merged_pos.append("nz" if len(group) > 1 else pos_tags[group_indices[0]])
                if len(group) > 1:
                    changed = True
                merge_map.append(group_indices)
                group = words[j + 1]
                group_indices = [j + 1]

        merged_words.append(group)
        merged_pos.append("nz" if len(group) > 1 else pos_tags[group_indices[0]])
        if len(group) > 1:
            changed = True
        merge_map.append(group_indices)
        i += 1

    return merged_words, merged_pos, merge_map, changed


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
    "locate_word_offsets",
    "candidate_char_spans",
    "token_values_from_word_values",
    "candidate_score_values",
    "candidate_score_values_from_token_spans",
    "rank_candidates_from_scores",
    "candidate_rank_from_word_scores",
    "candidate_rank_from_token_scores",
    "merge_single_chars",
]
