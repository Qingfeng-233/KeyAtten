from __future__ import annotations

from typing import Iterable, Sequence

from .word import segment_text
from ..utils import normalize_phrase


RECALL_CORE_PREFIXES = ("n", "eng", "vn")
RECALL_ALLOWED_PREFIXES = ("n", "eng", "a", "b", "v", "vn", "j", "l", "i")


def _is_core_pos(pos_tag: str) -> bool:
    return any(pos_tag.startswith(prefix) for prefix in RECALL_CORE_PREFIXES)


def _is_allowed_pos(pos_tag: str) -> bool:
    return any(pos_tag.startswith(prefix) for prefix in RECALL_ALLOWED_PREFIXES)


def _dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        normalized = normalize_phrase(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(phrase)
    return unique


def mine_recall_oriented_phrases(
    text: str,
    language: str = "zh",
    max_phrases: int = 50,
    max_ngram: int = 6,
) -> list[str]:
    """Mine high-recall phrase candidates for BIO candidate generation.

    The rule set is intentionally noun-biased:
    - single noun/proper-noun tokens
    - noun phrases ending in a noun/proper noun
    - adjective/attribute modifiers are allowed only before the core noun
    """
    if not text.strip():
        return []

    if not language.startswith("zh"):
        return []

    words, pos_tags = segment_text(text, language=language)
    if not words:
        return []

    phrases: list[str] = []
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + max_ngram) + 1):
            span_words = words[start:end]
            span_tags = pos_tags[start:end]
            if not _is_core_pos(span_tags[-1]):
                continue
            if not all(_is_allowed_pos(tag) for tag in span_tags):
                break
            if len(span_tags) > 1 and not all(
                _is_core_pos(tag) or tag.startswith(("a", "b", "v"))
                for tag in span_tags[:-1]
            ):
                break
            phrase = "".join(span_words).strip()
            if len(phrase) < 2 or len(phrase) > 16:
                continue
            phrases.append(phrase)

    return _dedupe_phrases(phrases)[:max_phrases]


def build_bio_positive_phrases(
    text: str,
    gold_keywords: Sequence[str],
    *,
    language: str = "zh",
    include_recall_phrases: bool = False,
    phrase_limit: int = 40,
) -> list[str]:
    positives = [kw.strip() for kw in gold_keywords if kw and kw.strip() and kw.strip() in text]
    if include_recall_phrases:
        positives.extend(
            phrase
            for phrase in mine_recall_oriented_phrases(
                text,
                language=language,
                max_phrases=phrase_limit,
            )
            if phrase in text
        )
    return _dedupe_phrases(positives)


def find_candidate_occurrences(text: str, candidate: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not candidate:
        return spans
    start = 0
    while True:
        idx = text.find(candidate, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(candidate)))
        start = idx + 1
    return spans
