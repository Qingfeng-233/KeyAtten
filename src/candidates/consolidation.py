from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from ..utils import normalize_phrase


_EDGE_PUNCT_RE = re.compile(r"^[\s\"'“”‘’《》〈〉「」『』（）()【】\[\]、，,。；;：:！？!?·]+|[\s\"'“”‘’《》〈〉「」『』（）()【】\[\]、，,。；;：:！？!?·]+$")


@dataclass(frozen=True)
class ScoredCandidate:
    text: str
    score: float
    span: tuple[int, int]


def clean_candidate_text(text: str) -> str:
    cleaned = _EDGE_PUNCT_RE.sub("", text.strip())
    return cleaned.strip()


def dedupe_candidate_texts(candidates: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        cleaned = clean_candidate_text(candidate)
        normalized = normalize_phrase(cleaned)
        if not cleaned or len(cleaned) < 2 or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(cleaned)
    return unique


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _overlap_ratio(span_a: tuple[int, int], span_b: tuple[int, int]) -> float:
    overlap = max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    if overlap <= 0:
        return 0.0
    shorter = min(span_a[1] - span_a[0], span_b[1] - span_b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _prefer_new_candidate(
    current: ScoredCandidate,
    existing: ScoredCandidate,
    *,
    score_margin: float,
) -> bool:
    if current.score > existing.score:
        return True
    if current.score < existing.score - score_margin:
        return False
    current_len = len(current.text)
    existing_len = len(existing.text)
    if current_len != existing_len:
        return current_len > existing_len
    return current.span[0] < existing.span[0]


def consolidate_scored_candidates(
    candidates: Sequence[ScoredCandidate],
    *,
    top_k: int,
    overlap_threshold: float = 0.8,
    score_margin: float = 0.03,
) -> list[str]:
    if top_k <= 0:
        return []

    ranked = sorted(candidates, key=lambda item: (-item.score, -len(item.text), item.span[0]))
    selected: list[ScoredCandidate] = []
    selected_norms: set[str] = set()

    for candidate in ranked:
        normalized = normalize_phrase(candidate.text)
        if not normalized or normalized in selected_norms:
            continue

        replaced = False
        suppressed = False
        for index, existing in enumerate(selected):
            overlap = _overlap_ratio(candidate.span, existing.span)
            if overlap < overlap_threshold:
                continue
            if not (_contains(candidate.span, existing.span) or _contains(existing.span, candidate.span)):
                continue

            if _prefer_new_candidate(candidate, existing, score_margin=score_margin):
                selected[index] = candidate
                selected_norms.discard(normalize_phrase(existing.text))
                selected_norms.add(normalized)
                replaced = True
            else:
                suppressed = True
            break

        if suppressed or replaced:
            continue

        selected.append(candidate)
        selected_norms.add(normalized)
        if len(selected) >= top_k:
            break

    return [candidate.text for candidate in selected[:top_k]]


__all__ = [
    "ScoredCandidate",
    "clean_candidate_text",
    "consolidate_scored_candidates",
    "dedupe_candidate_texts",
]
