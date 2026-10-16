from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


WEAK_PREFIXES = (
    "聊聊",
    "分为",
    "开始",
    "进行",
    "出现",
    "产生",
    "成为",
)


@dataclass(slots=True)
class CandidateFeatureRow:
    candidate_text: str
    qk_score: float
    received_attn_score: float
    fusion_attn_score: float
    char_len: int
    token_len: int
    first_occurrence_ratio: float
    occurrence_count: int
    begins_with_weak_prefix: bool
    is_contained_by_other_candidate: bool
    contains_other_candidate: bool


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(value / denominator)


def build_feature_row(
    *,
    document_text: str,
    candidate_text: str,
    qk_score: float,
    received_attn_score: float,
    fusion_attn_score: float,
    token_len: int,
    candidate_texts: Sequence[str],
) -> CandidateFeatureRow:
    first_index = document_text.find(candidate_text)
    occurrence_count = document_text.count(candidate_text) if candidate_text else 0
    begins_with_weak_prefix = any(candidate_text.startswith(prefix) for prefix in WEAK_PREFIXES)
    is_contained = any(
        candidate_text != other and candidate_text in other
        for other in candidate_texts
    )
    contains_other = any(
        candidate_text != other and other in candidate_text
        for other in candidate_texts
    )
    return CandidateFeatureRow(
        candidate_text=candidate_text,
        qk_score=float(qk_score),
        received_attn_score=float(received_attn_score),
        fusion_attn_score=float(fusion_attn_score),
        char_len=len(candidate_text),
        token_len=int(token_len),
        first_occurrence_ratio=_safe_ratio(first_index if first_index >= 0 else 0, len(document_text)),
        occurrence_count=occurrence_count,
        begins_with_weak_prefix=begins_with_weak_prefix,
        is_contained_by_other_candidate=is_contained,
        contains_other_candidate=contains_other,
    )


def build_feature_vector(row: CandidateFeatureRow | Mapping[str, object]) -> np.ndarray:
    if isinstance(row, Mapping):
        row = CandidateFeatureRow(
            candidate_text=str(row["candidate_text"]),
            qk_score=float(row["qk_score"]),
            received_attn_score=float(row["received_attn_score"]),
            fusion_attn_score=float(row["fusion_attn_score"]),
            char_len=int(row["char_len"]),
            token_len=int(row["token_len"]),
            first_occurrence_ratio=float(row["first_occurrence_ratio"]),
            occurrence_count=int(row["occurrence_count"]),
            begins_with_weak_prefix=bool(row["begins_with_weak_prefix"]),
            is_contained_by_other_candidate=bool(row["is_contained_by_other_candidate"]),
            contains_other_candidate=bool(row["contains_other_candidate"]),
        )
    return np.asarray(
        [
            row.qk_score,
            row.received_attn_score,
            row.fusion_attn_score,
            float(row.char_len),
            float(row.token_len),
            row.first_occurrence_ratio,
            float(row.occurrence_count),
            1.0 if row.begins_with_weak_prefix else 0.0,
            1.0 if row.is_contained_by_other_candidate else 0.0,
            1.0 if row.contains_other_candidate else 0.0,
        ],
        dtype=np.float32,
    )
