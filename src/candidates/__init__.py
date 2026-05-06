from __future__ import annotations

from .consolidation import ScoredCandidate, consolidate_scored_candidates
from .word import *
from .word import __all__ as _word_all

__all__ = [
    *_word_all,
    "BIOBoundaryHead",
    "ScoredCandidate",
    "bio_tags_to_spans",
    "build_bio_positive_phrases",
    "combine_word_scores",
    "consolidate_scored_candidates",
    "extract_keywords_relaxed",
    "extract_keywords_relaxed_windowed",
    "find_candidate_occurrences",
    "inverse_document_frequency",
    "mine_recall_oriented_phrases",
    "spans_to_text",
    "token_counter",
    "word_scores_from_token_values",
]


def __getattr__(name: str):
    if name in {
        "BIOBoundaryHead",
        "bio_tags_to_spans",
        "extract_keywords_relaxed",
        "extract_keywords_relaxed_windowed",
        "spans_to_text",
    }:
        from . import bio_head

        return getattr(bio_head, name)
    if name in {
        "build_bio_positive_phrases",
        "find_candidate_occurrences",
        "mine_recall_oriented_phrases",
    }:
        from . import bio_mining

        return getattr(bio_mining, name)
    if name in {
        "combine_word_scores",
        "inverse_document_frequency",
        "token_counter",
        "word_scores_from_token_values",
    }:
        from . import fusion

        return getattr(fusion, name)
    raise AttributeError(f"module 'keyatten.candidates' has no attribute '{name}'")
