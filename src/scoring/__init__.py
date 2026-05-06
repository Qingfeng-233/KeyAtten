from __future__ import annotations

from .utils import ATTENTION_METHODS, DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX, normalize_array

__all__ = [
    "ATTENTION_METHODS",
    "DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX",
    "attention_token_scores",
    "attention_word_scores",
    "attention_word_scores_with_raw",
    "batched_attention_word_scores",
    "build_model_bundle",
    "normalize_array",
    "rescore_with_new_words",
]


def __getattr__(name: str):
    if name == "build_model_bundle":
        from .model_bundle import build_model_bundle

        return build_model_bundle
    if name in {
        "attention_token_scores",
        "attention_word_scores",
        "attention_word_scores_with_raw",
        "batched_attention_word_scores",
        "rescore_with_new_words",
    }:
        from . import token_scoring

        return getattr(token_scoring, name)
    raise AttributeError(f"module 'keyatten.scoring' has no attribute '{name}'")
