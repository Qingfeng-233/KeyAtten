from __future__ import annotations

import math

import numpy as np

from ..candidates.word import PUNCT_RE


ATTENTION_METHODS = ("cls_attn", "received_attn", "samrank", "fusion_attn", "voted_attn", "excess_attn")
DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX = "核心关键词、关键实体、主题："


def normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _is_text_piece(piece: str) -> bool:
    stripped = piece.strip()
    if not stripped:
        return False
    return PUNCT_RE.fullmatch(stripped) is None


__all__ = [
    "ATTENTION_METHODS",
    "DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX",
    "normalize_array",
]
