from __future__ import annotations

from ..candidates.word import PUNCT_RE
from ..utils import normalize_array


ATTENTION_METHODS = ("cls_attn", "received_attn", "samrank", "fusion_attn", "voted_attn", "excess_attn")
DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX = "核心关键词、关键实体、主题："


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
