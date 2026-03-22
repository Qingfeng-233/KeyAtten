from __future__ import annotations

import re

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None


_SPACE_RE = re.compile(r"\s+")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PORTER = PorterStemmer() if PorterStemmer is not None else None


def normalize_phrase(text: str) -> str:
    text = text.strip().lower().replace("_", " ")
    if _CJK_RE.search(text):
        return _SPACE_RE.sub("", text)

    tokens = _LATIN_TOKEN_RE.findall(text)
    if not tokens:
        return _SPACE_RE.sub("", text)

    if _PORTER is not None:
        tokens = [_PORTER.stem(token) if token.isalpha() else token for token in tokens]
    return "".join(tokens)


__all__ = ["normalize_phrase"]
