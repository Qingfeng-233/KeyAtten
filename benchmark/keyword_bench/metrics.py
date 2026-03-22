from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

VENDOR_NLP_DIR = Path(__file__).resolve().parent.parent / ".vendor_nlp"
if VENDOR_NLP_DIR.exists() and str(VENDOR_NLP_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_NLP_DIR))

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


def _prf_at_k(predictions: List[str], golds: List[str], k: int) -> tuple[float, float, float]:
    gold_set = {normalize_phrase(item) for item in golds if normalize_phrase(item)}
    seen = set()
    hits = 0
    for item in predictions[:k]:
        normalized = normalize_phrase(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in gold_set:
            hits += 1
    precision = hits / max(min(k, len(predictions)), 1)
    recall = hits / max(len(gold_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_predictions(
    all_predictions: Iterable[List[str]],
    all_golds: Iterable[List[str]],
    ks: Iterable[int] = (5, 10),
) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {}
    for predictions, golds in zip(all_predictions, all_golds):
        for k in ks:
            precision, recall, f1 = _prf_at_k(predictions, golds, k)
            metrics.setdefault(f"p@{k}", []).append(precision)
            metrics.setdefault(f"r@{k}", []).append(recall)
            metrics.setdefault(f"f1@{k}", []).append(f1)

    return {name: float(np.mean(values)) for name, values in metrics.items()}
