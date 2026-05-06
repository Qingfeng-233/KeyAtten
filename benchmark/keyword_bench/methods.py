from __future__ import annotations

import math
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Sequence

import jieba.posseg as pseg
import networkx as nx
import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModel, AutoTokenizer

from .metrics import normalize_phrase

VENDOR_DIR = Path(__file__).resolve().parent.parent / ".vendor"


VALID_POS_PREFIXES = ("n", "nz", "eng", "v", "vn", "a", "m", "t")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX = "找关键词"
CLS_HEAD_SELECTION_METHOD_NAMES = {
    "topk": "cls_attn_headtopk",
    "softmax": "cls_attn_headsoftmax",
    "somp": "cls_attn_headsomp",
}
RECEIVED_HEAD_SELECTION_METHOD_NAMES = {
    "topk": "received_attn_headtopk",
    "softmax": "received_attn_headsoftmax",
    "somp": "received_attn_headsomp",
}


@dataclass(slots=True)
class Candidate:
    text: str
    word_start: int
    word_end: int


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
    if lowered in ENGLISH_STOP_WORDS:
        return False
    return bool(re.search(r"[a-z]", lowered))


def segment_text(text: str, language: str = "zh") -> tuple[List[str], List[str]]:
    if language.startswith("en"):
        words = EN_TOKEN_RE.findall(text)
        return words, ["eng"] * len(words)
    words: List[str] = []
    pos_tags: List[str] = []
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
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    joiner = "" if language.startswith("zh") else " "
    for start in range(len(words)):
        is_valid = is_valid_token(words[start], pos_tags[start]) if language.startswith("zh") else is_valid_english_token(words[start])
        if not is_valid:
            continue
        for end in range(start + 1, min(len(words), start + max_ngram) + 1):
            if language.startswith("zh"):
                if not all(is_valid_token(words[index], pos_tags[index]) for index in range(start, end)):
                    break
            else:
                if not all(is_valid_english_token(words[index]) for index in range(start, end)):
                    break
                if words[start].lower() in ENGLISH_STOP_WORDS or words[end - 1].lower() in ENGLISH_STOP_WORDS:
                    continue
            phrase = joiner.join(words[start:end]).strip()
            if language.startswith("en"):
                phrase = phrase.strip("- ")
            normalized = normalize_phrase(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(Candidate(text=phrase, word_start=start, word_end=end))
    return candidates


def _is_attention_candidate_token(word: str, language: str = "zh") -> bool:
    stripped = word.strip()
    if not stripped:
        return False
    if language.startswith("zh"):
        return not bool(PUNCT_RE.match(stripped))
    return bool(re.search(r"[A-Za-z0-9]", stripped))


def build_attention_gated_candidates(
    words: Sequence[str],
    gate_scores: Sequence[float],
    language: str = "zh",
    max_ngram: int = 4,
    threshold_mode: str = "mean",
    threshold_percentile: float = 75.0,
) -> List[Candidate]:
    if not words:
        return []

    gate_array = np.asarray(gate_scores, dtype=np.float32)
    if gate_array.size != len(words):
        raise ValueError("gate_scores must have the same length as words.")

    valid_token_mask = np.asarray([_is_attention_candidate_token(word, language=language) for word in words], dtype=bool)
    if not bool(valid_token_mask.any()):
        return []

    valid_scores = gate_array[valid_token_mask]
    if threshold_mode == "mean":
        threshold = float(valid_scores.mean())
    elif threshold_mode == "percentile":
        threshold = float(np.percentile(valid_scores, threshold_percentile))
    else:
        raise ValueError(f"Unsupported attention candidate threshold mode: {threshold_mode}")

    anchor_mask = valid_token_mask & (gate_array >= threshold)
    if not bool(anchor_mask.any()):
        valid_indices = np.flatnonzero(valid_token_mask)
        anchor_mask[valid_indices[int(np.argmax(gate_array[valid_indices]))]] = True

    candidates: List[Candidate] = []
    seen = set()
    joiner = "" if language.startswith("zh") else " "
    for start in range(len(words)):
        if not valid_token_mask[start]:
            continue
        has_anchor = False
        for end in range(start + 1, min(len(words), start + max_ngram) + 1):
            token_index = end - 1
            if not valid_token_mask[token_index]:
                break
            has_anchor = has_anchor or bool(anchor_mask[token_index])
            if not has_anchor:
                continue
            phrase = joiner.join(words[start:end]).strip()
            if language.startswith("en"):
                phrase = phrase.strip("- ")
            normalized = normalize_phrase(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(Candidate(text=phrase, word_start=start, word_end=end))
    return candidates


def _candidate_score(
    candidate: Candidate,
    word_scores: Sequence[float],
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
) -> float | None:
    span = np.asarray(word_scores[candidate.word_start : candidate.word_end], dtype=np.float32)
    if span.size == 0:
        return None

    if aggregation_mode == "mean":
        score = float(span.mean())
    elif aggregation_mode == "max":
        score = float(span.max())
    elif aggregation_mode == "top2_mean":
        top_values = np.sort(span)[-2:]
        score = float(top_values.mean())
    elif aggregation_mode == "sum_sqrt_len":
        score = float(span.sum() / math.sqrt(max(span.size, 1)))
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

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


def aggregate_candidate_scores(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
) -> List[str]:
    candidate_scores = candidate_score_values(
        candidates,
        word_scores,
        token_counts=token_counts,
        words=words,
        aggregation_mode=aggregation_mode,
        repeat_boost=repeat_boost,
    )
    return rank_candidates_from_scores(candidates, candidate_scores, top_k=len(candidates))


def candidate_score_values(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    token_counts: Dict[str, float] | None = None,
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

    # Fast path for the default benchmark setting used by the long-document runs.
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
        if score is None:
            continue
        scores[index] = float(score)
    return scores


def rank_candidates_from_scores(
    candidates: Sequence[Candidate],
    candidate_scores: Sequence[float],
    top_k: int = 30,
    dedup_nested: bool = False,
) -> List[str]:
    if not candidates:
        return []

    scores_array = np.asarray(candidate_scores, dtype=np.float32)
    finite_indices = np.flatnonzero(np.isfinite(scores_array))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(scores_array[finite_indices])[::-1]]

    if not dedup_nested:
        limit = min(top_k, sorted_indices.size)
        return [candidates[int(index)].text for index in sorted_indices[:limit]]

    # With nested deduplication
    ranked: List[str] = []
    selected_normalized: List[str] = []

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
        if len(ranked) >= top_k:
            break
    return ranked


def candidate_rank_from_word_scores(
    candidates: Sequence[Candidate],
    word_scores: Sequence[float],
    top_k: int = 30,
    token_counts: Dict[str, float] | None = None,
    words: Sequence[str] | None = None,
    aggregation_mode: str = "mean",
    repeat_boost: float = 0.0,
    candidate_starts: np.ndarray | None = None,
    candidate_ends: np.ndarray | None = None,
) -> List[str]:
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
    return rank_candidates_from_scores(candidates, candidate_scores, top_k=top_k)


def textrank_keywords(
    words: Sequence[str],
    pos_tags: Sequence[str],
    candidates: Sequence[Candidate],
    language: str = "zh",
    window: int = 4,
) -> List[str]:
    word_scores = textrank_word_scores(words, pos_tags, language=language, window=window)
    return aggregate_candidate_scores(candidates, word_scores)


def textrank_word_scores(
    words: Sequence[str],
    pos_tags: Sequence[str],
    language: str = "zh",
    window: int = 4,
) -> np.ndarray:
    graph = nx.Graph()
    if language.startswith("zh"):
        kept_indices = [index for index, (word, pos) in enumerate(zip(words, pos_tags)) if is_valid_token(word, pos)]
    else:
        kept_indices = [index for index, word in enumerate(words) if is_valid_english_token(word)]
    for left_pos, left_idx in enumerate(kept_indices):
        graph.add_node(left_idx)
        for right_idx in kept_indices[left_pos + 1 : left_pos + window]:
            graph.add_edge(left_idx, right_idx, weight=graph.get_edge_data(left_idx, right_idx, {}).get("weight", 0.0) + 1.0)
    scores = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    return np.asarray([scores.get(index, 0.0) for index in range(len(words))], dtype=np.float32)


def yake_keywords(text: str, top_k: int = 30, max_ngram: int = 4, language: str = "zh") -> List[str]:
    if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        import yake
    except ImportError as exc:
        raise RuntimeError("yake is not installed. Install it into a local target directory before running YAKE.")
    if yake is None:
        raise RuntimeError("yake is not installed. Install it into a local target directory before running YAKE.") from exc
    extractor = yake.KeywordExtractor(lan="zh" if language.startswith("zh") else "en", n=max_ngram, top=top_k)
    ranked = []
    seen = set()
    for phrase, _ in extractor.extract_keywords(text):
        normalized = normalize_phrase(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ranked.append(phrase)
    return ranked


def token_counter(words: Sequence[str], pos_tags: Sequence[str], language: str = "zh") -> Counter[str]:
    counts: Counter[str] = Counter()
    for word, pos_tag in zip(words, pos_tags):
        is_valid = is_valid_token(word, pos_tag) if language.startswith("zh") else is_valid_english_token(word)
        if not is_valid:
            continue
        normalized = normalize_phrase(word)
        if normalized:
            counts[normalized] += 1
    return counts


def inverse_document_frequency(token_sets: Iterable[Iterable[str]]) -> Dict[str, float]:
    token_sets_list = [set(tokens) for tokens in token_sets]
    doc_count = max(len(token_sets_list), 1)
    document_freq = Counter()
    for tokens in token_sets_list:
        document_freq.update(tokens)
    return {
        token: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
        for token, freq in document_freq.items()
    }


def word_scores_from_token_values(
    words: Sequence[str],
    pos_tags: Sequence[str],
    token_values: Dict[str, float],
    language: str = "zh",
) -> np.ndarray:
    scores = np.zeros(len(words), dtype=np.float32)
    for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        is_valid = is_valid_token(word, pos_tag) if language.startswith("zh") else is_valid_english_token(word)
        if not is_valid:
            continue
        scores[index] = float(token_values.get(normalize_phrase(word), 0.0))
    return scores


def combine_word_scores(
    primary_scores: Sequence[float],
    secondary_scores: Sequence[float],
    mode: str = "product",
) -> np.ndarray:
    primary = np.asarray(primary_scores, dtype=np.float32)
    secondary = np.asarray(secondary_scores, dtype=np.float32)
    if primary.shape != secondary.shape:
        raise ValueError("primary_scores and secondary_scores must have the same shape.")
    primary_norm = _normalize_array(primary)
    secondary_norm = _normalize_array(secondary)
    if mode == "product":
        return primary_norm * secondary_norm
    if mode == "sum":
        return 0.5 * primary_norm + 0.5 * secondary_norm
    raise ValueError(f"Unsupported combine mode: {mode}")


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum().item() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


def _resolve_embedding_pooling(model_name: str) -> str:
    if "qwen3-embedding" in model_name.lower():
        return "last_token"
    return "mean"


def _detect_is_causal(config) -> bool:
    if getattr(config, "is_decoder", False):
        return True
    architectures = getattr(config, "architectures", None) or []
    return any("CausalLM" in arch for arch in architectures)


def _tokenize_instruction_prefix(
    instruction_prefix: str | None,
    language: str,
) -> tuple[list[str], list[str]]:
    prefix = (instruction_prefix or "").strip()
    if not prefix:
        return [], []
    return segment_text(prefix, language=language)


def _build_content_mask(
    word_ids: Sequence[int | None],
    words: Sequence[str],
    pos_tags: Sequence[str],
    language: str,
) -> np.ndarray:
    mask = np.zeros(len(word_ids), dtype=bool)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= len(words):
            continue
        if language.startswith("en"):
            if is_valid_english_token(words[word_id]):
                mask[token_index] = True
        else:
            if is_valid_token(words[word_id], pos_tags[word_id]):
                mask[token_index] = True
    return mask


def _resolve_content_mask(
    word_ids: Sequence[int | None],
    content_mask: np.ndarray | None,
) -> np.ndarray:
    if content_mask is not None:
        return np.asarray(content_mask, dtype=bool)
    return np.asarray([word_id is not None for word_id in word_ids], dtype=bool)


def _compute_excess_token_scores(
    attention_map: np.ndarray,
    *,
    is_causal: bool,
    content_mask: np.ndarray,
) -> np.ndarray:
    n_tokens = attention_map.shape[0]
    excess_scores = np.zeros(n_tokens, dtype=np.float32)

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            score_view = excess_scores[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            score_view = excess_scores

        visible_content_count = int(row_content_mask.sum())
        if visible_content_count <= 1:
            continue

        row_content = row_visible[row_content_mask].astype(np.float32, copy=False)
        content_mass = float(row_content.sum())
        if content_mass <= 1e-10:
            continue

        p = row_content / content_mass
        log_visible_content = math.log(visible_content_count)
        entropy = float(-(p * np.log(np.clip(p, 1e-10, None))).sum())
        certainty = max(0.0, 1.0 - (entropy / log_visible_content)) if log_visible_content > 1e-10 else 0.0

        baseline = 1.0 / visible_content_count
        contribution = (content_mass * certainty) * (p - baseline)
        score_view[row_content_mask] += contribution.astype(np.float32, copy=False)

    return excess_scores


def _compute_sink_reallocated_received_scores(
    attention_map: np.ndarray,
    *,
    is_causal: bool,
    content_mask: np.ndarray,
) -> np.ndarray:
    n_tokens = attention_map.shape[0]
    reallocated_scores = np.zeros(n_tokens, dtype=np.float32)

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            score_view = reallocated_scores[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            score_view = reallocated_scores

        row_content = row_visible[row_content_mask].astype(np.float32, copy=False)
        content_mass = float(row_content.sum())
        if content_mass <= 1e-10:
            continue
        score_view[row_content_mask] += row_content / content_mass

    return reallocated_scores


def _compute_sink_reallocated_attention_map(
    attention_map: np.ndarray,
    *,
    is_causal: bool,
    content_mask: np.ndarray,
) -> np.ndarray:
    reallocated = np.zeros_like(attention_map, dtype=np.float32)
    n_tokens = attention_map.shape[0]

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            row_view = reallocated[row_index, : row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            row_view = reallocated[row_index]

        row_content = row_visible[row_content_mask].astype(np.float32, copy=False)
        content_mass = float(row_content.sum())
        if content_mass <= 1e-10:
            continue
        row_view[row_content_mask] = row_content / content_mass

    return reallocated


def _compute_debiased_received_scores(
    received_scores: np.ndarray,
    null_received_baseline: np.ndarray | None,
    *,
    gamma: float,
) -> np.ndarray | None:
    if null_received_baseline is None:
        return None
    baseline = np.asarray(null_received_baseline, dtype=np.float32)
    if baseline.shape != received_scores.shape:
        raise ValueError("null_received_baseline must have the same shape as received_scores.")
    return np.maximum(received_scores - (float(gamma) * baseline), 0.0).astype(np.float32, copy=False)


def _compute_excess_token_scores_torch(
    attention_map: torch.Tensor,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
) -> torch.Tensor:
    n_tokens = attention_map.shape[0]
    excess_scores = torch.zeros(n_tokens, dtype=torch.float32, device=attention_map.device)

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            score_view = excess_scores[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            score_view = excess_scores

        visible_content_count = int(row_content_mask.sum().item())
        if visible_content_count <= 1:
            continue

        row_content = row_visible[row_content_mask].float()
        content_mass = row_content.sum()
        if not bool((content_mass > 1e-10).item()):
            continue

        p = row_content / content_mass
        log_visible_content = math.log(visible_content_count)
        entropy = -(p * p.clamp_min(1e-10).log()).sum()
        certainty = max(0.0, 1.0 - (float(entropy.item()) / log_visible_content)) if log_visible_content > 1e-10 else 0.0

        baseline = 1.0 / visible_content_count
        contribution = (content_mass * certainty) * (p - baseline)
        score_view[row_content_mask] += contribution

    return excess_scores


def _compute_sink_reallocated_received_scores_torch(
    attention_map: torch.Tensor,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
) -> torch.Tensor:
    n_tokens = attention_map.shape[0]
    reallocated_scores = torch.zeros(n_tokens, dtype=torch.float32, device=attention_map.device)

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            score_view = reallocated_scores[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            score_view = reallocated_scores

        row_content = row_visible[row_content_mask].float()
        content_mass = row_content.sum()
        if not bool((content_mass > 1e-10).item()):
            continue
        score_view[row_content_mask] += row_content / content_mass

    return reallocated_scores


def _compute_sink_reallocated_attention_map_torch(
    attention_map: torch.Tensor,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
) -> torch.Tensor:
    reallocated = torch.zeros_like(attention_map, dtype=torch.float32)
    n_tokens = attention_map.shape[0]

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
            row_view = reallocated[row_index, : row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask
            row_view = reallocated[row_index]

        row_content = row_visible[row_content_mask].float()
        content_mass = row_content.sum()
        if not bool((content_mass > 1e-10).item()):
            continue
        row_view[row_content_mask] = row_content / content_mass

    return reallocated


def _compute_debiased_received_scores_torch(
    received_scores: torch.Tensor,
    null_received_baseline: torch.Tensor | None,
    *,
    gamma: float,
) -> torch.Tensor | None:
    if null_received_baseline is None:
        return None
    baseline = null_received_baseline.float()
    if baseline.shape != received_scores.shape:
        raise ValueError("null_received_baseline must have the same shape as received_scores.")
    return torch.clamp(received_scores - (float(gamma) * baseline), min=0.0)


def _build_bidirectional_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = inputs_embeds.shape[:2]
    full_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    if attention_mask is None:
        return full_mask
    key_mask = attention_mask[:, None, None, :].to(device=inputs_embeds.device, dtype=torch.bool)
    min_value = torch.finfo(inputs_embeds.dtype).min
    return full_mask.masked_fill(~key_mask, min_value)


def _patch_qwen3_forward_bidirectional(model: torch.nn.Module):
    if getattr(model, "_benchmark_original_forward", None) is not None:
        return lambda: None
    if getattr(model.config, "model_type", "") != "qwen3":
        raise ValueError("true_bidirectional_attention currently only supports Qwen3 models in this benchmark.")

    original_forward = model.forward

    def patched_forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        if attention_mask is not None and not isinstance(attention_mask, dict):
            local_inputs_embeds = inputs_embeds
            if local_inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("true_bidirectional_attention requires input_ids or inputs_embeds.")
                local_inputs_embeds = self.embed_tokens(input_ids)
            full_mask = _build_bidirectional_attention_mask(attention_mask, inputs_embeds=local_inputs_embeds)
            attention_mask = {"full_attention": full_mask}
            if getattr(self, "has_sliding_layers", False):
                attention_mask["sliding_attention"] = full_mask

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    model._benchmark_original_forward = original_forward
    model.forward = MethodType(patched_forward, model)

    def restore():
        original = getattr(model, "_benchmark_original_forward", None)
        if original is None:
            return
        model.forward = original
        delattr(model, "_benchmark_original_forward")

    return restore


def _run_attention_forward(
    model: torch.nn.Module,
    encoded_inputs: dict[str, torch.Tensor],
    *,
    true_bidirectional_attention: bool,
):
    restore_forward = None
    try:
        if true_bidirectional_attention:
            restore_forward = _patch_qwen3_forward_bidirectional(model)
        with torch.no_grad():
            return model(**encoded_inputs, output_attentions=True)
    finally:
        if restore_forward is not None:
            restore_forward()


def _build_noise_token_id_bank(tokenizer) -> np.ndarray:
    vocab_size = int(len(tokenizer))
    special_ids = {int(token_id) for token_id in (tokenizer.all_special_ids or []) if token_id is not None and token_id >= 0}
    token_ids = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    if not token_ids:
        token_ids = list(range(vocab_size))
    return np.asarray(token_ids, dtype=np.int64)


def _build_null_input_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_ids_per_item: Sequence[Sequence[int | None]],
    *,
    prefix_word_count: int,
    noise_token_ids: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    if noise_token_ids.size == 0:
        return input_ids

    noisy_input_ids = input_ids.clone()
    for item_index, word_ids in enumerate(word_ids_per_item):
        valid_token_count = int(attention_mask[item_index].sum().item())
        replace_positions = [
            token_index
            for token_index, word_id in enumerate(word_ids[:valid_token_count])
            if word_id is not None and word_id >= prefix_word_count
        ]
        if not replace_positions:
            continue
        sampled_ids = rng.choice(noise_token_ids, size=len(replace_positions), replace=True)
        noisy_input_ids[item_index, replace_positions] = torch.as_tensor(
            sampled_ids,
            dtype=noisy_input_ids.dtype,
            device=noisy_input_ids.device,
        )
    return noisy_input_ids


def build_model_bundle(
    model_name: str,
    device: str,
    is_causal_override: bool | None = None,
    true_bidirectional_attention: bool = False,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    detected_is_causal = _detect_is_causal(model.config)
    is_causal = detected_is_causal if is_causal_override is None else is_causal_override
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "max_length": int(getattr(model.config, "max_position_embeddings", 512)),
        "is_causal": is_causal,
        "detected_is_causal": detected_is_causal,
        "true_bidirectional_attention": bool(true_bidirectional_attention),
        "embedding_pooling": _resolve_embedding_pooling(model_name),
        "embedding_cache": OrderedDict(),
        "embedding_cache_max_entries": 20000,
        "noise_token_ids": _build_noise_token_id_bank(tokenizer),
    }


def embed_texts(
    model_bundle: dict,
    texts: Sequence[str],
    batch_size: int = 16,
    max_length: int = 512,
    progress_label: str | None = None,
    log_every_batches: int = 0,
) -> np.ndarray:
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    embedding_pooling = model_bundle.get("embedding_pooling", "mean")
    outputs: List[np.ndarray] = []
    total_batches = max(math.ceil(len(texts) / batch_size), 1)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded, output_attentions=False).last_hidden_state
            if embedding_pooling == "last_token":
                pooled = _last_token_pool(hidden, encoded["attention_mask"])
            else:
                pooled = _mean_pool(hidden, encoded["attention_mask"])
            outputs.append(torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy())
            if progress_label and (
                batch_index == 1
                or batch_index == total_batches
                or (log_every_batches > 0 and batch_index % log_every_batches == 0)
            ):
                print(f"[{progress_label}] batch {batch_index}/{total_batches}")
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def embed_texts_cached(
    model_bundle: dict,
    texts: Sequence[str],
    batch_size: int = 16,
    max_length: int = 32,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    cache: OrderedDict[str, np.ndarray] | None = model_bundle.get("embedding_cache")
    max_entries = int(model_bundle.get("embedding_cache_max_entries", 0) or 0)
    if cache is None or max_entries <= 0:
        return embed_texts(model_bundle, texts, batch_size=batch_size, max_length=max_length)

    outputs: List[np.ndarray | None] = [None] * len(texts)
    missing_indices: List[int] = []
    missing_texts: List[str] = []
    for index, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            cache.move_to_end(text)
            outputs[index] = cached
            continue
        missing_indices.append(index)
        missing_texts.append(text)

    if missing_texts:
        embedded = embed_texts(model_bundle, missing_texts, batch_size=batch_size, max_length=max_length)
        for index, text, vector in zip(missing_indices, missing_texts, embedded, strict=False):
            outputs[index] = vector
            cache[text] = vector
            cache.move_to_end(text)
            while len(cache) > max_entries:
                cache.popitem(last=False)

    return np.stack([np.asarray(vector, dtype=np.float32) for vector in outputs], axis=0)


def keybert_keywords_from_doc_embedding(
    candidates: Sequence[Candidate],
    doc_embedding: np.ndarray,
    model_bundle: dict,
    top_k: int = 30,
    batch_size: int = 16,
) -> List[str]:
    candidate_scores = keybert_candidate_scores_from_doc_embedding(
        candidates,
        doc_embedding,
        model_bundle,
        batch_size=batch_size,
    )
    return rank_candidates_from_scores(candidates, candidate_scores, top_k=top_k)


def keybert_candidate_scores_from_doc_embedding(
    candidates: Sequence[Candidate],
    doc_embedding: np.ndarray,
    model_bundle: dict,
    batch_size: int = 16,
    candidate_max_length: int = 32,
) -> np.ndarray:
    candidate_texts = [candidate.text for candidate in candidates]
    if not candidate_texts:
        return np.zeros(0, dtype=np.float32)
    candidate_embeddings = embed_texts_cached(
        model_bundle,
        candidate_texts,
        batch_size=max(batch_size, 64),
        max_length=candidate_max_length,
    )
    return np.asarray(candidate_embeddings @ doc_embedding, dtype=np.float32)


def keybert_keywords(
    text: str,
    candidates: Sequence[Candidate],
    model_bundle: dict,
    top_k: int = 30,
    batch_size: int = 16,
) -> List[str]:
    if not candidates:
        return []
    doc_embeddings = embed_texts(model_bundle, [text], batch_size=1)
    return keybert_keywords_from_doc_embedding(
        candidates,
        doc_embeddings[0],
        model_bundle,
        top_k=top_k,
        batch_size=batch_size,
    )


def keybert_word_scores(
    text: str,
    words: Sequence[str],
    pos_tags: Sequence[str],
    model_bundle: dict,
    language: str = "zh",
) -> np.ndarray:
    valid_indices = []
    valid_words = []
    for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        is_valid = is_valid_token(word, pos_tag) if language.startswith("zh") else is_valid_english_token(word)
        if not is_valid:
            continue
        valid_indices.append(index)
        valid_words.append(word)

    scores = np.zeros(len(words), dtype=np.float32)
    if not valid_words:
        return scores

    embeddings = embed_texts(model_bundle, [text] + valid_words)
    doc_embedding = embeddings[0]
    token_embeddings = embeddings[1:]
    similarities = token_embeddings @ doc_embedding
    for index, similarity in zip(valid_indices, similarities.tolist()):
        scores[index] = float(similarity)
    return scores


def _normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _aggregate_subwords_to_words(word_ids: List[int | None], token_scores: np.ndarray, word_count: int) -> np.ndarray:
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_scores[token_index])
        counts[word_id] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def _word_ids_to_tensor(word_ids: List[int | None], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [-1 if word_id is None else int(word_id) for word_id in word_ids],
        dtype=torch.long,
        device=device,
    )


def _aggregate_subwords_to_words_torch(
    word_ids: torch.Tensor,
    token_scores: torch.Tensor,
    word_count: int,
) -> np.ndarray:
    if word_count <= 0:
        return np.zeros(0, dtype=np.float32)

    valid_mask = (word_ids >= 0) & (word_ids < word_count)
    if not bool(valid_mask.any().item()):
        return np.zeros(word_count, dtype=np.float32)

    valid_word_ids = word_ids[valid_mask]
    valid_scores = token_scores[valid_mask].float()
    sums = torch.zeros(word_count, dtype=torch.float32, device=token_scores.device)
    counts = torch.zeros(word_count, dtype=torch.float32, device=token_scores.device)
    sums.scatter_add_(0, valid_word_ids, valid_scores)
    counts.scatter_add_(0, valid_word_ids, torch.ones_like(valid_scores))
    return (sums / counts.clamp_min(1.0)).cpu().numpy()


def _resolve_transformer_layers(model: torch.nn.Module) -> list[torch.nn.Module] | None:
    """Resolve transformer layers from various model architectures."""
    direct_model = getattr(model, "model", None)
    if direct_model is not None and hasattr(direct_model, "layers"):
        return list(direct_model.layers)

    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return list(encoder.layer)

    for backbone_name in ("bert", "roberta", "deberta", "deberta_v2", "mpnet", "electra", "xlm_roberta"):
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        encoder = getattr(backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return list(encoder.layer)

    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "layer"):
        return list(transformer.layer)

    return None


def _resolve_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module] | None:
    """Resolve attention modules from transformer layers."""
    layers = _resolve_transformer_layers(model)
    if not layers:
        return None

    modules: list[torch.nn.Module] = []
    for layer in layers:
        attention_module = getattr(layer, "self_attn", None)
        if attention_module is None:
            attention = getattr(layer, "attention", None)
            attention_module = getattr(attention, "self", None) if attention is not None else None
        if attention_module is None:
            attention_module = getattr(layer, "attn", None)
        if attention_module is None:
            return None
        modules.append(attention_module)
    return modules


def _extract_attention_weights_from_module_output(outputs) -> torch.Tensor | None:
    """Extract attention weights from module forward output."""
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        return None
    candidate = outputs[1]
    if torch.is_tensor(candidate):
        return candidate
    return None


def _replace_attention_weights_with_none(outputs):
    """Replace attention weights with None to save memory during streaming."""
    if not isinstance(outputs, tuple):
        return outputs
    if len(outputs) < 2 or not torch.is_tensor(outputs[1]):
        return outputs
    return (outputs[0], None, *outputs[2:])


def _resolve_layer_index(layer_index: int, layer_count: int) -> int:
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"Layer index {layer_index} is out of range for {layer_count} attention layers.")
    return resolved


def _scores_from_attention_map(
    word_ids: List[int | None],
    attention_map: np.ndarray,
    word_count: int,
    *,
    is_causal: bool = False,
    content_mask: np.ndarray | None = None,
    null_received_baseline: np.ndarray | None = None,
    null_debias_gamma: float = 1.0,
) -> Dict[str, np.ndarray]:
    resolved_content_mask = _resolve_content_mask(word_ids, content_mask)
    if is_causal and content_mask is not None:
        cls_scores = attention_map[-1].copy()
        cls_scores[~resolved_content_mask] = 0.0
        row_sum = cls_scores.sum()
        if row_sum > 1e-10:
            cls_scores /= row_sum
    else:
        cls_scores = attention_map[0]

    received_scores = attention_map.sum(axis=0)

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = _normalize_array(cls_scores) * _normalize_array(received_scores)
    excess_scores = _compute_excess_token_scores(
        attention_map,
        is_causal=is_causal,
        content_mask=resolved_content_mask,
    )
    sink_realloc_scores = _compute_sink_reallocated_received_scores(
        attention_map,
        is_causal=is_causal,
        content_mask=resolved_content_mask,
    )
    sink_realloc_attention = _compute_sink_reallocated_attention_map(
        attention_map,
        is_causal=is_causal,
        content_mask=resolved_content_mask,
    )
    if is_causal and content_mask is not None:
        sink_realloc_cls_scores = sink_realloc_attention[-1].copy()
    else:
        sink_realloc_cls_scores = sink_realloc_attention[0]
    sink_realloc_received_scores = sink_realloc_attention.sum(axis=0)
    sink_realloc_global_scores = sink_realloc_received_scores.copy()
    sink_realloc_redistributed = sink_realloc_attention * sink_realloc_global_scores[None, :]
    sink_realloc_col_sums = sink_realloc_redistributed.sum(axis=0, keepdims=True) + 1e-10
    sink_realloc_redistributed = np.divide(
        sink_realloc_redistributed,
        sink_realloc_col_sums,
        out=np.zeros_like(sink_realloc_redistributed),
        where=sink_realloc_col_sums > 0.0,
    )
    sink_realloc_proportional_scores = sink_realloc_redistributed.sum(axis=1)
    sink_realloc_samrank_scores = sink_realloc_global_scores + sink_realloc_proportional_scores
    sink_realloc_fusion_scores = _normalize_array(sink_realloc_cls_scores) * _normalize_array(sink_realloc_received_scores)
    debiased_received_scores = _compute_debiased_received_scores(
        received_scores,
        null_received_baseline,
        gamma=null_debias_gamma,
    )

    results = {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
        "excess_attn": _aggregate_subwords_to_words(word_ids, excess_scores, word_count),
        "sink_realloc_attn": _aggregate_subwords_to_words(word_ids, sink_realloc_scores, word_count),
        "sink_realloc_cls_attn": _aggregate_subwords_to_words(word_ids, sink_realloc_cls_scores, word_count),
        "sink_realloc_samrank": _aggregate_subwords_to_words(word_ids, sink_realloc_samrank_scores, word_count),
        "sink_realloc_fusion_attn": _aggregate_subwords_to_words(word_ids, sink_realloc_fusion_scores, word_count),
    }
    if debiased_received_scores is not None:
        results["received_attn_debiased"] = _aggregate_subwords_to_words(
            word_ids,
            debiased_received_scores,
            word_count,
        )
    return results


def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if bool(torch.isclose(min_value, max_value).item()):
        return torch.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _extract_cls_scores_torch(
    attention_map: torch.Tensor,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
) -> torch.Tensor:
    if is_causal:
        cls_scores = attention_map[-1].clone()
        cls_scores = cls_scores.masked_fill(~content_mask, 0.0)
        row_sum = cls_scores.sum()
        if bool((row_sum > 1e-10).item()):
            cls_scores = cls_scores / row_sum
        return cls_scores
    return attention_map[0]


def _compute_head_utility_torch(
    attention_map: torch.Tensor,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
) -> float:
    n_tokens = attention_map.shape[0]
    utility = attention_map.new_zeros(())

    for row_index in range(n_tokens):
        if is_causal:
            row_visible = attention_map[row_index, : row_index + 1]
            row_content_mask = content_mask[: row_index + 1]
        else:
            row_visible = attention_map[row_index]
            row_content_mask = content_mask

        visible_content_count = int(row_content_mask.sum().item())
        if visible_content_count <= 1:
            continue

        row_content = row_visible[row_content_mask].float()
        content_mass = row_content.sum()
        if not bool((content_mass > 1e-10).item()):
            continue

        p = row_content / content_mass
        log_visible_content = math.log(visible_content_count)
        certainty = 0.0
        if log_visible_content > 1e-10:
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            certainty = max(0.0, 1.0 - (float(entropy.item()) / log_visible_content))
        utility = utility + content_mass * certainty

    return float((utility / max(n_tokens, 1)).item())


def _compute_somp_head_score_torch(
    attention_map: torch.Tensor,
    *,
    content_mask: torch.Tensor,
    alpha: float,
    beta: float,
    local_window: int,
) -> float:
    if attention_map.shape[0] == 0:
        return float("-inf")

    last_row = attention_map[-1].float()
    row_content_mask = content_mask
    visible_content_count = int(row_content_mask.sum().item())
    if visible_content_count <= 1:
        return float("-inf")

    row_content = last_row[row_content_mask]
    content_mass = row_content.sum()
    if not bool((content_mass > 1e-10).item()):
        return float("-inf")

    normalized_row = row_content / content_mass
    sharpness = float(normalized_row.var(unbiased=False).item())

    visible_indices = torch.nonzero(row_content_mask, as_tuple=False).flatten()
    local_count = min(max(int(local_window), 1), visible_indices.numel())
    local_indices = visible_indices[-local_count:]
    local_mass = float(last_row[local_indices].sum().item() / content_mass.item())
    return float(alpha) * sharpness - float(beta) * local_mass


def _resolve_cls_head_strategies(strategies: Sequence[str] | None) -> List[str]:
    if not strategies:
        return []
    resolved: List[str] = []
    for strategy in strategies:
        normalized = strategy.strip().lower()
        if not normalized:
            continue
        if normalized not in CLS_HEAD_SELECTION_METHOD_NAMES:
            raise ValueError(f"Unsupported cls head selection strategy: {strategy}")
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


def _compute_cls_head_selection_scores_torch(
    word_ids: torch.Tensor,
    attention_heads: torch.Tensor,
    word_count: int,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
    strategies: Sequence[str],
    top_k: int,
    temperature: float,
    somp_alpha: float,
    somp_beta: float,
    somp_local_window: int,
) -> Dict[str, np.ndarray]:
    resolved_strategies = _resolve_cls_head_strategies(strategies)
    if not resolved_strategies:
        return {}

    head_count = int(attention_heads.shape[0])
    if head_count == 0:
        return {}

    per_head_cls_scores = []
    head_utilities = []
    for head_index in range(head_count):
        head_attention = attention_heads[head_index].float()
        per_head_cls_scores.append(
            _extract_cls_scores_torch(
                head_attention,
                is_causal=is_causal,
                content_mask=content_mask,
            )
        )
        head_utilities.append(
            _compute_head_utility_torch(
                head_attention,
                is_causal=is_causal,
                content_mask=content_mask,
            )
        )

    cls_stack = torch.stack(per_head_cls_scores, dim=0)
    utility_tensor = torch.as_tensor(head_utilities, dtype=torch.float32, device=attention_heads.device)

    results: Dict[str, np.ndarray] = {}
    if "topk" in resolved_strategies:
        keep_count = min(max(int(top_k), 1), head_count)
        selected = torch.topk(utility_tensor, k=keep_count, largest=True).indices
        aggregated_scores = cls_stack.index_select(0, selected).mean(dim=0)
        results["cls_attn_headtopk"] = _aggregate_subwords_to_words_torch(word_ids, aggregated_scores, word_count)

    if "softmax" in resolved_strategies:
        logits = utility_tensor - utility_tensor.max()
        weights = torch.softmax(logits * float(temperature), dim=0)
        aggregated_scores = (cls_stack * weights[:, None]).sum(dim=0)
        results["cls_attn_headsoftmax"] = _aggregate_subwords_to_words_torch(word_ids, aggregated_scores, word_count)

    if "somp" in resolved_strategies:
        somp_scores = []
        for head_index in range(head_count):
            somp_scores.append(
                _compute_somp_head_score_torch(
                    attention_heads[head_index].float(),
                    content_mask=content_mask,
                    alpha=somp_alpha,
                    beta=somp_beta,
                    local_window=somp_local_window,
                )
            )
        somp_tensor = torch.as_tensor(somp_scores, dtype=torch.float32, device=attention_heads.device)
        somp_logits = somp_tensor - somp_tensor.max()
        somp_weights = torch.softmax(somp_logits, dim=0)
        aggregated_scores = (cls_stack * somp_weights[:, None]).sum(dim=0)
        results[CLS_HEAD_SELECTION_METHOD_NAMES["somp"]] = _aggregate_subwords_to_words_torch(
            word_ids,
            aggregated_scores,
            word_count,
        )

    return results


def _compute_received_head_selection_scores_torch(
    word_ids: torch.Tensor,
    attention_heads: torch.Tensor,
    word_count: int,
    *,
    is_causal: bool,
    content_mask: torch.Tensor,
    strategies: Sequence[str],
    top_k: int,
    temperature: float,
    somp_alpha: float,
    somp_beta: float,
    somp_local_window: int,
) -> Dict[str, np.ndarray]:
    resolved_strategies = _resolve_cls_head_strategies(strategies)
    if not resolved_strategies:
        return {}

    head_count = int(attention_heads.shape[0])
    if head_count == 0:
        return {}

    per_head_received_scores = []
    head_utilities = []
    for head_index in range(head_count):
        head_attention = attention_heads[head_index].float()
        per_head_received_scores.append(head_attention.sum(dim=0))
        head_utilities.append(
            _compute_head_utility_torch(
                head_attention,
                is_causal=is_causal,
                content_mask=content_mask,
            )
        )

    received_stack = torch.stack(per_head_received_scores, dim=0)
    utility_tensor = torch.as_tensor(head_utilities, dtype=torch.float32, device=attention_heads.device)

    results: Dict[str, np.ndarray] = {}
    if "topk" in resolved_strategies:
        keep_count = min(max(int(top_k), 1), head_count)
        selected = torch.topk(utility_tensor, k=keep_count, largest=True).indices
        aggregated_scores = received_stack.index_select(0, selected).mean(dim=0)
        results[RECEIVED_HEAD_SELECTION_METHOD_NAMES["topk"]] = _aggregate_subwords_to_words_torch(
            word_ids,
            aggregated_scores,
            word_count,
        )

    if "softmax" in resolved_strategies:
        logits = utility_tensor - utility_tensor.max()
        weights = torch.softmax(logits * float(temperature), dim=0)
        aggregated_scores = (received_stack * weights[:, None]).sum(dim=0)
        results[RECEIVED_HEAD_SELECTION_METHOD_NAMES["softmax"]] = _aggregate_subwords_to_words_torch(
            word_ids,
            aggregated_scores,
            word_count,
        )

    if "somp" in resolved_strategies:
        somp_scores = []
        for head_index in range(head_count):
            somp_scores.append(
                _compute_somp_head_score_torch(
                    attention_heads[head_index].float(),
                    content_mask=content_mask,
                    alpha=somp_alpha,
                    beta=somp_beta,
                    local_window=somp_local_window,
                )
            )
        somp_tensor = torch.as_tensor(somp_scores, dtype=torch.float32, device=attention_heads.device)
        somp_logits = somp_tensor - somp_tensor.max()
        somp_weights = torch.softmax(somp_logits, dim=0)
        aggregated_scores = (received_stack * somp_weights[:, None]).sum(dim=0)
        results[RECEIVED_HEAD_SELECTION_METHOD_NAMES["somp"]] = _aggregate_subwords_to_words_torch(
            word_ids,
            aggregated_scores,
            word_count,
        )

    return results


def _scores_from_attention_map_torch(
    word_ids: torch.Tensor,
    attention_map: torch.Tensor,
    word_count: int,
    *,
    is_causal: bool = False,
    content_mask: torch.Tensor | None = None,
    null_received_baseline: torch.Tensor | None = None,
    null_debias_gamma: float = 1.0,
) -> Dict[str, np.ndarray]:
    attention = attention_map.float()
    if content_mask is None:
        content_mask = word_ids >= 0
    if is_causal and content_mask is not None:
        cls_scores = attention[-1].clone()
        cls_scores = cls_scores.masked_fill(~content_mask, 0.0)
        row_sum = cls_scores.sum()
        if bool((row_sum > 1e-10).item()):
            cls_scores = cls_scores / row_sum
    else:
        cls_scores = attention[0]
    received_scores = attention.sum(dim=0)

    global_scores = received_scores
    redistributed = attention * global_scores.unsqueeze(0)
    redistributed = redistributed / redistributed.sum(dim=0, keepdim=True).clamp_min(1e-10)
    proportional_scores = redistributed.sum(dim=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = _normalize_tensor(cls_scores) * _normalize_tensor(received_scores)
    excess_scores = _compute_excess_token_scores_torch(
        attention,
        is_causal=is_causal,
        content_mask=content_mask,
    )
    sink_realloc_scores = _compute_sink_reallocated_received_scores_torch(
        attention,
        is_causal=is_causal,
        content_mask=content_mask,
    )
    sink_realloc_attention = _compute_sink_reallocated_attention_map_torch(
        attention,
        is_causal=is_causal,
        content_mask=content_mask,
    )
    sink_realloc_cls_scores = _extract_cls_scores_torch(
        sink_realloc_attention,
        is_causal=is_causal,
        content_mask=content_mask,
    )
    sink_realloc_received_scores = sink_realloc_attention.sum(dim=0)
    sink_realloc_global_scores = sink_realloc_received_scores
    sink_realloc_redistributed = sink_realloc_attention * sink_realloc_global_scores.unsqueeze(0)
    sink_realloc_redistributed = sink_realloc_redistributed / sink_realloc_redistributed.sum(dim=0, keepdim=True).clamp_min(1e-10)
    sink_realloc_proportional_scores = sink_realloc_redistributed.sum(dim=1)
    sink_realloc_samrank_scores = sink_realloc_global_scores + sink_realloc_proportional_scores
    sink_realloc_fusion_scores = _normalize_tensor(sink_realloc_cls_scores) * _normalize_tensor(sink_realloc_received_scores)
    debiased_received_scores = _compute_debiased_received_scores_torch(
        received_scores,
        null_received_baseline,
        gamma=null_debias_gamma,
    )

    results = {
        "cls_attn": _aggregate_subwords_to_words_torch(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words_torch(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words_torch(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words_torch(word_ids, fusion_scores, word_count),
        "excess_attn": _aggregate_subwords_to_words_torch(word_ids, excess_scores, word_count),
        "sink_realloc_attn": _aggregate_subwords_to_words_torch(word_ids, sink_realloc_scores, word_count),
        "sink_realloc_cls_attn": _aggregate_subwords_to_words_torch(word_ids, sink_realloc_cls_scores, word_count),
        "sink_realloc_samrank": _aggregate_subwords_to_words_torch(word_ids, sink_realloc_samrank_scores, word_count),
        "sink_realloc_fusion_attn": _aggregate_subwords_to_words_torch(word_ids, sink_realloc_fusion_scores, word_count),
    }
    if debiased_received_scores is not None:
        results["received_attn_debiased"] = _aggregate_subwords_to_words_torch(
            word_ids,
            debiased_received_scores,
            word_count,
        )
    return results


def _aggregate_layer_word_scores(
    per_layer_scores: Sequence[Dict[str, np.ndarray]],
    layer_weights: Sequence[float] | None = None,
) -> Dict[str, np.ndarray]:
    if not per_layer_scores:
        return {}

    weights = np.asarray(layer_weights if layer_weights is not None else [1.0] * len(per_layer_scores), dtype=np.float32)
    if weights.size != len(per_layer_scores):
        raise ValueError("layer_weights must have the same length as per_layer_scores.")
    weight_sum = float(weights.sum())
    if math.isclose(weight_sum, 0.0):
        raise ValueError("layer_weights sum to zero.")
    normalized_weights = weights / weight_sum

    aggregated: Dict[str, np.ndarray] = {}
    for method_name in per_layer_scores[0]:
        stacked = np.stack([layer_scores[method_name] for layer_scores in per_layer_scores], axis=0)
        aggregated[method_name] = np.average(stacked, axis=0, weights=normalized_weights)
    return aggregated


def _compute_hidden_position_scaled_scores_torch(
    word_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    word_count: int,
    *,
    content_mask: torch.Tensor,
    top_k: int,
    scale_factor: float,
) -> Dict[str, np.ndarray]:
    if top_k <= 0 or word_count <= 0:
        return {}

    effective_mask = content_mask if content_mask is not None else (word_ids >= 0)
    content_states = hidden_states[effective_mask].float()
    if content_states.shape[0] < 3:
        return {}

    hidden_dim = int(content_states.shape[-1])
    keep_count = min(max(int(top_k), 1), hidden_dim)
    positions = torch.linspace(-1.0, 1.0, steps=int(content_states.shape[0]), device=content_states.device, dtype=content_states.dtype)
    position_centered = positions - positions.mean()
    state_centered = content_states - content_states.mean(dim=0, keepdim=True)
    cov = (state_centered * position_centered[:, None]).mean(dim=0)
    state_std = state_centered.pow(2).mean(dim=0).sqrt()
    position_std = position_centered.pow(2).mean().sqrt()
    position_corr = cov.abs() / (state_std * position_std + 1e-6)
    scaled_states = hidden_states.float().clone()
    damped_dims = torch.topk(position_corr, k=keep_count, largest=True).indices
    scaled_states[:, damped_dims] *= float(scale_factor)

    scaled_content_states = scaled_states[effective_mask]
    doc_state = scaled_content_states.mean(dim=0, keepdim=True)
    token_vectors = torch.nn.functional.normalize(scaled_states, dim=1)
    doc_vector = torch.nn.functional.normalize(doc_state, dim=1)
    token_scores = (token_vectors * doc_vector).sum(dim=1)
    token_scores = token_scores.masked_fill(~effective_mask, 0.0)
    return {
        "hidden_posscale": _aggregate_subwords_to_words_torch(word_ids, token_scores, word_count),
    }


def batched_attention_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
    progress_label: str | None = None,
    log_every_batches: int = 0,
    *,
    batch_pos_tags: Sequence[Sequence[str]] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
    cls_head_strategies: Sequence[str] | None = None,
    cls_head_top_k: int = 4,
    cls_head_temperature: float = 8.0,
    somp_alpha: float = 512.0,
    somp_beta: float = 1.0,
    somp_local_window: int = 8,
    null_debias_samples: int = 0,
    null_debias_gamma: float = 1.0,
    null_debias_seed: int = 13,
    stream_attention_scores: bool = True,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    if not batch_words:
        return []

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    is_causal = bool(model_bundle.get("is_causal", False))
    max_length = int(model_bundle.get("max_length", 512))
    prefix_words, prefix_pos_tags = _tokenize_instruction_prefix(instruction_prefix, language)
    resolved_head_strategies = _resolve_cls_head_strategies(cls_head_strategies)
    true_bidirectional_attention = bool(model_bundle.get("true_bidirectional_attention", False))
    null_debias_enabled = int(null_debias_samples) > 0
    null_rng = np.random.default_rng(null_debias_seed)
    noise_token_ids = np.asarray(model_bundle.get("noise_token_ids", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    attention_modules = (
        _resolve_attention_modules(model)
        if stream_attention_scores and not null_debias_enabled
        else None
    )

    results: List[Dict[int, Dict[str, np.ndarray]]] = []
    total_batches = max(math.ceil(len(batch_words) / batch_size), 1)
    for batch_index, start in enumerate(range(0, len(batch_words), batch_size), start=1):
        content_word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        pos_tag_batch = (
            [list(pt) for pt in batch_pos_tags[start : start + batch_size]]
            if batch_pos_tags is not None
            else None
        )
        word_batch = [prefix_words + words for words in content_word_batch]
        combined_pos_tag_batch = None
        if pos_tag_batch is not None:
            combined_pos_tag_batch = [prefix_pos_tags + pos_tags for pos_tags in pos_tag_batch]
        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        valid_token_counts = [int(encoded["attention_mask"][index].sum().item()) for index in range(len(word_batch))]
        word_id_tensors = [
            _word_ids_to_tensor(word_ids_per_item[index][: valid_token_counts[index]], device)
            for index in range(len(word_batch))
        ]

        if attention_modules is not None:
            prefix_word_count = len(prefix_words)
            per_item_scores_list = _collect_streamed_attention_scores(
                model,
                attention_modules,
                encoded,
                layer_indices=layer_indices,
                content_word_batch=content_word_batch,
                word_batch=word_batch,
                word_ids_per_item=word_ids_per_item,
                valid_token_counts=valid_token_counts,
                word_id_tensors=word_id_tensors,
                combined_pos_tag_batch=combined_pos_tag_batch,
                language=language,
                prefix_word_count=prefix_word_count,
                is_causal=is_causal,
                true_bidirectional_attention=true_bidirectional_attention,
                resolved_head_strategies=resolved_head_strategies,
                cls_head_top_k=cls_head_top_k,
                cls_head_temperature=cls_head_temperature,
                somp_alpha=somp_alpha,
                somp_beta=somp_beta,
                somp_local_window=somp_local_window,
            )
            results.extend(per_item_scores_list)
            if progress_label and (
                batch_index == 1
                or batch_index == total_batches
                or (log_every_batches > 0 and batch_index % log_every_batches == 0)
            ):
                print(f"[{progress_label}] batch {batch_index}/{total_batches}")
            continue

        outputs = _run_attention_forward(
            model,
            encoded,
            true_bidirectional_attention=true_bidirectional_attention,
        )

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        per_item_scores_list: List[Dict[int, Dict[str, np.ndarray]]] = [dict() for _ in word_batch]
        null_received_baselines: dict[int, list[torch.Tensor]] = {}

        if null_debias_enabled:
            null_accumulators = {
                layer_index: [
                    torch.zeros(valid_token_counts[item_index], dtype=torch.float32, device=device)
                    for item_index in range(len(word_batch))
                ]
                for layer_index in resolved_indices
            }
            prefix_word_count = len(prefix_words)
            for _ in range(int(null_debias_samples)):
                noisy_input_ids = _build_null_input_ids(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    word_ids_per_item,
                    prefix_word_count=prefix_word_count,
                    noise_token_ids=noise_token_ids,
                    rng=null_rng,
                )
                noisy_encoded = dict(encoded)
                noisy_encoded["input_ids"] = noisy_input_ids
                noisy_outputs = _run_attention_forward(
                    model,
                    noisy_encoded,
                    true_bidirectional_attention=true_bidirectional_attention,
                )
                for layer_index, resolved_index in resolved_indices.items():
                    noisy_layer_attention = noisy_outputs.attentions[resolved_index].mean(dim=1).detach()
                    for item_index in range(len(word_batch)):
                        valid_token_count = valid_token_counts[item_index]
                        null_accumulators[layer_index][item_index] += noisy_layer_attention[
                            item_index,
                            :valid_token_count,
                            :valid_token_count,
                        ].sum(dim=0).float()
                    del noisy_layer_attention
                del noisy_outputs
            sample_scale = float(max(int(null_debias_samples), 1))
            null_received_baselines = {
                layer_index: [baseline / sample_scale for baseline in baselines]
                for layer_index, baselines in null_accumulators.items()
            }

        for layer_index, resolved_index in resolved_indices.items():
            layer_outputs = outputs.attentions[resolved_index].detach()
            layer_attention = layer_outputs.mean(dim=1)
            prefix_word_count = len(prefix_words)
            for item_index, words in enumerate(word_batch):
                valid_token_count = valid_token_counts[item_index]
                attention_map = layer_attention[item_index, :valid_token_count, :valid_token_count]
                content_mask = None
                if is_causal and combined_pos_tag_batch is not None:
                    content_mask_np = _build_content_mask(
                        word_ids_per_item[item_index][:valid_token_count],
                        words,
                        combined_pos_tag_batch[item_index],
                        language,
                    )
                    content_mask = torch.as_tensor(content_mask_np, dtype=torch.bool, device=device)
                combined_scores = _scores_from_attention_map_torch(
                    word_id_tensors[item_index],
                    attention_map,
                    len(words),
                    is_causal=is_causal,
                    content_mask=content_mask,
                    null_received_baseline=(
                        null_received_baselines.get(layer_index, [None] * len(word_batch))[item_index]
                        if null_debias_enabled
                        else None
                    ),
                    null_debias_gamma=null_debias_gamma,
                )
                if resolved_head_strategies:
                    head_attention = layer_outputs[item_index, :, :valid_token_count, :valid_token_count]
                    effective_content_mask = content_mask if content_mask is not None else (word_id_tensors[item_index] >= 0)
                    combined_scores.update(
                        _compute_cls_head_selection_scores_torch(
                            word_id_tensors[item_index],
                            head_attention,
                            len(words),
                            is_causal=is_causal,
                            content_mask=effective_content_mask,
                            strategies=resolved_head_strategies,
                            top_k=cls_head_top_k,
                            temperature=cls_head_temperature,
                            somp_alpha=somp_alpha,
                            somp_beta=somp_beta,
                            somp_local_window=somp_local_window,
                        )
                    )
                    combined_scores.update(
                        _compute_received_head_selection_scores_torch(
                            word_id_tensors[item_index],
                            head_attention,
                            len(words),
                            is_causal=is_causal,
                            content_mask=effective_content_mask,
                            strategies=resolved_head_strategies,
                            top_k=cls_head_top_k,
                            temperature=cls_head_temperature,
                            somp_alpha=somp_alpha,
                            somp_beta=somp_beta,
                            somp_local_window=somp_local_window,
                        )
                    )
                content_word_count = len(content_word_batch[item_index])
                per_item_scores_list[item_index][layer_index] = {
                    method_name: method_scores[prefix_word_count : prefix_word_count + content_word_count]
                    for method_name, method_scores in combined_scores.items()
                }
            del layer_attention

        results.extend(per_item_scores_list)
        del outputs
        if progress_label and (
            batch_index == 1
            or batch_index == total_batches
            or (log_every_batches > 0 and batch_index % log_every_batches == 0)
        ):
            print(f"[{progress_label}] batch {batch_index}/{total_batches}")
    return results


def _collect_streamed_attention_scores(
    model: torch.nn.Module,
    attention_modules: Sequence[torch.nn.Module],
    encoded_inputs: dict[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    content_word_batch: Sequence[Sequence[str]],
    word_batch: Sequence[Sequence[str]],
    word_ids_per_item: Sequence[Sequence[int | None]],
    valid_token_counts: Sequence[int],
    word_id_tensors: Sequence[torch.Tensor],
    combined_pos_tag_batch: Sequence[Sequence[str]] | None,
    language: str,
    prefix_word_count: int,
    is_causal: bool,
    true_bidirectional_attention: bool,
    resolved_head_strategies: Sequence[str],
    cls_head_top_k: int,
    cls_head_temperature: float,
    somp_alpha: float,
    somp_beta: float,
    somp_local_window: int,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    """Collect attention scores using streaming (hook-based) method to save memory."""
    layer_count = len(attention_modules)
    resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
    requested_by_resolved = {resolved_index: layer_index for layer_index, resolved_index in resolved_indices.items()}
    per_item_scores_list: List[Dict[int, Dict[str, np.ndarray]]] = [dict() for _ in word_batch]
    original_forwards: list[tuple[torch.nn.Module, object]] = []

    def make_patched_forward(
        resolved_layer_index: int,
        original_forward,
    ):
        def patched_forward(self, *args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            attn_weights = _extract_attention_weights_from_module_output(outputs)
            if attn_weights is None:
                return outputs

            if resolved_layer_index in requested_by_resolved:
                requested_layer_index = requested_by_resolved[resolved_layer_index]
                layer_outputs = attn_weights.detach()
                layer_attention = layer_outputs.mean(dim=1)
                for item_index, words in enumerate(word_batch):
                    valid_token_count = valid_token_counts[item_index]
                    attention_map = layer_attention[item_index, :valid_token_count, :valid_token_count]
                    content_mask = None
                    if is_causal and combined_pos_tag_batch is not None:
                        content_mask_np = _build_content_mask(
                            word_ids_per_item[item_index][:valid_token_count],
                            words,
                            combined_pos_tag_batch[item_index],
                            language,
                        )
                        content_mask = torch.as_tensor(content_mask_np, dtype=torch.bool, device=attention_map.device)
                    combined_scores = _scores_from_attention_map_torch(
                        word_id_tensors[item_index],
                        attention_map,
                        len(words),
                        is_causal=is_causal,
                        content_mask=content_mask,
                    )
                    if resolved_head_strategies:
                        head_attention = layer_outputs[item_index, :, :valid_token_count, :valid_token_count]
                        effective_content_mask = content_mask if content_mask is not None else (word_id_tensors[item_index] >= 0)
                        combined_scores.update(
                            _compute_cls_head_selection_scores_torch(
                                word_id_tensors[item_index],
                                head_attention,
                                len(words),
                                is_causal=is_causal,
                                content_mask=effective_content_mask,
                                strategies=resolved_head_strategies,
                                top_k=cls_head_top_k,
                                temperature=cls_head_temperature,
                                somp_alpha=somp_alpha,
                                somp_beta=somp_beta,
                                somp_local_window=somp_local_window,
                            )
                        )
                        combined_scores.update(
                            _compute_received_head_selection_scores_torch(
                                word_id_tensors[item_index],
                                head_attention,
                                len(words),
                                is_causal=is_causal,
                                content_mask=effective_content_mask,
                                strategies=resolved_head_strategies,
                                top_k=cls_head_top_k,
                                temperature=cls_head_temperature,
                                somp_alpha=somp_alpha,
                                somp_beta=somp_beta,
                                somp_local_window=somp_local_window,
                            )
                        )
                    content_word_count = len(content_word_batch[item_index])
                    per_item_scores_list[item_index][requested_layer_index] = {
                        method_name: method_scores[prefix_word_count : prefix_word_count + content_word_count]
                        for method_name, method_scores in combined_scores.items()
                    }
                del layer_attention
                del layer_outputs

            return _replace_attention_weights_with_none(outputs)

        return patched_forward

    for resolved_layer_index, module in enumerate(attention_modules):
        original_forward = module.forward
        original_forwards.append((module, original_forward))
        module.forward = MethodType(make_patched_forward(resolved_layer_index, original_forward), module)

    try:
        outputs = _run_attention_forward(
            model,
            encoded_inputs,
            true_bidirectional_attention=true_bidirectional_attention,
        )
        # Access outputs to trigger hooks
        _ = outputs.last_hidden_state
    finally:
        for module, original_forward in original_forwards:
            module.forward = original_forward

    return per_item_scores_list


def batched_hidden_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
    progress_label: str | None = None,
    log_every_batches: int = 0,
    *,
    batch_pos_tags: Sequence[Sequence[str]] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
    hidden_pos_top_k: int = 0,
    hidden_pos_scale_factor: float = 0.25,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    if not batch_words or int(hidden_pos_top_k) <= 0:
        return []

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    is_causal = bool(model_bundle.get("is_causal", False))
    max_length = int(model_bundle.get("max_length", 512))
    prefix_words, prefix_pos_tags = _tokenize_instruction_prefix(instruction_prefix, language)

    results: List[Dict[int, Dict[str, np.ndarray]]] = []
    total_batches = max(math.ceil(len(batch_words) / batch_size), 1)
    for batch_index, start in enumerate(range(0, len(batch_words), batch_size), start=1):
        content_word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        pos_tag_batch = (
            [list(pt) for pt in batch_pos_tags[start : start + batch_size]]
            if batch_pos_tags is not None
            else None
        )
        word_batch = [prefix_words + words for words in content_word_batch]
        combined_pos_tag_batch = None
        if pos_tag_batch is not None:
            combined_pos_tag_batch = [prefix_pos_tags + pos_tags for pos_tags in pos_tag_batch]

        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        valid_token_counts = [int(encoded["attention_mask"][index].sum().item()) for index in range(len(word_batch))]
        word_id_tensors = [
            _word_ids_to_tensor(word_ids_per_item[index][: valid_token_counts[index]], device)
            for index in range(len(word_batch))
        ]

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=False, output_hidden_states=True)

        layer_count = len(outputs.hidden_states) - 1
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        per_item_scores_list: List[Dict[int, Dict[str, np.ndarray]]] = [dict() for _ in word_batch]

        for layer_index, resolved_index in resolved_indices.items():
            layer_hidden = outputs.hidden_states[resolved_index + 1].detach()
            prefix_word_count = len(prefix_words)
            for item_index, words in enumerate(word_batch):
                valid_token_count = valid_token_counts[item_index]
                token_hidden = layer_hidden[item_index, :valid_token_count, :]
                content_mask = None
                if is_causal and combined_pos_tag_batch is not None:
                    content_mask_np = _build_content_mask(
                        word_ids_per_item[item_index][:valid_token_count],
                        words,
                        combined_pos_tag_batch[item_index],
                        language,
                    )
                    content_mask = torch.as_tensor(content_mask_np, dtype=torch.bool, device=device)
                else:
                    content_mask = word_id_tensors[item_index] >= 0
                hidden_scores = _compute_hidden_position_scaled_scores_torch(
                    word_id_tensors[item_index],
                    token_hidden,
                    len(words),
                    content_mask=content_mask,
                    top_k=hidden_pos_top_k,
                    scale_factor=hidden_pos_scale_factor,
                )
                content_word_count = len(content_word_batch[item_index])
                per_item_scores_list[item_index][layer_index] = {
                    method_name: method_scores[prefix_word_count : prefix_word_count + content_word_count]
                    for method_name, method_scores in hidden_scores.items()
                }
            del layer_hidden

        results.extend(per_item_scores_list)
        del outputs
        if progress_label and (
            batch_index == 1
            or batch_index == total_batches
            or (log_every_batches > 0 and batch_index % log_every_batches == 0)
        ):
            print(f"[{progress_label}] batch {batch_index}/{total_batches}")
    return results


def attention_word_scores(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
    *,
    pos_tags: Sequence[str] | None = None,
    language: str = "zh",
    instruction_prefix: str | None = None,
    cls_head_strategies: Sequence[str] | None = None,
    cls_head_top_k: int = 4,
    cls_head_temperature: float = 8.0,
    somp_alpha: float = 512.0,
    somp_beta: float = 1.0,
    somp_local_window: int = 8,
) -> Dict[str, np.ndarray]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    batched_scores = batched_attention_word_scores(
        [list(words)],
        model_bundle,
        layer_indices=effective_layer_indices,
        batch_size=1,
        batch_pos_tags=[list(pos_tags)] if pos_tags is not None else None,
        language=language,
        instruction_prefix=instruction_prefix,
        cls_head_strategies=cls_head_strategies,
        cls_head_top_k=cls_head_top_k,
        cls_head_temperature=cls_head_temperature,
        somp_alpha=somp_alpha,
        somp_beta=somp_beta,
        somp_local_window=somp_local_window,
    )
    if not batched_scores:
        return {}
    per_layer_scores = [batched_scores[0][index] for index in effective_layer_indices]
    return _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)
