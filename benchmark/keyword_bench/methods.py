from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import jieba.posseg as pseg
import networkx as nx
import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModel, AutoTokenizer

from .metrics import normalize_phrase

VENDOR_DIR = Path(__file__).resolve().parent.parent / ".vendor"


VALID_POS_PREFIXES = ("n", "nz", "eng", "v", "vn")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


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
) -> List[str]:
    if not candidates:
        return []

    scores_array = np.asarray(candidate_scores, dtype=np.float32)
    finite_indices = np.flatnonzero(np.isfinite(scores_array))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(scores_array[finite_indices])[::-1]]
    limit = min(top_k, sorted_indices.size)
    return [candidates[int(index)].text for index in sorted_indices[:limit]]


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


def build_model_bundle(model_name: str, device: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def embed_texts(
    model_bundle: dict,
    texts: Sequence[str],
    batch_size: int = 16,
    progress_label: str | None = None,
    log_every_batches: int = 0,
) -> np.ndarray:
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    outputs: List[np.ndarray] = []
    total_batches = max(math.ceil(len(texts) / batch_size), 1)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded, output_attentions=False).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            outputs.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            if progress_label and (
                batch_index == 1
                or batch_index == total_batches
                or (log_every_batches > 0 and batch_index % log_every_batches == 0)
            ):
                print(f"[{progress_label}] batch {batch_index}/{total_batches}")
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


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
) -> np.ndarray:
    candidate_texts = [candidate.text for candidate in candidates]
    if not candidate_texts:
        return np.zeros(0, dtype=np.float32)
    candidate_embeddings = embed_texts(model_bundle, candidate_texts, batch_size=batch_size)
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


def _resolve_layer_index(layer_index: int, layer_count: int) -> int:
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"Layer index {layer_index} is out of range for {layer_count} attention layers.")
    return resolved


def _scores_from_attention_map(
    word_ids: List[int | None],
    attention_map: np.ndarray,
    word_count: int,
) -> Dict[str, np.ndarray]:
    cls_scores = attention_map[0]
    received_scores = attention_map.sum(axis=0)

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = _normalize_array(cls_scores) * _normalize_array(received_scores)

    return {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
    }


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


def batched_attention_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
    progress_label: str | None = None,
    log_every_batches: int = 0,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    if not batch_words:
        return []

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    results: List[Dict[int, Dict[str, np.ndarray]]] = []
    total_batches = max(math.ceil(len(batch_words) / batch_size), 1)
    for batch_index, start in enumerate(range(0, len(batch_words), batch_size), start=1):
        word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        attention_by_layer = {
            layer_index: outputs.attentions[resolved_index].mean(dim=1).detach().cpu().numpy()
            for layer_index, resolved_index in resolved_indices.items()
        }

        for item_index, words in enumerate(word_batch):
            per_item_scores: Dict[int, Dict[str, np.ndarray]] = {}
            valid_token_count = int(encoded["attention_mask"][item_index].sum().item())
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            for layer_index in layer_indices:
                attention_map = attention_by_layer[layer_index][item_index][:valid_token_count, :valid_token_count]
                per_item_scores[layer_index] = _scores_from_attention_map(
                    valid_word_ids,
                    attention_map,
                    len(words),
                )
            results.append(per_item_scores)
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
) -> Dict[str, np.ndarray]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    batched_scores = batched_attention_word_scores(
        [list(words)],
        model_bundle,
        layer_indices=effective_layer_indices,
        batch_size=1,
    )
    if not batched_scores:
        return {}
    per_layer_scores = [batched_scores[0][index] for index in effective_layer_indices]
    return _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)
