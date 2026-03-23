from __future__ import annotations

import math
import re
import sys
from collections import Counter, OrderedDict
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


def build_model_bundle(model_name: str, device: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "embedding_pooling": _resolve_embedding_pooling(model_name),
        "embedding_cache": OrderedDict(),
        "embedding_cache_max_entries": 20000,
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


def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if bool(torch.isclose(min_value, max_value).item()):
        return torch.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _scores_from_attention_map_torch(
    word_ids: torch.Tensor,
    attention_map: torch.Tensor,
    word_count: int,
) -> Dict[str, np.ndarray]:
    attention = attention_map.float()
    cls_scores = attention[0]
    received_scores = attention.sum(dim=0)

    global_scores = received_scores
    redistributed = attention * global_scores.unsqueeze(0)
    redistributed = redistributed / redistributed.sum(dim=0, keepdim=True).clamp_min(1e-10)
    proportional_scores = redistributed.sum(dim=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = _normalize_tensor(cls_scores) * _normalize_tensor(received_scores)

    return {
        "cls_attn": _aggregate_subwords_to_words_torch(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words_torch(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words_torch(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words_torch(word_ids, fusion_scores, word_count),
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
        valid_token_counts = [int(encoded["attention_mask"][index].sum().item()) for index in range(len(word_batch))]
        word_id_tensors = [
            _word_ids_to_tensor(word_ids_per_item[index][: valid_token_counts[index]], device)
            for index in range(len(word_batch))
        ]

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        per_item_scores_list: List[Dict[int, Dict[str, np.ndarray]]] = [dict() for _ in word_batch]

        for layer_index, resolved_index in resolved_indices.items():
            layer_attention = outputs.attentions[resolved_index].mean(dim=1).detach()
            for item_index, words in enumerate(word_batch):
                valid_token_count = valid_token_counts[item_index]
                attention_map = layer_attention[item_index, :valid_token_count, :valid_token_count]
                per_item_scores_list[item_index][layer_index] = _scores_from_attention_map_torch(
                    word_id_tensors[item_index],
                    attention_map,
                    len(words),
                )
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
