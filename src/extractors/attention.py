from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..scoring import (
    ATTENTION_METHODS,
    DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX,
    attention_token_scores,
    attention_word_scores,
    batched_attention_word_scores,
    build_model_bundle,
)
from ..candidates import (
    Candidate,
    WordWeight,
    build_candidates,
    candidate_char_spans,
    candidate_rank_from_token_scores,
    candidate_rank_from_word_scores,
    locate_word_offsets,
    segment_text,
    token_values_from_word_values,
)
from ..candidates.fusion import combine_word_scores, token_counter, word_scores_from_token_values

if TYPE_CHECKING:
    from .bio import BIOExtractor


HYBRID_METHODS = tuple(f"{method_name}_idf" for method_name in ATTENTION_METHODS)
ALL_METHODS = ATTENTION_METHODS + HYBRID_METHODS
CANDIDATE_SOURCES = ("word", "token_span", "bio")


class KeyAttenExtractor:
    def __init__(
        self,
        model: str,
        language: str = "zh",
        device: str = "cpu",
        backend: str = "auto",
        onnx_path: str | None = None,
        user_dict: str | Sequence[str] | dict[str, str | tuple[int | None, str | None]] | None = None,
        layer_index: int | None = None,
        layer_indices: list[int] | None = None,
        layer_weights: list[float] | None = None,
        instruction_prefix: str | None = None,
        is_causal_override: bool | None = None,
        dedup_nested_for_topk5: bool = False,
        dedup_nested: bool = False,
        candidate_scoring: str = "word",
        dtype: str | None = "auto",
        bio_model_path: str | None = None,
        cache_enabled: bool = False,
        cache_dir: str | Path = "cache",
    ) -> None:
        if not model:
            raise ValueError("model is required.")
        if language not in {"zh", "en"}:
            raise ValueError("language must be 'zh' or 'en'.")
        if layer_indices is not None and not layer_indices:
            raise ValueError("layer_indices must not be empty.")
        if layer_weights is not None:
            if layer_indices is None:
                raise ValueError("layer_weights requires layer_indices.")
            if len(layer_weights) != len(layer_indices):
                raise ValueError("layer_weights must have the same length as layer_indices.")
        if backend not in {"auto", "torch", "onnx"}:
            raise ValueError("backend must be one of {'auto', 'torch', 'onnx'}.")
        if backend == "onnx" and layer_indices is not None:
            raise ValueError("ONNX backend currently supports only a single exported attention layer.")
        if candidate_scoring not in {"word", "token_span", "bio"}:
            raise ValueError("candidate_scoring must be one of {'word', 'token_span', 'bio'}.")
        if candidate_scoring == "bio" and bio_model_path is None:
            raise ValueError("candidate_scoring='bio' requires bio_model_path.")

        self.model = model
        self.language = language
        self.device = device
        self.backend = backend
        self.onnx_path = onnx_path
        self.user_dict = user_dict
        self.layer_index = layer_index
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.layer_weights = list(layer_weights) if layer_weights is not None else None
        self.instruction_prefix = instruction_prefix
        self.is_causal_override = is_causal_override
        self.dedup_nested_for_topk5 = dedup_nested_for_topk5
        self.dedup_nested = dedup_nested
        self.candidate_scoring = candidate_scoring
        self.dtype = dtype
        self.bio_model_path = bio_model_path
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = Path(cache_dir)
        self.model_bundle: dict | None = None
        self.idf_lookup: dict[str, float] | None = None
        self._idf_doc_count = 0
        self._idf_document_freq: Counter[str] = Counter()
        self._bio_extractor: BIOExtractor | None = None

    def extract_keywords(
        self,
        text: str | Sequence[str],
        method: str = "received_attn",
        top_k: int = 10,
        idf_lookup: dict[str, float] | None = None,
        pos_tags: Sequence[str] | None = None,
    ) -> list[str]:
        self._validate_method(method, allow_hybrid=True)
        cache_path = self._keyword_cache_path(
            text=text,
            method=method,
            top_k=top_k,
            idf_lookup=idf_lookup,
            pos_tags=pos_tags,
        )
        if cache_path is not None:
            cached = self._read_keyword_cache(cache_path)
            if cached is not None:
                return cached

        if self.candidate_scoring == "word":
            result = self._extract_keywords_from_word_cache(
                text=text,
                method=method,
                top_k=top_k,
                idf_lookup=idf_lookup,
                pos_tags=pos_tags,
            )
            if cache_path is not None:
                self._write_keyword_cache(cache_path, result)
            return result

        words, pos_tags, candidates, candidate_starts, candidate_ends, token_counts = self._prepare_document(text, pos_tags=pos_tags)
        if not candidates:
            if cache_path is not None:
                self._write_keyword_cache(cache_path, [])
            return []
        if self.candidate_scoring == "token_span":
            result = self._rank_candidates_with_token_span(
                text=text,
                words=words,
                pos_tags=pos_tags,
                candidates=candidates,
                token_counts=token_counts,
                method=method,
                top_k=top_k,
                idf_lookup=idf_lookup,
            )
            if cache_path is not None:
                self._write_keyword_cache(cache_path, result)
            return result
        if self.candidate_scoring == "bio":
            result = self._rank_candidates_with_bio(
                text=text,
                method=method,
                top_k=top_k,
                idf_lookup=idf_lookup,
            )
            if cache_path is not None:
                self._write_keyword_cache(cache_path, result)
            return result

    def _rank_candidates_with_bio(
        self,
        *,
        text: str | Sequence[str],
        method: str,
        top_k: int,
        idf_lookup: dict[str, float] | None,
    ) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("candidate_scoring='bio' requires raw string text input.")

        bio_ext = self._get_bio_extractor()
        bio_candidates = bio_ext.extract_spans(text)
        if not bio_candidates:
            return []

        # Score BIO candidates with attention token scores
        token_method_scores, token_offsets = attention_token_scores(
            text,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )
        base_method = method.removesuffix("_idf")
        token_scores = token_method_scores[base_method]

        if method.endswith("_idf"):
            lookup = self._require_idf_lookup(idf_lookup)
            words, pos_tags = self._resolve_document_words(text)
            token_counts = dict(token_counter(words, pos_tags, language=self.language))
            tfidf_word_scores = self._tfidf_word_scores(words, pos_tags, token_counts, lookup)
            word_offsets = locate_word_offsets(text, words)
            tfidf_token_scores = token_values_from_word_values(token_offsets, word_offsets, tfidf_word_scores)
            token_scores = combine_word_scores(token_scores, tfidf_token_scores, mode="product")

        # Score each BIO candidate by its token span overlap with attention scores
        scored: list[tuple[float, str]] = []
        for kw in bio_candidates:
            best_score = 0.0
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx < 0:
                    break
                end = idx + len(kw)
                # Find tokens overlapping [idx, end)
                span_score = 0.0
                span_count = 0
                for t_idx, (t_start, t_end) in enumerate(token_offsets):
                    t_start = int(t_start)
                    t_end = int(t_end)
                    if t_end <= t_start:
                        continue
                    if t_start < end and t_end > idx:
                        span_score += float(token_scores[t_idx])
                        span_count += 1
                if span_count > 0:
                    best_score = max(best_score, span_score / span_count)
                start = idx + 1
            scored.append((best_score, kw))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [kw for _, kw in scored[:top_k]]

    def _rank_candidates_with_token_span(
        self,
        *,
        text: str | Sequence[str],
        words: Sequence[str],
        pos_tags: Sequence[str],
        candidates: Sequence[Candidate],
        token_counts: dict[str, float],
        method: str,
        top_k: int,
        idf_lookup: dict[str, float] | None,
    ) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("candidate_scoring='token_span' requires raw string text input.")

        word_offsets = locate_word_offsets(text, words)
        candidate_spans = candidate_char_spans(candidates, word_offsets)
        token_method_scores, token_offsets = attention_token_scores(
            text,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )
        base_method = method.removesuffix("_idf")
        token_scores = token_method_scores[base_method]
        if method.endswith("_idf"):
            lookup = self._require_idf_lookup(idf_lookup)
            tfidf_word_scores = self._tfidf_word_scores(words, pos_tags, token_counts, lookup)
            tfidf_token_scores = token_values_from_word_values(token_offsets, word_offsets, tfidf_word_scores)
            token_scores = combine_word_scores(token_scores, tfidf_token_scores, mode="product")

        return candidate_rank_from_token_scores(
            candidates,
            candidate_spans,
            token_offsets,
            token_scores,
            top_k=top_k,
            dedup_nested=self._use_nested_dedup(top_k),
        )

    def extract_word_weights(
        self,
        text: str | Sequence[str],
        method: str = "received_attn",
        pos_tags: Sequence[str] | None = None,
    ) -> list[WordWeight]:
        self._validate_method(method, allow_hybrid=False)
        words, pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)
        scores_by_method = attention_word_scores(
            words,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            pos_tags=pos_tags,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )
        word_scores = scores_by_method[method]

        return [
            WordWeight(
                word=word,
                index=index,
                weight=float(word_scores[index]),
                pos_tag=pos_tags[index],
            )
            for index, word in enumerate(words)
        ]

    def fit_idf(
        self,
        texts: list[str | Sequence[str]],
        pos_tags_batch: Sequence[Sequence[str] | None] | None = None,
    ) -> dict[str, float]:
        if pos_tags_batch is not None and len(pos_tags_batch) != len(texts):
            raise ValueError("pos_tags_batch must have the same length as texts.")
        self._idf_doc_count = 0
        self._idf_document_freq = Counter()
        for index, text in enumerate(texts):
            pos_tags = pos_tags_batch[index] if pos_tags_batch is not None else None
            self._add_idf_document(text, pos_tags=pos_tags)
        self.idf_lookup = self._build_idf_lookup_from_state()
        return dict(self.idf_lookup)

    def update_idf(
        self,
        texts: list[str | Sequence[str]],
        pos_tags_batch: Sequence[Sequence[str] | None] | None = None,
    ) -> dict[str, float]:
        if pos_tags_batch is not None and len(pos_tags_batch) != len(texts):
            raise ValueError("pos_tags_batch must have the same length as texts.")
        for index, text in enumerate(texts):
            pos_tags = pos_tags_batch[index] if pos_tags_batch is not None else None
            self._add_idf_document(text, pos_tags=pos_tags)
        self.idf_lookup = self._build_idf_lookup_from_state()
        return dict(self.idf_lookup)

    def extract_keywords_batch(
        self,
        texts: list[str | Sequence[str]],
        method: str = "received_attn",
        top_k: int = 10,
        idf_lookup: dict[str, float] | None = None,
        pos_tags_batch: Sequence[Sequence[str] | None] | None = None,
    ) -> list[list[str]]:
        self._validate_method(method, allow_hybrid=True)
        if not texts:
            return []
        if pos_tags_batch is not None and len(pos_tags_batch) != len(texts):
            raise ValueError("pos_tags_batch must have the same length as texts.")
        if self.candidate_scoring == "token_span":
            return [
                self.extract_keywords(
                    text,
                    method=method,
                    top_k=top_k,
                    idf_lookup=idf_lookup,
                    pos_tags=pos_tags_batch[index] if pos_tags_batch is not None else None,
                )
                for index, text in enumerate(texts)
            ]

        prepared = [
            self._prepare_document(
                text,
                pos_tags=pos_tags_batch[index] if pos_tags_batch is not None else None,
            )
            for index, text in enumerate(texts)
        ]
        batch_words = [item[0] for item in prepared]
        batch_pos_tags_list = [item[1] for item in prepared]
        effective_layer_indices = (
            list(self.layer_indices)
            if self.layer_indices is not None
            else [self._resolve_effective_layer_index()]
        )
        per_doc_layer_scores = batched_attention_word_scores(
            batch_words,
            self._get_model_bundle(),
            layer_indices=effective_layer_indices,
            batch_pos_tags=batch_pos_tags_list,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )

        results: list[list[str]] = []
        base_method = method.removesuffix("_idf")
        for (words, pos_tags, candidates, candidate_starts, candidate_ends, token_counts), per_doc_scores in zip(
            prepared,
            per_doc_layer_scores,
        ):
            if not candidates:
                results.append([])
                continue

            per_layer_scores = [per_doc_scores[index] for index in effective_layer_indices]
            if len(per_layer_scores) == 1:
                scores_by_method = per_layer_scores[0]
            else:
                scores_by_method = {
                    method_name: np.average(
                        np.stack([scores[method_name] for scores in per_layer_scores], axis=0),
                        axis=0,
                        weights=self.layer_weights,
                    )
                    for method_name in per_layer_scores[0]
                }

            word_scores = scores_by_method[base_method]
            if method.endswith("_idf"):
                lookup = self._require_idf_lookup(idf_lookup)
                tfidf_word_scores = self._tfidf_word_scores(words, pos_tags, token_counts, lookup)
                word_scores = combine_word_scores(word_scores, tfidf_word_scores, mode="product")

            results.append(
                candidate_rank_from_word_scores(
                    candidates,
                    word_scores,
                    top_k=top_k,
                    dedup_nested=self._use_nested_dedup(top_k),
                    token_counts=token_counts,
                    words=words,
                    candidate_starts=candidate_starts,
                    candidate_ends=candidate_ends,
                )
            )
        return results

    def _prepare_document(
        self,
        text: str | Sequence[str],
        pos_tags: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str], list[Candidate], np.ndarray, np.ndarray, dict[str, float]]:
        words, pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)
        candidates = build_candidates(words, pos_tags, language=self.language)
        candidate_starts = np.fromiter((candidate.word_start for candidate in candidates), dtype=np.int32, count=len(candidates))
        candidate_ends = np.fromiter((candidate.word_end for candidate in candidates), dtype=np.int32, count=len(candidates))
        counts = dict(token_counter(words, pos_tags, language=self.language))
        return words, pos_tags, candidates, candidate_starts, candidate_ends, counts

    def _resolve_document_words(
        self,
        text: str | Sequence[str],
        pos_tags: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        if isinstance(text, str):
            return segment_text(text, language=self.language, user_dict=self.user_dict)

        words = [str(word) for word in text]
        if any(not word.strip() for word in words):
            raise ValueError("External token input must not contain empty tokens.")

        if pos_tags is not None and len(pos_tags) != len(words):
            raise ValueError("pos_tags must have the same length as words.")

        if pos_tags is None:
            default_pos = "eng" if self.language.startswith("en") else "n"
            return words, [default_pos] * len(words)

        return words, [str(tag) for tag in pos_tags]

    def _resolve_word_scores(
        self,
        words: Sequence[str],
        pos_tags: Sequence[str],
        method: str,
        token_counts: dict[str, float],
        idf_lookup: dict[str, float] | None,
    ) -> np.ndarray:
        self._validate_method(method, allow_hybrid=True)
        scores_by_method = attention_word_scores(
            words,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            pos_tags=pos_tags,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )
        base_method = method.removesuffix("_idf")
        word_scores = scores_by_method[base_method]
        if method.endswith("_idf"):
            lookup = self._require_idf_lookup(idf_lookup)
            tfidf_word_scores = self._tfidf_word_scores(words, pos_tags, token_counts, lookup)
            word_scores = combine_word_scores(word_scores, tfidf_word_scores, mode="product")
        return word_scores

    def _add_idf_document(
        self,
        text: str | Sequence[str],
        pos_tags: Sequence[str] | None = None,
    ) -> None:
        words, resolved_pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)
        tokens = set(token_counter(words, resolved_pos_tags, language=self.language).keys())
        self._idf_doc_count += 1
        self._idf_document_freq.update(tokens)

    def _build_idf_lookup_from_state(self) -> dict[str, float]:
        doc_count = max(self._idf_doc_count, 1)
        return {
            token: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
            for token, freq in self._idf_document_freq.items()
        }

    def _extract_keywords_from_word_cache(
        self,
        *,
        text: str | Sequence[str],
        method: str,
        top_k: int,
        idf_lookup: dict[str, float] | None,
        pos_tags: Sequence[str] | None,
    ) -> list[str]:
        document = self._load_or_build_word_document_cache(text=text, pos_tags=pos_tags)
        candidates = [
            Candidate(
                text=str(candidate["text"]),
                word_start=int(candidate["word_start"]),
                word_end=int(candidate["word_end"]),
            )
            for candidate in document["candidates"]
        ]
        if not candidates:
            return []

        words = [str(word) for word in document["words"]]
        resolved_pos_tags = [str(tag) for tag in document["pos_tags"]]
        token_counts = {str(key): float(value) for key, value in document["token_counts"].items()}
        scores_by_method = {
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in document["scores_by_method"].items()
        }
        base_method = method.removesuffix("_idf")
        word_scores = scores_by_method[base_method]
        if method.endswith("_idf"):
            lookup = self._require_idf_lookup(idf_lookup)
            tfidf_word_scores = self._tfidf_word_scores(words, resolved_pos_tags, token_counts, lookup)
            word_scores = combine_word_scores(word_scores, tfidf_word_scores, mode="product")

        candidate_starts = np.fromiter((candidate.word_start for candidate in candidates), dtype=np.int32, count=len(candidates))
        candidate_ends = np.fromiter((candidate.word_end for candidate in candidates), dtype=np.int32, count=len(candidates))
        return candidate_rank_from_word_scores(
            candidates,
            word_scores,
            top_k=top_k,
            dedup_nested=self._use_nested_dedup(top_k),
            token_counts=token_counts,
            words=words,
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
        )

    def _load_or_build_word_document_cache(
        self,
        *,
        text: str | Sequence[str],
        pos_tags: Sequence[str] | None,
    ) -> dict:
        cache_path = self._word_document_cache_path(text=text, pos_tags=pos_tags)
        if cache_path is not None:
            cached = self._read_word_document_cache(cache_path)
            if cached is not None:
                return cached

        words, resolved_pos_tags, candidates, _candidate_starts, _candidate_ends, token_counts = self._prepare_document(
            text,
            pos_tags=pos_tags,
        )
        scores_by_method = attention_word_scores(
            words,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            pos_tags=resolved_pos_tags,
            language=self.language,
            instruction_prefix=self._resolve_instruction_prefix(),
        )
        document = {
            "version": 1,
            "words": list(words),
            "pos_tags": list(resolved_pos_tags),
            "candidates": [
                {
                    "text": candidate.text,
                    "word_start": int(candidate.word_start),
                    "word_end": int(candidate.word_end),
                }
                for candidate in candidates
            ],
            "token_counts": {str(key): float(value) for key, value in token_counts.items()},
            "scores_by_method": {
                method_name: np.asarray(scores, dtype=np.float32).tolist()
                for method_name, scores in scores_by_method.items()
            },
        }
        if cache_path is not None:
            self._write_json_cache(cache_path, document)
        return document

    def _tfidf_word_scores(
        self,
        words: Sequence[str],
        pos_tags: Sequence[str],
        token_counts: dict[str, float],
        idf_lookup: dict[str, float],
    ) -> np.ndarray:
        tfidf_values = {token: count * idf_lookup.get(token, 0.0) for token, count in token_counts.items()}
        return word_scores_from_token_values(words, pos_tags, tfidf_values, language=self.language)

    def _require_idf_lookup(self, idf_lookup: dict[str, float] | None) -> dict[str, float]:
        lookup = idf_lookup if idf_lookup is not None else self.idf_lookup
        if lookup is None:
            raise ValueError("IDF-based methods require idf_lookup or a prior fit_idf() call.")
        return lookup

    def _get_model_bundle(self) -> dict:
        if self.model_bundle is None:
            self.model_bundle = build_model_bundle(
                self.model,
                self.device,
                backend=self.backend,
                onnx_path=self.onnx_path,
                layer_index=self.layer_index if self.layer_index is not None else -1,
                layer_indices=self.layer_indices,
                is_causal_override=self.is_causal_override,
                dtype=self.dtype,
            )
        return self.model_bundle

    def _resolve_effective_layer_index(self) -> int:
        if self.layer_index is not None:
            return self.layer_index
        model_bundle = self._get_model_bundle()
        if model_bundle.get("is_causal"):
            recommended = model_bundle.get("recommended_layer_index")
            if recommended is not None:
                return int(recommended)
        return -1

    def _resolve_instruction_prefix(self) -> str | None:
        if self.instruction_prefix is not None:
            prefix = self.instruction_prefix.strip()
            return prefix or None
        if self._get_model_bundle().get("is_causal") and self.language.startswith("zh"):
            return DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX
        return None

    def _get_bio_extractor(self) -> BIOExtractor:
        from .bio import BIOExtractor

        if self._bio_extractor is None:
            if not self.bio_model_path:
                raise ValueError("bio_model_path is required for candidate_scoring='bio'.")
            self._bio_extractor = BIOExtractor(
                checkpoint_path=self.bio_model_path, device=self.device
            )
        return self._bio_extractor

    def _use_nested_dedup(self, top_k: int) -> bool:
        if self.dedup_nested:
            return True
        return self.dedup_nested_for_topk5 and int(top_k) <= 5

    def _keyword_cache_path(
        self,
        *,
        text: str | Sequence[str],
        method: str,
        top_k: int,
        idf_lookup: dict[str, float] | None,
        pos_tags: Sequence[str] | None,
    ) -> Path | None:
        if not self.cache_enabled:
            return None

        payload = {
            "version": 1,
            "model": self.model,
            "language": self.language,
            "backend": self.backend,
            "onnx_path": self.onnx_path,
            "layer_index": self.layer_index,
            "layer_indices": self.layer_indices,
            "layer_weights": self.layer_weights,
            "instruction_prefix": self.instruction_prefix,
            "is_causal_override": self.is_causal_override,
            "dedup_nested_for_topk5": self.dedup_nested_for_topk5,
            "dedup_nested": self.dedup_nested,
            "candidate_scoring": self.candidate_scoring,
            "dtype": self.dtype,
            "bio_model_path": self.bio_model_path,
            "user_dict": self._stable_cache_value(self.user_dict),
            "method": method,
            "top_k": int(top_k),
            "text": self._stable_cache_value(text),
            "pos_tags": self._stable_cache_value(pos_tags),
            "idf_lookup": self._idf_cache_fingerprint(idf_lookup),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.cache_dir / "keyatten_keywords" / f"{digest}.json"

    def _word_document_cache_path(
        self,
        *,
        text: str | Sequence[str],
        pos_tags: Sequence[str] | None,
    ) -> Path | None:
        if not self.cache_enabled:
            return None

        payload = {
            "version": 1,
            "model": self.model,
            "language": self.language,
            "backend": self.backend,
            "onnx_path": self.onnx_path,
            "layer_index": self.layer_index,
            "layer_indices": self.layer_indices,
            "layer_weights": self.layer_weights,
            "instruction_prefix": self.instruction_prefix,
            "is_causal_override": self.is_causal_override,
            "candidate_scoring": self.candidate_scoring,
            "dtype": self.dtype,
            "user_dict": self._stable_cache_value(self.user_dict),
            "text": self._stable_cache_value(text),
            "pos_tags": self._stable_cache_value(pos_tags),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.cache_dir / "keyatten_documents" / f"{digest}.json"

    @staticmethod
    def _stable_cache_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): KeyAttenExtractor._stable_cache_value(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Sequence):
            return [KeyAttenExtractor._stable_cache_value(item) for item in value]
        return repr(value)

    def _idf_cache_fingerprint(self, idf_lookup: dict[str, float] | None) -> str | None:
        lookup = idf_lookup if idf_lookup is not None else self.idf_lookup
        if lookup is None:
            return None
        normalized = {
            str(key): round(float(value), 12)
            for key, value in sorted(lookup.items(), key=lambda item: str(item[0]))
        }
        encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _read_keyword_cache(path: Path) -> list[str] | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        keywords = payload.get("keywords") if isinstance(payload, dict) else None
        if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
            return None
        return list(keywords)

    @staticmethod
    def _read_word_document_cache(path: Path) -> dict | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        if not isinstance(payload, dict):
            return None
        required_keys = {"words", "pos_tags", "candidates", "token_counts", "scores_by_method"}
        if not required_keys.issubset(payload):
            return None
        if not isinstance(payload["words"], list) or not isinstance(payload["pos_tags"], list):
            return None
        if not isinstance(payload["candidates"], list):
            return None
        if not isinstance(payload["token_counts"], dict) or not isinstance(payload["scores_by_method"], dict):
            return None
        return payload

    @staticmethod
    def _write_keyword_cache(path: Path, keywords: Sequence[str]) -> None:
        KeyAttenExtractor._write_json_cache(path, {"keywords": list(keywords)})

    @staticmethod
    def _write_json_cache(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        tmp_path.replace(path)

    @staticmethod
    def _validate_method(
        method: str,
        allow_hybrid: bool,
    ) -> None:
        allowed_methods = ATTENTION_METHODS if not allow_hybrid else ALL_METHODS
        if method not in allowed_methods:
            raise ValueError(f"Unsupported method: {method}. Expected one of {allowed_methods}.")
        if method.endswith("_idf") and not allow_hybrid:
            raise ValueError(f"{method} is not supported for word-weight extraction.")


def extract_keywords(
    text: str | Sequence[str],
    model: str,
    language: str = "zh",
    method: str = "received_attn",
    top_k: int = 10,
    device: str = "cpu",
    backend: str = "auto",
    onnx_path: str | None = None,
    user_dict: str | Sequence[str] | dict[str, str | tuple[int | None, str | None]] | None = None,
    idf_lookup: dict[str, float] | None = None,
    layer_index: int | None = None,
    pos_tags: Sequence[str] | None = None,
    instruction_prefix: str | None = None,
    is_causal_override: bool | None = None,
    dedup_nested_for_topk5: bool = False,
    dedup_nested: bool = False,
    candidate_scoring: str = "word",
    bio_model_path: str | None = None,
    cache_enabled: bool = False,
    cache_dir: str | Path = "cache",
) -> list[str]:
    extractor = KeyAttenExtractor(
        model=model,
        language=language,
        device=device,
        backend=backend,
        onnx_path=onnx_path,
        user_dict=user_dict,
        layer_index=layer_index,
        instruction_prefix=instruction_prefix,
        is_causal_override=is_causal_override,
        dedup_nested_for_topk5=dedup_nested_for_topk5,
        dedup_nested=dedup_nested,
        candidate_scoring=candidate_scoring,
        bio_model_path=bio_model_path,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )
    return extractor.extract_keywords(
        text=text,
        method=method,
        top_k=top_k,
        idf_lookup=idf_lookup,
        pos_tags=pos_tags,
    )


__all__ = [
    "ALL_METHODS",
    "CANDIDATE_SOURCES",
    "HYBRID_METHODS",
    "KeyAttenExtractor",
    "extract_keywords",
]
