from __future__ import annotations

from typing import Sequence

import numpy as np

from .attention import (
    ATTENTION_METHODS,
    DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX,
    attention_token_scores,
    attention_word_scores,
    attention_word_scores_with_raw,
    batched_attention_word_scores,
    build_model_bundle,
    rescore_with_new_words,
)
from .candidates import (
    Candidate,
    WordWeight,
    build_candidates,
    candidate_char_spans,
    candidate_rank_from_token_scores,
    candidate_rank_from_word_scores,
    candidate_score_values_from_token_spans,
    gravity_candidates,
    locate_word_offsets,
    merge_single_chars,
    rank_candidates_from_scores,
    segment_text,
    token_values_from_word_values,
)
from .hybrid import combine_word_scores, inverse_document_frequency, token_counter, word_scores_from_token_values


HYBRID_METHODS = tuple(f"{method_name}_idf" for method_name in ATTENTION_METHODS)
ALL_METHODS = ATTENTION_METHODS + HYBRID_METHODS


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
        attn_merge: bool = False,
        merge_threshold: float = 0.3,
        instruction_prefix: str | None = None,
        is_causal_override: bool | None = None,
        dedup_nested_for_topk5: bool = False,
        dedup_nested: bool = False,
        candidate_scoring: str = "word",
        enable_gravity: bool = False,
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
        if candidate_scoring not in {"word", "token_span"}:
            raise ValueError("candidate_scoring must be one of {'word', 'token_span'}.")
        if attn_merge and candidate_scoring == "token_span":
            raise ValueError("candidate_scoring='token_span' does not support attn_merge.")
        if enable_gravity and candidate_scoring != "token_span":
            raise ValueError("enable_gravity requires candidate_scoring='token_span'.")

        self.model = model
        self.language = language
        self.device = device
        self.backend = backend
        self.onnx_path = onnx_path
        self.user_dict = user_dict
        self.layer_index = layer_index
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.layer_weights = list(layer_weights) if layer_weights is not None else None
        self.attn_merge = attn_merge
        self.merge_threshold = merge_threshold
        self.instruction_prefix = instruction_prefix
        self.is_causal_override = is_causal_override
        self.dedup_nested_for_topk5 = dedup_nested_for_topk5
        self.dedup_nested = dedup_nested
        self.candidate_scoring = candidate_scoring
        self.enable_gravity = enable_gravity
        self.model_bundle: dict | None = None
        self.idf_lookup: dict[str, float] | None = None

    def extract_keywords(
        self,
        text: str | Sequence[str],
        method: str = "received_attn",
        top_k: int = 10,
        idf_lookup: dict[str, float] | None = None,
        pos_tags: Sequence[str] | None = None,
    ) -> list[str]:
        self._validate_method(method, allow_hybrid=True)

        if self.attn_merge and self.language.startswith("zh") and self._resolve_instruction_prefix() is None:
            return self._extract_keywords_with_merge(text, method, top_k, idf_lookup, pos_tags=pos_tags)

        words, pos_tags, candidates, candidate_starts, candidate_ends, token_counts = self._prepare_document(text, pos_tags=pos_tags)
        if not candidates:
            return []
        if self.candidate_scoring == "token_span":
            return self._rank_candidates_with_token_span(
                text=text,
                words=words,
                pos_tags=pos_tags,
                candidates=candidates,
                token_counts=token_counts,
                method=method,
                top_k=top_k,
                idf_lookup=idf_lookup,
            )

        word_scores = self._resolve_word_scores(
            words=words,
            pos_tags=pos_tags,
            method=method,
            token_counts=token_counts,
            idf_lookup=idf_lookup,
        )
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

        if self.enable_gravity:
            all_candidates = list(candidates)
            all_spans = list(candidate_spans)
            existing_normalized = {c.text.lower() for c in all_candidates}
            gravity_cands, gravity_spans = gravity_candidates(
                text, token_scores, token_offsets,
            )
            for gc, gs in zip(gravity_cands, gravity_spans):
                if gc.text.lower() not in existing_normalized:
                    all_candidates.append(gc)
                    all_spans.append(gs)
                    existing_normalized.add(gc.text.lower())
            all_scores = candidate_score_values_from_token_spans(
                all_spans, token_offsets, token_scores, aggregation_mode="mean",
            )
            return rank_candidates_from_scores(
                all_candidates, all_scores,
                top_k=top_k,
                dedup_nested=self._use_nested_dedup(top_k),
            )

        return candidate_rank_from_token_scores(
            candidates,
            candidate_spans,
            token_offsets,
            token_scores,
            top_k=top_k,
            dedup_nested=self._use_nested_dedup(top_k),
        )

    def _extract_keywords_with_merge(
        self,
        text: str | Sequence[str],
        method: str,
        top_k: int,
        idf_lookup: dict[str, float] | None,
        pos_tags: Sequence[str] | None = None,
    ) -> list[str]:
        # Step 1: coarse segmentation
        words, pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)

        # Step 2: forward pass — get word scores + raw attention map
        scores_by_method, attn_map, word_ids = attention_word_scores_with_raw(
            words,
            self._get_model_bundle(),
            layer_index=self._resolve_effective_layer_index(),
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            pos_tags=pos_tags,
            language=self.language,
        )

        # Step 3: merge single chars using attention
        merged_words, merged_pos, merge_map, changed = merge_single_chars(
            list(words), list(pos_tags), attn_map, list(word_ids), self.merge_threshold,
        )

        # Step 4: rescore if merging happened
        if changed:
            base_method = method.removesuffix("_idf")
            rescored = rescore_with_new_words(attn_map, list(word_ids), words, merged_words, merge_map)
            word_scores = rescored[base_method]
        else:
            base_method = method.removesuffix("_idf")
            word_scores = scores_by_method[base_method]
            merged_words = list(words)
            merged_pos = list(pos_tags)

        # Step 5: IDF hybrid if needed
        token_counts = dict(token_counter(merged_words, merged_pos, language=self.language))
        if method.endswith("_idf"):
            lookup = self._require_idf_lookup(idf_lookup)
            tfidf_word_scores = self._tfidf_word_scores(merged_words, merged_pos, token_counts, lookup)
            word_scores = combine_word_scores(word_scores, tfidf_word_scores, mode="product")

        # Step 6: build candidates and rank
        candidates = build_candidates(merged_words, merged_pos, language=self.language)
        if not candidates:
            return []
        candidate_starts = np.fromiter((c.word_start for c in candidates), dtype=np.int32, count=len(candidates))
        candidate_ends = np.fromiter((c.word_end for c in candidates), dtype=np.int32, count=len(candidates))

        return candidate_rank_from_word_scores(
            candidates,
            word_scores,
            top_k=top_k,
            dedup_nested=self._use_nested_dedup(top_k),
            token_counts=token_counts,
            words=merged_words,
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
        )

    def extract_word_weights(
        self,
        text: str | Sequence[str],
        method: str = "received_attn",
        pos_tags: Sequence[str] | None = None,
    ) -> list[WordWeight]:
        self._validate_method(method, allow_hybrid=False)
        words, pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)

        if self.attn_merge and self.language.startswith("zh") and self._resolve_instruction_prefix() is None:
            scores_by_method, attn_map, word_ids = attention_word_scores_with_raw(
                words,
                self._get_model_bundle(),
                layer_index=self._resolve_effective_layer_index(),
                layer_indices=self.layer_indices,
                layer_weights=self.layer_weights,
                pos_tags=pos_tags,
                language=self.language,
            )
            merged_words, merged_pos, merge_map, changed = merge_single_chars(
                list(words), list(pos_tags), attn_map, list(word_ids), self.merge_threshold,
            )
            if changed:
                rescored = rescore_with_new_words(attn_map, list(word_ids), words, merged_words, merge_map)
                word_scores = rescored[method]
                words, pos_tags = merged_words, merged_pos
            else:
                word_scores = scores_by_method[method]
        else:
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
        token_sets = []
        for index, text in enumerate(texts):
            pos_tags = pos_tags_batch[index] if pos_tags_batch is not None else None
            words, resolved_pos_tags = self._resolve_document_words(text, pos_tags=pos_tags)
            token_sets.append(token_counter(words, resolved_pos_tags, language=self.language).keys())
        self.idf_lookup = inverse_document_frequency(token_sets)
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

        # attn_merge: fall back to per-document extraction (no batch optimization)
        if self.attn_merge and self.language.startswith("zh") and self._resolve_instruction_prefix() is None:
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

    def _use_nested_dedup(self, top_k: int) -> bool:
        if self.dedup_nested:
            return True
        return self.dedup_nested_for_topk5 and int(top_k) <= 5

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
    enable_gravity: bool = False,
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
        enable_gravity=enable_gravity,
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
    "HYBRID_METHODS",
    "KeyAttenExtractor",
    "extract_keywords",
]
