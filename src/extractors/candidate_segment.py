"""Candidate-segment attention keyword extraction.

This module packages the 2026-05-05 mainline experiment into a reusable
inference API:

    document text
    -> BIO candidate generator
    -> explicit candidate segment prompt
    -> attention-only candidate reranking

Unlike the base ``KeyAttenExtractor``, this route depends on a trained LoRA
adapter and a BIO candidate generator checkpoint.
"""
from __future__ import annotations

import random
from typing import Any, List, Sequence

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

from .bio import BIOExtractor
from .qk_lora import _resolve_layer_arg


DEFAULT_SEGMENT_INSTRUCTION = "为这篇文章选择最重要的关键词。"


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "torch is required for CandidateSegmentAttentionExtractor. "
            "Install with: pip install torch>=2.0"
        )
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for CandidateSegmentAttentionExtractor. "
            "Install with: pip install transformers>=4.30"
        )


def _candidate_token_scores(
    attention_map: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    attn_mean = attention_map.mean(dim=1)
    pad_mask = attention_mask.unsqueeze(-1).float()
    return (attn_mean * pad_mask).sum(dim=1)


def _char_span_to_token_span(
    offsets: Sequence[tuple[int, int]] | Sequence[Sequence[int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    token_indices: list[int] = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if int(token_end) <= int(token_start):
            continue
        if int(token_start) < char_end and int(token_end) > char_start:
            token_indices.append(token_index)
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def _build_segment_text(
    text: str,
    candidates: Sequence[str],
    *,
    instruction: str,
    number_candidates: bool,
) -> tuple[str, list[dict[str, Any]]]:
    header = f"{instruction}\n\n文章：\n{text}\n\n候选：\n"
    parts = [header]
    spans: list[dict[str, Any]] = []
    cursor = len(header)
    for index, candidate in enumerate(candidates, 1):
        prefix = f"[{index}] " if number_candidates else "<候选>\n"
        suffix = "\n"
        parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        parts.append(candidate)
        cursor += len(candidate)
        end = cursor
        parts.append(suffix)
        cursor += len(suffix)
        spans.append({"text": candidate, "char_span": (start, end)})
    return "".join(parts), spans


class CandidateSegmentAttentionExtractor:
    """Keyword extraction with BIO candidates and attention reranking.

    Args:
        model: HuggingFace model name or local path.
        adapter_path: Optional LoRA adapter path. If omitted, runs zero-shot.
        bio_model_path: BIO checkpoint path used to produce candidate phrases.
        language: Currently only ``'zh'`` is supported.
        device: PyTorch device string.
        layer: Attention layer index or ``'auto'``.
        max_length: Max token length for the document + candidate segment input.
        max_candidates: Number of BIO candidates exposed to the reranker.
        bio_profile: BIO candidate profile, e.g. ``clean`` or ``high_recall``.
        candidate_order: ``bio`` or ``random``.
        candidate_seed: Seed used when ``candidate_order='random'`` and no
            per-call seed list is provided.
        instruction: Prompt header for the candidate-segment input.
        number_candidates: Whether to prefix candidates with ``[1]``, ``[2]``.
        dtype: ``auto``, ``float32``, ``float16``, or ``bfloat16``.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        adapter_path: str | None = None,
        bio_model_path: str,
        language: str = "zh",
        device: str = "cpu",
        layer: str = "auto",
        max_length: int = 1024,
        max_candidates: int = 30,
        bio_profile: str = "high_recall",
        candidate_order: str = "random",
        candidate_seed: int = 42,
        instruction: str = DEFAULT_SEGMENT_INSTRUCTION,
        number_candidates: bool = False,
        dtype: str = "auto",
    ) -> None:
        _require_torch()
        if language != "zh":
            raise ValueError("CandidateSegmentAttentionExtractor currently supports only language='zh'.")
        if candidate_order not in {"bio", "random"}:
            raise ValueError("candidate_order must be one of {'bio', 'random'}.")
        if max_candidates < 2:
            raise ValueError("max_candidates must be >= 2.")

        self.model_name = model
        self.adapter_path = adapter_path
        self.bio_model_path = bio_model_path
        self.language = language
        self.device = device
        self.layer_arg = layer
        self.max_length = max_length
        self.max_candidates = max_candidates
        self.bio_profile = bio_profile
        self.candidate_order = candidate_order
        self.candidate_seed = candidate_seed
        self.instruction = instruction
        self.number_candidates = number_candidates
        self.dtype = dtype

        self._model = None
        self._tokenizer = None
        self._layer_idx: int | None = None
        self._bio_extractor: BIOExtractor | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        amp_dtype = None
        if self.dtype == "auto":
            if self.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
        elif self.dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        elif self.dtype == "float16":
            amp_dtype = torch.float16
        elif self.dtype != "float32":
            raise ValueError("dtype must be one of {'auto', 'float32', 'float16', 'bfloat16'}.")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                trust_remote_code=True,
            )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token or self._tokenizer.unk_token

        base_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=amp_dtype,
            attn_implementation="eager",
        )
        self._layer_idx, _ = _resolve_layer_arg(self.layer_arg, base_model)

        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "peft is required to load LoRA adapters. "
                    "Install with: pip install peft>=0.10.0"
                )
            base_model = PeftModel.from_pretrained(base_model, self.adapter_path)

        base_model.to(self.device)
        base_model.eval()
        self._model = base_model

    def _get_bio_extractor(self) -> BIOExtractor:
        if self._bio_extractor is None:
            self._bio_extractor = BIOExtractor(self.bio_model_path, device=self.device)
        return self._bio_extractor

    def _score_candidates_once(
        self,
        text: str,
        candidates: Sequence[str],
    ) -> list[tuple[str, float]]:
        segment_text, spans = _build_segment_text(
            text,
            candidates,
            instruction=self.instruction,
            number_candidates=self.number_candidates,
        )
        encoded = self._tokenizer(
            segment_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
        usable: list[dict[str, Any]] = []
        for span in spans:
            token_span = _char_span_to_token_span(offsets, *span["char_span"])
            if token_span is None:
                continue
            start, end = token_span
            if end > int(encoded["input_ids"].shape[1]):
                continue
            usable.append({"text": span["text"], "token_span": token_span})
        if len(usable) < 2:
            return []

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            token_scores = _candidate_token_scores(outputs.attentions[self._layer_idx], attention_mask).float()

        scored: list[tuple[str, float]] = []
        for candidate in usable:
            start, end = candidate["token_span"]
            span_scores = token_scores[0, start:end]
            if span_scores.numel() == 0:
                continue
            scored.append((candidate["text"], float(span_scores.mean().detach().cpu())))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def extract_keywords_with_scores(
        self,
        text: str,
        top_k: int = 10,
        *,
        random_seeds: Sequence[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Extract keywords and return candidate scores.

        When ``random_seeds`` is provided and ``candidate_order='random'``, the
        extractor reruns the candidate shuffle for each seed and averages scores
        per candidate. This is the preferred path when you want to reduce the
        order-sensitivity observed in the original experiment.
        """
        self._ensure_loaded()
        if not text.strip():
            return []

        bio = self._get_bio_extractor()
        scored_candidates = bio.extract_spans_profile(text, profile=self.bio_profile)
        candidates = [candidate for candidate, _ in scored_candidates[: self.max_candidates]]
        if len(candidates) < 2:
            return []

        if self.candidate_order == "random":
            seeds = list(random_seeds) if random_seeds else [self.candidate_seed]
        else:
            seeds = [self.candidate_seed]

        aggregate: dict[str, list[float]] = {}
        for seed in seeds:
            ordered = list(candidates)
            if self.candidate_order == "random":
                random.Random(int(seed)).shuffle(ordered)
            for candidate, score in self._score_candidates_once(text, ordered):
                aggregate.setdefault(candidate, []).append(score)

        averaged = [
            (candidate, sum(scores) / len(scores))
            for candidate, scores in aggregate.items()
            if scores
        ]
        averaged.sort(key=lambda item: (-item[1], item[0]))
        return averaged[:top_k]

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        *,
        random_seeds: Sequence[int] | None = None,
    ) -> List[str]:
        return [candidate for candidate, _ in self.extract_keywords_with_scores(text, top_k=top_k, random_seeds=random_seeds)]

    def extract_keywords_batch(
        self,
        texts: Sequence[str],
        top_k: int = 10,
        *,
        random_seeds: Sequence[int] | None = None,
    ) -> List[List[str]]:
        return [self.extract_keywords(text, top_k=top_k, random_seeds=random_seeds) for text in texts]


__all__ = [
    "CandidateSegmentAttentionExtractor",
    "DEFAULT_SEGMENT_INSTRUCTION",
]
