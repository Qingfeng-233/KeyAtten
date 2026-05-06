from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from ..candidates.bio_head import (
    BIOBoundaryHead,
    bio_tags_to_spans,
    extract_keywords_relaxed,
    extract_keywords_relaxed_windowed,
    spans_to_text,
)
from ..candidates.bio_mining import mine_recall_oriented_phrases


_EDGE_PUNCT = "，。！？；：、（）()《》“”\"'‘’[]【】<>\n\r\t "
_EDGE_FUNCTION_WORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "并",
    "而",
    "或",
    "是",
    "在",
    "对",
    "把",
    "将",
    "被",
    "让",
    "给",
    "向",
    "从",
    "由",
    "于",
    "中",
    "上",
    "下",
    "里",
    "外",
}
_LEADING_VERB_CHARS = {
    "聊",
    "说",
    "讲",
    "谈",
    "论",
    "看",
    "听",
    "做",
    "搞",
    "用",
    "把",
    "将",
    "让",
    "给",
    "到",
    "向",
    "为",
    "从",
    "就",
    "来",
    "去",
    "要",
    "想",
    "能",
    "会",
    "该",
    "可",
    "被",
    "使",
    "令",
    "谓",
}
_TRAILING_FRAGMENT_CHARS = {
    "题",
    "分",
    "逐",
    "出",
    "下",
    "这",
    "的",
    "了",
    "着",
}
_ASCII_SHORT_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+\-_/]*$")
_CJK_SINGLE_CHAR = re.compile(r"^[\u4e00-\u9fff]$")


def _clean_span_text(text: str) -> str:
    candidate = text.strip(_EDGE_PUNCT)
    while len(candidate) > 1 and candidate[:1] in _EDGE_FUNCTION_WORDS:
        candidate = candidate[1:].lstrip(_EDGE_PUNCT)
    while len(candidate) > 1 and candidate[-1:] in _EDGE_FUNCTION_WORDS:
        candidate = candidate[:-1].rstrip(_EDGE_PUNCT)
    return candidate.strip(_EDGE_PUNCT)


def _strip_clean_span_edges(text: str) -> str:
    candidate = text
    while len(candidate) > 2 and candidate[:1] in _LEADING_VERB_CHARS:
        candidate = candidate[1:]
    while len(candidate) > 2 and candidate[-1:] in _TRAILING_FRAGMENT_CHARS:
        candidate = candidate[:-1]
    return candidate


def _is_clean_span_candidate(text: str) -> bool:
    if not text:
        return False
    if _CJK_SINGLE_CHAR.fullmatch(text):
        return False
    if len(text) == 1 and not _ASCII_SHORT_TOKEN.fullmatch(text):
        return False
    return True


def _postprocess_clean_spans(spans: list[tuple[str, float]]) -> list[tuple[str, float]]:
    dedup: dict[str, float] = {}
    for text, score in spans:
        cleaned = _clean_span_text(text)
        cleaned = _strip_clean_span_edges(cleaned)
        cleaned = _clean_span_text(cleaned)
        if not _is_clean_span_candidate(cleaned):
            continue
        previous = dedup.get(cleaned)
        if previous is None or score > previous:
            dedup[cleaned] = score
    return sorted(dedup.items(), key=lambda item: item[1], reverse=True)


def _merge_recall_phrases(
    text: str,
    spans: list[tuple[str, float]],
    *,
    phrase_limit: int = 80,
    base_score: float = 0.95,
) -> list[tuple[str, float]]:
    dedup: dict[str, tuple[str, float]] = {}
    for candidate, score in spans:
        key = candidate.lower()
        previous = dedup.get(key)
        if previous is None or score > previous[1]:
            dedup[key] = (candidate, score)

    for rank, phrase in enumerate(
        mine_recall_oriented_phrases(text, language="zh", max_phrases=phrase_limit)
    ):
        cleaned = _clean_span_text(phrase)
        cleaned = _strip_clean_span_edges(cleaned)
        cleaned = _clean_span_text(cleaned)
        if not _is_clean_span_candidate(cleaned) or cleaned not in text:
            continue
        score = base_score - rank * 1e-4
        key = cleaned.lower()
        previous = dedup.get(key)
        if previous is None or score > previous[1]:
            dedup[key] = (cleaned, score)
    return sorted(dedup.values(), key=lambda item: item[1], reverse=True)


class BIOExtractor:
    """Load a trained BIO boundary head and extract keyword candidates from text."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_name = ckpt["model_name"]
        layer_index = int(ckpt.get("layer_index", -1))
        freeze_backbone = bool(ckpt.get("freeze_backbone", True))
        self.max_length = int(ckpt.get("max_length", 512))
        trust_remote_code = bool(ckpt.get("trust_remote_code", True))

        model_state = ckpt.get("model_state_dict")
        local_config = checkpoint_path.parents[2] / "models" / "bert-base-chinese-test"
        tokenizer_name = ckpt.get("tokenizer_name", model_name)
        if model_state is not None and local_config.exists():
            tokenizer_name = str(local_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, local_files_only=local_config.exists()
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                if self.tokenizer.eos_token is not None
                else self.tokenizer.unk_token
            )

        init_model_name = model_name
        init_from_config = False
        if model_state is not None and local_config.exists():
            init_model_name = str(local_config)
            init_from_config = True

        self.model = BIOBoundaryHead(
            init_model_name,
            layer_index=layer_index,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
            init_from_config=init_from_config,
        )
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.classifier.load_state_dict(ckpt["classifier_state"])
            self.model.crf.load_state_dict(ckpt["crf_state"])
        self.model.to(self.device)
        self.model.eval()

    def extract_spans(self, text: str) -> list[str]:
        """Extract keyword candidates from text using BIO decoding.

        Returns a list of unique candidate strings (preserving first-occurrence order).
        """
        if not text.strip():
            return []

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            [encoded["attention_mask"]], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            result = self.model(input_ids=input_ids, attention_mask=attention_mask)
            emissions = result["emissions"]
            decoded_tags = self.model.decode(emissions, attention_mask)

        tags = decoded_tags[0]
        seq_len = int(attention_mask[0].sum().item())
        tags = tags[:seq_len]

        spans = bio_tags_to_spans(tags)
        offsets = encoded["offset_mapping"][:seq_len]
        keywords = spans_to_text(text, spans, offsets)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(kw)
        return unique

    def extract_spans_relaxed(
        self,
        text: str,
        max_spans: int = 20,
        b_threshold: float = 0.15,
        window_stride: int = 128,
        window_strides: Sequence[int] | None = None,
        threshold_schedule: Sequence[float] | None = None,
        max_expand_steps: int = 1,
        max_subspan_width: int = 0,
    ) -> list[tuple[str, float]]:
        """Extract more keyword candidates using emission probabilities.

        Instead of Viterbi (single best path), uses softmax on emissions to
        find all positions where P(B) > b_threshold, then extends with I tags.
        Returns (keyword, confidence) tuples sorted by confidence descending.
        Viterbi results are included first with a bonus to preserve their priority.
        """
        if not text.strip():
            return []

        return extract_keywords_relaxed_windowed(
            text,
            self.tokenizer,
            self.model,
            self.device,
            self.max_length,
            max_spans=max_spans,
            b_threshold=b_threshold,
            window_stride=window_stride,
            window_strides=window_strides,
            threshold_schedule=threshold_schedule,
            max_expand_steps=max_expand_steps,
            max_subspan_width=max_subspan_width,
        )

    def extract_spans_profile(
        self,
        text: str,
        *,
        profile: str = "balanced",
    ) -> list[tuple[str, float]]:
        profile_name = profile.strip().lower()
        if profile_name == "clean":
            spans = self.extract_spans_relaxed(
                text,
                max_spans=80,
                b_threshold=0.10,
                window_stride=128,
                window_strides=(64, 128),
                threshold_schedule=(0.15, 0.10, 0.05),
                max_expand_steps=1,
                max_subspan_width=0,
            )
            return _merge_recall_phrases(text, _postprocess_clean_spans(spans))[:80]
        if profile_name == "high_recall":
            return self.extract_spans_relaxed(
                text,
                max_spans=200,
                b_threshold=0.05,
                window_stride=128,
                window_strides=(64, 128, 192),
                threshold_schedule=(0.15, 0.10, 0.05, 0.0),
                max_expand_steps=1,
                max_subspan_width=6,
            )
        return self.extract_spans_relaxed(
            text,
            max_spans=120,
            b_threshold=0.08,
            window_stride=128,
            window_strides=(64, 128),
            threshold_schedule=(0.15, 0.10, 0.05),
            max_expand_steps=1,
            max_subspan_width=2,
        )

    def extract_spans_batch(self, texts: Sequence[str]) -> list[list[str]]:
        """Extract keyword candidates for a batch of texts."""
        return [self.extract_spans(text) for text in texts]
