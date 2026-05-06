"""QK LoRA keyword extraction using contrastive Q[EOS]·K[i] scoring.

Fine-tuned LoRA adapters on Q/K projection layers produce dot-product scores
that directly rank keyword tokens higher than non-keyword tokens.

Requires: torch, transformers
Optional: peft (for loading LoRA adapters)

Usage::

    from keyatten import QKLoRAExtractor

    extractor = QKLoRAExtractor(
        model="Qwen/Qwen3-Embedding-0.6B",
        adapter_path="models/qk_lora/best_adapter",
        device="cuda",
    )
    keywords = extractor.extract_keywords("一段中文文本", top_k=10)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..scoring import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX
from ..candidates.consolidation import ScoredCandidate
from ..candidates import (
    build_candidates,
    candidate_char_spans,
    char_scores_from_tokens,
    locate_word_offsets,
    segment_text,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


# ── Dependency checks ─────────────────────────────────────────────────


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for QK LoRA. Install with: pip install torch>=2.0")
    if AutoModel is None:
        raise ImportError("transformers is required for QK LoRA. Install with: pip install transformers>=4.30")


# ── Model introspection helpers ───────────────────────────────────────


def _resolve_transformer_layers(model) -> list | None:
    """Walk model attributes to find the list of Transformer layers."""
    candidates = [model]

    direct_model = getattr(model, "model", None)
    if direct_model is not None:
        candidates.append(direct_model)

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested_model = getattr(base_model, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)

    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return list(candidate.layers)
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return list(language_model.layers)
        nested_model = getattr(candidate, "model", None)
        if nested_model is not None and hasattr(nested_model, "layers"):
            return list(nested_model.layers)
    return None


def _resolve_qk_attention_module(layer):
    """Find the attention sub-module that exposes q_proj and k_proj."""
    for candidate in [
        getattr(layer, "self_attn", None),
        getattr(layer, "attention", None),
        layer,
    ]:
        if candidate is None:
            continue
        if hasattr(candidate, "q_proj") and hasattr(candidate, "k_proj"):
            return candidate
    return None


def _supported_qk_layer_indices(model) -> List[int]:
    resolved_layers = _resolve_transformer_layers(model)
    if not resolved_layers:
        return []
    return [
        idx for idx, layer in enumerate(resolved_layers)
        if _resolve_qk_attention_module(layer) is not None
    ]


def _recommended_layer_index(layer_count: int | None) -> int | None:
    if layer_count is None or layer_count <= 0:
        return None
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))


def _resolve_layer_arg(layer_arg: str, model) -> Tuple[int, int | None]:
    """Resolve a layer argument ('auto' or integer) to a concrete layer index."""
    layer_count = None
    for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(model.config, attr_name, None)
        if isinstance(value, int) and value > 0:
            layer_count = value
            break

    supported_qk_layers = _supported_qk_layer_indices(model)

    if layer_count is None:
        resolved_layers = _resolve_transformer_layers(model)
        if resolved_layers:
            layer_count = len(resolved_layers)

    if layer_arg.strip().lower() == "auto":
        if supported_qk_layers:
            raw_recommended = _recommended_layer_index(
                layer_count if layer_count is not None else len(supported_qk_layers)
            )
            if raw_recommended is None:
                return supported_qk_layers[-1], layer_count
            recommended = min(
                supported_qk_layers,
                key=lambda idx: (abs(idx - raw_recommended), -idx),
            )
            return recommended, layer_count
        recommended = _recommended_layer_index(layer_count)
        if recommended is None:
            if layer_count is None:
                return -1, None
            return layer_count - 1, layer_count
        return recommended, layer_count

    resolved = int(layer_arg)
    if layer_count is not None and resolved < 0:
        resolved = layer_count + resolved
    if layer_count is not None and (resolved < 0 or resolved >= layer_count):
        raise ValueError(f"Layer index {layer_arg} is out of range for {layer_count} layers.")
    if supported_qk_layers and resolved not in supported_qk_layers:
        supported_desc = ", ".join(str(idx) for idx in supported_qk_layers)
        raise ValueError(
            f"Layer index {resolved} does not expose q_proj/k_proj. "
            f"Supported layers: {supported_desc}"
        )
    return resolved, layer_count


# ── Core QK scoring ───────────────────────────────────────────────────


def compute_qk_scores(
    model,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    layer_idx: int,
) -> "torch.Tensor":
    """Extract Q[EOS]·K[i] dot-product scores from a specific layer.

    Returns: ``(batch, seq_len)`` score tensor.
    """
    _require_torch()
    q_store: dict = {}
    k_store: dict = {}

    resolved_layers = _resolve_transformer_layers(model)
    if not resolved_layers:
        raise AttributeError("Could not resolve transformer layers from model for QK scoring.")
    target_layer = _resolve_qk_attention_module(resolved_layers[layer_idx])
    if target_layer is None:
        raise AttributeError(f"Layer {layer_idx} does not expose q_proj/k_proj for QK scoring.")

    def q_hook(module, input, output):
        q_store["q"] = output

    def k_hook(module, input, output):
        k_store["k"] = output

    hq = target_layer.q_proj.register_forward_hook(q_hook)
    hk = target_layer.k_proj.register_forward_hook(k_hook)

    try:
        model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
    finally:
        hq.remove()
        hk.remove()

    Q = q_store["q"].float()
    K = k_store["k"].float()

    batch_size, seq_len, _ = Q.shape
    head_dim = target_layer.head_dim
    num_heads = Q.shape[-1] // head_dim
    num_kv_heads = K.shape[-1] // head_dim
    groups = max(1, num_heads // max(1, num_kv_heads))

    Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    K = K.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    K = K.repeat_interleave(groups, dim=1)

    eos_idx = attention_mask.sum(dim=1) - 1
    Q_eos = Q[torch.arange(batch_size), :, eos_idx, :].unsqueeze(2)
    scale = head_dim ** 0.5
    scores = (Q_eos * K).sum(dim=-1) / scale
    scores = scores.mean(dim=1)

    return scores


# ── Public API ────────────────────────────────────────────────────────


class QKLoRAExtractor:
    """Keyword extraction using QK LoRA contrastive scoring.

    This extractor loads a base embedding model and optionally a LoRA adapter
    trained with contrastive QK learning. Keywords are ranked by Q[EOS]·K[i]
    dot-product scores averaged over candidate character spans.

    Args:
        model: HuggingFace model name or local path.
        adapter_path: Path to the LoRA adapter directory (optional).
        language: ``'zh'`` or ``'en'``.
        device: PyTorch device string (``'cpu'``, ``'cuda'``).
        layer: Layer index for QK scoring. ``'auto'`` selects ~75% depth.
        max_length: Maximum token length for input truncation.
        instruction_prefix: Prefix prepended to input text. Defaults to the
            standard Chinese causal instruction prefix for ``language='zh'``.
        dtype: Model dtype — ``'auto'``, ``'float32'``, ``'float16'``, or ``'bfloat16'``.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        adapter_path: str | None = None,
        language: str = "zh",
        device: str = "cpu",
        layer: str = "auto",
        max_length: int = 512,
        instruction_prefix: str | None = None,
        dtype: str = "auto",
    ) -> None:
        _require_torch()
        if language not in {"zh", "en"}:
            raise ValueError("language must be 'zh' or 'en'.")

        self.model_name = model
        self.adapter_path = adapter_path
        self.language = language
        self.device = device
        self.layer_arg = layer
        self.max_length = max_length
        self.dtype = dtype

        if instruction_prefix is not None:
            self.instruction_prefix = instruction_prefix
        elif language == "zh":
            self.instruction_prefix = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX
        else:
            self.instruction_prefix = ""

        self._model = None
        self._tokenizer = None
        self._layer_idx: int | None = None

    def _encode_text(
        self,
        text: str,
    ) -> tuple["torch.Tensor", "torch.Tensor", list[tuple[int, int]], int]:
        full_text = self.instruction_prefix + text
        prefix_len = len(self.instruction_prefix)
        encoding = self._tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = [tuple(map(int, pair)) for pair in encoding["offset_mapping"][0].tolist()]
        return input_ids, attention_mask, offset_mapping, prefix_len

    def _qk_char_scores(
        self,
        text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
        input_ids, attention_mask, offset_mapping, prefix_len = self._encode_text(text)

        with torch.no_grad():
            scores = compute_qk_scores(self._model, input_ids, attention_mask, self._layer_idx)

        scores_np = scores[0].detach().cpu().numpy()
        text_token_offsets: list[tuple[int, int]] = []
        text_token_scores: list[float] = []
        for token_index, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_end <= tok_start or tok_start < prefix_len:
                continue
            char_start = max(0, tok_start - prefix_len)
            char_end = min(len(text), tok_end - prefix_len)
            if char_end <= char_start:
                continue
            text_token_offsets.append((char_start, char_end))
            text_token_scores.append(float(scores_np[token_index]))

        char_scores = char_scores_from_tokens(
            text_token_offsets,
            text_token_scores,
            len(text),
            normalize=False,
        )
        return char_scores, text_token_offsets, np.asarray(text_token_scores, dtype=np.float32)

    @staticmethod
    def _resolve_candidate_scores(
        text: str,
        words: Sequence[str],
        candidates,
        char_scores: np.ndarray,
    ) -> list[ScoredCandidate]:
        if not candidates:
            return []
        word_offsets = locate_word_offsets(text, words)
        spans = candidate_char_spans(candidates, word_offsets)
        return [
            ScoredCandidate(
                text=candidate.text,
                score=float(char_scores[span[0]:span[1]].mean()) if span[1] > span[0] else 0.0,
                span=span,
            )
            for candidate, span in zip(candidates, spans, strict=False)
        ]

    def _ensure_loaded(self) -> None:
        """Lazy-load the model, tokenizer, and optional LoRA adapter."""
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

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, trust_remote_code=True,
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False, trust_remote_code=True,
            )

        base_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=amp_dtype,
        )

        self._layer_idx, _layer_count = _resolve_layer_arg(self.layer_arg, base_model)

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

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from a single document.

        Args:
            text: Input document text.
            top_k: Maximum number of keywords to return.

        Returns:
            A list of keyword strings ranked by QK score.
        """
        self._ensure_loaded()

        words, pos_tags = segment_text(text, language=self.language)
        candidates = build_candidates(words, pos_tags, language=self.language)
        if not candidates:
            return []
        char_scores, _, _ = self._qk_char_scores(text)
        scored_candidates = self._resolve_candidate_scores(text, words, candidates, char_scores)
        ranked = sorted(scored_candidates, key=lambda item: (-item.score, item.span[0]))
        return [candidate.text for candidate in ranked[:top_k]]

    def extract_keywords_batch(self, texts: List[str], top_k: int = 10) -> List[List[str]]:
        """Extract keywords from multiple documents.

        Args:
            texts: List of input document texts.
            top_k: Maximum number of keywords per document.

        Returns:
            A list of keyword lists, one per input document.
        """
        return [self.extract_keywords(text, top_k=top_k) for text in texts]


__all__ = [
    "QKLoRAExtractor",
    "compute_qk_scores",
]
