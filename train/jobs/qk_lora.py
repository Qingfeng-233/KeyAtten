#!/usr/bin/env python3
"""
QK LoRA Training: Contrastive QK Learning on Qwen3-Embedding

Fine-tunes LoRA adapters on Q/K projection layers so that keyword tokens
receive higher Q[EOS]·K[i] dot-product scores than non-keyword tokens.

Training and inference are fully aligned:
  Train: Q[EOS]·K[i] → BCE loss (keyword tokens get high scores)
  Infer: Q[EOS]·K[i] → candidate ranking

Requirements:
  pip install torch transformers peft jieba

Data layout (place under benchmark/data/):
  data/train.tsv          — CSL training split (optional, unless --disable-csl)
  data/test.tsv           — CSL test split (optional, unless --disable-csl)
  data/shencecup/raw/     — ShenCeCup labeled docs
  data/multi_domain.jsonl  — (optional) LLM-annotated multi-domain JSONL

Usage:
  python train_qk_lora.py --model Qwen/Qwen3-Embedding-0.6B --epochs 20
  python train_qk_lora.py --disable-csl --md-max-keywords 6
  python train_qk_lora.py --smoke   # quick sanity check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as _exc:
    raise RuntimeError("peft is required. Install with: pip install peft>=0.10.0") from _exc

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODELS_ROOT = PROJECT_ROOT / "models"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from keyword_bench.data import (
    Document,
    load_csl_split,
    load_multi_domain_jsonl,
    load_shencecup_labeled,
)
from keyword_bench.metrics import evaluate_predictions
from keyatten.candidates import build_candidates, segment_text
from keyatten.candidates.bio_mining import find_candidate_occurrences
from keyatten.scoring import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX

DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_LAYER = "auto"
INSTRUCTION_PREFIX = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QK LoRA: Contrastive QK Learning.")
    p.add_argument(
        "--root-dir",
        default=None,
        help="Experiment root containing external/ and data/ (defaults: benchmark/, repo/, then repo/测试沙箱).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model name or path")
    p.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help="attention layer index for QK scoring, or 'auto' for the recommended middle-upper layer",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None, help="output directory (default: auto-generated)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience")
    p.add_argument("--max-length", type=int, default=512, help="max token length per document")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10, help="top-k for dev evaluation")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--dev-limit", type=int, default=None)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--shence-train-limit", type=int, default=None, help="optional cap for ShenCe training docs after split")
    p.add_argument("--shence-dev-limit", type=int, default=None, help="optional cap for ShenCe dev docs after split")
    p.add_argument("--shence-test-limit", type=int, default=None, help="optional cap for ShenCe held-out test docs after split")
    p.add_argument("--shence-test-pool-size", type=int, default=None, help="override held-out ShenCe pool size before train/dev/test split")
    p.add_argument("--split-seed", type=int, default=42, help="random seed for the ShenCe and multi-domain splits")
    p.add_argument("--md-max-keywords", type=int, default=4, help="cap keywords per multi-domain doc (0=no cap)")
    p.add_argument("--disable-csl", action="store_true", help="skip CSL train/test data entirely")
    p.add_argument("--disable-multi-domain", action="store_true", help="skip optional multi-domain training data even if present")
    p.add_argument("--smoke", action="store_true", help="quick sanity check with small data")
    p.add_argument("--bio-candidate-checkpoint", type=str, default=None, help="optional BIO checkpoint for candidate generation")
    p.add_argument(
        "--training-target",
        choices=("token", "candidate"),
        default="token",
        help="training supervision target: token BCE or candidate-level BCE on BIO candidates",
    )
    p.add_argument(
        "--bio-candidate-mode",
        choices=("auto", "explicit", "profile"),
        default="auto",
        help="candidate extraction mode: auto=explicit when params are provided, else profile",
    )
    p.add_argument("--bio-candidate-max-spans", type=int, default=None)
    p.add_argument("--bio-candidate-b-threshold", type=float, default=None)
    p.add_argument(
        "--bio-candidate-profile",
        choices=("balanced", "clean", "high_recall"),
        default="balanced",
    )
    p.add_argument("--bio-candidate-cache", type=str, default=None, help="optional JSONL cache for BIO candidates")
    p.add_argument("--write-bio-candidate-cache", action="store_true", help="write missing BIO candidates to cache")
    p.add_argument(
        "--candidate-label-mode",
        choices=("strict", "soft"),
        default="strict",
        help="candidate label policy: strict exact match or soft overlap labels",
    )
    return p.parse_args()


# ── Data preparation ──────────────────────────────────────────────────


def _iter_root_candidates(user_root: str | None) -> List[Path]:
    candidates = []
    if user_root:
        candidates.append(Path(user_root))
    candidates.extend(
        [
            BENCHMARK_DIR,
            BENCHMARK_DIR.parent,
            BENCHMARK_DIR.parent / "测试沙箱",
        ]
    )

    resolved: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            normalized = candidate.resolve()
        except FileNotFoundError:
            normalized = candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _resolve_csl_paths(root_dir: str | None) -> tuple[Path, Path]:
    for root in _iter_root_candidates(root_dir):
        direct_root = root / "external" / "CSL" / "benchmark" / "kg"
        if (direct_root / "train.tsv").exists() and (direct_root / "test.tsv").exists():
            return direct_root / "train.tsv", direct_root / "test.tsv"

        alt_root = root / "data" / "CSL" / "benchmark" / "kg"
        if (alt_root / "train.tsv").exists() and (alt_root / "test.tsv").exists():
            return alt_root / "train.tsv", alt_root / "test.tsv"

    searched = ", ".join(str(path) for path in _iter_root_candidates(root_dir))
    raise FileNotFoundError(
        "Could not locate CSL benchmark files. Searched roots: "
        f"{searched}"
    )


def _resolve_shence_root(root_dir: str | None) -> Path:
    for root in _iter_root_candidates(root_dir):
        raw_dir = root / "data" / "shencecup" / "raw"
        if (raw_dir / "all_docs.txt").exists() and (raw_dir / "train_docs_keywords.txt").exists():
            return root

    searched = ", ".join(str(path) for path in _iter_root_candidates(root_dir))
    raise FileNotFoundError(
        "Could not locate ShenCeCup labeled data. Searched roots: "
        f"{searched}"
    )


def _resolve_multi_domain_path(root_dir: str | None) -> Path | None:
    for root in _iter_root_candidates(root_dir):
        candidates = [
            root / "data" / "multi_domain.jsonl",
            root / "data" / "multi_domain" / "annotated.jsonl",
            root / "benchmark" / "data" / "multi_domain.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _recommended_layer_index(layer_count: int | None) -> int | None:
    if layer_count is None or layer_count <= 0:
        return None
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))


def _resolve_transformer_layers(model) -> list | None:
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
    candidates = [
        getattr(layer, "self_attn", None),
        getattr(layer, "attention", None),
        layer,
    ]
    for candidate in candidates:
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


def _resolve_layer_arg(layer_arg: str, model) -> tuple[int, int | None]:
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
            raw_recommended = _recommended_layer_index(layer_count if layer_count is not None else len(supported_qk_layers))
            if raw_recommended is None:
                return supported_qk_layers[-1], layer_count
            recommended = min(supported_qk_layers, key=lambda idx: (abs(idx - raw_recommended), -idx))
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
            f"Layer index {resolved} does not expose q_proj/k_proj for QK scoring. "
            f"Supported layers: {supported_desc}"
        )
    return resolved, layer_count


def _gpu_memory_snapshot(device: str) -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "peak_memory_allocated_gb": None,
            "peak_memory_reserved_gb": None,
        }
    return {
        "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        "peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 4),
    }


def _keyword_char_spans(text: str, keywords: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for kw in keywords:
        if not kw:
            continue
        pos = 0
        while True:
            idx = text.find(kw, pos)
            if idx == -1:
                break
            spans.append((idx, idx + len(kw)))
            pos = idx + 1
    return spans


def tokenize_and_label(
    text: str,
    keywords: List[str],
    tokenizer,
    instruction_prefix: str,
    max_length: int,
) -> dict:
    full_text = instruction_prefix + text
    prefix_len = len(instruction_prefix)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )

    offset_mapping: List[Tuple[int, int]] = encoding["offset_mapping"]
    input_ids: List[int] = encoding["input_ids"]
    attention_mask: List[int] = encoding["attention_mask"]

    kw_spans_in_full = [
        (s + prefix_len, e + prefix_len)
        for s, e in _keyword_char_spans(text, keywords)
    ]

    seq_len = len(input_ids)
    labels = [-1.0] * seq_len

    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue
        if tok_start < prefix_len:
            continue
        labels[tok_idx] = 0.0
        for kw_start, kw_end in kw_spans_in_full:
            if tok_start < kw_end and tok_end > kw_start:
                labels[tok_idx] = 1.0
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
    }


def _extract_bio_candidates(
    bio_extractor,
    text: str,
    bio_candidate_mode: str,
    bio_candidate_max_spans: Optional[int],
    bio_candidate_b_threshold: Optional[float],
    bio_candidate_profile: str,
) -> List[Tuple[str, float]]:
    use_explicit = bio_candidate_mode == "explicit"
    if bio_candidate_mode == "auto":
        use_explicit = (
            bio_candidate_max_spans is not None
            or bio_candidate_b_threshold is not None
        )
    if use_explicit:
        max_spans = 50 if bio_candidate_max_spans is None else bio_candidate_max_spans
        b_threshold = 0.15 if bio_candidate_b_threshold is None else bio_candidate_b_threshold
        return bio_extractor.extract_spans_relaxed(
            text,
            max_spans=max_spans,
            b_threshold=b_threshold,
        )
    return bio_extractor.extract_spans_profile(
        text,
        profile=bio_candidate_profile,
    )


def _doc_cache_key(doc: Document) -> str:
    digest = hashlib.sha1(doc.text.encode("utf-8")).hexdigest()
    return f"{doc.doc_id}:{digest}"


def _load_bio_candidate_cache(path: str | None) -> Dict[str, List[Tuple[str, float]]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    cache: Dict[str, List[Tuple[str, float]]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            candidates = item.get("candidates", [])
            cache[item["key"]] = [(str(text), float(score)) for text, score in candidates]
    return cache


def _write_bio_candidate_cache(path: str | None, cache: Dict[str, List[Tuple[str, float]]]) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for key, candidates in sorted(cache.items()):
            handle.write(json.dumps({"key": key, "candidates": candidates}, ensure_ascii=False) + "\n")


def _candidate_label(candidate_text: str, gold_texts: set[str], mode: str) -> float:
    if candidate_text in gold_texts:
        return 1.0
    if mode == "strict":
        return 0.0
    for gold_text in gold_texts:
        if len(gold_text) >= 2 and gold_text in candidate_text:
            return 0.8
        if len(candidate_text) >= 2 and candidate_text in gold_text:
            return 0.6
    return 0.0


def _char_span_to_token_span(
    offset_mapping: Sequence[Tuple[int, int]],
    prefix_len: int,
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    token_indices: List[int] = []
    full_start = char_start + prefix_len
    full_end = char_end + prefix_len
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end or tok_start < prefix_len:
            continue
        if tok_start < full_end and tok_end > full_start:
            token_indices.append(tok_idx)
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def build_candidate_supervision(
    text: str,
    keywords: Sequence[str],
    offset_mapping: Sequence[Tuple[int, int]],
    prefix_len: int,
    *,
    bio_extractor,
    bio_candidate_mode: str,
    bio_candidate_max_spans: Optional[int],
    bio_candidate_b_threshold: Optional[float],
    bio_candidate_profile: str,
    candidate_cache_entry: Optional[List[Tuple[str, float]]] = None,
    candidate_label_mode: str = "strict",
) -> List[dict]:
    candidate_specs: List[dict] = []
    seen_texts: set[str] = set()
    gold_texts = {kw.strip() for kw in keywords if kw and kw.strip()}
    raw_candidates = candidate_cache_entry
    if raw_candidates is None:
        raw_candidates = _extract_bio_candidates(
            bio_extractor,
            text,
            bio_candidate_mode,
            bio_candidate_max_spans,
            bio_candidate_b_threshold,
            bio_candidate_profile,
        )
    for candidate_text, bio_score in raw_candidates:
        normalized_text = candidate_text.strip()
        if not normalized_text or normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        token_spans: List[Tuple[int, int]] = []
        for char_start, char_end in find_candidate_occurrences(text, normalized_text):
            token_span = _char_span_to_token_span(offset_mapping, prefix_len, char_start, char_end)
            if token_span is not None:
                token_spans.append(token_span)
        if not token_spans:
            continue
        candidate_specs.append(
            {
                "text": normalized_text,
                "label": _candidate_label(normalized_text, gold_texts, candidate_label_mode),
                "bio_score": float(bio_score),
                "token_spans": token_spans,
            }
        )
    return candidate_specs


class TokenDataset(Dataset):
    def __init__(
        self,
        docs,
        tokenizer,
        instruction_prefix: str,
        max_length: int,
        *,
        training_target: str = "token",
        bio_extractor=None,
        bio_candidate_mode: str = "auto",
        bio_candidate_max_spans: Optional[int] = None,
        bio_candidate_b_threshold: Optional[float] = None,
        bio_candidate_profile: str = "balanced",
        bio_candidate_cache: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        write_bio_candidate_cache: bool = False,
        candidate_label_mode: str = "strict",
    ) -> None:
        self.items: List[dict] = []
        for index, doc in enumerate(docs, start=1):
            item = tokenize_and_label(doc.text, doc.keywords, tokenizer, instruction_prefix, max_length)
            item["doc"] = doc
            if training_target == "candidate":
                if bio_extractor is None:
                    raise ValueError("Candidate-level training requires --bio-candidate-checkpoint.")
                cache_key = _doc_cache_key(doc)
                cache_entry = bio_candidate_cache.get(cache_key) if bio_candidate_cache is not None else None
                if cache_entry is None and write_bio_candidate_cache and bio_candidate_cache is not None:
                    cache_entry = _extract_bio_candidates(
                        bio_extractor,
                        doc.text,
                        bio_candidate_mode,
                        bio_candidate_max_spans,
                        bio_candidate_b_threshold,
                        bio_candidate_profile,
                    )
                    bio_candidate_cache[cache_key] = cache_entry
                item["candidate_supervision"] = build_candidate_supervision(
                    doc.text,
                    doc.keywords,
                    item["offset_mapping"],
                    len(instruction_prefix),
                    bio_extractor=bio_extractor,
                    bio_candidate_mode=bio_candidate_mode,
                    bio_candidate_max_spans=bio_candidate_max_spans,
                    bio_candidate_b_threshold=bio_candidate_b_threshold,
                    bio_candidate_profile=bio_candidate_profile,
                    candidate_cache_entry=cache_entry,
                    candidate_label_mode=candidate_label_mode,
                )
                if index % 50 == 0 or index == len(docs):
                    print(f"[pretokenize] candidate docs {index}/{len(docs)}")
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _collate_fn(batch: List[dict]) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list, attention_mask_list, labels_list = [], [], []
    candidate_supervision_list = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids_list.append(item["input_ids"] + [0] * pad)
        attention_mask_list.append(item["attention_mask"] + [0] * pad)
        labels_list.append(item["labels"] + [-1.0] * pad)
        candidate_supervision_list.append(item.get("candidate_supervision", []))
    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.float32),
        "candidate_supervision": candidate_supervision_list,
    }


# ── QK scoring ────────────────────────────────────────────────────────


def compute_qk_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Extract Q[EOS]·K[i] dot-product scores from a specific layer.

    Returns: (batch, seq_len) score tensor.
    """
    q_store = {}
    k_store = {}

    resolved_layers = _resolve_transformer_layers(model)
    if not resolved_layers:
        raise AttributeError("Could not resolve transformer layers from model for QK scoring.")
    target_layer = _resolve_qk_attention_module(resolved_layers[layer_idx])
    if target_layer is None:
        raise AttributeError(
            f"Layer {layer_idx} does not expose q_proj/k_proj for QK scoring."
        )

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


# ── Evaluation ────────────────────────────────────────────────────────


def score_candidates_with_qk(
    doc: Document,
    tokenizer,
    model,
    device: str,
    instruction_prefix: str,
    max_length: int,
    layer_idx: int,
    top_k: int,
    bio_extractor=None,
    bio_candidate_mode: str = "auto",
    bio_candidate_max_spans: Optional[int] = None,
    bio_candidate_b_threshold: Optional[float] = None,
    bio_candidate_profile: str = "balanced",
) -> List[str]:
    model.eval()

    if bio_extractor is not None:
        candidates = [
            text
            for text, _ in _extract_bio_candidates(
                bio_extractor,
                doc.text,
                bio_candidate_mode,
                bio_candidate_max_spans,
                bio_candidate_b_threshold,
                bio_candidate_profile,
            )
        ]
    else:
        words, pos_tags = segment_text(doc.text, language="zh")
        candidates = [candidate.text for candidate in build_candidates(words, pos_tags, language="zh")]
    if not candidates:
        return []

    full_text = instruction_prefix + doc.text
    prefix_len = len(instruction_prefix)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        scores = compute_qk_scores(model, input_ids, attention_mask, layer_idx)

    scores = scores[0].cpu().numpy()

    char_to_score: Dict[int, float] = {}
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end or tok_start < prefix_len:
            continue
        char_start = tok_start - prefix_len
        char_end = tok_end - prefix_len
        sc = float(scores[tok_idx])
        for c in range(char_start, char_end):
            if c not in char_to_score or sc > char_to_score[c]:
                char_to_score[c] = sc

    candidate_scores: Dict[str, float] = {}
    for candidate_text in candidates:
        spans = find_candidate_occurrences(doc.text, candidate_text)
        if not spans:
            continue
        best_score = None
        for char_start, char_end in spans:
            cand_scores = [char_to_score[c] for c in range(char_start, char_end) if c in char_to_score]
            score = sum(cand_scores) / len(cand_scores) if cand_scores else 0.0
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            candidate_scores[candidate_text] = best_score

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
    return [text for text, _ in sorted_candidates[:top_k]]


def evaluate_f1_with_qk(
    model,
    tokenizer,
    docs: List[Document],
    device: str,
    instruction_prefix: str,
    max_length: int,
    layer_idx: int,
    top_k: int,
    bio_extractor=None,
    bio_candidate_mode: str = "auto",
    bio_candidate_max_spans: Optional[int] = None,
    bio_candidate_b_threshold: Optional[float] = None,
    bio_candidate_profile: str = "balanced",
) -> Dict[str, float]:
    predictions = []
    golds = []
    for doc in docs:
        pred = score_candidates_with_qk(
            doc, tokenizer, model, device,
            instruction_prefix, max_length, layer_idx, top_k,
            bio_extractor=bio_extractor,
            bio_candidate_mode=bio_candidate_mode,
            bio_candidate_max_spans=bio_candidate_max_spans,
            bio_candidate_b_threshold=bio_candidate_b_threshold,
            bio_candidate_profile=bio_candidate_profile,
        )
        predictions.append(pred)
        golds.append(doc.keywords)
    return evaluate_predictions(predictions, golds)


def compute_candidate_bce_loss(
    scores: torch.Tensor,
    candidate_supervision: Sequence[Sequence[dict]],
    criterion: nn.Module,
) -> Optional[torch.Tensor]:
    candidate_losses: List[torch.Tensor] = []
    for batch_idx, batch_candidates in enumerate(candidate_supervision):
        for candidate in batch_candidates:
            occurrence_scores: List[torch.Tensor] = []
            for start_idx, end_idx in candidate["token_spans"]:
                if end_idx <= start_idx:
                    continue
                span_scores = scores[batch_idx, start_idx:end_idx]
                if span_scores.numel() == 0:
                    continue
                occurrence_scores.append(span_scores.mean())
            if not occurrence_scores:
                continue
            candidate_score = torch.stack(occurrence_scores).max()
            label = torch.tensor([candidate["label"]], dtype=torch.float32, device=scores.device)
            candidate_losses.append(criterion(candidate_score.unsqueeze(0), label).mean())
    if not candidate_losses:
        return None
    return torch.stack(candidate_losses).mean()


# ── Training ──────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_ROOT / "qk_lora" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output dir: {output_dir}")

    bio_extractor = None
    if args.bio_candidate_checkpoint:
        from keyatten import BIOExtractor

        bio_extractor = BIOExtractor(
            checkpoint_path=args.bio_candidate_checkpoint,
            device="cuda" if args.device == "cuda" else "cpu",
        )
        print(f"[info] Using BIO candidate checkpoint: {args.bio_candidate_checkpoint}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available, falling back to CPU")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_limit = 100 if args.smoke else args.train_limit
    dev_limit = 50 if args.smoke else args.dev_limit
    test_limit = 100 if args.smoke else args.test_limit
    shence_dev_limit = 25 if args.smoke else args.shence_dev_limit
    shence_test_limit = 25 if args.smoke else args.shence_test_limit

    print(f"[info] Model: {args.model}")
    print(f"[info] Layer: {args.layer}")
    if args.training_target == "candidate":
        print("[info] Method: Candidate-level QK BCE on BIO candidate spans")
    else:
        print(f"[info] Method: Contrastive QK Learning (Q[EOS]·K[i] BCE)")

    # ── Load data ──

    _split_rng = random.Random(args.split_seed)

    if args.disable_csl:
        print("[info] CSL disabled by flag, skipping")
        csl_train, csl_test = [], []
    else:
        print("[info] Loading CSL data...")
        csl_train_path, csl_test_path = _resolve_csl_paths(args.root_dir)
        csl_limit = train_limit if train_limit else 2000
        csl_train = load_csl_split(csl_train_path, "train", limit=csl_limit)
        csl_test = load_csl_split(csl_test_path, "test", limit=test_limit)

    print("[info] Loading ShenCeCup data...")
    shence_root = _resolve_shence_root(args.root_dir)
    shence_all = load_shencecup_labeled(shence_root)
    _split_rng.shuffle(shence_all)
    shence_test_size = args.shence_test_pool_size if args.shence_test_pool_size is not None else min(200, len(shence_all) // 5)
    shence_test = shence_all[:shence_test_size]
    shence_train = shence_all[shence_test_size:]
    dev_docs = shence_test[:shence_test_size // 2]
    shence_test_final = shence_test[shence_test_size // 2:]

    if args.shence_train_limit is not None:
        shence_train = shence_train[: args.shence_train_limit]
    if shence_dev_limit is not None:
        dev_docs = dev_docs[:shence_dev_limit]
    if shence_test_limit is not None:
        shence_test_final = shence_test_final[:shence_test_limit]

    multi_domain_path = None if args.disable_multi_domain else _resolve_multi_domain_path(args.root_dir)
    if multi_domain_path is not None and multi_domain_path.exists():
        print("[info] Loading multi-domain data...")
        md_all = load_multi_domain_jsonl(multi_domain_path)
        _split_rng.shuffle(md_all)
        md_test_size = min(1000, len(md_all) // 5)
        md_test = md_all[:md_test_size]
        md_train = md_all[md_test_size:]
    else:
        print("[info] Multi-domain data not found or disabled, skipping")
        md_train, md_test = [], []

    # ── Extractive filter: only keep keywords that appear in source text ──

    def _filter_extractive(docs: list, name: str) -> list:
        total_kw, kept_kw, dropped_docs = 0, 0, 0
        filtered = []
        for doc in docs:
            extractive = [kw for kw in doc.keywords if kw in doc.text]
            total_kw += len(doc.keywords)
            kept_kw += len(extractive)
            if extractive:
                d = copy(doc)
                d.keywords = extractive
                filtered.append(d)
            else:
                dropped_docs += 1
        kept_ratio = 0.0 if total_kw == 0 else kept_kw / total_kw
        print(f"[info] {name}: keywords {kept_kw}/{total_kw} kept ({kept_ratio:.1%}), {dropped_docs} docs dropped")
        return filtered

    csl_train = _filter_extractive(csl_train, "CSL train")
    shence_train = _filter_extractive(shence_train, "ShenCe train")
    md_train = _filter_extractive(md_train, "MD train")

    # Cap multi-domain keywords per doc
    if args.md_max_keywords > 0 and md_train:
        capped = 0
        capped_docs = []
        for doc in md_train:
            if len(doc.keywords) > args.md_max_keywords:
                d = copy(doc)
                d.keywords = doc.keywords[:args.md_max_keywords]
                capped_docs.append(d)
                capped += 1
            else:
                capped_docs.append(doc)
        md_train = capped_docs
        print(f"[info] MD train: capped {capped}/{len(md_train)} docs to max {args.md_max_keywords} keywords")

    train_docs = csl_train + shence_train + md_train
    _split_rng.shuffle(train_docs)
    if train_limit is not None:
        train_docs = train_docs[:train_limit]
    print(f"[info] === Data Split (seed=42, extractive-filtered) ===")
    print(f"[info] Train: {len(train_docs)} (CSL={len(csl_train)}, ShenCe={len(shence_train)}, MD={len(md_train)})")
    print(f"[info] Dev (ShenCe): {len(dev_docs)}")
    print(f"[info] Test: ShenCe={len(shence_test_final)}, MD={len(md_test)}, CSL={len(csl_test)}")

    # ── Tokenize & build dataset ──

    print(f"[info] Loading tokenizer from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception as exc:
        print(f"[warn] Failed to load fast tokenizer ({exc}), falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    print("[info] Pre-tokenizing training data...")
    bio_candidate_cache = _load_bio_candidate_cache(args.bio_candidate_cache)
    if args.bio_candidate_cache:
        print(f"[info] BIO candidate cache loaded: {len(bio_candidate_cache)} docs")
    train_dataset = TokenDataset(
        train_docs,
        tokenizer,
        INSTRUCTION_PREFIX,
        args.max_length,
        training_target=args.training_target,
        bio_extractor=bio_extractor,
        bio_candidate_mode=args.bio_candidate_mode,
        bio_candidate_max_spans=args.bio_candidate_max_spans,
        bio_candidate_b_threshold=args.bio_candidate_b_threshold,
        bio_candidate_profile=args.bio_candidate_profile,
        bio_candidate_cache=bio_candidate_cache,
        write_bio_candidate_cache=args.write_bio_candidate_cache,
        candidate_label_mode=args.candidate_label_mode,
    )
    if args.write_bio_candidate_cache and args.bio_candidate_cache:
        _write_bio_candidate_cache(args.bio_candidate_cache, bio_candidate_cache)
        print(f"[info] BIO candidate cache written: {args.bio_candidate_cache} ({len(bio_candidate_cache)} docs)")

    if args.training_target == "candidate":
        pos_count = sum(
            1
            for item in train_dataset.items
            for candidate in item.get("candidate_supervision", [])
            if candidate["label"] > 0.5
        )
        neg_count = sum(
            1
            for item in train_dataset.items
            for candidate in item.get("candidate_supervision", [])
            if candidate["label"] <= 0.5
        )
        print(f"[info] Candidate labels: {pos_count} pos / {neg_count} neg", end="")
    else:
        pos_count = sum(1 for item in train_dataset.items for lbl in item["labels"] if lbl > 0.5)
        neg_count = sum(1 for item in train_dataset.items for lbl in item["labels"] if 0.0 <= lbl <= 0.5)
        print(f"[info] Token labels: {pos_count} pos / {neg_count} neg", end="")
    pos_weight = neg_count / max(pos_count, 1)
    print(f" → pos_weight={pos_weight:.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ── Model setup ──

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[info] AMP: {'bfloat16' if use_amp else 'disabled'}")

    print(f"[info] Loading base model from {args.model}...")
    base_model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else None,
    )
    layer_idx, layer_count = _resolve_layer_arg(args.layer, base_model)
    if layer_count is not None:
        print(f"[info] Resolved QK layer: {layer_idx} / {layer_count - 1}")
    else:
        print(f"[info] Resolved QK layer: {layer_idx}")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model.to(device)
    total_params = sum(param.numel() for param in lora_model.parameters())
    trainable_params = sum(param.numel() for param in lora_model.parameters() if param.requires_grad)
    print(f"[info] Parameters: trainable={trainable_params:,} total={total_params:,}")

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # ── Resume from checkpoint ──

    training_log = []
    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    start_epoch = 1

    checkpoint_path = output_dir / "checkpoint.pt"
    latest_adapter_dir = output_dir / "latest_adapter"
    if checkpoint_path.exists() and latest_adapter_dir.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        training_log = ckpt["training_log"]
        lora_model.load_adapter(str(latest_adapter_dir), adapter_name="default")
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        tqdm.write(f"[info] Resumed from epoch {ckpt['epoch']}, best F1@10={best_f1:.4f}")

    # ── Training loop ──

    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        lora_model.train()
        total_loss = 0.0
        steps = 0
        epoch_start = time.perf_counter()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    ncols=100, ascii=True, leave=False, mininterval=2.0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                scores = compute_qk_scores(lora_model, input_ids, attention_mask, layer_idx)
                if args.training_target == "candidate":
                    loss = compute_candidate_bce_loss(
                        scores,
                        batch["candidate_supervision"],
                        criterion,
                    )
                    if loss is None:
                        continue
                else:
                    valid_mask = labels >= 0.0
                    if not valid_mask.any():
                        continue
                    loss = criterion(scores[valid_mask], labels[valid_mask]).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / steps:.4f}")

        avg_loss = total_loss / max(steps, 1)
        tqdm.write(f"[epoch {epoch}/{args.epochs}] loss={avg_loss:.4f} | evaluating dev...")

        dev_metrics = evaluate_f1_with_qk(
            lora_model, tokenizer, dev_docs, device,
            INSTRUCTION_PREFIX, args.max_length, layer_idx, args.top_k,
            bio_extractor=bio_extractor,
            bio_candidate_mode=args.bio_candidate_mode,
            bio_candidate_max_spans=args.bio_candidate_max_spans,
            bio_candidate_b_threshold=args.bio_candidate_b_threshold,
            bio_candidate_profile=args.bio_candidate_profile,
        )
        f1_10 = dev_metrics.get("f1@10", 0.0)
        f1_5 = dev_metrics.get("f1@5", 0.0)
        tqdm.write(f"[epoch {epoch}] dev: F1@10={f1_10:.4f} F1@5={f1_5:.4f}")

        epoch_seconds = round(time.perf_counter() - epoch_start, 2)
        memory_stats = _gpu_memory_snapshot(device)
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_f1@10": f1_10,
            "dev_f1@5": f1_5,
            "epoch_seconds": epoch_seconds,
            **memory_stats,
        }
        training_log.append(log_entry)
        (output_dir / "training_log.json").write_text(
            json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if f1_10 > best_f1:
            best_f1 = f1_10
            best_epoch = epoch
            patience_counter = 0
            adapter_dir = output_dir / "best_adapter"
            adapter_dir.mkdir(exist_ok=True)
            lora_model.save_pretrained(str(adapter_dir))
            tqdm.write(f"[epoch {epoch}] *** New best F1@10={best_f1:.4f} — adapter saved ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"[info] Early stopping at epoch {epoch} (patience={args.patience})")
                break

        latest_adapter_dir.mkdir(exist_ok=True)
        lora_model.save_pretrained(str(latest_adapter_dir))
        torch.save({
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "training_log": training_log,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, str(checkpoint_path))

    # ── Final evaluation ──

    print(f"\n[info] Best epoch={best_epoch}, dev F1@10={best_f1:.4f}")
    print("[info] Final evaluation on held-out test sets...")
    total_training_seconds = round(time.perf_counter() - training_start, 2)

    if (output_dir / "best_adapter").exists():
        lora_model.load_adapter(str(output_dir / "best_adapter"), adapter_name="default")

    test_sets = {}
    if shence_test_final:
        test_sets["shence_test"] = shence_test_final
    if md_test:
        test_sets["md_test"] = md_test
    if csl_test:
        test_sets["csl_test"] = csl_test

    all_test_metrics = {}
    for ts_name, ts_docs in test_sets.items():
        metrics = evaluate_f1_with_qk(
            lora_model, tokenizer, ts_docs, device,
            INSTRUCTION_PREFIX, args.max_length, layer_idx, args.top_k,
            bio_extractor=bio_extractor,
            bio_candidate_mode=args.bio_candidate_mode,
            bio_candidate_max_spans=args.bio_candidate_max_spans,
            bio_candidate_b_threshold=args.bio_candidate_b_threshold,
            bio_candidate_profile=args.bio_candidate_profile,
        )
        f1_10 = metrics.get("f1@10", 0.0)
        f1_5 = metrics.get("f1@5", 0.0)
        print(f"[test/{ts_name}] F1@5={f1_5:.4f}  F1@10={f1_10:.4f}  ({len(ts_docs)} docs)")
        all_test_metrics[ts_name] = metrics

    final_eval = {
        "model": args.model,
        "layer": layer_idx,
        "requested_layer": args.layer,
        "attention_layer_count": layer_count,
        "best_epoch": best_epoch,
        "best_dev_f1@10": best_f1,
        "bio_candidate_mode": args.bio_candidate_mode,
        "bio_candidate_profile": args.bio_candidate_profile,
        "total_training_seconds": total_training_seconds,
        "test_metrics": all_test_metrics,
        "data_split": {
            "train": len(train_docs),
            "dev_shence": len(dev_docs),
            "test_shence": len(shence_test_final),
            "test_md": len(md_test),
            "test_csl": len(csl_test),
            "split_seed": args.split_seed,
            "shence_test_pool_size": shence_test_size,
        },
        "config": {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target": "q_proj + k_proj",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "pos_weight": pos_weight,
            "candidate_label_mode": args.candidate_label_mode,
            "training_method": "candidate_qk_bce" if args.training_target == "candidate" else "contrastive_qk",
        },
        "model_stats": {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "best_epoch_log": next((item for item in training_log if item["epoch"] == best_epoch), None),
        },
    }
    (output_dir / "final_eval.json").write_text(
        json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[info] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
