#!/usr/bin/env python3
"""
Attention LoRA Training: Teach the model to attend to keyword tokens.

Instead of QK readout (Q[EOS]·K[i]), directly supervise the EOS token's
attention distribution so that attention mass concentrates on keyword tokens.

Train: KL(keyword_target || EOS_attention_to_text_tokens)
Infer: received_attn on BIO clean candidates
"""
from __future__ import annotations

import argparse, json, random, sys, time
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
    raise RuntimeError("peft required. pip install peft>=0.10.0") from _exc

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODELS_ROOT = PROJECT_ROOT / "models"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from keyword_bench.data import (
    Document, load_csl_split, load_multi_domain_jsonl, load_shencecup_labeled,
)
from keyword_bench.metrics import evaluate_predictions
from keyatten.candidates.bio_mining import find_candidate_occurrences
from keyatten.scoring import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX

DATA_DIR = BENCHMARK_DIR / "data"
INSTRUCTION_PREFIX = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX


def parse_args():
    p = argparse.ArgumentParser(description="Attention LoRA: EOS-to-keyword attention supervision.")
    p.add_argument("--root-dir", default=None)
    p.add_argument("--model", default=str(MODELS_ROOT / "Qwen3-Embedding-0.6B"))
    p.add_argument("--layer", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--dev-limit", type=int, default=None)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--md-max-keywords", type=int, default=4)
    p.add_argument("--disable-csl", action="store_true")
    p.add_argument("--disable-multi-domain", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--bio-candidate-checkpoint", type=str,
                   default=str(PROJECT_ROOT / "train" / "remote_pull_resume16_epoch13" / "best_full_ckpt.pt"))
    p.add_argument("--bio-candidate-profile", choices=("balanced", "clean", "high_recall"), default="clean")
    p.add_argument("--loss-type", choices=("kl", "bce", "focal", "ranking"), default="ranking",
                   help="attention supervision loss type")
    p.add_argument("--ranking-margin", type=float, default=0.3, help="margin for ranking loss")
    p.add_argument("--loss-target", choices=("eos_row", "col_sum"), default="eos_row",
                   help="which attention signal to supervise: eos_row=EOS-to-token, col_sum=received_attn")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="focal loss gamma")
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj",
                   help="comma-separated LoRA target modules")
    p.add_argument("--attn-method", choices=("received_attn", "samrank", "eos_attn"), default="received_attn",
                   help="attention method for inference evaluation")
    p.add_argument("--bio-aux-weight", type=float, default=0.0)
    p.add_argument("--bio-aux-margin", type=float, default=0.1)
    p.add_argument("--bio-aux-max-candidates", type=int, default=40)
    p.add_argument("--bio-aux-label-mode", choices=("strict", "soft"), default="soft")
    p.add_argument("--bio-aux-loss-type", choices=("ranking", "kl"), default="ranking",
                   help="candidate-level loss: ranking=pairwise hinge, kl=soft KL on candidate softmax")
    p.add_argument("--attn-loss-weight", type=float, default=1.0,
                   help="weight for token-level attn_loss; set 0 to disable (use only candidate-level bio aux)")
    p.add_argument("--tasc-mode", choices=("none", "lin", "hidden"), default="none",
                   help="TaSc rescaling: lin=per-vocab scalar gate, hidden=MLP on hidden states")
    p.add_argument("--soft-target", action="store_true",
                   help="enable label smoothing for KL target")
    p.add_argument("--soft-alpha", type=float, default=0.1,
                   help="label smoothing alpha (0=hard target)")
    return p.parse_args()


# ── Helpers (reused from train_qk_lora) ──

def _iter_root_candidates(user_root):
    candidates = []
    if user_root: candidates.append(Path(user_root))
    candidates.extend([BENCHMARK_DIR, BENCHMARK_DIR.parent, BENCHMARK_DIR.parent / "测试沙箱"])
    seen, resolved = set(), []
    for c in candidates:
        n = c.resolve()
        k = str(n)
        if k not in seen: seen.add(k); resolved.append(n)
    return resolved

def _resolve_csl_paths(root_dir):
    for root in _iter_root_candidates(root_dir):
        for sub in [root / "external" / "CSL" / "benchmark" / "kg", root / "data" / "CSL" / "benchmark" / "kg"]:
            if (sub / "train.tsv").exists() and (sub / "test.tsv").exists():
                return sub / "train.tsv", sub / "test.tsv"
    raise FileNotFoundError("CSL not found")

def _resolve_shence_root(root_dir):
    for root in _iter_root_candidates(root_dir):
        raw = root / "data" / "shencecup" / "raw"
        if (raw / "all_docs.txt").exists() and (raw / "train_docs_keywords.txt").exists():
            return root
    raise FileNotFoundError("ShenCeCup not found")

def _resolve_multi_domain_path(root_dir):
    for root in _iter_root_candidates(root_dir):
        for c in [root / "data" / "multi_domain.jsonl", root / "data" / "multi_domain" / "annotated.jsonl"]:
            if c.exists(): return c
    return None

def _recommended_layer(layer_count):
    if not layer_count or layer_count <= 0: return None
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))

def _resolve_layer_arg(layer_arg, model):
    layer_count = None
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0: layer_count = v; break
    if layer_arg.strip().lower() == "auto":
        rec = _recommended_layer(layer_count)
        return (rec if rec is not None else layer_count - 1), layer_count
    idx = int(layer_arg)
    if layer_count and idx < 0: idx += layer_count
    return idx, layer_count

def _keyword_char_spans(text, keywords):
    spans = []
    for kw in keywords:
        if not kw: continue
        pos = 0
        while True:
            idx = text.find(kw, pos)
            if idx < 0: break
            spans.append((idx, idx + len(kw)))
            pos = idx + 1
    return spans

def _filter_extractive(docs, name):
    total_kw, kept_kw, dropped = 0, 0, 0
    filtered = []
    for doc in docs:
        ext = [kw for kw in doc.keywords if kw in doc.text]
        total_kw += len(doc.keywords); kept_kw += len(ext)
        if ext: d = copy(doc); d.keywords = ext; filtered.append(d)
        else: dropped += 1
    ratio = 0.0 if not total_kw else kept_kw / total_kw
    print(f"[info] {name}: {kept_kw}/{total_kw} kw kept ({ratio:.1%}), {dropped} docs dropped")
    return filtered


# ── Tokenize with keyword mask ──

def tokenize_with_kw_mask(text, keywords, tokenizer, instruction_prefix, max_length):
    full_text = instruction_prefix + text
    prefix_len = len(instruction_prefix)
    enc = tokenizer(full_text, max_length=max_length, truncation=True, padding=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    kw_spans = [(s + prefix_len, e + prefix_len) for s, e in _keyword_char_spans(text, keywords)]

    seq_len = len(enc["input_ids"])
    kw_mask = [0.0] * seq_len  # 1.0 for keyword tokens, 0.0 for non-keyword text tokens, -1 for ignore
    for tok_idx, (ts, te) in enumerate(offsets):
        if ts == te or ts < prefix_len:
            kw_mask[tok_idx] = -1.0  # ignore: special/prefix tokens
            continue
        for ks, ke in kw_spans:
            if ts < ke and te > ks:
                kw_mask[tok_idx] = 1.0
                break

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "kw_mask": kw_mask,
        "offset_mapping": offsets,
    }


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


def _char_span_to_token_span(offset_mapping: Sequence[Tuple[int, int]], prefix_len: int,
                             char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    token_indices = []
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


def build_bio_candidate_supervision(doc, offset_mapping, prefix_len, bio_extractor,
                                    bio_profile, max_candidates, label_mode):
    candidate_specs = []
    seen_texts = set()
    gold_texts = {kw.strip() for kw in doc.keywords if kw and kw.strip()}
    raw_candidates = bio_extractor.extract_spans_profile(doc.text, profile=bio_profile)[:max_candidates]
    for candidate_text, bio_score in raw_candidates:
        normalized_text = candidate_text.strip()
        if not normalized_text or normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        token_spans = []
        for char_start, char_end in find_candidate_occurrences(doc.text, normalized_text):
            token_span = _char_span_to_token_span(offset_mapping, prefix_len, char_start, char_end)
            if token_span is not None:
                token_spans.append(token_span)
        if not token_spans:
            continue
        candidate_specs.append({
            "text": normalized_text,
            "label": _candidate_label(normalized_text, gold_texts, label_mode),
            "bio_score": float(bio_score),
            "token_spans": token_spans,
        })
    return candidate_specs


class AttnDataset(Dataset):
    def __init__(self, docs, tokenizer, instruction_prefix, max_length,
                 bio_extractor=None, bio_profile="clean", bio_aux_max_candidates=40,
                 bio_aux_label_mode="soft"):
        self.items = []
        prefix_len = len(instruction_prefix)
        for i, doc in enumerate(docs, 1):
            item = tokenize_with_kw_mask(doc.text, doc.keywords, tokenizer, instruction_prefix, max_length)
            item["doc"] = doc
            if bio_extractor is not None:
                item["candidate_supervision"] = build_bio_candidate_supervision(
                    doc, item["offset_mapping"], prefix_len, bio_extractor,
                    bio_profile, bio_aux_max_candidates, bio_aux_label_mode,
                )
            self.items.append(item)
            if i % 200 == 0 or i == len(docs):
                print(f"[pretokenize] {i}/{len(docs)}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def _collate_fn(batch):
    max_len = max(len(it["input_ids"]) for it in batch)
    ids, masks, kw_masks = [], [], []
    docs, candidate_supervision = [], []
    for it in batch:
        pad = max_len - len(it["input_ids"])
        ids.append(it["input_ids"] + [0] * pad)
        masks.append(it["attention_mask"] + [0] * pad)
        kw_masks.append(it["kw_mask"] + [-1.0] * pad)
        docs.append(it["doc"])
        candidate_supervision.append(it.get("candidate_supervision", []))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "kw_mask": torch.tensor(kw_masks, dtype=torch.float32),
        "docs": docs,
        "candidate_supervision": candidate_supervision,
    }


# ── TaSc modules ──

class LinTaSc(nn.Module):
    """每个 vocab id 一个标量参数 u，token 重要性 = sigmoid(u_v * sum(embed_v))。
    论文 TaSc 原版：先用一个 token-level 标量门控 scale attention signal，再走原下游 pipeline。
    fp32 buffer 以避免 bf16 autocast 下数值精度损失。"""
    def __init__(self, vocab_size: int, embeddings_weight: torch.Tensor):
        super().__init__()
        self.u = nn.Parameter(torch.ones(vocab_size))
        with torch.no_grad():
            self.register_buffer("embed_sum",
                                 embeddings_weight.sum(dim=-1).detach().float().clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, S) -> output: (B, S) in [0, 1]
        return torch.sigmoid(self.u[input_ids] * self.embed_sum[input_ids])


class HiddenScaler(nn.Module):
    """从指定层 hidden_states 学 token-level 门控：sigmoid(MLP(hidden_states))。
    强制 fp32 forward 避免 bf16 autocast 下 sigmoid 数值不稳。"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, D) or (S, D) -> output: (B, S) or (S,) in [0, 1]
        return torch.sigmoid(self.proj(hidden_states.float()).squeeze(-1))


# ── Attention loss ──

def compute_attn_loss(attention_map, kw_mask, attention_mask, loss_type, loss_target="eos_row", focal_gamma=2.0, ranking_margin=0.3,
                     token_score_override=None, soft_target=False, soft_alpha=0.1):
    """
    attention_map: (batch, num_heads, seq, seq) — raw attention weights (softmax output)
    kw_mask: (batch, seq) — 1.0=keyword, 0.0=non-keyword text, -1=ignore
    attention_mask: (batch, seq)
    loss_target: 'eos_row' = supervise EOS→token attention row
                 'col_sum' = supervise column-sum (received_attn, aligned with inference)
    token_score_override: optional (batch, seq) tensor that bypasses internal signal extraction
                          (used for TaSc-modulated signal — option A: scale-then-normalize)
    soft_target: KL only — use label-smoothed target instead of hard keyword-uniform target
    soft_alpha: KL soft target smoothing strength
    """
    batch_size = attention_map.shape[0]

    if token_score_override is not None:
        signal = token_score_override
    else:
        # Mean over heads → (batch, seq, seq)
        attn_mean = attention_map.mean(dim=1)

        if loss_target == "eos_row":
            # EOS-to-all attention: (batch, seq)
            eos_idx = attention_mask.sum(dim=1) - 1
            signal = attn_mean[torch.arange(batch_size, device=attn_mean.device), eos_idx]
        else:
            # Column sum = received_attn: how much each token is attended to by all others
            # Mask out padding rows before summing
            pad_mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            masked_attn = attn_mean * pad_mask  # zero out rows from pad tokens
            signal = masked_attn.sum(dim=1)  # (batch, seq) — column sum

    losses = []
    for b in range(batch_size):
        valid_mask = kw_mask[b] >= 0.0
        if not valid_mask.any():
            continue

        p = signal[b][valid_mask]
        kw_label = kw_mask[b][valid_mask]

        p_sum = p.sum()
        if p_sum < 1e-8:
            continue
        p_norm = p / p_sum

        if loss_type == "kl":
            kw_count = kw_label.sum()
            if kw_count < 1e-6:
                continue
            if soft_target:
                # Label smoothing: keyword tokens share (1-α), all valid tokens get α/N evenly
                num_valid = valid_mask.sum().float()
                target = kw_label / kw_count * (1 - soft_alpha) + soft_alpha / num_valid
            else:
                target = kw_label / kw_count
            log_p = torch.log(p_norm + 1e-8)
            log_t = torch.log(target + 1e-8)
            kl = (target * (log_t - log_p)).sum()
            losses.append(kl)

        elif loss_type == "bce":
            # bf16 autocast 不允许 binary_cross_entropy (sigmoid 已经在 p_norm 之前隐含)，
            # 强制 fp32 计算
            with torch.amp.autocast("cuda", enabled=False):
                p_fp32 = p_norm.float().clamp(1e-7, 1 - 1e-7)
                lbl_fp32 = kw_label.float()
                losses.append(nn.functional.binary_cross_entropy(p_fp32, lbl_fp32, reduction="mean"))

        elif loss_type == "focal":
            with torch.amp.autocast("cuda", enabled=False):
                p_fp32 = p_norm.float().clamp(1e-7, 1 - 1e-7)
                lbl_fp32 = kw_label.float()
                bce = nn.functional.binary_cross_entropy(p_fp32, lbl_fp32, reduction="none")
                p_t = p_fp32 * lbl_fp32 + (1 - p_fp32) * (1 - lbl_fp32)
                focal_weight = (1 - p_t) ** focal_gamma
                losses.append((focal_weight * bce).mean())

        elif loss_type == "ranking":
            # Pairwise ranking: keyword attention > non-keyword attention + margin
            kw_scores = p_norm[kw_label > 0.5]
            neg_scores = p_norm[kw_label <= 0.5]
            if kw_scores.numel() == 0 or neg_scores.numel() == 0:
                continue
            # Sample pairs to avoid O(n*m)
            n_pairs = min(kw_scores.numel() * neg_scores.numel(), 64)
            idx_pos = torch.randint(0, kw_scores.numel(), (n_pairs,), device=p.device)
            idx_neg = torch.randint(0, neg_scores.numel(), (n_pairs,), device=p.device)
            pos = kw_scores[idx_pos]  # (n_pairs,)
            neg = neg_scores[idx_neg]  # (n_pairs,)
            # margin hinge loss: max(0, margin - (pos - neg))
            violations = torch.clamp(ranking_margin - (pos - neg), min=0.0)
            losses.append(violations.mean())

    if not losses:
        return None
    return torch.stack(losses).mean()


def attention_signal(attention_map, attention_mask, loss_target="col_sum"):
    batch_size = attention_map.shape[0]
    attn_mean = attention_map.mean(dim=1)
    if loss_target == "eos_row":
        eos_idx = attention_mask.sum(dim=1) - 1
        return attn_mean[torch.arange(batch_size, device=attn_mean.device), eos_idx]
    pad_mask = attention_mask.unsqueeze(-1).float()
    return (attn_mean * pad_mask).sum(dim=1)


def compute_bio_candidate_ranking_loss(token_scores, candidate_supervision,
                                       margin=0.1, max_pairs=64):
    losses = []
    for batch_idx, batch_candidates in enumerate(candidate_supervision):
        scored = []
        for candidate in batch_candidates:
            occurrence_scores = []
            for start_idx, end_idx in candidate["token_spans"]:
                if end_idx <= start_idx:
                    continue
                span_scores = token_scores[batch_idx, start_idx:end_idx]
                if span_scores.numel() > 0:
                    occurrence_scores.append(span_scores.mean())
            if occurrence_scores:
                scored.append((torch.stack(occurrence_scores).max(), float(candidate["label"])))
        positives = [score for score, label in scored if label > 0.5]
        negatives = [score for score, label in scored if label <= 0.5]
        if not positives or not negatives:
            continue
        pos_scores = torch.stack(positives)
        neg_scores = torch.stack(negatives)
        n_pairs = min(pos_scores.numel() * neg_scores.numel(), max_pairs)
        idx_pos = torch.randint(0, pos_scores.numel(), (n_pairs,), device=token_scores.device)
        idx_neg = torch.randint(0, neg_scores.numel(), (n_pairs,), device=token_scores.device)
        violations = torch.clamp(margin - (pos_scores[idx_pos] - neg_scores[idx_neg]), min=0.0)
        losses.append(violations.mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_bio_candidate_kl_loss(token_scores, candidate_supervision, soft_alpha=0.1):
    """Candidate-level KL with soft target — analogous to token-level soft KL.

    For each doc:
      - Aggregate token_scores → one score per BIO candidate (max over occurrences, mean over tokens)
      - softmax over candidates → predicted distribution P
      - target = gold_mask/gold_count * (1-α) + α/N    (label-smoothed: gold candidates share 1-α, all share α/N)
      - loss = KL(target || P)
    """
    losses = []
    for batch_idx, batch_candidates in enumerate(candidate_supervision):
        cand_scores = []
        cand_labels = []
        for candidate in batch_candidates:
            occurrence_scores = []
            for start_idx, end_idx in candidate["token_spans"]:
                if end_idx <= start_idx:
                    continue
                span_scores = token_scores[batch_idx, start_idx:end_idx]
                if span_scores.numel() > 0:
                    occurrence_scores.append(span_scores.mean())
            if occurrence_scores:
                cand_scores.append(torch.stack(occurrence_scores).max())
                cand_labels.append(float(candidate["label"]))
        if len(cand_scores) < 2:
            continue
        scores_t = torch.stack(cand_scores)  # (N,)
        labels_t = torch.tensor(cand_labels, device=scores_t.device, dtype=scores_t.dtype)  # (N,)

        gold_mask = labels_t > 0.5
        gold_count = gold_mask.float().sum()
        if gold_count < 0.5:
            continue
        N = float(scores_t.numel())

        # Predicted distribution over candidates
        log_p = torch.nn.functional.log_softmax(scores_t, dim=0)  # (N,)

        # Soft target: gold candidates share (1-α), all candidates uniform α/N
        target = gold_mask.float() / gold_count * (1.0 - soft_alpha) + soft_alpha / N

        log_t = torch.log(target.clamp(min=1e-8))
        kl = (target * (log_t - log_p)).sum()
        losses.append(kl)
    if not losses:
        return None
    return torch.stack(losses).mean()


# ── Evaluation with attention ──

def score_candidates_with_attn(doc, tokenizer, model, device, instruction_prefix,
                                max_length, layer_idx, top_k, bio_extractor,
                                bio_profile, attn_method, tasc_module=None):
    model.eval()
    candidates = [c for c, _ in bio_extractor.extract_spans_profile(doc.text, profile=bio_profile)]
    if not candidates:
        return []

    full_text = instruction_prefix + doc.text
    prefix_len = len(instruction_prefix)
    enc = tokenizer(full_text, max_length=max_length, truncation=True, padding=False,
                    return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = enc["offset_mapping"][0].tolist()
    enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}

    need_hidden = (tasc_module is not None) and isinstance(tasc_module, HiddenScaler)
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True, output_hidden_states=need_hidden)

    valid_len = int(enc["attention_mask"][0].sum().item())
    # Stay in torch on GPU until TaSc is applied (if any), then move to numpy.
    attn_map_t = outputs.attentions[layer_idx].mean(dim=1)[0, :valid_len, :valid_len]

    # Compute token scores based on method
    if attn_method == "eos_attn":
        # Use EOS row: how much EOS attends to each token (aligned with eos_row training)
        token_scores_t = attn_map_t[valid_len - 1, :]
    elif attn_method == "received_attn":
        token_scores_t = attn_map_t.sum(dim=0)
    else:  # samrank
        idx_t = torch.arange(1, valid_len + 1, device=device, dtype=attn_map_t.dtype) / valid_len
        token_scores_t = attn_map_t.sum(dim=0) * idx_t

    # Apply TaSc rescaling (option A: scale signal directly)
    if tasc_module is not None:
        if isinstance(tasc_module, LinTaSc):
            tasc_t = tasc_module(enc["input_ids"][0, :valid_len])
        else:  # HiddenScaler
            tasc_t = tasc_module(outputs.hidden_states[layer_idx][0, :valid_len])
        token_scores_t = token_scores_t * tasc_t

    token_scores = token_scores_t.detach().cpu().float().numpy()

    # Map token scores to char offsets
    char_scores = {}
    for tok_idx, (ts, te) in enumerate(offset_mapping[:valid_len]):
        if ts == te or ts < prefix_len: continue
        cs, ce = ts - prefix_len, te - prefix_len
        sc = float(token_scores[tok_idx])
        for c in range(cs, ce):
            if c not in char_scores or sc > char_scores[c]:
                char_scores[c] = sc

    # Score candidates
    candidate_scores = {}
    for cand in candidates:
        spans = find_candidate_occurrences(doc.text, cand)
        if not spans: continue
        best = None
        for cs, ce in spans:
            vals = [char_scores[c] for c in range(cs, ce) if c in char_scores]
            if vals:
                s = sum(vals) / len(vals)
                if best is None or s > best: best = s
        if best is not None:
            candidate_scores[cand] = best

    sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_cands[:top_k]]


def evaluate_f1(model, tokenizer, docs, device, instruction_prefix, max_length,
                layer_idx, top_k, bio_extractor, bio_profile, attn_method, tasc_module=None):
    preds, golds = [], []
    for doc in docs:
        pred = score_candidates_with_attn(doc, tokenizer, model, device, instruction_prefix,
                                           max_length, layer_idx, top_k, bio_extractor,
                                           bio_profile, attn_method, tasc_module=tasc_module)
        preds.append(pred)
        golds.append(doc.keywords)
    return evaluate_predictions(preds, golds)


# ── Main ──

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_ROOT / "attn_lora" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output: {output_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"; print("[warn] CUDA unavailable, CPU fallback")
    if device == "cuda": torch.backends.cudnn.benchmark = True

    train_limit = 100 if args.smoke else args.train_limit
    dev_limit = 50 if args.smoke else args.dev_limit
    test_limit = 100 if args.smoke else args.test_limit

    print(f"[info] Loss: {args.loss_type} | LoRA targets: {args.lora_targets} | Eval method: {args.attn_method}")

    # ── Data ──
    rng = random.Random(args.split_seed)

    if args.disable_csl:
        csl_train, csl_test = [], []
    else:
        csl_tp, csl_tep = _resolve_csl_paths(args.root_dir)
        csl_train = load_csl_split(csl_tp, "train", limit=train_limit or 2000)
        csl_test = load_csl_split(csl_tep, "test", limit=test_limit)

    shence_root = _resolve_shence_root(args.root_dir)
    shence_all = load_shencecup_labeled(shence_root)
    rng.shuffle(shence_all)
    test_size = min(200, len(shence_all) // 5)
    shence_test = shence_all[:test_size]
    shence_train = shence_all[test_size:]
    dev_docs = shence_test[:test_size // 2]
    shence_test_final = shence_test[test_size // 2:]
    if dev_limit: dev_docs = dev_docs[:dev_limit]

    md_path = None if args.disable_multi_domain else _resolve_multi_domain_path(args.root_dir)
    if md_path and md_path.exists():
        md_all = load_multi_domain_jsonl(md_path); rng.shuffle(md_all)
        md_test_size = min(1000, len(md_all) // 5)
        md_test, md_train = md_all[:md_test_size], md_all[md_test_size:]
        if args.md_max_keywords > 0:
            capped = []
            for d in md_train:
                if len(d.keywords) > args.md_max_keywords:
                    d = copy(d); d.keywords = d.keywords[:args.md_max_keywords]
                capped.append(d)
            md_train = capped
    else:
        md_train, md_test = [], []

    csl_train = _filter_extractive(csl_train, "CSL train")
    shence_train = _filter_extractive(shence_train, "ShenCe train")
    md_train = _filter_extractive(md_train, "MD train")

    train_docs = csl_train + shence_train + md_train
    rng.shuffle(train_docs)
    if train_limit: train_docs = train_docs[:train_limit]

    print(f"[info] Train: {len(train_docs)} (CSL={len(csl_train)}, ShenCe={len(shence_train)}, MD={len(md_train)})")
    print(f"[info] Dev: {len(dev_docs)} | Test: ShenCe={len(shence_test_final)}, MD={len(md_test)}, CSL={len(csl_test)}")

    # ── Tokenizer ──
    print(f"[info] Loading tokenizer from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # ── BIO extractor for eval ──
    bio_extractor = None
    if args.bio_candidate_checkpoint:
        from keyatten import BIOExtractor
        bio_extractor = BIOExtractor(args.bio_candidate_checkpoint, device="cuda" if device == "cuda" else "cpu")
        print(f"[info] BIO extractor loaded: {args.bio_candidate_checkpoint}")

    print("[info] Pre-tokenizing...")
    bio_aux_extractor = bio_extractor if args.bio_aux_weight > 0.0 else None
    train_dataset = AttnDataset(
        train_docs, tokenizer, INSTRUCTION_PREFIX, args.max_length,
        bio_extractor=bio_aux_extractor,
        bio_profile=args.bio_candidate_profile,
        bio_aux_max_candidates=args.bio_aux_max_candidates,
        bio_aux_label_mode=args.bio_aux_label_mode,
    )

    pos_count = sum(1 for it in train_dataset.items for m in it["kw_mask"] if m > 0.5)
    neg_count = sum(1 for it in train_dataset.items for m in it["kw_mask"] if 0.0 <= m <= 0.5)
    print(f"[info] Token labels: {pos_count} kw / {neg_count} non-kw text")
    if args.bio_aux_weight > 0.0:
        cand_pos = sum(1 for it in train_dataset.items for c in it.get("candidate_supervision", []) if c["label"] > 0.5)
        cand_neg = sum(1 for it in train_dataset.items for c in it.get("candidate_supervision", []) if c["label"] <= 0.5)
        print(f"[info] BIO aux candidates: {cand_pos} pos / {cand_neg} neg | weight={args.bio_aux_weight}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate_fn, num_workers=0, pin_memory=(device == "cuda"))

    # ── Model ──
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[info] AMP: {'bf16' if use_amp else 'off'}")

    print(f"[info] Loading model from {args.model}...")
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                            torch_dtype=amp_dtype if use_amp else None,
                                            attn_implementation='eager')
    layer_idx, layer_count = _resolve_layer_arg(args.layer, base_model)
    print(f"[info] Layer: {layer_idx} / {layer_count - 1 if layer_count else '?'}")

    lora_targets = [t.strip() for t in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets, bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model.to(device)

    # ── TaSc module (optional) ──
    tasc_module = None
    if args.tasc_mode == "lin":
        try:
            base_for_embed = lora_model.get_base_model()
        except Exception:
            base_for_embed = base_model
        embed_w = base_for_embed.get_input_embeddings().weight
        vocab_size = base_for_embed.config.vocab_size
        tasc_module = LinTaSc(vocab_size, embed_w).to(device)
        print(f"[info] TaSc=lin (vocab_size={vocab_size}, params={sum(p.numel() for p in tasc_module.parameters())})")
    elif args.tasc_mode == "hidden":
        tasc_module = HiddenScaler(base_model.config.hidden_size).to(device)
        print(f"[info] TaSc=hidden (hidden_size={base_model.config.hidden_size}, params={sum(p.numel() for p in tasc_module.parameters())})")

    trainable = list(lora_model.parameters())
    if tasc_module is not None:
        trainable += list(tasc_module.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # ── Training loop ──
    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    start_epoch = 1
    training_log = []

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
        if tasc_module is not None and ckpt.get("tasc_state_dict"):
            tasc_module.load_state_dict(ckpt["tasc_state_dict"])
        print(f"[info] Resumed from epoch {ckpt['epoch']}, best F1@10={best_f1:.4f}")

    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        lora_model.train()
        total_loss, total_attn_loss, total_bio_loss, bio_steps, steps = 0.0, 0.0, 0.0, 0, 0
        epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100, ascii=True, leave=False, mininterval=2.0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            kw_mask = batch["kw_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_attentions=True,
                                     output_hidden_states=(args.tasc_mode == "hidden"))
                # Get attention at target layer
                attn = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)

                # TaSc-modulated token signal (option A: scale before downstream pipeline)
                token_score_override = None
                if tasc_module is not None:
                    attn_sig = attention_signal(attn, attention_mask, args.loss_target)
                    if args.tasc_mode == "lin":
                        tasc_t = tasc_module(input_ids)
                    else:  # hidden
                        tasc_t = tasc_module(outputs.hidden_states[layer_idx])
                    token_score_override = attn_sig * tasc_t  # (B, S)

                attn_loss = compute_attn_loss(
                    attn, kw_mask, attention_mask, args.loss_type, args.loss_target,
                    args.focal_gamma, args.ranking_margin,
                    token_score_override=token_score_override,
                    soft_target=args.soft_target, soft_alpha=args.soft_alpha,
                )
                bio_loss = None
                if args.bio_aux_weight > 0.0:
                    if token_score_override is not None:
                        token_scores = token_score_override
                    else:
                        token_scores = attention_signal(attn, attention_mask, args.loss_target)
                    if args.bio_aux_loss_type == "kl":
                        bio_loss = compute_bio_candidate_kl_loss(
                            token_scores, batch["candidate_supervision"],
                            soft_alpha=args.soft_alpha,
                        )
                    else:
                        bio_loss = compute_bio_candidate_ranking_loss(
                            token_scores, batch["candidate_supervision"],
                            margin=args.bio_aux_margin,
                        )

                # Combine weighted losses (skip None or zero-weight components)
                loss_terms = []
                if attn_loss is not None and args.attn_loss_weight > 0.0:
                    loss_terms.append(args.attn_loss_weight * attn_loss)
                if bio_loss is not None and args.bio_aux_weight > 0.0:
                    loss_terms.append(args.bio_aux_weight * bio_loss)
                if not loss_terms:
                    continue
                loss = sum(loss_terms)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if attn_loss is not None:
                total_attn_loss += attn_loss.item()
            if bio_loss is not None:
                total_bio_loss += bio_loss.item()
                bio_steps += 1
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/steps:.4f}", attn=f"{total_attn_loss/steps:.4f}")

        avg_loss = total_loss / max(steps, 1)
        avg_attn_loss = total_attn_loss / max(steps, 1)
        avg_bio_loss = total_bio_loss / max(bio_steps, 1)
        tqdm.write(f"[epoch {epoch}] loss={avg_loss:.4f} attn={avg_attn_loss:.4f} bio={avg_bio_loss:.4f} | evaluating dev...")

        dev_metrics = evaluate_f1(lora_model, tokenizer, dev_docs, device, INSTRUCTION_PREFIX,
                                   args.max_length, layer_idx, args.top_k, bio_extractor,
                                   args.bio_candidate_profile, args.attn_method,
                                   tasc_module=tasc_module)
        f1_10 = dev_metrics.get("f1@10", 0.0)
        f1_5 = dev_metrics.get("f1@5", 0.0)
        tqdm.write(f"[epoch {epoch}] dev: F1@10={f1_10:.4f} F1@5={f1_5:.4f}")

        epoch_secs = round(time.perf_counter() - epoch_start, 2)
        training_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "attn_loss": avg_attn_loss,
            "bio_aux_loss": avg_bio_loss if bio_steps else None,
            "dev_f1@10": dev_metrics["f1@10"],
            "dev_f1@5": dev_metrics.get("f1@5", 0),
            "epoch_seconds": epoch_secs,
        })
        (output_dir / "training_log.json").write_text(json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8")

        if f1_10 > best_f1:
            best_f1, best_epoch, patience_counter = f1_10, epoch, 0
            ad = output_dir / "best_adapter"; ad.mkdir(exist_ok=True)
            lora_model.save_pretrained(str(ad))
            if tasc_module is not None:
                torch.save(tasc_module.state_dict(), str(ad / "tasc.pt"))
            tqdm.write(f"[epoch {epoch}] *** New best F1@10={best_f1:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"[info] Early stop at epoch {epoch}")
                break

        latest_adapter_dir.mkdir(exist_ok=True)
        lora_model.save_pretrained(str(latest_adapter_dir))
        if tasc_module is not None:
            torch.save(tasc_module.state_dict(), str(latest_adapter_dir / "tasc.pt"))
        torch.save({"epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                     "patience_counter": patience_counter, "training_log": training_log,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "tasc_state_dict": tasc_module.state_dict() if tasc_module is not None else None},
                    str(checkpoint_path))

    # ── Final eval ──
    print(f"\n[info] Best epoch={best_epoch}, dev F1@10={best_f1:.4f}")
    if (output_dir / "best_adapter").exists():
        lora_model.load_adapter(str(output_dir / "best_adapter"), adapter_name="default")
        if tasc_module is not None:
            best_tasc_pt = output_dir / "best_adapter" / "tasc.pt"
            if best_tasc_pt.exists():
                tasc_module.load_state_dict(torch.load(str(best_tasc_pt), map_location=device))
                print(f"[info] Loaded best TaSc state from {best_tasc_pt}")

    test_sets = {}
    if shence_test_final: test_sets["shence_test"] = shence_test_final
    if md_test: test_sets["md_test"] = md_test
    if csl_test: test_sets["csl_test"] = csl_test

    all_test = {}
    for name, docs in test_sets.items():
        m = evaluate_f1(lora_model, tokenizer, docs, device, INSTRUCTION_PREFIX,
                         args.max_length, layer_idx, args.top_k, bio_extractor,
                         args.bio_candidate_profile, args.attn_method,
                         tasc_module=tasc_module)
        print(f"[test/{name}] F1@5={m.get('f1@5',0):.4f}  F1@10={m.get('f1@10',0):.4f}")
        all_test[name] = m

    final = {
        "model": args.model, "layer": layer_idx, "best_epoch": best_epoch, "best_dev_f1@10": best_f1,
        "loss_type": args.loss_type, "lora_targets": lora_targets, "attn_method": args.attn_method,
        "total_training_seconds": round(time.perf_counter() - training_start, 2),
        "test_metrics": all_test,
        "config": {"lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "lr": args.lr,
                    "batch_size": args.batch_size, "max_length": args.max_length,
                    "bio_aux_weight": args.bio_aux_weight, "bio_aux_margin": args.bio_aux_margin,
                    "bio_aux_max_candidates": args.bio_aux_max_candidates,
                    "bio_aux_label_mode": args.bio_aux_label_mode,
                    "attn_loss_weight": args.attn_loss_weight,
                    "bio_aux_loss_type": args.bio_aux_loss_type,
                    "tasc_mode": args.tasc_mode,
                    "soft_target": args.soft_target,
                    "soft_alpha": args.soft_alpha if args.soft_target else None,
                    "loss_target": args.loss_target},
    }
    (output_dir / "final_eval.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
