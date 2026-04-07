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
  data/train.tsv          — CSL training split
  data/test.tsv           — CSL test split
  data/shencecup/raw/     — ShenCeCup labeled docs
  data/multi_domain.jsonl  — (optional) LLM-annotated multi-domain JSONL

Usage:
  python train_qk_lora.py --model Qwen/Qwen3-Embedding-0.6B --epochs 20
  python train_qk_lora.py --smoke   # quick sanity check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR.parent))

from keyword_bench.data import (
    Document,
    load_csl_split,
    load_multi_domain_jsonl,
    load_shencecup_labeled,
)
from keyword_bench.metrics import evaluate_predictions
from keyatten.candidates import build_candidates, segment_text
from keyatten.attention import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX

DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_LAYER = 21
INSTRUCTION_PREFIX = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QK LoRA: Contrastive QK Learning.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model name or path")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="attention layer index for QK scoring")
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
    p.add_argument("--md-max-keywords", type=int, default=4, help="cap keywords per multi-domain doc (0=no cap)")
    p.add_argument("--smoke", action="store_true", help="quick sanity check with small data")
    return p.parse_args()


# ── Data preparation ──────────────────────────────────────────────────


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


class TokenDataset(Dataset):
    def __init__(self, docs, tokenizer, instruction_prefix: str, max_length: int) -> None:
        self.items: List[dict] = []
        for doc in docs:
            item = tokenize_and_label(doc.text, doc.keywords, tokenizer, instruction_prefix, max_length)
            item["doc"] = doc
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _collate_fn(batch: List[dict]) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids_list.append(item["input_ids"] + [0] * pad)
        attention_mask_list.append(item["attention_mask"] + [0] * pad)
        labels_list.append(item["labels"] + [-1.0] * pad)
    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.float32),
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

    inner = model.base_model.model
    if hasattr(inner, "model"):
        inner = inner.model
    target_layer = inner.layers[layer_idx].self_attn

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
) -> List[str]:
    model.eval()

    words, pos_tags = segment_text(doc.text, language="zh")
    candidates = build_candidates(words, pos_tags, language="zh")
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
    for cand in candidates:
        char_start = sum(len(words[i]) for i in range(cand.word_start))
        char_end = sum(len(words[i]) for i in range(cand.word_end))
        cand_scores = [char_to_score[c] for c in range(char_start, char_end) if c in char_to_score]
        score = sum(cand_scores) / len(cand_scores) if cand_scores else 0.0
        if cand.text not in candidate_scores or score > candidate_scores[cand.text]:
            candidate_scores[cand.text] = score

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
) -> Dict[str, float]:
    predictions = []
    golds = []
    for doc in docs:
        pred = score_candidates_with_qk(
            doc, tokenizer, model, device,
            instruction_prefix, max_length, layer_idx, top_k
        )
        predictions.append(pred)
        golds.append(doc.keywords)
    return evaluate_predictions(predictions, golds)


# ── Training ──────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else BENCHMARK_DIR / "qk_lora_adapter" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output dir: {output_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available, falling back to CPU")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_limit = 100 if args.smoke else args.train_limit
    dev_limit = 50 if args.smoke else args.dev_limit
    test_limit = 100 if args.smoke else args.test_limit
    layer_idx = args.layer

    print(f"[info] Model: {args.model}")
    print(f"[info] Layer: {layer_idx}")
    print(f"[info] Method: Contrastive QK Learning (Q[EOS]·K[i] BCE)")

    # ── Load data ──

    _split_rng = random.Random(42)

    print("[info] Loading CSL data...")
    csl_limit = train_limit if train_limit else 2000
    csl_train = load_csl_split(DATA_DIR / "train.tsv", "train", limit=csl_limit)
    csl_test = load_csl_split(DATA_DIR / "test.tsv", "test", limit=test_limit)

    print("[info] Loading ShenCeCup data...")
    shence_all = load_shencecup_labeled(BENCHMARK_DIR)
    _split_rng.shuffle(shence_all)
    shence_test_size = min(200, len(shence_all) // 5)
    shence_test = shence_all[:shence_test_size]
    shence_train = shence_all[shence_test_size:]
    dev_docs = shence_test[:shence_test_size // 2]
    shence_test_final = shence_test[shence_test_size // 2:]

    multi_domain_path = DATA_DIR / "multi_domain.jsonl"
    if multi_domain_path.exists():
        print("[info] Loading multi-domain data...")
        md_all = load_multi_domain_jsonl(multi_domain_path)
        _split_rng.shuffle(md_all)
        md_test_size = min(1000, len(md_all) // 5)
        md_test = md_all[:md_test_size]
        md_train = md_all[md_test_size:]
    else:
        print("[info] multi_domain.jsonl not found, skipping")
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
        print(f"[info] {name}: keywords {kept_kw}/{total_kw} kept ({kept_kw / total_kw:.1%}), {dropped_docs} docs dropped")
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
    print(f"[info] === Data Split (seed=42, extractive-filtered) ===")
    print(f"[info] Train: {len(train_docs)} (CSL={len(csl_train)}, ShenCe={len(shence_train)}, MD={len(md_train)})")
    print(f"[info] Dev (ShenCe): {len(dev_docs)}")
    print(f"[info] Test: ShenCe={len(shence_test_final)}, MD={len(md_test)}, CSL={len(csl_test)}")

    # ── Tokenize & build dataset ──

    print(f"[info] Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    print("[info] Pre-tokenizing training data...")
    train_dataset = TokenDataset(train_docs, tokenizer, INSTRUCTION_PREFIX, args.max_length)

    pos_count = sum(1 for item in train_dataset.items for lbl in item["labels"] if lbl > 0.5)
    neg_count = sum(1 for item in train_dataset.items for lbl in item["labels"] if 0.0 <= lbl <= 0.5)
    pos_weight = neg_count / max(pos_count, 1)
    print(f"[info] Token labels: {pos_count} pos / {neg_count} neg → pos_weight={pos_weight:.2f}")

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

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

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

    for epoch in range(start_epoch, args.epochs + 1):
        lora_model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    ncols=100, ascii=True, leave=False, mininterval=2.0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                scores = compute_qk_scores(lora_model, input_ids, attention_mask, layer_idx)
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
            INSTRUCTION_PREFIX, args.max_length, layer_idx, args.top_k
        )
        f1_10 = dev_metrics.get("f1@10", 0.0)
        f1_5 = dev_metrics.get("f1@5", 0.0)
        tqdm.write(f"[epoch {epoch}] dev: F1@10={f1_10:.4f} F1@5={f1_5:.4f}")

        log_entry = {"epoch": epoch, "train_loss": avg_loss, "dev_f1@10": f1_10, "dev_f1@5": f1_5}
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
            INSTRUCTION_PREFIX, args.max_length, layer_idx, args.top_k
        )
        f1_10 = metrics.get("f1@10", 0.0)
        f1_5 = metrics.get("f1@5", 0.0)
        print(f"[test/{ts_name}] F1@5={f1_5:.4f}  F1@10={f1_10:.4f}  ({len(ts_docs)} docs)")
        all_test_metrics[ts_name] = metrics

    final_eval = {
        "model": args.model,
        "layer": layer_idx,
        "best_epoch": best_epoch,
        "best_dev_f1@10": best_f1,
        "test_metrics": all_test_metrics,
        "data_split": {
            "train": len(train_docs),
            "dev_shence": len(dev_docs),
            "test_shence": len(shence_test_final),
            "test_md": len(md_test),
            "test_csl": len(csl_test),
            "split_seed": 42,
        },
        "config": {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target": "q_proj + k_proj",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "pos_weight": pos_weight,
            "training_method": "contrastive_qk",
        },
    }
    (output_dir / "final_eval.json").write_text(
        json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[info] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
