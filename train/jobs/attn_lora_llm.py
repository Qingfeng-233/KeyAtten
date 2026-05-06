#!/usr/bin/env python3
"""Train GTE attention LoRA from LLM ranked keyword soft labels."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, "/root/Keyatten")
sys.path.insert(0, "/root/Keyatten/benchmark")
sys.path.insert(0, "/root/Keyatten/train")

from keyword_bench.data import Document, load_shencecup_labeled
from train.jobs.attn_lora import (
    INSTRUCTION_PREFIX,
    _resolve_layer_arg,
    attention_signal,
    evaluate_f1,
)


PROJECT_ROOT = Path("/root/Keyatten")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="/root/Keyatten/train/data/llm_labels_1k.jsonl")
    parser.add_argument("--model", default="/root/Keyatten/models/gte-small-zh")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--loss-target", choices=("col_sum", "eos_row"), default="col_sum")
    parser.add_argument("--attn-method", choices=("received_attn", "samrank", "eos_attn"), default="received_attn")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj")
    parser.add_argument("--bio-ckpt", default="/root/Keyatten/train/remote_pull_resume16_epoch13/best_full_ckpt.pt")
    parser.add_argument("--bio-profile", choices=("clean", "balanced", "high_recall"), default="clean")
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def ranked_keywords(record: dict) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for item in record.get("llm_keywords", []):
        if isinstance(item, dict):
            kw = str(item.get("keyword", "")).strip()
            rank = int(item.get("rank", len(out) + 1))
        else:
            kw = str(item).strip()
            rank = len(out) + 1
        if kw:
            out.append((rank, kw))
    return out


def keyword_spans(text: str, keyword: str) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    while keyword:
        idx = text.find(keyword, pos)
        if idx < 0:
            break
        spans.append((idx, idx + len(keyword)))
        pos = idx + 1
    return spans


def build_teacher_dist(text: str, llm_keywords: list[tuple[int, str]], tokenizer, max_length: int) -> dict | None:
    full_text = INSTRUCTION_PREFIX + text
    prefix_len = len(INSTRUCTION_PREFIX)
    enc = tokenizer(full_text, max_length=max_length, truncation=True, padding=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    token_scores = [0.0] * len(enc["input_ids"])
    valid = [False] * len(enc["input_ids"])

    char_scores: dict[int, float] = {}
    for rank, keyword in llm_keywords:
        score = 1.0 / max(rank, 1)
        for start, end in keyword_spans(text, keyword):
            for char_idx in range(start, end):
                char_scores[char_idx] = max(char_scores.get(char_idx, 0.0), score)

    for tok_idx, (start, end) in enumerate(offsets):
        if start == end or start < prefix_len:
            continue
        valid[tok_idx] = True
        local_start, local_end = start - prefix_len, end - prefix_len
        vals = [char_scores.get(i, 0.0) for i in range(local_start, local_end)]
        token_scores[tok_idx] = max(vals) if vals else 0.0

    total = sum(token_scores[i] for i, ok in enumerate(valid) if ok)
    if total <= 0:
        return None
    teacher = [0.0] * len(token_scores)
    for idx, score in enumerate(token_scores):
        if valid[idx]:
            teacher[idx] = score / total
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "teacher_dist": teacher,
        "valid_mask": [1.0 if ok else 0.0 for ok in valid],
    }


class LlmLabelDataset(Dataset):
    def __init__(self, labels_path: Path, tokenizer, max_length: int) -> None:
        self.items = []
        with labels_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                item = build_teacher_dist(record["text"], ranked_keywords(record), tokenizer, max_length)
                if item is None:
                    continue
                item["doc_id"] = record.get("doc_id", f"line-{line_no}")
                self.items.append(item)
                if len(self.items) % 200 == 0:
                    print(f"[pretokenize] {len(self.items)} usable labels", flush=True)
        print(f"[info] usable labels={len(self.items)}", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    max_len = max(len(item["input_ids"]) for item in batch)
    ids, masks, teachers, valids, doc_ids = [], [], [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        ids.append(item["input_ids"] + [0] * pad)
        masks.append(item["attention_mask"] + [0] * pad)
        teachers.append(item["teacher_dist"] + [0.0] * pad)
        valids.append(item["valid_mask"] + [0.0] * pad)
        doc_ids.append(item["doc_id"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "teacher_dist": torch.tensor(teachers, dtype=torch.float32),
        "valid_mask": torch.tensor(valids, dtype=torch.float32),
        "doc_ids": doc_ids,
    }


def llm_kl_loss(token_scores: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor | None:
    losses = []
    for idx in range(token_scores.shape[0]):
        valid = valid_mask[idx] > 0
        if not valid.any():
            continue
        target = teacher_dist[idx][valid]
        if target.sum() <= 0:
            continue
        logits = token_scores[idx][valid].float()
        log_p = torch.nn.functional.log_softmax(logits, dim=0)
        losses.append(torch.nn.functional.kl_div(log_p, target.float(), reduction="sum"))
    if not losses:
        return None
    return torch.stack(losses).mean()


def load_dev_docs(seed: int, limit: int) -> list[Document]:
    rng = random.Random(seed)
    docs = load_shencecup_labeled(PROJECT_ROOT)
    rng.shuffle(docs)
    docs = docs[: min(200, len(docs) // 5)]
    return docs[:limit]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir or f"/root/Keyatten/models/exp_llm_smoke_1k/{datetime.now():%Y%m%d_%H%M}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output={output_dir}", flush=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    dataset = LlmLabelDataset(Path(args.labels), tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    base = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else None,
        attn_implementation="eager",
    )
    layer_idx, layer_count = _resolve_layer_arg(args.layer, base)
    print(f"[info] layer={layer_idx}/{layer_count - 1 if layer_count else '?'} amp={'bf16' if use_amp else 'off'}")
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_targets.split(",") if x.strip()],
        bias="none",
    )
    model = get_peft_model(base, config).to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(len(loader) * args.epochs, 1),
    )

    from keyatten import BIOExtractor

    bio = BIOExtractor(args.bio_ckpt, device=device)
    dev_docs = load_dev_docs(args.seed, args.dev_limit)
    best_f1 = -1.0
    training_log = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100, ascii=True, mininterval=2.0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_dist = batch["teacher_dist"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                token_scores = attention_signal(outputs.attentions[layer_idx], attention_mask, args.loss_target)
                loss = llm_kl_loss(token_scores, teacher_dist, valid_mask)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / max(steps, 1):.4f}")

        avg_loss = total_loss / max(steps, 1)
        metrics = evaluate_f1(
            model,
            tokenizer,
            dev_docs,
            device,
            INSTRUCTION_PREFIX,
            args.max_length,
            layer_idx,
            args.top_k,
            bio,
            args.bio_profile,
            args.attn_method,
        )
        f1_10 = metrics.get("f1@10", 0.0)
        print(f"[epoch {epoch}] loss={avg_loss:.4f} dev_F1@10={f1_10:.4f}", flush=True)
        training_log.append({"epoch": epoch, "loss": avg_loss, "dev": metrics})
        (output_dir / "training_log.json").write_text(json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8")
        latest = output_dir / "latest_adapter"
        latest.mkdir(exist_ok=True)
        model.save_pretrained(str(latest))
        if f1_10 > best_f1:
            best_f1 = f1_10
            best = output_dir / "best_adapter"
            best.mkdir(exist_ok=True)
            model.save_pretrained(str(best))
            print(f"[epoch {epoch}] new best F1@10={best_f1:.4f}", flush=True)

    final = {
        "labels": args.labels,
        "model": args.model,
        "layer": layer_idx,
        "best_dev_f1@10": best_f1,
        "seconds": round(time.perf_counter() - started, 2),
        "config": vars(args),
    }
    (output_dir / "final_eval.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] done output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
