from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from keyword_bench.data import (
    build_csl_eval_sets,
    build_english_eval_sets,
    build_shencecup_eval_sets,
)
from keyword_bench.hidden_state_head import (
    HiddenStateKeywordHead,
    TokenizedLabelExample,
    collate_token_examples,
    compute_pos_weight,
    masked_bce_with_logits_loss,
    tokenize_with_keyword_labels,
)
from keyword_bench.output_paths import resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hidden-state token classifier for keywords."
    )
    parser.add_argument("--root-dir", default=".", help="Project root")
    parser.add_argument(
        "--output-dir",
        default="outputs_hidden_head",
        help="Output dir under 测试沙箱/Outputs",
    )
    parser.add_argument("--model", default="thenlper/gte-small-zh")
    parser.add_argument("--train-dataset", default="csl_train_sample")
    parser.add_argument("--dev-dataset", default="csl_dev")
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--dev-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--derived-limit", type=int, default=200)
    parser.add_argument("--english-limit", type=int, default=120)
    parser.add_argument("--shencecup-limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone and train only linear head",
    )
    parser.add_argument(
        "--unfreeze-backbone", action="store_true", help="Train backbone as well"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_examples(docs, tokenizer, max_length: int) -> list[TokenizedLabelExample]:
    examples: list[TokenizedLabelExample] = []
    for doc in docs:
        if not doc.text.strip():
            continue
        if not doc.keywords:
            continue
        examples.append(
            tokenize_with_keyword_labels(
                doc.text, doc.keywords, tokenizer=tokenizer, max_length=max_length
            )
        )
    return examples


def evaluate(
    model: HiddenStateKeywordHead,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_bce_with_logits_loss(logits, labels, pos_weight=pos_weight)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits)
            valid_mask = labels >= 0
            if not bool(valid_mask.any()):
                continue
            y_true = labels[valid_mask]
            y_pred = (probs[valid_mask] >= 0.5).float()
            total_tp += float(((y_pred == 1.0) & (y_true == 1.0)).sum().item())
            total_fp += float(((y_pred == 1.0) & (y_true == 0.0)).sum().item())
            total_fn += float(((y_pred == 0.0) & (y_true == 1.0)).sum().item())
    if (total_tp + total_fp + total_fn) <= 0.0:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_tp + total_fn, 1e-8)
    f1 = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root_dir = Path(args.root_dir).resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {args.train_dataset, args.dev_dataset}
    all_eval_sets = {}
    if any(name.startswith("csl_") for name in requested):
        all_eval_sets.update(
            build_csl_eval_sets(
                root_dir,
                train_limit=args.train_limit,
                dev_limit=args.dev_limit,
                test_limit=args.test_limit,
                derived_limit=args.derived_limit,
            )
        )
    if any(name.startswith("shencecup_") for name in requested):
        all_eval_sets.update(
            build_shencecup_eval_sets(root_dir, shencecup_limit=args.shencecup_limit)
        )
    if any(
        name.startswith(("semeval", "krapivin", "pubmed", "lis2000"))
        for name in requested
    ):
        all_eval_sets.update(
            build_english_eval_sets(root_dir, english_limit=args.english_limit)
        )
    if args.train_dataset not in all_eval_sets:
        raise ValueError(f"Unknown train dataset: {args.train_dataset}")
    if args.dev_dataset not in all_eval_sets:
        raise ValueError(f"Unknown dev dataset: {args.dev_dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must provide a valid pad_token_id.")

    train_docs = all_eval_sets[args.train_dataset]
    dev_docs = all_eval_sets[args.dev_dataset]
    train_examples = build_examples(
        train_docs, tokenizer=tokenizer, max_length=args.max_length
    )
    dev_examples = build_examples(
        dev_docs, tokenizer=tokenizer, max_length=args.max_length
    )

    if not train_examples:
        raise RuntimeError("No valid training examples were built.")

    collate_fn = lambda items: collate_token_examples(
        items, pad_token_id=int(tokenizer.pad_token_id)
    )
    train_loader = DataLoader(
        train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    freeze_backbone = bool(not args.unfreeze_backbone)
    model = HiddenStateKeywordHead(
        args.model, layer_index=args.layer_index, freeze_backbone=freeze_backbone
    )
    device = torch.device(args.device)
    model.to(device)

    pos_weight_value = compute_pos_weight(train_examples)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    train_log: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_bce_with_logits_loss(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        dev_metrics = evaluate(model, dev_loader, device=device, pos_weight=pos_weight)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "dev_loss": float(dev_metrics["loss"]),
            "dev_precision": float(dev_metrics["precision"]),
            "dev_recall": float(dev_metrics["recall"]),
            "dev_f1": float(dev_metrics["f1"]),
        }
        train_log.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if dev_metrics["f1"] > best_f1:
            best_f1 = float(dev_metrics["f1"])
            checkpoint_path = output_dir / "best_hidden_head.pt"
            torch.save(
                {
                    "classifier_state": model.classifier.state_dict(),
                    "model_name": args.model,
                    "layer_index": int(args.layer_index),
                    "freeze_backbone": freeze_backbone,
                    "max_length": int(args.max_length),
                    "tokenizer_name": args.model,
                    "pos_weight": float(pos_weight_value),
                },
                checkpoint_path,
            )

    report = {
        "model": args.model,
        "train_dataset": args.train_dataset,
        "dev_dataset": args.dev_dataset,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "freeze_backbone": freeze_backbone,
        "layer_index": int(args.layer_index),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "pos_weight": float(pos_weight_value),
        "best_dev_f1": float(best_f1),
        "history": train_log,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
