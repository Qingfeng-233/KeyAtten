#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as exc:
    raise RuntimeError("peft required. pip install peft>=0.10.0") from exc

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODELS_ROOT = PROJECT_ROOT / "models"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from keyword_bench.data import Document, load_multi_domain_jsonl
from keyword_bench.metrics import evaluate_predictions
from keyatten import BIOExtractor


DEFAULT_INSTRUCTION = "为这篇文章选择最重要的关键词。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train attention over an explicit document + BIO candidate segment.",
    )
    parser.add_argument("--model", default=str(MODELS_ROOT / "Qwen3-Embedding-0.6B"))
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "multi_domain.jsonl"))
    parser.add_argument("--eval-data", default=str(PROJECT_ROOT / "data" / "news_annotated.jsonl"))
    parser.add_argument("--bio-ckpt", default=str(MODELS_ROOT / "bio_ckipbert_extractive_ep13" / "bio_model_full.pt"))
    parser.add_argument("--bio-profile", choices=("clean", "balanced", "high_recall"), default="clean")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--candidate-order", choices=("bio", "random"), default="random")
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument("--eval-random-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--number-candidates", action="store_true", help="Prefix candidates with [1], [2], ... for legacy ablation.")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--dev-limit", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj")
    parser.add_argument("--soft-alpha", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-dir", default=None, help="Load an existing LoRA adapter for evaluation.")
    parser.add_argument("--zero-shot", action="store_true", help="Only run candidate-segment evaluation; skip LoRA training.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _recommended_layer(layer_count: int | None) -> int | None:
    if not layer_count or layer_count <= 0:
        return None
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))


def _resolve_layer_arg(layer_arg: str, model: torch.nn.Module) -> tuple[int, int | None]:
    layer_count = None
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            layer_count = value
            break
    if layer_arg.strip().lower() == "auto":
        recommended = _recommended_layer(layer_count)
        if recommended is not None:
            return recommended, layer_count
        if layer_count is None:
            raise ValueError("Cannot resolve layer='auto': model config has no layer count.")
        return layer_count - 1, layer_count
    index = int(layer_arg)
    if layer_count and index < 0:
        index += layer_count
    return index, layer_count


def _load_news_jsonl(path: Path, limit: int | None = None) -> list[Document]:
    docs: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"].strip()
            keywords = [kw for kw in obj.get("keywords", []) if kw in text]
            if text and keywords:
                docs.append(
                    Document(
                        doc_id=str(obj.get("id", len(docs) + 1)),
                        text=text,
                        keywords=keywords,
                        meta={"source": obj.get("source"), "title": obj.get("title")},
                    )
                )
            if limit is not None and len(docs) >= limit:
                break
    return docs


def _filter_extractive(docs: list[Document], name: str) -> list[Document]:
    filtered: list[Document] = []
    kept = total = dropped = 0
    for doc in docs:
        keywords = [kw for kw in doc.keywords if kw in doc.text]
        total += len(doc.keywords)
        kept += len(keywords)
        if keywords:
            copied = copy(doc)
            copied.keywords = keywords
            filtered.append(copied)
        else:
            dropped += 1
    ratio = 0.0 if total == 0 else kept / total
    print(f"[info] {name}: {kept}/{total} extractive keywords kept ({ratio:.1%}), dropped_docs={dropped}")
    return filtered


def _candidate_label(candidate: str, gold_keywords: set[str]) -> float:
    if candidate in gold_keywords:
        return 1.0
    # Boundary-relaxed labels are intentionally not used here. This experiment
    # tests exact BIO candidate ranking before adding boundary repair.
    return 0.0


def build_segment_text(
    doc: Document,
    candidates: list[str],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    number_candidates: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    header = f"{instruction}\n\n文章：\n{doc.text}\n\n候选：\n"
    parts = [header]
    spans: list[dict[str, Any]] = []
    cursor = len(header)
    gold = {kw.strip() for kw in doc.keywords if kw.strip()}
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
        spans.append(
            {
                "text": candidate,
                "char_span": (start, end),
                "label": _candidate_label(candidate, gold),
            }
        )
    return "".join(parts), spans


def _char_span_to_token_span(
    offsets: list[tuple[int, int]] | list[list[int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    token_indices: list[int] = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if int(token_start) < char_end and int(token_end) > char_start:
            token_indices.append(token_index)
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def build_item(
    doc: Document,
    tokenizer,
    bio: BIOExtractor,
    *,
    bio_profile: str,
    max_candidates: int,
    max_length: int,
    order: str,
    rng: random.Random,
    number_candidates: bool,
) -> dict[str, Any] | None:
    scored = bio.extract_spans_profile(doc.text, profile=bio_profile)
    candidates = [candidate for candidate, _ in scored[:max_candidates]]
    if order == "random":
        candidates = candidates[:]
        rng.shuffle(candidates)
    if len(candidates) < 2:
        return None
    segment_text, candidate_spans = build_segment_text(
        doc,
        candidates,
        number_candidates=number_candidates,
    )
    encoded = tokenizer(
        segment_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    usable_candidates: list[dict[str, Any]] = []
    for candidate in candidate_spans:
        token_span = _char_span_to_token_span(offsets, *candidate["char_span"])
        if token_span is None:
            continue
        start, end = token_span
        if end > len(encoded["input_ids"]):
            continue
        usable = dict(candidate)
        usable["token_span"] = token_span
        usable_candidates.append(usable)
    if len(usable_candidates) < 2 or not any(item["label"] > 0.5 for item in usable_candidates):
        return None
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "candidates": usable_candidates,
        "doc": doc,
    }


class CandidateSegmentDataset(Dataset):
    def __init__(
        self,
        docs: list[Document],
        tokenizer,
        bio: BIOExtractor,
        *,
        bio_profile: str,
        max_candidates: int,
        max_length: int,
        order: str,
        seed: int,
        number_candidates: bool,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        rng = random.Random(seed)
        for index, doc in enumerate(docs, 1):
            item = build_item(
                doc,
                tokenizer,
                bio,
                bio_profile=bio_profile,
                max_candidates=max_candidates,
                max_length=max_length,
                order=order,
                rng=rng,
                number_candidates=number_candidates,
            )
            if item is not None:
                self.items.append(item)
            if index % 100 == 0 or index == len(docs):
                print(f"[pretokenize] {index}/{len(docs)} usable={len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_candidate_segments(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad)
        masks.append(item["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "candidates": [item["candidates"] for item in batch],
        "docs": [item["doc"] for item in batch],
    }


def candidate_token_scores(attention_map: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attn_mean = attention_map.mean(dim=1)
    pad_mask = attention_mask.unsqueeze(-1).float()
    return (attn_mean * pad_mask).sum(dim=1)


def score_candidates_from_token_scores(
    token_scores: torch.Tensor,
    batch_candidates: list[list[dict[str, Any]]],
) -> list[list[tuple[str, torch.Tensor, float]]]:
    batch_results: list[list[tuple[str, torch.Tensor, float]]] = []
    for batch_index, candidates in enumerate(batch_candidates):
        scored: list[tuple[str, torch.Tensor, float]] = []
        for candidate in candidates:
            start, end = candidate["token_span"]
            span_scores = token_scores[batch_index, start:end]
            if span_scores.numel() == 0:
                continue
            scored.append((candidate["text"], span_scores.mean(), float(candidate["label"])))
        batch_results.append(scored)
    return batch_results


def candidate_soft_kl_loss(
    token_scores: torch.Tensor,
    batch_candidates: list[list[dict[str, Any]]],
    *,
    soft_alpha: float,
) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    for scored in score_candidates_from_token_scores(token_scores, batch_candidates):
        if len(scored) < 2:
            continue
        scores = torch.stack([item[1] for item in scored])
        labels = torch.tensor([item[2] for item in scored], device=scores.device, dtype=scores.dtype)
        gold_mask = labels > 0.5
        gold_count = gold_mask.float().sum()
        if gold_count < 0.5:
            continue
        n_candidates = float(scores.numel())
        target = gold_mask.float() / gold_count * (1.0 - soft_alpha) + soft_alpha / n_candidates
        log_p = torch.nn.functional.log_softmax(scores, dim=0)
        log_t = torch.log(target.clamp(min=1e-8))
        losses.append((target * (log_t - log_p)).sum())
    if not losses:
        return None
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_candidate_segment(
    model,
    tokenizer,
    bio: BIOExtractor,
    docs: list[Document],
    *,
    device: str,
    layer_index: int,
    bio_profile: str,
    max_candidates: int,
    max_length: int,
    order: str,
    seed: int,
    number_candidates: bool,
    top_k: int,
) -> dict[str, Any]:
    dataset = CandidateSegmentDataset(
        docs,
        tokenizer,
        bio,
        bio_profile=bio_profile,
        max_candidates=max_candidates,
        max_length=max_length,
        order=order,
        seed=seed,
        number_candidates=number_candidates,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_candidate_segments)
    predictions: list[list[str]] = []
    golds: list[list[str]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        token_scores = candidate_token_scores(outputs.attentions[layer_index], attention_mask).float()
        scored = score_candidates_from_token_scores(token_scores, batch["candidates"])[0]
        ranked = sorted(scored, key=lambda item: float(item[1].detach().cpu()), reverse=True)
        predictions.append([item[0] for item in ranked[:top_k]])
        golds.append(batch["docs"][0].keywords)
    metrics = evaluate_predictions(predictions, golds)
    metrics["usable_docs"] = len(dataset)
    metrics["total_docs"] = len(docs)
    return metrics


def average_metric_dicts(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_list:
        return {}
    averaged: dict[str, Any] = {"runs": metrics_list}
    numeric_keys = [
        key
        for key, value in metrics_list[0].items()
        if isinstance(value, int | float) and key not in {"usable_docs", "total_docs"}
    ]
    for key in numeric_keys:
        averaged[key] = sum(float(metrics[key]) for metrics in metrics_list) / len(metrics_list)
    averaged["usable_docs"] = metrics_list[0].get("usable_docs")
    averaged["total_docs"] = metrics_list[0].get("total_docs")
    averaged["num_runs"] = len(metrics_list)
    return averaged


@torch.no_grad()
def evaluate_with_optional_seeds(
    model,
    tokenizer,
    bio: BIOExtractor,
    docs: list[Document],
    *,
    device: str,
    layer_index: int,
    bio_profile: str,
    max_candidates: int,
    max_length: int,
    order: str,
    seed: int,
    eval_random_seeds: list[int] | None,
    number_candidates: bool,
    top_k: int,
) -> dict[str, Any]:
    if order == "random" and eval_random_seeds:
        runs = [
            evaluate_candidate_segment(
                model,
                tokenizer,
                bio,
                docs,
                device=device,
                layer_index=layer_index,
                bio_profile=bio_profile,
                max_candidates=max_candidates,
                max_length=max_length,
                order=order,
                seed=run_seed,
                number_candidates=number_candidates,
                top_k=top_k,
            )
            for run_seed in eval_random_seeds
        ]
        averaged = average_metric_dicts(runs)
        averaged["seeds"] = eval_random_seeds
        return averaged
    return evaluate_candidate_segment(
        model,
        tokenizer,
        bio,
        docs,
        device=device,
        layer_index=layer_index,
        bio_profile=bio_profile,
        max_candidates=max_candidates,
        max_length=max_length,
        order=order,
        seed=seed,
        number_candidates=number_candidates,
        top_k=top_k,
    )


def load_training_docs(args: argparse.Namespace) -> tuple[list[Document], list[Document]]:
    docs = load_multi_domain_jsonl(Path(args.data))
    docs = _filter_extractive(docs, "train-data")
    rng = random.Random(args.candidate_seed)
    rng.shuffle(docs)
    if args.smoke:
        train_limit = min(args.train_limit or 80, 80)
        dev_limit = min(args.dev_limit or 20, 20)
    else:
        train_limit = args.train_limit
        dev_limit = args.dev_limit
    dev_docs = docs[: dev_limit or 100]
    train_docs = docs[dev_limit or 100 :]
    if train_limit is not None:
        train_docs = train_docs[:train_limit]
    return train_docs, dev_docs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_ROOT / "candidate_segment_attn" / datetime.now().strftime("%Y%m%d_%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA unavailable; using CPU")

    print(f"[info] output={output_dir}")
    print(f"[info] model={args.model} dtype={args.dtype} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    bio_device = device if device == "cuda" else "cpu"
    bio = BIOExtractor(args.bio_ckpt, device=bio_device)
    dtype = _torch_dtype(args.dtype)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype if device == "cuda" and dtype != torch.float32 else None,
        attn_implementation="eager",
    ).to(device)
    layer_index, layer_count = _resolve_layer_arg(args.layer, model)
    print(f"[info] layer={layer_index}/{layer_count - 1 if layer_count else '?'}")

    if args.adapter_dir:
        print(f"[info] loading adapter={args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir).to(device)

    eval_docs = _load_news_jsonl(Path(args.eval_data), limit=args.eval_limit)
    zero_metrics = evaluate_with_optional_seeds(
        model,
        tokenizer,
        bio,
        eval_docs,
        device=device,
        layer_index=layer_index,
        bio_profile=args.bio_profile,
        max_candidates=args.max_candidates,
        max_length=args.max_length,
        order=args.candidate_order,
        seed=args.candidate_seed,
        eval_random_seeds=args.eval_random_seeds,
        number_candidates=args.number_candidates,
        top_k=args.top_k,
    )
    print(f"[zero-shot] {json.dumps(zero_metrics, ensure_ascii=False)}")

    if args.zero_shot or args.adapter_dir:
        result_name = "adapter_metrics.json" if args.adapter_dir else "zero_shot_metrics.json"
        payload = {"config": vars(args), "metrics": zero_metrics}
        (output_dir / result_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    train_docs, dev_docs = load_training_docs(args)
    train_dataset = CandidateSegmentDataset(
        train_docs,
        tokenizer,
        bio,
        bio_profile=args.bio_profile,
        max_candidates=args.max_candidates,
        max_length=args.max_length,
        order=args.candidate_order,
        seed=args.candidate_seed,
        number_candidates=args.number_candidates,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_candidate_segments,
        num_workers=0,
    )
    print(f"[info] train_docs={len(train_docs)} usable_train={len(train_dataset)} dev_docs={len(dev_docs)} eval_docs={len(eval_docs)}")
    if len(train_dataset) == 0:
        raise RuntimeError("No usable training examples. Check BIO candidates, max_length, or labels.")

    lora_targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 2),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    log_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100, ascii=True, leave=False, mininterval=2.0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            token_scores = candidate_token_scores(outputs.attentions[layer_index], attention_mask)
            loss = candidate_soft_kl_loss(
                token_scores,
                batch["candidates"],
                soft_alpha=args.soft_alpha,
            )
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / max(steps, 1):.4f}")

        avg_loss = total_loss / max(steps, 1)
        dev_metrics = evaluate_with_optional_seeds(
            model,
            tokenizer,
            bio,
            dev_docs,
            device=device,
            layer_index=layer_index,
            bio_profile=args.bio_profile,
            max_candidates=args.max_candidates,
            max_length=args.max_length,
            order=args.candidate_order,
            seed=args.candidate_seed,
            eval_random_seeds=args.eval_random_seeds,
            number_candidates=args.number_candidates,
            top_k=args.top_k,
        )
        eval_metrics = evaluate_with_optional_seeds(
            model,
            tokenizer,
            bio,
            eval_docs,
            device=device,
            layer_index=layer_index,
            bio_profile=args.bio_profile,
            max_candidates=args.max_candidates,
            max_length=args.max_length,
            order=args.candidate_order,
            seed=args.candidate_seed,
            eval_random_seeds=args.eval_random_seeds,
            number_candidates=args.number_candidates,
            top_k=args.top_k,
        )
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev": dev_metrics,
            "eval": eval_metrics,
            "elapsed_seconds": round(time.perf_counter() - started_at, 2),
        }
        log_rows.append(row)
        (output_dir / "training_log.json").write_text(json.dumps(log_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[epoch {epoch}] loss={avg_loss:.4f} "
            f"dev_f1@10={dev_metrics.get('f1@10', 0):.4f} "
            f"news55_f1@10={eval_metrics.get('f1@10', 0):.4f}",
            flush=True,
        )
        if dev_metrics.get("f1@10", 0.0) > best_f1:
            best_f1 = dev_metrics.get("f1@10", 0.0)
            adapter_dir = output_dir / "best_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            print(f"[epoch {epoch}] saved best_adapter f1@10={best_f1:.4f}")

    final_payload = {
        "config": vars(args),
        "zero_shot": zero_metrics,
        "training_log": log_rows,
    }
    (output_dir / "final_eval.json").write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
