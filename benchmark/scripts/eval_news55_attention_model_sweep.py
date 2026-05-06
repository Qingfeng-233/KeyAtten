#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

import torch

from benchmark.keyword_bench.metrics import evaluate_predictions
from keyatten import KeyAttenExtractor


DEFAULT_MODELS = [
    str(PROJECT_ROOT / "models" / "gte-small-zh"),
    str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B"),
    str(PROJECT_ROOT / "models" / "ckiplab-bert-base-chinese-ner"),
]


def load_news55(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"].strip()
            keywords = [kw for kw in obj.get("keywords", []) if kw in text]
            if text and keywords:
                docs.append(
                    {
                        "id": str(obj.get("id", len(docs) + 1)),
                        "text": text,
                        "keywords": keywords,
                    }
                )
            if limit is not None and len(docs) >= limit:
                break
    return docs


def eval_one(model_path: str, docs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    print(f"[model] {model_path}", flush=True)
    extractor = KeyAttenExtractor(
        model=model_path,
        language="zh",
        device=args.device,
        layer_index=args.layer_index,
        instruction_prefix=args.instruction_prefix,
        is_causal_override=args.is_causal_override,
        candidate_scoring=args.candidate_scoring,
        dtype=args.dtype,
    )

    model_result: dict[str, Any] = {"model": model_path, "methods": {}}
    for method in args.methods:
        predictions: list[list[str]] = []
        golds: list[list[str]] = []
        method_started_at = time.perf_counter()
        for index, doc in enumerate(docs, 1):
            try:
                prediction = extractor.extract_keywords(doc["text"], method=method, top_k=args.top_k)
            except Exception as exc:  # noqa: BLE001 - long sweeps should keep going.
                print(
                    f"[error] model={model_path} method={method} doc={doc['id']} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                prediction = []
            predictions.append(prediction)
            golds.append(doc["keywords"])
            if index % args.log_every == 0:
                print(f"  [{method}] {index}/{len(docs)}", flush=True)

        metrics = evaluate_predictions(predictions, golds)
        model_result["methods"][method] = metrics
        print(
            f"  [{method}] F1@5={metrics.get('f1@5', 0):.4f} "
            f"F1@10={metrics.get('f1@10', 0):.4f} "
            f"P@10={metrics.get('p@10', 0):.4f} "
            f"R@10={metrics.get('r@10', 0):.4f} "
            f"({time.perf_counter() - method_started_at:.1f}s)",
            flush=True,
        )

    model_result["seconds"] = round(time.perf_counter() - started_at, 2)
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_result


def normalize_is_causal(value: str | None) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "news_annotated.jsonl"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["received_attn", "cls_attn", "samrank", "fusion_attn"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--candidate-scoring", default="word", choices=["word", "token_span"])
    parser.add_argument("--instruction-prefix", default=None)
    parser.add_argument("--is-causal-override", default=None, choices=["true", "false", "auto"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "news55_attention_model_sweep_local.json"))
    args = parser.parse_args()
    args.is_causal_override = normalize_is_causal(args.is_causal_override)

    docs = load_news55(Path(args.data), limit=args.limit)
    output: dict[str, Any] = {
        "dataset": {
            "path": args.data,
            "docs": len(docs),
            "gold": sum(len(doc["keywords"]) for doc in docs),
        },
        "config": vars(args),
        "results": [],
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    for model in args.models:
        try:
            output["results"].append(eval_one(model, docs, args))
        except Exception as exc:  # noqa: BLE001 - record failed models in the sweep output.
            print(f"[model-error] {model}: {type(exc).__name__}: {exc}", flush=True)
            output["results"].append({"model": model, "error": f"{type(exc).__name__}: {exc}"})
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[progress] saved {output_path}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
