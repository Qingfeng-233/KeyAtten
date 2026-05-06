"""Gemini flash-lite evaluation on ShenCeCup held-out 100 docs (test_final).

Uses the same seed=42 split as train_qk_lora.py:
  all 1000 docs -> shuffle(seed=42) -> first 200 = test_pool
  test_pool[:100] = dev, test_pool[100:] = test_final (100 docs)

Only test_final is evaluated here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from keyword_bench.data import load_shencecup_labeled  # noqa: E402
from keyword_bench.metrics import evaluate_predictions  # noqa: E402
from keyword_bench.output_paths import resolve_output_dir  # noqa: E402
from llm_keyword_benchmark import (  # noqa: E402
    build_messages,
    extract_json_array,
    request_keywords,
    truncate_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini evaluation on ShenCeCup held-out 100 docs.")
    parser.add_argument("--output-dir", default="outputs_gemini_heldout_100")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def get_heldout_100() -> list:
    """Reproduce the exact test_final 100 docs from train_qk_lora.py split."""
    data_root = REPO_ROOT / "data"
    all_docs = load_shencecup_labeled(data_root)
    if not all_docs:
        print("[ERROR] Cannot find shencecup data.")
        return []
    rng = random.Random(42)
    rng.shuffle(all_docs)
    test_size = min(200, len(all_docs) // 5)
    test_pool = all_docs[:test_size]
    dev_docs = test_pool[:test_size // 2]
    test_final = test_pool[test_size // 2 :]
    print(f"[info] ShenCeCup total={len(all_docs)}, test_pool={test_size}, "
          f"dev={len(dev_docs)}, test_final={len(test_final)}")
    return test_final


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        import getpass
        api_key = getpass.getpass("API key: ").strip()

    docs = get_heldout_100()
    print(f"[info] Evaluating {len(docs)} held-out docs with {args.model}")

    predictions: list[list[str]] = [[] for _ in docs]
    raw_items: list[dict[str, Any] | None] = [None for _ in docs]

    def run_one(index_and_doc: tuple[int, Any]) -> tuple[int, dict[str, Any]]:
        index, doc = index_and_doc
        text = truncate_text(doc.text, args.max_chars)
        started = time.perf_counter()
        keywords, raw_response = request_keywords(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            text=text,
            language=doc.language,
            top_k=args.top_k,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        elapsed = time.perf_counter() - started
        return index, {
            "doc_id": doc.doc_id,
            "language": doc.language,
            "gold": doc.keywords,
            "prediction": keywords,
            "latency_seconds": elapsed,
            "response": raw_response,
        }

    completed = 0
    with futures.ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as executor:
        future_map = {
            executor.submit(run_one, (i, doc)): i
            for i, doc in enumerate(docs)
        }
        for future in futures.as_completed(future_map):
            idx = future_map[future]
            _, payload = future.result()
            predictions[idx] = payload["prediction"]
            raw_items[idx] = payload
            completed += 1
            if completed % 10 == 0 or completed == len(docs):
                print(f"[progress] {completed}/{len(docs)} done, "
                      f"last latency={payload['latency_seconds']:.2f}s")

    finalized = [item for item in raw_items if item is not None]
    metrics = evaluate_predictions(predictions, [doc.keywords for doc in docs], ks=(5, 10))
    mean_latency = sum(item["latency_seconds"] for item in finalized) / max(len(finalized), 1)

    result = {
        "model": args.model,
        "dataset": "shencecup_heldout_100",
        "doc_count": len(docs),
        "metrics": metrics,
        "mean_latency_seconds": mean_latency,
        "items": finalized,
    }

    (output_dir / "shencecup_heldout_100.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"Docs: {len(docs)} held-out test_final")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"Mean latency: {mean_latency:.2f}s")
    print(f"Results saved to: {output_dir / 'shencecup_heldout_100.json'}")


if __name__ == "__main__":
    main()
