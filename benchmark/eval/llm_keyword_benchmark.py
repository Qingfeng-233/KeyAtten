from __future__ import annotations

import argparse
import concurrent.futures as futures
import getpass
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from keyword_bench.data import build_all_eval_sets  # noqa: E402
from keyword_bench.metrics import evaluate_predictions  # noqa: E402
from keyword_bench.output_paths import resolve_output_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible LLM keyword extraction.")
    parser.add_argument("--root-dir", default=str(REPO_ROOT), help="Project root directory containing data/")
    parser.add_argument("--output-dir", default="outputs_llm_keyword_benchmark")
    parser.add_argument("--datasets", nargs="+", default=["shencecup_labeled", "semeval2010_fulltext"])
    parser.add_argument("--shencecup-limit", type=int, default=20)
    parser.add_argument("--english-limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def append_log(output_dir: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def resolve_api_key(env_name: str) -> str:
    api_key = os.environ.get(env_name)
    if api_key:
        return api_key.strip()
    return getpass.getpass("API key: ").strip()


def build_messages(text: str, language: str, top_k: int) -> list[dict[str, str]]:
    if language.startswith("zh"):
        system = (
            "You are a keyword extractor. Extract the most important keywords or key phrases from the given text. "
            "Prefer words or phrases that explicitly appear in the text. Do not explain, do not number, return only JSON array."
        )
        user = (
            f"Extract no more than {top_k} keywords, ranked by importance.\n"
            "Requirements:\n"
            "1. Keywords should be directly from the original text\n"
            "2. Prefer nouns, proper nouns, key verb-object phrases\n"
            "3. Return format must be JSON array, e.g. [\"keyword1\", \"keyword2\"]\n\n"
            f"Text:\n{text}"
        )
    else:
        system = (
            "You extract keywords from documents. Prefer exact phrases appearing in the text. "
            "Return JSON array only, with no explanation."
        )
        user = (
            f"Extract up to {top_k} keywords or keyphrases ranked by importance.\n"
            "Requirements:\n"
            "1. Prefer phrases that appear verbatim in the document\n"
            "2. Prefer nouns, named entities, and salient technical phrases\n"
            "3. Output must be a JSON array like [\"keyword 1\", \"keyword 2\"]\n\n"
            f"Document:\n{text}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_json_array(text: str) -> list[str]:
    stripped = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", stripped, re.IGNORECASE)
    if fenced_match:
        stripped = fenced_match.group(1)

    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[[\s\S]*\]", stripped)
    if array_match:
        try:
            payload = json.loads(array_match.group(0))
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass

    lines = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            lines.append(line)
    return lines


def request_keywords(
    *,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    language: str,
    top_k: int,
    timeout: int,
    max_retries: int,
) -> tuple[list[str], dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": build_messages(text, language=language, top_k=top_k),
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return extract_json_array(content)[:top_k], data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2.0 * attempt, 8.0))
    assert last_error is not None
    raise last_error


def truncate_text(text: str, max_chars: int) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = resolve_api_key(args.api_key_env)
    datasets = build_all_eval_sets(
        args.root_dir,
        english_limit=args.english_limit,
        shencecup_limit=args.shencecup_limit,
        train_limit=0,
        dev_limit=0,
        test_limit=0,
        derived_limit=0,
    )

    summary: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        docs = datasets.get(dataset_name, [])
        if not docs:
            append_log(output_dir, f"{dataset_name}: no documents found, skip")
            continue

        append_log(output_dir, f"{dataset_name}: start {len(docs)} docs")
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
                executor.submit(run_one, (index, doc)): index
                for index, doc in enumerate(docs)
            }
            for future in futures.as_completed(future_map):
                index = future_map[future]
                item = future.result()
                _, payload = item
                predictions[index] = payload["prediction"]
                raw_items[index] = payload
                completed += 1
                append_log(
                    output_dir,
                    f"{dataset_name}: {completed}/{len(docs)} docs, "
                    f"{payload['latency_seconds']:.2f}s, pred={payload['prediction'][:5]}",
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        finalized_items = [item for item in raw_items if item is not None]
        metrics = evaluate_predictions(predictions, [doc.keywords for doc in docs], ks=(5, 10))
        mean_latency = sum(item["latency_seconds"] for item in finalized_items) / max(len(finalized_items), 1)
        summary["datasets"][dataset_name] = {
            "doc_count": len(docs),
            "metrics": metrics,
            "mean_latency_seconds": mean_latency,
        }

        (output_dir / f"{dataset_name}.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "doc_count": len(docs),
                    "metrics": metrics,
                    "mean_latency_seconds": mean_latency,
                    "items": finalized_items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        append_log(output_dir, f"{dataset_name}: metrics={metrics}, mean_latency={mean_latency:.2f}s")

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
