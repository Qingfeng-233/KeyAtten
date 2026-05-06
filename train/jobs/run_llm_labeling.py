#!/usr/bin/env python3
"""Batch LLM keyword labeling for attention distillation."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Keyatten")
sys.path.insert(0, "/root/Keyatten/benchmark")

from keyword_bench.data import Document, load_multi_domain_jsonl, load_shencecup_labeled

from train.jobs.llm_teacher import QwenKeywordTeacher


PROJECT_ROOT = Path("/root/Keyatten")
MD_PATH = PROJECT_ROOT / "train" / "data" / "multi_domain.jsonl"


def load_train_docs(seed: int, limit: int | None, source: str) -> list[Document]:
    rng = random.Random(seed)
    docs: list[Document] = []
    if source in ("shence", "mixed"):
        shence = load_shencecup_labeled(PROJECT_ROOT)
        rng.shuffle(shence)
        docs.extend(shence[200:])
    if source in ("md", "mixed"):
        md = load_multi_domain_jsonl(MD_PATH)
        rng.shuffle(md)
        docs.extend(md)
    rng.shuffle(docs)
    if limit:
        docs = docs[:limit]
    return docs


def load_sanity_docs(seed: int, limit: int) -> list[Document]:
    rng = random.Random(seed)
    docs = load_shencecup_labeled(PROJECT_ROOT)
    rng.shuffle(docs)
    return docs[:limit]


def write_record(handle, doc: Document, llm_keywords: list[str], raw_output: str) -> None:
    obj = {
        "doc_id": doc.doc_id,
        "text": doc.text,
        "gold_keywords": doc.keywords,
        "llm_keywords": [{"rank": idx, "keyword": kw} for idx, kw in enumerate(llm_keywords, 1)],
        "raw_output": raw_output,
    }
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/Keyatten/models/Qwen3.5-4B")
    parser.add_argument("--output", default="/root/Keyatten/train/data/llm_labels_1k.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--source", choices=("mixed", "shence", "md"), default="mixed")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32", "auto"), default="bfloat16")
    parser.add_argument("--max-gpu-memory", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = load_sanity_docs(args.seed, args.limit) if args.sanity else load_train_docs(args.seed, args.limit, args.source)
    if args.start_index:
        docs = docs[args.start_index:]
    if args.num_shards > 1:
        docs = [doc for idx, doc in enumerate(docs) if idx % args.num_shards == args.shard_index]
    print(f"[info] docs={len(docs)} output={out_path}", flush=True)

    teacher = QwenKeywordTeacher(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_gpu_memory=args.max_gpu_memory,
        load_in_4bit=args.load_in_4bit,
    )
    started = time.perf_counter()
    hit_docs = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, doc in enumerate(docs, 1):
            t0 = time.perf_counter()
            try:
                kws, raw = teacher.generate_keywords(doc.text, top_k=args.top_k)
            except Exception as exc:
                kws, raw = [], f"[ERROR] {type(exc).__name__}: {exc}"
            hit_docs += int(bool(kws))
            write_record(handle, doc, kws, raw)
            if idx % 1 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[label] {idx}/{len(docs)} kws={len(kws)} hit_docs={hit_docs} "
                    f"doc_sec={time.perf_counter() - t0:.2f} total_min={elapsed / 60:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
