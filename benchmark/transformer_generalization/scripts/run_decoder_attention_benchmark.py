from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDOR_NLP = PROJECT_ROOT / ".vendor_nlp"
if VENDOR_NLP.exists() and str(VENDOR_NLP) not in sys.path:
    sys.path.insert(0, str(VENDOR_NLP))

from transformers import AutoModelForCausalLM, AutoTokenizer

from keyword_bench.data import build_all_eval_sets
from keyword_bench.methods import build_candidates, candidate_rank_from_word_scores, segment_text
from keyword_bench.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", default=["csl_test", "shencecup_labeled"])
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--shencecup-limit", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="transformer_generalization/results/qwen_decoder_benchmark.json")
    return parser.parse_args()


def _aggregate_subwords_to_words(word_ids: list[int | None], token_scores: np.ndarray, word_count: int) -> np.ndarray:
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_scores[token_index])
        counts[word_id] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def _resolve_anchor_index(input_ids: torch.Tensor, tokenizer, anchor: str) -> int:
    if anchor == "last_token":
        return input_ids.shape[1] - 1
    if anchor == "eos_token" and tokenizer.eos_token_id is not None:
        eos_positions = (input_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel():
            return int(eos_positions[-1].item())
    return input_ids.shape[1] - 1


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_path = (root_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.to(args.device)
    model.eval()

    datasets = build_all_eval_sets(root_dir, test_limit=args.test_limit, shencecup_limit=args.shencecup_limit)
    results = []
    anchors = ["last_token", "eos_token"]
    methods = anchors + ["received_attn"]

    for dataset_name in args.datasets:
        docs = datasets[dataset_name]
        predictions = {name: [] for name in methods}
        golds = [doc.keywords for doc in docs]
        token_lengths = []
        for doc in docs:
            words, pos_tags = segment_text(doc.text, language=doc.language)
            candidates = build_candidates(words, pos_tags, language=doc.language)
            encoded = tokenizer(
                list(words),
                is_split_into_words=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            word_ids = encoded.word_ids(batch_index=0)
            token_lengths.append(int(encoded["input_ids"].shape[1]))
            encoded = {key: value.to(args.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded, output_attentions=True, return_dict=True)
            attention_map = outputs.attentions[-1][0].mean(dim=0).float().detach().cpu().numpy()
            received_scores = attention_map.sum(axis=0)
            received_word_scores = _aggregate_subwords_to_words(word_ids, received_scores, len(words))
            predictions["received_attn"].append(
                candidate_rank_from_word_scores(candidates, received_word_scores, top_k=args.top_k)
            )

            for anchor in anchors:
                anchor_index = _resolve_anchor_index(encoded["input_ids"], tokenizer, anchor)
                anchor_scores = attention_map[anchor_index]
                anchor_word_scores = _aggregate_subwords_to_words(word_ids, anchor_scores, len(words))
                predictions[anchor].append(
                    candidate_rank_from_word_scores(candidates, anchor_word_scores, top_k=args.top_k)
                )

        average_token_length = float(np.mean(token_lengths)) if token_lengths else 0.0
        for method_name, method_predictions in predictions.items():
            metrics = evaluate_predictions(method_predictions, golds)
            results.append(
                {
                    "model": args.model,
                    "dataset": dataset_name,
                    "method": method_name,
                    "doc_count": len(docs),
                    "avg_token_length": average_token_length,
                    **metrics,
                }
            )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
