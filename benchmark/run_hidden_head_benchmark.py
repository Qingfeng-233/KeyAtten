from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from keyword_bench.data import (
    build_csl_eval_sets,
    build_english_eval_sets,
    build_shencecup_eval_sets,
)
from keyword_bench.hidden_state_head import (
    HiddenStateKeywordHead,
    aggregate_token_probs_to_words,
)
from keyword_bench.methods import (
    build_candidates,
    candidate_rank_from_word_scores,
    combine_word_scores,
    inverse_document_frequency,
    segment_text,
    token_counter,
    word_scores_from_token_values,
)
from keyword_bench.metrics import evaluate_predictions
from keyword_bench.output_paths import resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hidden-state keyword head on benchmark datasets."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", default="outputs_hidden_head_eval")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to best_hidden_head.pt"
    )
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--datasets", nargs="+", default=["csl_dev", "csl_test"])
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--dev-limit", type=int, default=250)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--derived-limit", type=int, default=200)
    parser.add_argument("--english-limit", type=int, default=120)
    parser.add_argument("--shencecup-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--candidate-aggregation",
        default="mean",
        choices=["mean", "max", "top2_mean", "sum_sqrt_len"],
    )
    parser.add_argument("--repeat-boost", type=float, default=0.0)
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    if "classifier_state" not in payload:
        raise RuntimeError("Invalid checkpoint: missing 'classifier_state'.")
    return payload


def _resolve_model_source(root_dir: Path, model_name: str) -> str:
    local_model_dir = root_dir / "models" / model_name.replace("/", "__")
    if (local_model_dir / "download.ok").exists():
        return str(local_model_dir)
    return model_name


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model_name = args.model or str(
        checkpoint.get("model_name") or "thenlper/gte-small-zh"
    )
    model_source = _resolve_model_source(root_dir, model_name)
    max_length = int(checkpoint.get("max_length", 512))
    layer_index = int(checkpoint.get("layer_index", -1))

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must provide a pad token id.")

    model = HiddenStateKeywordHead(
        model_source, layer_index=layer_index, freeze_backbone=True
    )
    model.classifier.load_state_dict(checkpoint["classifier_state"])
    model.to(device)
    model.eval()

    requested = set(args.datasets)
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

    rows: list[dict] = []
    for dataset_name in args.datasets:
        if dataset_name not in all_eval_sets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        docs = all_eval_sets[dataset_name]
        predictions_hidden: list[list[str]] = []
        predictions_hidden_idf: list[list[str]] = []
        golds = [doc.keywords for doc in docs]

        preprocessed = []
        for doc in docs:
            words, pos_tags = segment_text(doc.text, language=doc.language)
            candidates = build_candidates(words, pos_tags, language=doc.language)
            candidate_starts = np.fromiter(
                (c.word_start for c in candidates),
                dtype=np.int32,
                count=len(candidates),
            )
            candidate_ends = np.fromiter(
                (c.word_end for c in candidates), dtype=np.int32, count=len(candidates)
            )
            token_counts = token_counter(words, pos_tags, language=doc.language)
            preprocessed.append(
                {
                    "doc": doc,
                    "words": words,
                    "pos_tags": pos_tags,
                    "candidates": candidates,
                    "candidate_starts": candidate_starts,
                    "candidate_ends": candidate_ends,
                    "token_counts": token_counts,
                }
            )

        idf_lookup = inverse_document_frequency(
            item["token_counts"].keys() for item in preprocessed
        )

        for start in range(0, len(preprocessed), args.batch_size):
            batch = preprocessed[start : start + args.batch_size]
            encoded = tokenizer(
                [item["words"] for item in batch],
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            word_ids_per_item = [
                encoded.word_ids(batch_index=index) for index in range(len(batch))
            ]
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                masks = encoded["attention_mask"].detach().cpu().numpy()

            for item_index, item in enumerate(batch):
                valid_token_count = int(masks[item_index].sum())
                token_probs = probs[item_index, :valid_token_count]
                word_scores = aggregate_token_probs_to_words(
                    word_ids_per_item[item_index][:valid_token_count],
                    token_probs,
                    word_count=len(item["words"]),
                )
                tfidf_values = {
                    token: count * idf_lookup.get(token, 0.0)
                    for token, count in item["token_counts"].items()
                }
                tfidf_word_scores = word_scores_from_token_values(
                    item["words"],
                    item["pos_tags"],
                    tfidf_values,
                    language=item["doc"].language,
                )
                hidden_idf_scores = combine_word_scores(
                    word_scores, tfidf_word_scores, mode="product"
                )

                predictions_hidden.append(
                    candidate_rank_from_word_scores(
                        item["candidates"],
                        word_scores,
                        top_k=args.top_k,
                        token_counts=item["token_counts"],
                        words=item["words"],
                        aggregation_mode=args.candidate_aggregation,
                        repeat_boost=args.repeat_boost,
                        candidate_starts=item["candidate_starts"],
                        candidate_ends=item["candidate_ends"],
                    )
                )
                predictions_hidden_idf.append(
                    candidate_rank_from_word_scores(
                        item["candidates"],
                        hidden_idf_scores,
                        top_k=args.top_k,
                        token_counts=item["token_counts"],
                        words=item["words"],
                        aggregation_mode=args.candidate_aggregation,
                        repeat_boost=args.repeat_boost,
                        candidate_starts=item["candidate_starts"],
                        candidate_ends=item["candidate_ends"],
                    )
                )

        metrics_hidden = evaluate_predictions(predictions_hidden, golds)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "method": "hidden_state_head",
                "doc_count": len(docs),
                **metrics_hidden,
            }
        )
        metrics_hidden_idf = evaluate_predictions(predictions_hidden_idf, golds)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "method": "hidden_state_head_idf",
                "doc_count": len(docs),
                **metrics_hidden_idf,
            }
        )

    (output_dir / "hidden_head_benchmark_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
