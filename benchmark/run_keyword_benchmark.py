from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

from keyword_bench.data import build_all_eval_sets
from keyword_bench.methods import (
    batched_attention_word_scores,
    build_candidates,
    build_model_bundle,
    candidate_score_values,
    candidate_rank_from_word_scores,
    combine_word_scores,
    embed_texts,
    inverse_document_frequency,
    keybert_candidate_scores_from_doc_embedding,
    rank_candidates_from_scores,
    segment_text,
    token_counter,
    textrank_word_scores,
    word_scores_from_token_values,
    yake_keywords,
)
from keyword_bench.metrics import evaluate_predictions


MODEL_REGISTRY = [
    {"name": "bert-base-chinese", "benchmark_family": "none", "benchmark_score": None},
    {"name": "BAAI/bge-small-zh-v1.5", "benchmark_family": "c-mteb", "benchmark_score": 57.82},
    {"name": "moka-ai/m3e-base", "benchmark_family": "none", "benchmark_score": None},
    {"name": "moka-ai/m3e-small", "benchmark_family": "none", "benchmark_score": None},
    {"name": "thenlper/gte-small-zh", "benchmark_family": "c-mteb", "benchmark_score": 60.08},
    {"name": "thenlper/gte-base-zh", "benchmark_family": "c-mteb", "benchmark_score": 65.92},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "benchmark_family": "none", "benchmark_score": None},
    {"name": "distilbert-base-uncased", "benchmark_family": "none", "benchmark_score": None},
    {"name": "prajjwal1/bert-tiny", "benchmark_family": "none", "benchmark_score": None},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root.")
    parser.add_argument("--output-dir", default="outputs", help="Where to save outputs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "csl_dev",
            "csl_test",
            "csl_test_short",
            "csl_test_medium",
            "csl_test_long",
            "csl_test_kw_le_4",
            "csl_test_kw_ge_5",
        ],
    )
    parser.add_argument("--models", nargs="+", default=[item["name"] for item in MODEL_REGISTRY])
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--dev-limit", type=int, default=250)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--derived-limit", type=int, default=200)
    parser.add_argument("--english-limit", type=int, default=120)
    parser.add_argument("--shencecup-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-batch-size", type=int, default=4, help="Batch size for encoder attention forward passes.")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="Batch size for embedding-based methods like KeyBERT.")
    parser.add_argument("--log-every-docs", type=int, default=100, help="Log and persist progress every N documents.")
    parser.add_argument("--log-every-batches", type=int, default=10, help="Log progress every N model batches.")
    parser.add_argument("--skip-yake", action="store_true")
    parser.add_argument(
        "--candidate-aggregation",
        default="mean",
        choices=["mean", "max", "top2_mean", "sum_sqrt_len"],
        help="How to aggregate word scores into phrase scores.",
    )
    parser.add_argument("--repeat-boost", type=float, default=0.0, help="Boost repeated words inside phrases.")
    parser.add_argument(
        "--attention-layer-specs",
        nargs="+",
        default=["last"],
        help="Attention layer presets: last, second_last, third_last, mean_last2, mean_last3, or layer:<index>.",
    )
    return parser.parse_args()


def parse_attention_layer_specs(layer_specs: List[str]) -> List[dict]:
    parsed = []
    for spec in layer_specs:
        if spec == "last":
            parsed.append({"name": "last", "indices": [-1]})
        elif spec == "second_last":
            parsed.append({"name": "second_last", "indices": [-2]})
        elif spec == "third_last":
            parsed.append({"name": "third_last", "indices": [-3]})
        elif spec == "mean_last2":
            parsed.append({"name": "mean_last2", "indices": [-1, -2]})
        elif spec == "mean_last3":
            parsed.append({"name": "mean_last3", "indices": [-1, -2, -3]})
        elif spec.startswith("layer:"):
            layer_value = int(spec.split(":", 1)[1])
            normalized_name = f"layer_{layer_value}".replace("-", "neg")
            parsed.append({"name": normalized_name, "indices": [layer_value]})
        else:
            raise ValueError(f"Unsupported attention layer spec: {spec}")
    return parsed


def append_debug_log(output_dir: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with (output_dir / "run_debug.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_progress(output_dir: Path, payload: dict) -> None:
    (output_dir / "run_progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_dataset_methods(
    docs,
    model_bundle,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    top_k: int,
    run_yake: bool,
    attention_configs: List[dict],
    candidate_aggregation: str,
    repeat_boost: float,
    attention_batch_size: int,
    embedding_batch_size: int,
    log_every_docs: int,
    log_every_batches: int,
) -> Dict[str, List[List[str]]]:
    predictions: Dict[str, List[List[str]]] = {
        "termfreq": [],
        "tfidf": [],
        "textrank": [],
        "textrank_idf": [],
        "keybert": [],
        "keybert_idf": [],
    }
    for config in attention_configs:
        suffix = "" if config["name"] == "last" else f"@{config['name']}"
        predictions[f"cls_attn{suffix}"] = []
        predictions[f"received_attn{suffix}"] = []
        predictions[f"samrank{suffix}"] = []
        predictions[f"fusion_attn{suffix}"] = []
        predictions[f"cls_attn_idf{suffix}"] = []
        predictions[f"received_attn_idf{suffix}"] = []
        predictions[f"samrank_idf{suffix}"] = []
        predictions[f"fusion_attn_idf{suffix}"] = []
    if run_yake:
        predictions["yake"] = []

    preprocessed = []
    total_docs = len(docs)
    append_debug_log(output_dir, f"{model_name} / {dataset_name}: preprocessing {total_docs} docs")
    write_progress(
        output_dir,
        {
            "model": model_name,
            "dataset": dataset_name,
            "phase": "preprocessing",
            "processed_docs": 0,
            "total_docs": total_docs,
        },
    )
    for doc_index, doc in enumerate(docs, start=1):
        words, pos_tags = segment_text(doc.text, language=doc.language)
        candidates = build_candidates(words, pos_tags, language=doc.language)
        preprocessed.append(
            {
                "doc": doc,
                "words": words,
                "pos_tags": pos_tags,
                "candidates": candidates,
                "candidate_starts": np.fromiter((candidate.word_start for candidate in candidates), dtype=np.int32, count=len(candidates)),
                "candidate_ends": np.fromiter((candidate.word_end for candidate in candidates), dtype=np.int32, count=len(candidates)),
                "token_counts": token_counter(words, pos_tags, language=doc.language),
            }
        )
        if doc_index == 1 or doc_index == total_docs or (log_every_docs > 0 and doc_index % log_every_docs == 0):
            append_debug_log(output_dir, f"{model_name} / {dataset_name}: preprocessed {doc_index}/{total_docs} docs")
            write_progress(
                output_dir,
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "phase": "preprocessing",
                    "processed_docs": doc_index,
                    "total_docs": total_docs,
                },
            )

    idf_lookup = inverse_document_frequency(item["token_counts"].keys() for item in preprocessed)
    append_debug_log(output_dir, f"{model_name} / {dataset_name}: IDF lookup ready ({len(idf_lookup)} unique tokens)")
    doc_embeddings = embed_texts(
        model_bundle,
        [item["doc"].text for item in preprocessed],
        batch_size=embedding_batch_size,
        progress_label=f"{dataset_name} keybert",
        log_every_batches=log_every_batches,
    )
    append_debug_log(output_dir, f"{model_name} / {dataset_name}: KeyBERT document embeddings ready")
    required_layer_indices = []
    seen_layer_indices = set()
    for config in attention_configs:
        for layer_index in config["indices"]:
            if layer_index in seen_layer_indices:
                continue
            seen_layer_indices.add(layer_index)
            required_layer_indices.append(layer_index)
    attention_scores_cache = batched_attention_word_scores(
        [item["words"] for item in preprocessed],
        model_bundle,
        layer_indices=required_layer_indices,
        batch_size=attention_batch_size,
        progress_label=f"{dataset_name} attention",
        log_every_batches=log_every_batches,
    )
    append_debug_log(
        output_dir,
        f"{model_name} / {dataset_name}: attention cache ready for {len(required_layer_indices)} layer spec(s)",
    )
    write_progress(
        output_dir,
        {
            "model": model_name,
            "dataset": dataset_name,
            "phase": "ranking",
            "processed_docs": 0,
            "total_docs": total_docs,
        },
    )

    for doc_index, (item, doc_embedding, per_doc_layer_scores) in enumerate(
        zip(preprocessed, doc_embeddings, attention_scores_cache),
        start=1,
    ):
        doc = item["doc"]
        words = item["words"]
        pos_tags = item["pos_tags"]
        candidates = item["candidates"]
        candidate_starts = item["candidate_starts"]
        candidate_ends = item["candidate_ends"]
        token_counts = item["token_counts"]
        predictions["termfreq"].append(
            candidate_rank_from_word_scores(
                candidates,
                word_scores_from_token_values(words, pos_tags, token_counts, language=doc.language),
                top_k=top_k,
                token_counts=token_counts,
                words=words,
                aggregation_mode=candidate_aggregation,
                repeat_boost=repeat_boost,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
            )
        )
        tfidf_values = {token: count * idf_lookup.get(token, 0.0) for token, count in token_counts.items()}
        tfidf_word_scores = word_scores_from_token_values(words, pos_tags, tfidf_values, language=doc.language)
        tfidf_candidate_scores = candidate_score_values(
            candidates,
            tfidf_word_scores,
            token_counts=token_counts,
            words=words,
            aggregation_mode=candidate_aggregation,
            repeat_boost=repeat_boost,
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
        )
        predictions["tfidf"].append(
            candidate_rank_from_word_scores(
                candidates,
                tfidf_word_scores,
                top_k=top_k,
                token_counts=token_counts,
                words=words,
                aggregation_mode=candidate_aggregation,
                repeat_boost=repeat_boost,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
            )
        )
        textrank_scores = textrank_word_scores(words, pos_tags, language=doc.language)
        predictions["textrank"].append(
            candidate_rank_from_word_scores(
                candidates,
                textrank_scores,
                top_k=top_k,
                token_counts=token_counts,
                words=words,
                aggregation_mode=candidate_aggregation,
                repeat_boost=repeat_boost,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
            )
        )
        predictions["textrank_idf"].append(
            candidate_rank_from_word_scores(
                candidates,
                combine_word_scores(textrank_scores, tfidf_word_scores, mode="product"),
                top_k=top_k,
                token_counts=token_counts,
                words=words,
                aggregation_mode=candidate_aggregation,
                repeat_boost=repeat_boost,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
            )
        )
        keybert_candidate_scores = keybert_candidate_scores_from_doc_embedding(
            candidates,
            doc_embedding,
            model_bundle,
            batch_size=embedding_batch_size,
        )
        predictions["keybert"].append(rank_candidates_from_scores(candidates, keybert_candidate_scores, top_k=top_k))
        predictions["keybert_idf"].append(
            rank_candidates_from_scores(
                candidates,
                combine_word_scores(keybert_candidate_scores, tfidf_candidate_scores, mode="product"),
                top_k=top_k,
            )
        )
        for config in attention_configs:
            suffix = "" if config["name"] == "last" else f"@{config['name']}"
            layer_scores = [per_doc_layer_scores[layer_index] for layer_index in config["indices"]]
            if len(layer_scores) == 1:
                word_scores_by_method = layer_scores[0]
            else:
                word_scores_by_method = {
                    method_name: np.average(
                        np.stack([scores[method_name] for scores in layer_scores], axis=0),
                        axis=0,
                    )
                    for method_name in layer_scores[0]
                }
            for method_name, word_scores in word_scores_by_method.items():
                predictions[f"{method_name}{suffix}"].append(
                    candidate_rank_from_word_scores(
                        candidates,
                        word_scores,
                        top_k=top_k,
                        token_counts=token_counts,
                        words=words,
                        aggregation_mode=candidate_aggregation,
                        repeat_boost=repeat_boost,
                        candidate_starts=candidate_starts,
                        candidate_ends=candidate_ends,
                    )
                )
                hybrid_word_scores = combine_word_scores(word_scores, tfidf_word_scores, mode="product")
                predictions[f"{method_name}_idf{suffix}"].append(
                    candidate_rank_from_word_scores(
                        candidates,
                        hybrid_word_scores,
                        top_k=top_k,
                        token_counts=token_counts,
                        words=words,
                        aggregation_mode=candidate_aggregation,
                        repeat_boost=repeat_boost,
                        candidate_starts=candidate_starts,
                        candidate_ends=candidate_ends,
                    )
                )
        if run_yake:
            predictions["yake"].append(yake_keywords(doc.text, top_k=top_k, language=doc.language)[:top_k])
        if doc_index == 1 or doc_index == total_docs or (log_every_docs > 0 and doc_index % log_every_docs == 0):
            append_debug_log(output_dir, f"{model_name} / {dataset_name}: ranked {doc_index}/{total_docs} docs")
            write_progress(
                output_dir,
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "phase": "ranking",
                    "processed_docs": doc_index,
                    "total_docs": total_docs,
                },
            )
    return predictions


def compute_correlations(rows: List[dict], metric_name: str = "f1@10") -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        if row.get("benchmark_score") is None:
            continue
        key = f"{row['dataset']}::{row['method']}"
        grouped.setdefault(key, []).append(row)

    results = []
    for key, items in grouped.items():
        if len(items) < 3:
            continue
        scores = [item["benchmark_score"] for item in items]
        metrics = [item[metric_name] for item in items]
        pearson = pearsonr(scores, metrics)
        spearman = spearmanr(scores, metrics)
        dataset_name, method_name = key.split("::", 1)
        results.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "metric": metric_name,
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "num_models": len(items),
            }
        )
    return sorted(results, key=lambda item: (item["dataset"], item["method"]))


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_eval_sets = build_all_eval_sets(
        root_dir,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        derived_limit=args.derived_limit,
        english_limit=args.english_limit,
        shencecup_limit=args.shencecup_limit,
    )

    selected_sets = {name: all_eval_sets[name] for name in args.datasets}
    registry = {item["name"]: item for item in MODEL_REGISTRY}
    all_rows: List[dict] = []
    attention_configs = parse_attention_layer_specs(args.attention_layer_specs)
    append_debug_log(output_dir, f"Benchmark start: {len(args.models)} model(s), {len(selected_sets)} dataset(s)")

    for model_name in args.models:
        append_debug_log(output_dir, f"Running model: {model_name}")
        local_model_dir = root_dir / "models" / model_name.replace("/", "__")
        model_source = str(local_model_dir) if (local_model_dir / "download.ok").exists() else model_name
        model_bundle = build_model_bundle(model_source, args.device)
        for dataset_name, docs in selected_sets.items():
            append_debug_log(output_dir, f"Dataset start: {dataset_name} ({len(docs)} docs)")
            write_progress(
                output_dir,
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "phase": "starting",
                    "processed_docs": 0,
                    "total_docs": len(docs),
                },
            )
            predictions = run_dataset_methods(
                docs,
                model_bundle,
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=output_dir,
                top_k=args.top_k,
                run_yake=not args.skip_yake,
                attention_configs=attention_configs,
                candidate_aggregation=args.candidate_aggregation,
                repeat_boost=args.repeat_boost,
                attention_batch_size=args.attention_batch_size,
                embedding_batch_size=args.embedding_batch_size,
                log_every_docs=args.log_every_docs,
                log_every_batches=args.log_every_batches,
            )
            golds = [doc.keywords for doc in docs]
            for method_name, method_predictions in predictions.items():
                metrics = evaluate_predictions(method_predictions, golds)
                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "method": method_name,
                    "doc_count": len(docs),
                    **metrics,
                    "benchmark_family": registry.get(model_name, {}).get("benchmark_family"),
                    "benchmark_score": registry.get(model_name, {}).get("benchmark_score"),
                }
                all_rows.append(row)
                append_debug_log(
                    output_dir,
                    f"{model_name} / {dataset_name} / {method_name}: "
                    f"f1@5={row.get('f1@5', 0.0):.4f} "
                    f"f1@10={row.get('f1@10', 0.0):.4f} "
                    f"r@10={row.get('r@10', 0.0):.4f}"
                )
            (output_dir / "keyword_benchmark_results.partial.json").write_text(
                json.dumps(all_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_progress(
                output_dir,
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "phase": "completed_dataset",
                    "processed_docs": len(docs),
                    "total_docs": len(docs),
                    "result_rows": len(all_rows),
                },
            )

    (output_dir / "keyword_benchmark_results.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    correlations = compute_correlations(all_rows, metric_name="f1@10")
    (output_dir / "embedding_attention_correlations.json").write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    append_debug_log(output_dir, "Correlation summary")
    for row in correlations:
        append_debug_log(
            output_dir,
            f"{row['dataset']} / {row['method']}: "
            f"pearson={row['pearson_r']:.4f}, spearman={row['spearman_rho']:.4f}, n={row['num_models']}"
        )
    write_progress(
        output_dir,
        {
            "phase": "completed_all",
            "result_rows": len(all_rows),
            "correlation_rows": len(correlations),
        },
    )


if __name__ == "__main__":
    main()
