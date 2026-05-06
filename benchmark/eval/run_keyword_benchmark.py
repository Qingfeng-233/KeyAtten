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
    DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX,
    batched_attention_word_scores,
    batched_hidden_word_scores,
    build_attention_gated_candidates,
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
from keyword_bench.output_paths import resolve_output_dir


MODEL_REGISTRY = [
    {"name": "bert-base-chinese", "benchmark_family": "none", "benchmark_score": None},
    {"name": "BAAI/bge-small-zh-v1.5", "benchmark_family": "c-mteb", "benchmark_score": 57.82},
    {"name": "moka-ai/m3e-base", "benchmark_family": "none", "benchmark_score": None},
    {"name": "moka-ai/m3e-small", "benchmark_family": "none", "benchmark_score": None},
    {"name": "thenlper/gte-small-zh", "benchmark_family": "c-mteb", "benchmark_score": 60.08},
    {"name": "thenlper/gte-base-zh", "benchmark_family": "c-mteb", "benchmark_score": 65.92},
    {"name": "Qwen/Qwen3-Embedding-0.6B", "benchmark_family": "none", "benchmark_score": None},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "benchmark_family": "none", "benchmark_score": None},
    {"name": "distilbert-base-uncased", "benchmark_family": "none", "benchmark_score": None},
    {"name": "prajjwal1/bert-tiny", "benchmark_family": "none", "benchmark_score": None},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root.")
    parser.add_argument("--output-dir", default="outputs", help="Where to save outputs. Relative paths resolve under 测试沙箱/Outputs.")
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
        "--instruction-prefix-zh-causal",
        default=DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX,
        help="Instruction prefix automatically prepended for causal models on Chinese datasets. Empty string disables it.",
    )
    parser.add_argument(
        "--is-causal-override",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override benchmark readout mode. Use false to treat decoder attention as non-causal for control experiments.",
    )
    parser.add_argument(
        "--true-bidirectional-attention",
        action="store_true",
        help="Monkey-patch supported decoder models to replace the internal causal mask with full bidirectional visibility during attention scoring.",
    )
    parser.add_argument(
        "--cls-head-strategies",
        nargs="+",
        default=[],
        choices=["topk", "softmax", "somp"],
        help="Optional unsupervised head selection strategies applied only to cls_attn on the benchmark side.",
    )
    parser.add_argument(
        "--cls-head-top-k",
        type=int,
        default=4,
        help="Number of heads kept by cls_attn_headtopk.",
    )
    parser.add_argument(
        "--cls-head-temperature",
        type=float,
        default=8.0,
        help="Softmax temperature tau for cls_attn_headsoftmax.",
    )
    parser.add_argument(
        "--somp-alpha",
        type=float,
        default=512.0,
        help="Alpha coefficient in SOMP head weighting exp(alpha * variance - beta * local_mass).",
    )
    parser.add_argument(
        "--somp-beta",
        type=float,
        default=1.0,
        help="Beta coefficient in SOMP head weighting exp(alpha * variance - beta * local_mass).",
    )
    parser.add_argument(
        "--somp-local-window",
        type=int,
        default=8,
        help="Trailing local window size used by the SOMP locality penalty.",
    )
    parser.add_argument(
        "--null-debias-samples",
        type=int,
        default=0,
        help="Number of random no-semantic forward passes used to estimate the received_attn position baseline.",
    )
    parser.add_argument(
        "--null-debias-gamma",
        type=float,
        default=1.0,
        help="Scale factor gamma in max(received_attn - gamma * null_baseline, 0).",
    )
    parser.add_argument(
        "--null-debias-seed",
        type=int,
        default=13,
        help="Random seed for null-baseline token replacement.",
    )
    parser.add_argument(
        "--hidden-pos-top-k",
        type=int,
        default=0,
        help="If > 0, build an independent hidden_posscale method by damping the top-k absolute-position channels.",
    )
    parser.add_argument(
        "--hidden-pos-scale-factor",
        type=float,
        default=0.25,
        help="Scaling factor applied to the detected absolute-position hidden channels.",
    )
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
    parser.add_argument(
        "--rise-layers",
        nargs="+",
        default=[],
        help="Optional ordered layer specs used to build rise_attn from normalized received_attn deltas.",
    )
    parser.add_argument(
        "--attention-gated-candidates",
        action="store_true",
        help="Build alternative candidate sets from received_attn gates instead of POS filtering for attention-based methods.",
    )
    parser.add_argument(
        "--attention-gated-threshold-mode",
        choices=["mean", "percentile"],
        default="mean",
        help="Threshold rule for selecting high-attention anchor tokens.",
    )
    parser.add_argument(
        "--attention-gated-threshold-percentile",
        type=float,
        default=75.0,
        help="Percentile used when attention-gated-threshold-mode=percentile.",
    )
    parser.add_argument(
        "--attention-gated-max-ngram",
        type=int,
        default=4,
        help="Maximum n-gram length for attention-gated candidate expansion.",
    )
    parser.add_argument(
        "--disable-stream-attention-scores",
        action="store_true",
        help="Disable streaming attention scores collection (use full attention outputs instead).",
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


def parse_rise_layer_config(layer_specs: List[str]) -> dict | None:
    if not layer_specs:
        return None

    parsed = parse_attention_layer_specs(layer_specs)
    indices: List[int] = []
    for config in parsed:
        indices.extend(config["indices"])
    if len(indices) < 2:
        raise ValueError("rise_layers must contain at least two layer specs.")

    normalized_parts = [str(index).replace("-", "neg") for index in indices]
    return {"name": f"rise_{'_'.join(normalized_parts)}", "indices": indices}


def _normalize_word_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


def _compute_rise_scores(
    per_doc_layer_scores: Dict[int, Dict[str, np.ndarray]],
    layer_indices: List[int],
) -> np.ndarray:
    previous_scores: np.ndarray | None = None
    rise_scores: np.ndarray | None = None

    for layer_index in layer_indices:
        current_scores = _normalize_word_scores(per_doc_layer_scores[layer_index]["received_attn"])
        if previous_scores is not None:
            delta = np.maximum(current_scores - previous_scores, 0.0)
            rise_scores = delta if rise_scores is None else rise_scores + delta
        previous_scores = current_scores

    if rise_scores is None:
        if previous_scores is None:
            return np.zeros(0, dtype=np.float32)
        rise_scores = np.zeros_like(previous_scores)
    return _normalize_word_scores(rise_scores)


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
    instruction_prefix_zh_causal: str,
    cls_head_strategies: List[str],
    cls_head_top_k: int,
    cls_head_temperature: float,
    somp_alpha: float,
    somp_beta: float,
    somp_local_window: int,
    null_debias_samples: int,
    null_debias_gamma: float,
    null_debias_seed: int,
    hidden_pos_top_k: int,
    hidden_pos_scale_factor: float,
    rise_layer_config: dict | None,
    attention_gated_candidates: bool,
    attention_gated_threshold_mode: str,
    attention_gated_threshold_percentile: float,
    attention_gated_max_ngram: int,
    stream_attention_scores: bool = True,
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
        predictions[f"excess_attn{suffix}"] = []
        predictions[f"sink_realloc_attn{suffix}"] = []
        predictions[f"sink_realloc_cls_attn{suffix}"] = []
        predictions[f"sink_realloc_samrank{suffix}"] = []
        predictions[f"sink_realloc_fusion_attn{suffix}"] = []
        predictions[f"cls_attn_idf{suffix}"] = []
        predictions[f"received_attn_idf{suffix}"] = []
        predictions[f"samrank_idf{suffix}"] = []
        predictions[f"fusion_attn_idf{suffix}"] = []
        predictions[f"excess_attn_idf{suffix}"] = []
        predictions[f"sink_realloc_attn_idf{suffix}"] = []
        predictions[f"sink_realloc_cls_attn_idf{suffix}"] = []
        predictions[f"sink_realloc_samrank_idf{suffix}"] = []
        predictions[f"sink_realloc_fusion_attn_idf{suffix}"] = []
        if null_debias_samples > 0:
            predictions[f"received_attn_debiased{suffix}"] = []
            predictions[f"received_attn_debiased_idf{suffix}"] = []
        if hidden_pos_top_k > 0:
            predictions[f"hidden_posscale{suffix}"] = []
            predictions[f"hidden_posscale_idf{suffix}"] = []
        if attention_gated_candidates:
            predictions[f"cls_attn_attncand{suffix}"] = []
            predictions[f"received_attn_attncand{suffix}"] = []
            predictions[f"samrank_attncand{suffix}"] = []
            predictions[f"fusion_attn_attncand{suffix}"] = []
            predictions[f"excess_attn_attncand{suffix}"] = []
            predictions[f"sink_realloc_attn_attncand{suffix}"] = []
            predictions[f"sink_realloc_cls_attn_attncand{suffix}"] = []
            predictions[f"sink_realloc_samrank_attncand{suffix}"] = []
            predictions[f"sink_realloc_fusion_attn_attncand{suffix}"] = []
            predictions[f"cls_attn_attncand_idf{suffix}"] = []
            predictions[f"received_attn_attncand_idf{suffix}"] = []
            predictions[f"samrank_attncand_idf{suffix}"] = []
            predictions[f"fusion_attn_attncand_idf{suffix}"] = []
            predictions[f"excess_attn_attncand_idf{suffix}"] = []
            predictions[f"sink_realloc_attn_attncand_idf{suffix}"] = []
            predictions[f"sink_realloc_cls_attn_attncand_idf{suffix}"] = []
            predictions[f"sink_realloc_samrank_attncand_idf{suffix}"] = []
            predictions[f"sink_realloc_fusion_attn_attncand_idf{suffix}"] = []
            if null_debias_samples > 0:
                predictions[f"received_attn_debiased_attncand{suffix}"] = []
                predictions[f"received_attn_debiased_attncand_idf{suffix}"] = []
            if hidden_pos_top_k > 0:
                predictions[f"hidden_posscale_attncand{suffix}"] = []
                predictions[f"hidden_posscale_attncand_idf{suffix}"] = []
        if "topk" in cls_head_strategies:
            predictions[f"cls_attn_headtopk{suffix}"] = []
            predictions[f"cls_attn_headtopk_idf{suffix}"] = []
            predictions[f"received_attn_headtopk{suffix}"] = []
            predictions[f"received_attn_headtopk_idf{suffix}"] = []
        if "softmax" in cls_head_strategies:
            predictions[f"cls_attn_headsoftmax{suffix}"] = []
            predictions[f"cls_attn_headsoftmax_idf{suffix}"] = []
            predictions[f"received_attn_headsoftmax{suffix}"] = []
            predictions[f"received_attn_headsoftmax_idf{suffix}"] = []
        if "somp" in cls_head_strategies:
            predictions[f"cls_attn_headsomp{suffix}"] = []
            predictions[f"cls_attn_headsomp_idf{suffix}"] = []
            predictions[f"received_attn_headsomp{suffix}"] = []
            predictions[f"received_attn_headsomp_idf{suffix}"] = []
    if rise_layer_config is not None:
        rise_suffix = f"@{rise_layer_config['name']}"
        predictions[f"rise_attn{rise_suffix}"] = []
        predictions[f"rise_attn_idf{rise_suffix}"] = []
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
    if rise_layer_config is not None:
        for layer_index in rise_layer_config["indices"]:
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
        batch_pos_tags=[item["pos_tags"] for item in preprocessed],
        language=preprocessed[0]["doc"].language if preprocessed else "zh",
        instruction_prefix=(
            instruction_prefix_zh_causal
            if preprocessed
            and model_bundle.get("is_causal")
            and preprocessed[0]["doc"].language.startswith("zh")
            and instruction_prefix_zh_causal.strip()
            else None
        ),
        cls_head_strategies=cls_head_strategies,
        cls_head_top_k=cls_head_top_k,
        cls_head_temperature=cls_head_temperature,
        somp_alpha=somp_alpha,
        somp_beta=somp_beta,
        somp_local_window=somp_local_window,
        null_debias_samples=null_debias_samples,
        null_debias_gamma=null_debias_gamma,
        null_debias_seed=null_debias_seed,
        stream_attention_scores=stream_attention_scores,
    )
    append_debug_log(
        output_dir,
        f"{model_name} / {dataset_name}: attention cache ready for {len(required_layer_indices)} layer spec(s)",
    )
    hidden_scores_cache: List[Dict[int, Dict[str, np.ndarray]]] | None = None
    if hidden_pos_top_k > 0:
        hidden_scores_cache = batched_hidden_word_scores(
            [item["words"] for item in preprocessed],
            model_bundle,
            layer_indices=required_layer_indices,
            batch_size=attention_batch_size,
            progress_label=f"{dataset_name} hidden",
            log_every_batches=log_every_batches,
            batch_pos_tags=[item["pos_tags"] for item in preprocessed],
            language=preprocessed[0]["doc"].language if preprocessed else "zh",
            instruction_prefix=(
                instruction_prefix_zh_causal
                if preprocessed
                and model_bundle.get("is_causal")
                and preprocessed[0]["doc"].language.startswith("zh")
                and instruction_prefix_zh_causal.strip()
                else None
            ),
            hidden_pos_top_k=hidden_pos_top_k,
            hidden_pos_scale_factor=hidden_pos_scale_factor,
        )
        append_debug_log(
            output_dir,
            f"{model_name} / {dataset_name}: hidden cache ready for {len(required_layer_indices)} layer spec(s)",
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
            if hidden_scores_cache is not None:
                hidden_layer_scores = [hidden_scores_cache[doc_index - 1][layer_index] for layer_index in config["indices"]]
                if len(hidden_layer_scores) == 1:
                    word_scores_by_method.update(hidden_layer_scores[0])
                else:
                    word_scores_by_method.update(
                        {
                            method_name: np.average(
                                np.stack([scores[method_name] for scores in hidden_layer_scores], axis=0),
                                axis=0,
                            )
                            for method_name in hidden_layer_scores[0]
                        }
                    )
            gated_candidates = None
            gated_candidate_starts = None
            gated_candidate_ends = None
            if attention_gated_candidates:
                gated_candidates = build_attention_gated_candidates(
                    words,
                    word_scores_by_method["received_attn"],
                    language=doc.language,
                    max_ngram=attention_gated_max_ngram,
                    threshold_mode=attention_gated_threshold_mode,
                    threshold_percentile=attention_gated_threshold_percentile,
                )
                gated_candidate_starts = np.fromiter(
                    (candidate.word_start for candidate in gated_candidates),
                    dtype=np.int32,
                    count=len(gated_candidates),
                )
                gated_candidate_ends = np.fromiter(
                    (candidate.word_end for candidate in gated_candidates),
                    dtype=np.int32,
                    count=len(gated_candidates),
                )
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
                if attention_gated_candidates and gated_candidates is not None:
                    attncand_key = f"{method_name}_attncand{suffix}"
                    attncand_idf_key = f"{method_name}_attncand_idf{suffix}"
                    if attncand_key not in predictions:
                        predictions[attncand_key] = []
                    if attncand_idf_key not in predictions:
                        predictions[attncand_idf_key] = []
                    predictions[attncand_key].append(
                        candidate_rank_from_word_scores(
                            gated_candidates,
                            word_scores,
                            top_k=top_k,
                            token_counts=token_counts,
                            words=words,
                            aggregation_mode=candidate_aggregation,
                            repeat_boost=repeat_boost,
                            candidate_starts=gated_candidate_starts,
                            candidate_ends=gated_candidate_ends,
                        )
                    )
                    predictions[attncand_idf_key].append(
                        candidate_rank_from_word_scores(
                            gated_candidates,
                            hybrid_word_scores,
                            top_k=top_k,
                            token_counts=token_counts,
                            words=words,
                            aggregation_mode=candidate_aggregation,
                            repeat_boost=repeat_boost,
                            candidate_starts=gated_candidate_starts,
                            candidate_ends=gated_candidate_ends,
                        )
                    )
        if rise_layer_config is not None:
            rise_suffix = f"@{rise_layer_config['name']}"
            rise_scores = _compute_rise_scores(per_doc_layer_scores, rise_layer_config["indices"])
            predictions[f"rise_attn{rise_suffix}"].append(
                candidate_rank_from_word_scores(
                    candidates,
                    rise_scores,
                    top_k=top_k,
                    token_counts=token_counts,
                    words=words,
                    aggregation_mode=candidate_aggregation,
                    repeat_boost=repeat_boost,
                    candidate_starts=candidate_starts,
                    candidate_ends=candidate_ends,
                )
            )
            rise_idf_scores = combine_word_scores(rise_scores, tfidf_word_scores, mode="product")
            predictions[f"rise_attn_idf{rise_suffix}"].append(
                candidate_rank_from_word_scores(
                    candidates,
                    rise_idf_scores,
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
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_causal_override = None if args.is_causal_override == "auto" else args.is_causal_override == "true"
    rise_layer_config = parse_rise_layer_config(args.rise_layers)

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
        model_bundle = build_model_bundle(
            model_source,
            args.device,
            is_causal_override=is_causal_override,
            true_bidirectional_attention=args.true_bidirectional_attention,
        )
        append_debug_log(
            output_dir,
            f"{model_name}: detected_is_causal={model_bundle.get('detected_is_causal')} "
            f"effective_is_causal={model_bundle.get('is_causal')}",
        )
        if args.true_bidirectional_attention:
            append_debug_log(output_dir, f"{model_name}: true_bidirectional_attention=True")
        if args.cls_head_strategies:
            append_debug_log(
                output_dir,
                f"{model_name}: cls_head_strategies={args.cls_head_strategies} "
                f"top_k={args.cls_head_top_k} temperature={args.cls_head_temperature}",
            )
            if "somp" in args.cls_head_strategies:
                append_debug_log(
                    output_dir,
                    f"{model_name}: somp_alpha={args.somp_alpha} "
                    f"somp_beta={args.somp_beta} somp_local_window={args.somp_local_window}",
                )
        if args.null_debias_samples > 0:
            append_debug_log(
                output_dir,
                f"{model_name}: null_debias_samples={args.null_debias_samples} "
                f"gamma={args.null_debias_gamma} seed={args.null_debias_seed}",
            )
        if args.hidden_pos_top_k > 0:
            append_debug_log(
                output_dir,
                f"{model_name}: hidden_pos_top_k={args.hidden_pos_top_k} "
                f"hidden_pos_scale_factor={args.hidden_pos_scale_factor}",
            )
        if args.attention_gated_candidates:
            append_debug_log(
                output_dir,
                f"{model_name}: attention_gated_candidates=True "
                f"threshold_mode={args.attention_gated_threshold_mode} "
                f"threshold_percentile={args.attention_gated_threshold_percentile} "
                f"max_ngram={args.attention_gated_max_ngram}",
            )
        if rise_layer_config is not None:
            append_debug_log(
                output_dir,
                f"{model_name}: rise_layers={rise_layer_config['indices']} name={rise_layer_config['name']}",
            )
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
                instruction_prefix_zh_causal=args.instruction_prefix_zh_causal,
                cls_head_strategies=args.cls_head_strategies,
                cls_head_top_k=args.cls_head_top_k,
                cls_head_temperature=args.cls_head_temperature,
                somp_alpha=args.somp_alpha,
                somp_beta=args.somp_beta,
                somp_local_window=args.somp_local_window,
                null_debias_samples=args.null_debias_samples,
                null_debias_gamma=args.null_debias_gamma,
                null_debias_seed=args.null_debias_seed,
                hidden_pos_top_k=args.hidden_pos_top_k,
                hidden_pos_scale_factor=args.hidden_pos_scale_factor,
                rise_layer_config=rise_layer_config,
                attention_gated_candidates=args.attention_gated_candidates,
                attention_gated_threshold_mode=args.attention_gated_threshold_mode,
                attention_gated_threshold_percentile=args.attention_gated_threshold_percentile,
                attention_gated_max_ngram=args.attention_gated_max_ngram,
                stream_attention_scores=not args.disable_stream_attention_scores,
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
