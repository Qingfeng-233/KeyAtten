"""
ShenCeCup 100-doc strict held-out evaluation.

Reproduces the seed=42 split from train_qk_lora.py to get held-out test set,
then evaluates QK LoRA adapter and zero-shot baselines.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
EVAL_DIR = BENCHMARK_ROOT / "eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(EVAL_DIR))

import run_keyword_benchmark as benchmark_runner
from keyword_bench.data import Document, load_shencecup_labeled
from keyword_bench.metrics import evaluate_predictions
from keyword_bench.methods import DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX, build_model_bundle
from keyatten.candidates import build_candidates, segment_text
from keyatten.candidates.bio_mining import find_candidate_occurrences

TOP_K = 10
MAX_LENGTH = 512

SUMMARY_ORDER = [
    ("qk_lora_heldout", "QK Scoring LoRA (held-out 100)"),
    ("samrank", "samrank"),
    ("samrank_idf", "samrank_idf"),
    ("cls_attn", "cls_attn"),
    ("cls_attn_idf", "cls_attn_idf"),
    ("received_attn", "received_attn"),
    ("received_attn_idf", "received_attn_idf"),
    ("tfidf", "TF-IDF"),
]


def compute_qk_scores(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
    q_store = {}
    k_store = {}

    inner = model.base_model.model
    if hasattr(inner, "model"):
        inner = inner.model
    target_layer = inner.layers[layer_idx].self_attn

    def q_hook(module, inputs, output):
        q_store["q"] = output

    def k_hook(module, inputs, output):
        k_store["k"] = output

    handle_q = target_layer.q_proj.register_forward_hook(q_hook)
    handle_k = target_layer.k_proj.register_forward_hook(k_hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
    finally:
        handle_q.remove()
        handle_k.remove()

    q_tensor = q_store["q"].float()
    k_tensor = k_store["k"].float()

    batch_size, seq_len, _ = q_tensor.shape
    head_dim = target_layer.head_dim
    num_heads = q_tensor.shape[-1] // head_dim
    num_kv_heads = k_tensor.shape[-1] // head_dim
    groups = max(1, num_heads // max(1, num_kv_heads))

    q_tensor = q_tensor.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k_tensor = k_tensor.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    k_tensor = k_tensor.repeat_interleave(groups, dim=1)

    eos_idx = attention_mask.sum(dim=1) - 1
    q_eos = q_tensor[torch.arange(batch_size), :, eos_idx, :].unsqueeze(2)
    scores = (q_eos * k_tensor).sum(dim=-1) / (head_dim ** 0.5)
    return torch.sigmoid(scores.mean(dim=1))


def score_candidates_with_qk(
    doc: Document,
    tokenizer,
    model,
    device: str,
    layer_idx: int,
    bio_extractor=None,
    bio_candidate_mode: str = "auto",
    bio_candidate_max_spans: Optional[int] = None,
    bio_candidate_b_threshold: Optional[float] = None,
    bio_candidate_profile: str = "balanced",
) -> List[str]:
    if bio_extractor is not None:
        use_explicit = bio_candidate_mode == "explicit"
        if bio_candidate_mode == "auto":
            use_explicit = (
                bio_candidate_max_spans is not None
                or bio_candidate_b_threshold is not None
            )
        if use_explicit:
            max_spans = 50 if bio_candidate_max_spans is None else bio_candidate_max_spans
            b_threshold = 0.15 if bio_candidate_b_threshold is None else bio_candidate_b_threshold
            candidates = [
                text
                for text, _ in bio_extractor.extract_spans_relaxed(
                    doc.text,
                    max_spans=max_spans,
                    b_threshold=b_threshold,
                )
            ]
        else:
            candidates = [
                text
                for text, _ in bio_extractor.extract_spans_profile(
                    doc.text,
                    profile=bio_candidate_profile,
                )
            ]
    else:
        words, pos_tags = segment_text(doc.text, language="zh")
        candidates = [candidate.text for candidate in build_candidates(words, pos_tags, language="zh")]
    if not candidates:
        return []

    full_text = DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX + doc.text
    prefix_len = len(DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX)
    encoding = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.inference_mode():
        scores = compute_qk_scores(model, input_ids, attention_mask, layer_idx)[0].cpu().numpy()

    char_to_score: Dict[int, float] = {}
    for token_index, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end or tok_start < prefix_len:
            continue
        char_start = tok_start - prefix_len
        char_end = tok_end - prefix_len
        score = float(scores[token_index])
        for char_index in range(char_start, char_end):
            if char_index not in char_to_score or score > char_to_score[char_index]:
                char_to_score[char_index] = score

    candidate_scores: Dict[str, float] = {}
    for candidate_text in candidates:
        occurrences = find_candidate_occurrences(doc.text, candidate_text)
        if not occurrences:
            continue
        best_score = None
        for start_char, end_char in occurrences:
            span_scores = [char_to_score[idx] for idx in range(start_char, end_char) if idx in char_to_score]
            score = float(sum(span_scores) / len(span_scores)) if span_scores else 0.0
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            candidate_scores[candidate_text] = best_score

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, _ in ranked[:TOP_K]]


def evaluate_qk_lora(
    docs: List[Document],
    tokenizer,
    model,
    device: str,
    layer_idx: int,
    bio_extractor=None,
    bio_candidate_mode: str = "auto",
    bio_candidate_max_spans: Optional[int] = None,
    bio_candidate_b_threshold: Optional[float] = None,
    bio_candidate_profile: str = "balanced",
) -> Dict[str, float]:
    predictions = []
    golds = []
    for index, doc in enumerate(docs, start=1):
        predictions.append(
            score_candidates_with_qk(
                doc,
                tokenizer,
                model,
                device,
                layer_idx,
                bio_extractor=bio_extractor,
                bio_candidate_mode=bio_candidate_mode,
                bio_candidate_max_spans=bio_candidate_max_spans,
                bio_candidate_b_threshold=bio_candidate_b_threshold,
                bio_candidate_profile=bio_candidate_profile,
            )
        )
        golds.append(doc.keywords)
        if index % 20 == 0 or index == len(docs):
            print(f"[qk] processed {index}/{len(docs)}")
    return evaluate_predictions(predictions, golds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "Qwen__Qwen3-Embedding-0.6B"))
    parser.add_argument("--baseline-model", default=str(REPO_ROOT / "models" / "thenlper__gte-small-zh"))
    parser.add_argument("--baseline-layer", type=int, default=-1)
    parser.add_argument("--adapter-dir", default=str(REPO_ROOT / "models" / "qk_lora" / "best_adapter"))
    parser.add_argument("--qk-layer", type=int, default=None)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "outputs_shence_heldout"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bio-candidate-checkpoint", default=None)
    parser.add_argument(
        "--bio-candidate-mode",
        choices=("auto", "explicit", "profile"),
        default="auto",
        help="candidate extraction mode: auto=explicit when params are provided, else profile",
    )
    parser.add_argument("--bio-candidate-max-spans", type=int, default=None)
    parser.add_argument("--bio-candidate-b-threshold", type=float, default=None)
    parser.add_argument(
        "--bio-candidate-profile",
        choices=("balanced", "clean", "high_recall"),
        default="balanced",
    )
    return parser.parse_args()


def _recommended_layer_index(layer_count: int | None) -> int:
    if layer_count is None or layer_count <= 0:
        return 21
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))


def _resolve_qk_layer_arg(qk_layer_arg: int | None, base_model) -> int:
    if qk_layer_arg is not None:
        return int(qk_layer_arg)
    layer_count = None
    config = getattr(base_model, "config", None)
    if config is not None:
        for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            value = getattr(config, attr_name, None)
            if isinstance(value, int) and value > 0:
                layer_count = value
                break
    return _recommended_layer_index(layer_count)


def build_shence_split(data_root: str | Path) -> Dict[str, List[Document]]:
    splitter = random.Random(42)
    all_docs = load_shencecup_labeled(data_root)
    splitter.shuffle(all_docs)
    test_size = min(200, len(all_docs) // 5)
    test_pool = all_docs[:test_size]
    train_docs = all_docs[test_size:]
    dev_docs = test_pool[: test_size // 2]
    test_final_docs = test_pool[test_size // 2 :]
    return {
        "all": all_docs,
        "train": train_docs,
        "dev": dev_docs,
        "test_final": test_final_docs,
    }


def run_zero_shot_baselines(
    docs: List[Document],
    device: str,
    model_name: str,
    output_dir: Path,
    layer_index: int,
) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "zero_shot_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = build_model_bundle(model_name, device=device)
    try:
        predictions = benchmark_runner.run_dataset_methods(
            docs=docs,
            model_bundle=model_bundle,
            model_name=model_name,
            dataset_name="shence_test_final",
            output_dir=log_dir,
            top_k=TOP_K,
            run_yake=False,
            attention_configs=[{"name": "last", "indices": [layer_index]}],
            candidate_aggregation="mean",
            repeat_boost=0.0,
            attention_batch_size=1,
            embedding_batch_size=8,
            log_every_docs=25,
            log_every_batches=5,
            instruction_prefix_zh_causal=DEFAULT_ZH_CAUSAL_INSTRUCTION_PREFIX,
            cls_head_strategies=[],
            cls_head_top_k=4,
            cls_head_temperature=8.0,
            somp_alpha=512.0,
            somp_beta=1.0,
            somp_local_window=8,
            null_debias_samples=0,
            null_debias_gamma=1.0,
            null_debias_seed=13,
            hidden_pos_top_k=0,
            hidden_pos_scale_factor=0.25,
            rise_layer_config=None,
            attention_gated_candidates=False,
            attention_gated_threshold_mode="mean",
            attention_gated_threshold_percentile=75.0,
            attention_gated_max_ngram=4,
        )
    finally:
        model = model_bundle.get("model")
        if model is not None:
            del model
        tokenizer = model_bundle.get("tokenizer")
        if tokenizer is not None:
            del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    golds = [doc.keywords for doc in docs]
    metrics = {}
    for method_name, _ in SUMMARY_ORDER[1:]:
        metrics[method_name] = evaluate_predictions(predictions[method_name], golds)
    return metrics


def write_summary(
    results: Dict[str, Dict[str, float]],
    split_docs: List[Document],
    output_dir: Path,
    model_name: str,
    adapter_dir: str,
    qk_layer_idx: int,
) -> None:
    lines = [
        "# ShenCeCup Held-out Evaluation",
        "",
        f"- Documents: {len(split_docs)}",
        f"- Model: `{model_name}`",
        f"- LoRA Adapter: `{adapter_dir}`",
        f"- QK layer: `{qk_layer_idx}`",
        "",
        "| Method | F1@5 | F1@10 | R@10 |",
        "|------|------|-------|------|",
    ]
    for method_key, label in SUMMARY_ORDER:
        metrics = results[method_key]
        lines.append(
            f"| {label} | {metrics['f1@5']:.4f} | {metrics['f1@10']:.4f} | {metrics['r@10']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bio_extractor = None
    if args.bio_candidate_checkpoint:
        from keyatten import BIOExtractor

        bio_extractor = BIOExtractor(
            checkpoint_path=args.bio_candidate_checkpoint,
            device=device,
        )

    splits = build_shence_split(args.data_root)
    heldout_docs = splits["test_final"]
    print(f"[info] ShenCe all={len(splits['all'])} train={len(splits['train'])} dev={len(splits['dev'])} heldout={len(heldout_docs)}")

    print(f"[info] Running zero-shot baselines with {args.baseline_model} ...")
    zero_shot_results = run_zero_shot_baselines(
        heldout_docs,
        device=device,
        model_name=args.baseline_model,
        output_dir=output_dir,
        layer_index=args.baseline_layer,
    )

    print("[info] Loading QK LoRA model...")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    qk_layer_idx = _resolve_qk_layer_arg(args.qk_layer, base_model)
    print(f"[info] QK layer index={qk_layer_idx}")
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    try:
        qk_metrics = evaluate_qk_lora(
            heldout_docs,
            tokenizer,
            model,
            device,
            qk_layer_idx,
            bio_extractor=bio_extractor,
            bio_candidate_mode=args.bio_candidate_mode,
            bio_candidate_max_spans=args.bio_candidate_max_spans,
            bio_candidate_b_threshold=args.bio_candidate_b_threshold,
            bio_candidate_profile=args.bio_candidate_profile,
        )
    finally:
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    results = {"qk_lora_heldout": qk_metrics}
    results.update(zero_shot_results)

    payload = {
        "metadata": {
            "device": device,
            "model_dir": args.model,
            "baseline_model_dir": args.baseline_model,
            "baseline_layer_index": args.baseline_layer,
            "adapter_dir": args.adapter_dir,
            "layer_index": qk_layer_idx,
            "bio_candidate_mode": args.bio_candidate_mode,
            "bio_candidate_profile": args.bio_candidate_profile,
            "top_k": TOP_K,
            "max_length": MAX_LENGTH,
        },
        "split": {
            "all": len(splits["all"]),
            "train": len(splits["train"]),
            "dev": len(splits["dev"]),
            "test_final": len(heldout_docs),
            "test_final_doc_ids": [doc.doc_id for doc in heldout_docs],
        },
        "results": results,
    }
    (output_dir / "heldout_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(results, heldout_docs, output_dir, args.model, args.adapter_dir, qk_layer_idx)
    print(f"[done] Results written to {output_dir}")


if __name__ == "__main__":
    main()
