from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from keyword_bench.data import build_all_eval_sets
from keyword_bench.methods import (
    attention_word_scores,
    build_candidates,
    build_model_bundle,
    candidate_rank_from_word_scores,
    keybert_word_scores,
    segment_text,
)
from keyword_bench.output_paths import resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root.")
    parser.add_argument("--dataset", required=True, help="Dataset name in build_all_eval_sets().")
    parser.add_argument("--model", required=True, help="HF model name or local model path.")
    parser.add_argument("--doc-ids", nargs="*", default=[], help="Specific doc ids to visualize.")
    parser.add_argument("--doc-limit", type=int, default=3, help="Number of documents to export when doc ids are omitted.")
    parser.add_argument("--output-dir", default="outputs_case_study", help="Directory for JSON and PNG outputs. Relative paths resolve under 测试沙箱/Outputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-layer-spec",
        default="last",
        help="last, second_last, third_last, mean_last2, mean_last3, or layer:<index>.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def parse_attention_layer_spec(spec: str) -> dict:
    if spec == "last":
        return {"name": "last", "indices": [-1]}
    if spec == "second_last":
        return {"name": "second_last", "indices": [-2]}
    if spec == "third_last":
        return {"name": "third_last", "indices": [-3]}
    if spec == "mean_last2":
        return {"name": "mean_last2", "indices": [-1, -2]}
    if spec == "mean_last3":
        return {"name": "mean_last3", "indices": [-1, -2, -3]}
    if spec.startswith("layer:"):
        return {"name": spec.replace(":", "_"), "indices": [int(spec.split(":", 1)[1])]}
    raise ValueError(f"Unsupported attention layer spec: {spec}")


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def select_docs(docs: list, doc_ids: list[str], doc_limit: int) -> list:
    if doc_ids:
        index = {doc.doc_id: doc for doc in docs}
        return [index[doc_id] for doc_id in doc_ids if doc_id in index]
    ranked = sorted(docs, key=lambda doc: (doc.meta.get("char_len", 0), doc.meta.get("keyword_count", 0)), reverse=True)
    return ranked[:doc_limit]


def render_heatmap(words: list[str], left_scores: np.ndarray, right_scores: np.ndarray, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for heatmap export.") from exc

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font_name for font_name in preferred_fonts if font_name in available_fonts), None)
    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False

    left = _normalize(left_scores)
    right = _normalize(right_scores)
    matrix = np.vstack([left, right])
    fig_width = max(12, len(words) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=55, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Embedding", "Attention"], fontsize=10)
    ax.set_title(title)
    for row_index, row_scores in enumerate(matrix):
        for col_index, score in enumerate(row_scores):
            if score >= 0.7:
                ax.text(col_index, row_index, f"{score:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_eval_sets = build_all_eval_sets(root_dir)
    docs = all_eval_sets[args.dataset]
    selected_docs = select_docs(docs, args.doc_ids, args.doc_limit)
    attention_config = parse_attention_layer_spec(args.attention_layer_spec)

    model_path = Path(args.model)
    model_source = str(model_path.resolve()) if model_path.exists() else args.model
    model_bundle = build_model_bundle(model_source, args.device)

    exported = []
    for doc in selected_docs:
        words, pos_tags = segment_text(doc.text, language=doc.language)
        candidates = build_candidates(words, pos_tags, language=doc.language)
        embedding_scores = keybert_word_scores(doc.text, words, pos_tags, model_bundle, language=doc.language)
        attention_scores = attention_word_scores(words, model_bundle, layer_indices=attention_config["indices"])["cls_attn"]
        ranked_embedding = candidate_rank_from_word_scores(candidates, embedding_scores, top_k=args.top_k)
        ranked_attention = candidate_rank_from_word_scores(candidates, attention_scores, top_k=args.top_k)

        safe_doc_id = doc.doc_id.replace("/", "_")
        png_path = output_dir / f"{args.dataset}__{safe_doc_id}__{attention_config['name']}.png"
        json_path = output_dir / f"{args.dataset}__{safe_doc_id}__{attention_config['name']}.json"

        render_heatmap(
            words,
            embedding_scores,
            attention_scores,
            png_path,
            title=f"{doc.doc_id} | {args.model} | {attention_config['name']}",
        )

        payload = {
            "dataset": args.dataset,
            "doc_id": doc.doc_id,
            "model": args.model,
            "attention_layer_spec": attention_config["name"],
            "gold_keywords": doc.keywords,
            "embedding_top_keywords": ranked_embedding,
            "attention_top_keywords": ranked_attention,
            "words": words,
            "embedding_scores": [float(value) for value in embedding_scores.tolist()],
            "attention_scores": [float(value) for value in attention_scores.tolist()],
            "meta": doc.meta,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append({"doc_id": doc.doc_id, "json": str(json_path), "png": str(png_path)})

    summary_path = output_dir / "case_study_manifest.json"
    summary_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} case studies to {output_dir}")


if __name__ == "__main__":
    main()
