from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BASELINE = "thenlper/gte-small-zh"
DEFAULT_CANDIDATE = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_METHODS = [
    "keybert",
    "keybert_idf",
    "cls_attn",
    "received_attn",
    "samrank",
    "fusion_attn",
    "cls_attn_idf",
    "received_attn_idf",
    "samrank_idf",
    "fusion_attn_idf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render same-caliber embedding comparison tables from benchmark JSON.")
    parser.add_argument("--results", required=True, help="Path to keyword_benchmark_results.json")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE)
    parser.add_argument("--datasets", nargs="+", default=["csl_test", "shencecup_labeled"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", help="Optional markdown output path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (row["dataset"], row["model"], row["method"]): row
        for row in rows
    }


def fmt_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def build_markdown(
    rows_by_key: dict[tuple[str, str, str], dict],
    baseline_model: str,
    candidate_model: str,
    datasets: list[str],
    methods: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Embedding 同口径分数表")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append(f"- baseline: `{baseline_model}`")
    lines.append(f"- candidate: `{candidate_model}`")
    lines.append(f"- datasets: `{', '.join(datasets)}`")
    lines.append(f"- methods: `{', '.join(methods)}`")
    lines.append("- 指标主列使用 `F1@10`，辅助列使用 `R@10`")
    lines.append("")

    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| 方法 | baseline F1@10 | baseline R@10 | candidate F1@10 | candidate R@10 | ΔF1@10 | ΔR@10 |")
        lines.append("|------|---------------:|--------------:|----------------:|---------------:|-------:|------:|")
        for method in methods:
            baseline = rows_by_key.get((dataset, baseline_model, method))
            candidate = rows_by_key.get((dataset, candidate_model, method))
            baseline_f1 = baseline.get("f1@10") if baseline else None
            baseline_r = baseline.get("r@10") if baseline else None
            candidate_f1 = candidate.get("f1@10") if candidate else None
            candidate_r = candidate.get("r@10") if candidate else None
            delta_f1 = None if baseline_f1 is None or candidate_f1 is None else candidate_f1 - baseline_f1
            delta_r = None if baseline_r is None or candidate_r is None else candidate_r - baseline_r
            lines.append(
                f"| `{method}` | {fmt_score(baseline_f1)} | {fmt_score(baseline_r)} | "
                f"{fmt_score(candidate_f1)} | {fmt_score(candidate_r)} | {fmt_delta(delta_f1)} | {fmt_delta(delta_r)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.results))
    rows_by_key = index_rows(rows)
    markdown = build_markdown(
        rows_by_key=rows_by_key,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        datasets=list(args.datasets),
        methods=list(args.methods),
    )
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
