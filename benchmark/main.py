#!/usr/bin/env python3
"""Command-line entry point; run from the repo root as `python -m benchmark.main <command> [args...]`."""
from __future__ import annotations

from benchmark.cli import Commands, run_cli

COMMANDS: Commands = {
    "keyword": ("Run the main keyword benchmark.", "benchmark.eval.run_keyword_benchmark"),
    "hidden-head": ("Evaluate a hidden-state keyword head checkpoint.", "benchmark.eval.run_hidden_head_benchmark"),
    "llm-keyword": ("Benchmark an OpenAI-compatible LLM keyword extractor.", "benchmark.eval.llm_keyword_benchmark"),
    "bio-local-matrix": ("Run the local BIO/QK matrix evaluation.", "benchmark.experiments.eval_bio_local_matrix"),
    "bio-qk-combo": ("Run BIO-only, BIO+QK, and fused BIO/QK evaluation.", "benchmark.experiments.eval_bio_qk_combo"),
    "shence-heldout": ("Run ShenCe strict held-out evaluation.", "benchmark.experiments.run_shence_heldout_eval"),
    "gemini-heldout": ("Run Gemini held-out helper.", "benchmark.experiments.gemini_heldout_100"),
    "test-llm-keywords": ("Run the LLM keyword smoke script.", "benchmark.experiments.test_llm_keywords"),
    "download-hf-assets": ("Download Hugging Face model assets.", "benchmark.tools.download_hf_assets"),
    "gte-onnx-probe": ("Validate gte-small-zh ONNX attention export.", "benchmark.tools.gte_onnx_probe"),
    "remote-hidden-head": ("Run the remote hidden-head helper.", "benchmark.tools.remote_hidden_head_runner"),
    "render-embedding-comparison": ("Render embedding comparison reports.", "benchmark.tools.render_embedding_comparison"),
}


def main(argv: list[str] | None = None) -> None:
    run_cli("Unified local entry point for Keyatten benchmark jobs.", COMMANDS, argv)


if __name__ == "__main__":
    main()
