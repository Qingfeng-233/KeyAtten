#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path


def _run_module(module_name: str, argv: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    benchmark_root = project_root / "benchmark"
    eval_root = benchmark_root / "eval"
    for path in (project_root, benchmark_root, eval_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    module = importlib.import_module(module_name)
    previous_argv = sys.argv[:]
    sys.argv = [module_name, *argv]
    try:
        module.main()
    finally:
        sys.argv = previous_argv


def _runner(module_name: str) -> Callable[[argparse.Namespace], None]:
    def run(args: argparse.Namespace) -> None:
        _run_module(module_name, args.extra_args)

    return run


COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "keyword": ("Run the main keyword benchmark.", _runner("benchmark.eval.run_keyword_benchmark")),
    "hidden-head": ("Evaluate a hidden-state keyword head checkpoint.", _runner("benchmark.eval.run_hidden_head_benchmark")),
    "llm-keyword": ("Benchmark an OpenAI-compatible LLM keyword extractor.", _runner("benchmark.eval.llm_keyword_benchmark")),
    "bio-local-matrix": ("Run the local BIO/QK matrix evaluation.", _runner("benchmark.scripts.eval_bio_local_matrix")),
    "bio-qk-combo": ("Run BIO-only, BIO+QK, and fused BIO/QK evaluation.", _runner("benchmark.scripts.eval_bio_qk_combo")),
    "shence-heldout": ("Run ShenCe strict held-out evaluation.", _runner("benchmark.scripts.run_shence_heldout_eval")),
    "gemini-heldout": ("Run Gemini held-out helper.", _runner("benchmark.scripts.gemini_heldout_100")),
    "test-llm-keywords": ("Run the LLM keyword smoke script.", _runner("benchmark.scripts.test_llm_keywords")),
    "download-hf-assets": ("Download Hugging Face model assets.", _runner("benchmark.tools.download_hf_assets")),
    "gte-onnx-probe": ("Validate gte-small-zh ONNX attention export.", _runner("benchmark.tools.gte_onnx_probe")),
    "remote-hidden-head": ("Run the remote hidden-head helper.", _runner("benchmark.tools.remote_hidden_head_runner")),
    "render-embedding-comparison": ("Render embedding comparison reports.", _runner("benchmark.tools.render_embedding_comparison")),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified local entry point for Keyatten benchmark jobs.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments passed through to the selected command.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, runner = COMMANDS[args.command]
    runner(args)


if __name__ == "__main__":
    main()
