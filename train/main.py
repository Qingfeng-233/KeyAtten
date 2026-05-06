#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path


def _run_module(module_name: str, argv: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "benchmark", project_root / "train"):
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
    "eval-news55": ("Evaluate the clean news55 annotation set.", _runner("train.eval.news55")),
    "eval-bio": ("Run the legacy BIO-only benchmark evaluation.", _runner("train.eval.bio")),
    "eval-fusion": ("Run attention LoRA + BIO fusion evaluation.", _runner("train.eval.fusion")),
    "train-bio": ("Train the BIO boundary model.", _runner("train.jobs.bio_boundary")),
    "train-attn-lora": ("Train attention LoRA from supervised labels.", _runner("train.jobs.attn_lora")),
    "train-attn-lora-llm": ("Train attention LoRA from LLM labels.", _runner("train.jobs.attn_lora_llm")),
    "train-candidate-segment-attn": ("Train attention over explicit BIO candidate segments.", _runner("train.jobs.candidate_segment_attn")),
    "train-qk-lora": ("Train the legacy QK LoRA experiment.", _runner("train.jobs.qk_lora")),
    "label-llm": ("Run LLM keyword labeling.", _runner("train.jobs.run_llm_labeling")),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified local entry point for Keyatten training and evaluation jobs.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected command.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, runner = COMMANDS[args.command]
    runner(args)


if __name__ == "__main__":
    main()
