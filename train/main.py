#!/usr/bin/env python3
"""Command-line entry point; run from the repo root as `python -m train.main <command> [args...]`."""
from __future__ import annotations

from benchmark.cli import Commands, run_cli

COMMANDS: Commands = {
    "eval-news55": ("Evaluate the clean news55 annotation set.", "train.eval.news55"),
    "eval-bio": ("Run the legacy BIO-only benchmark evaluation.", "train.eval.bio"),
    "eval-fusion": ("Run attention LoRA + BIO fusion evaluation.", "train.eval.fusion"),
    "train-bio": ("Train the BIO boundary model.", "train.jobs.bio_boundary"),
    "train-attn-lora": ("Train attention LoRA from supervised labels.", "train.jobs.attn_lora"),
    "train-attn-lora-llm": ("Train attention LoRA from LLM labels.", "train.jobs.attn_lora_llm"),
    "train-candidate-segment-attn": ("Train attention over explicit BIO candidate segments.", "train.jobs.candidate_segment_attn"),
    "train-qk-lora": ("Train the legacy QK LoRA experiment.", "train.jobs.qk_lora"),
    "label-llm": ("Run LLM keyword labeling.", "train.jobs.run_llm_labeling"),
}


def main(argv: list[str] | None = None) -> None:
    run_cli("Unified local entry point for Keyatten training and evaluation jobs.", COMMANDS, argv)


if __name__ == "__main__":
    main()
