from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkCommand:
    script: str
    description: str


COMMANDS: dict[str, BenchmarkCommand] = {
    "keywords-eval": BenchmarkCommand(
        script="run_keyword_benchmark.py",
        description="Run main keyword benchmark evaluation.",
    ),
    "hidden-head-train": BenchmarkCommand(
        script="train_hidden_state_head.py",
        description="Train hidden-state keyword head.",
    ),
    "hidden-head-eval": BenchmarkCommand(
        script="run_hidden_head_benchmark.py",
        description="Evaluate hidden-state keyword head checkpoint.",
    ),
    "onnx-probe": BenchmarkCommand(
        script="gte_onnx_probe.py",
        description="Validate ONNX attention export and runtime path.",
    ),
    "llm-eval": BenchmarkCommand(
        script="llm_keyword_benchmark.py",
        description="Run LLM-based keyword benchmark.",
    ),
    "hf-assets-download": BenchmarkCommand(
        script="download_hf_assets.py",
        description="Download benchmark-related Hugging Face assets.",
    ),
    "attention-case-study": BenchmarkCommand(
        script="generate_attention_case_study.py",
        description="Generate attention case-study artifacts.",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_dir() -> Path:
    return _repo_root() / "benchmark"


def _print_help() -> None:
    print("KeyAtten Benchmark CLI")
    print("")
    print("Usage:")
    print("  keyatten-benchmark <command> [script args]")
    print("  python -m keyatten.benchmark_cli <command> [script args]")
    print("")
    print("Commands:")
    for name, item in COMMANDS.items():
        print(f"  {name:<20} {item.description}")
    print("")
    print("Tips:")
    print("  Use `keyatten-benchmark list` to print command table.")
    print("  Script args are forwarded as-is to the target script.")


def _run_command(command: str, script_args: Sequence[str]) -> int:
    item = COMMANDS[command]
    script_path = _benchmark_dir() / item.script
    if not script_path.exists():
        print(f"[keyatten-benchmark] Missing script: {script_path}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script_path), *script_args]
    completed = subprocess.run(cmd, cwd=str(_repo_root()))
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    if not parsed_argv or parsed_argv[0] in {"-h", "--help", "help", "list"}:
        _print_help()
        return 0

    command = parsed_argv[0]
    script_args = parsed_argv[1:]
    if command not in COMMANDS:
        print(f"[keyatten-benchmark] Unknown command: {command}", file=sys.stderr)
        print("Run `keyatten-benchmark --help` for available commands.", file=sys.stderr)
        return 2
    return _run_command(command, script_args)


if __name__ == "__main__":
    raise SystemExit(main())
