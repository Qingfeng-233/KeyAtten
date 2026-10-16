"""Shared command dispatcher for the `benchmark.main` and `train.main` entry points.

Each entry point declares a table of `name -> (help, module_path)`; `run_cli`
parses the command name, then imports that module and calls its `main()` with
the remaining arguments as `sys.argv`.
"""
from __future__ import annotations

import argparse
import importlib
import sys

Commands = dict[str, tuple[str, str]]


def _run_module(module_name: str, argv: list[str]) -> None:
    """Import a job module by its repo-root dotted path and call its main() with argv.

    Must be launched from the repo root via `python -m <pkg>.main` so that
    `benchmark`, `train` resolve as packages; keyatten comes from `pip install -e .`.
    """
    module = importlib.import_module(module_name)
    previous_argv = sys.argv[:]
    sys.argv = [module_name, *argv]
    try:
        module.main()
    finally:
        sys.argv = previous_argv


def run_cli(description: str, commands: Commands, argv: list[str] | None = None) -> None:
    """Parse `<command> [args...]` and run the module mapped to that command."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=sorted(commands))
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected command.",
    )
    args = parser.parse_args(argv)
    _, module_name = commands[args.command]
    _run_module(module_name, args.extra_args)
