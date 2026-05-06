from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    try:
        from benchmark.main import main as benchmark_main
    except ImportError as exc:
        raise RuntimeError("benchmark.main is required for the benchmark CLI.") from exc
    benchmark_main(list(argv) if argv is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
