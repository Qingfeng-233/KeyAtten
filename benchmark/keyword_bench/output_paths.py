from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "测试沙箱"
OUTPUTS_ROOT = SANDBOX_ROOT / "Outputs"


def sandbox_outputs_root() -> Path:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_ROOT


def resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()
    return (sandbox_outputs_root() / path).resolve()


def resolve_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (sandbox_outputs_root() / path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
