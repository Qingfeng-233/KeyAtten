#!/usr/bin/env python3
"""Download and verify all required model assets for KeyAtten.

Supports HuggingFace and ModelScope (魔搭) as download sources.
ModelScope is preferred for users in mainland China.

Models layout (under models/):
    gte-small-zh/              — PyTorch base model
    gte_small_zh_onnx/         — ONNX model + tokenizer
    Qwen3-Embedding-0.6B/      — Qwen3 base model (for QK LoRA)
    qk_qwen0.6B/               — QK LoRA outputs (adapters tracked by git)
    qk_qwen4B/                 — QK LoRA 4B outputs

Usage:
    python -m keyatten.download_models                          # download all
    python -m keyatten.download_models --model gte-small-zh
    python -m keyatten.download_models --source modelscope      # or huggingface
    python -m keyatten.download_models --check                   # check only
    python -m keyatten.download_models --model qwen3-embed-0.6b --download-qk  # also train QK
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
MODELSCOPE_BASE_URL = "https://modelscope.cn/api/v1"

# All model definitions
# Each entry: target_dir, required files, and repo ids for both sources
MODEL_REGISTRY = {
    "gte-small-zh": {
        "target": "gte-small-zh",
        "hf": "thenlper/gte-small-zh",
        "ms": "AI-ModelScope/gte-small-zh",
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.txt", "special_tokens_map.json", "pytorch_model.bin"],
    },
    "gte-small-zh-onnx": {
        "target": "gte_small_zh_onnx",
        # ONNX model shares tokenizer with the PyTorch version
        "hf": "thenlper/gte-small-zh",
        "ms": "AI-ModelScope/gte-small-zh",
        "extra_files": ["gte_small_zh.onnx"],
        # Shared tokenizer files (copied if PyTorch version already exists)
        "tokenizer_files": ["config.json", "tokenizer.json", "tokenizer_config.json",
                            "vocab.txt", "special_tokens_map.json"],
    },
    "qwen3-embed-0.6b": {
        "target": "Qwen3-Embedding-0.6B",
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "ms": "Qwen/Qwen3-Embedding-0.6B",
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "merges.txt", "vocab.json", "model.safetensors"],
    },
}

# QK LoRA adapter files that should be git-tracked
QK_ADAPTER_DIRS = {
    "qk_qwen0.6B": ["best_adapter", "latest_adapter"],
    "qk_qwen4B": ["best_adapter", "latest_adapter"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KeyAtten model assets.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope",
                        help="Download source (default: modelscope for Chinese network).")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="Only check missing files.")
    parser.add_argument("--download-qk", action="store_true", help="Also download QK LoRA base models.")
    return parser.parse_args()


# ── Download helpers ──────────────────────────────────────────────────

def _download_file_hf(repo_id: str, filename: str, dest: Path, timeout: int) -> bool:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    resp = requests.get(url, headers=HF_HEADERS, timeout=timeout, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def _download_file_modelscope(repo_id: str, filename: str, dest: Path, timeout: int) -> bool:
    url = f"{MODELSCOPE_BASE_URL}/models/{repo_id}/repo/download"
    params = {"revision": "master", "filepath": filename}
    resp = requests.get(url, headers=HF_HEADERS, params=params, timeout=timeout, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def _check_files(model_dir: Path, files: list[str]) -> list[str]:
    return [f for f in files if not (model_dir / f).exists()]


def _download_model(
    repo_id: str,
    target_dir: Path,
    files: list[str],
    timeout: int,
    workers: int,
    source: str,
) -> tuple[str, list[str], list[str], list[str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = [], [], []
    download_fn = _download_file_modelscope if source == "modelscope" else _download_file_hf

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for filename in files:
            dest = target_dir / filename
            if dest.exists():
                skipped.append(filename)
                continue
            futures[pool.submit(download_fn, repo_id, filename, dest, timeout)] = filename

        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
                downloaded.append(filename)
            except Exception as e:
                failed.append(filename)
                print(f"    ERROR downloading {filename}: {e}")

    (target_dir / "download.ok").write_text(f"source={source}\n")
    return target_dir.name, downloaded, skipped, failed


def _copy_tokenizer_to_onnx(onnx_dir: Path, pytorch_dir: Path) -> list[str]:
    """Copy tokenizer files from PyTorch model dir to ONNX model dir."""
    copied = []
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.txt", "special_tokens_map.json"]:
        src = pytorch_dir / f
        if src.exists() and not (onnx_dir / f).exists():
            onnx_dir.mkdir(parents=True, exist_ok=True)
            (onnx_dir / f).write_bytes(src.read_bytes())
            copied.append(f)
    return copied


# ── Public API ─────────────────────────────────────────────────────────

def check_all_models(models_dir: Path | None = None) -> dict[str, dict]:
    """Check all model assets. Returns {model_name: {ok, missing}}."""
    if models_dir is None:
        models_dir = MODELS_DIR
    results = {}
    for name, cfg in MODEL_REGISTRY.items():
        target = models_dir / cfg["target"]
        all_files = cfg.get("files", []) + cfg.get("extra_files", [])
        missing = _check_files(target, all_files)
        results[name] = {"ok": len(missing) == 0, "missing": missing, "dir": str(target)}

    # Check QK adapters
    for qk_dir, adapter_subdirs in QK_ADAPTER_DIRS.items():
        qk_path = models_dir / qk_dir
        for subdir in adapter_subdirs:
            adapter_path = qk_path / subdir
            key = f"{qk_dir}/{subdir}"
            if not adapter_path.exists():
                results[key] = {"ok": False, "missing": ["directory not found"], "dir": str(adapter_path)}
            else:
                required = ["adapter_config.json", "adapter_model.safetensors"]
                missing = _check_files(adapter_path, required)
                results[key] = {"ok": len(missing) == 0, "missing": missing, "dir": str(adapter_path)}

    return results


def download_model(
    name: str,
    models_dir: Path | None = None,
    source: str = "modelscope",
    timeout: int = 600,
    workers: int = 4,
) -> dict:
    """Download a single model. Returns status dict."""
    if models_dir is None:
        models_dir = MODELS_DIR
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[name]
    target = models_dir / cfg["target"]
    repo_id = cfg["ms"] if source == "modelscope" else cfg["hf"]

    # Check existing
    all_files = cfg.get("files", []) + cfg.get("extra_files", [])
    missing = _check_files(target, all_files)
    if not missing:
        return {"name": name, "ok": True, "downloaded": [], "skipped": all_files, "failed": [],
                "message": "already complete"}

    # For ONNX model, try copying tokenizer from PyTorch version first
    if name == "gte-small-zh-onnx":
        pytorch_dir = models_dir / MODEL_REGISTRY["gte-small-zh"]["target"]
        if pytorch_dir.exists():
            copied = _copy_tokenizer_to_onnx(target, pytorch_dir)
            if copied:
                print(f"    Copied tokenizer files from {pytorch_dir.name}")

    # Download remaining files
    remaining = _check_files(target, all_files)
    if not remaining:
        return {"name": name, "ok": True, "downloaded": [], "skipped": all_files, "failed": [],
                "message": "already complete (after tokenizer copy)"}

    _, downloaded, skipped, failed = _download_model(
        repo_id, target, remaining, timeout, workers, source,
    )
    ok = len(failed) == 0
    return {
        "name": name, "ok": ok, "downloaded": downloaded,
        "skipped": skipped, "failed": failed,
        "message": "ok" if ok else f"{len(failed)} file(s) failed",
    }


def download_all(
    models_dir: Path | None = None,
    source: str = "modelscope",
    timeout: int = 600,
    workers: int = 4,
) -> dict[str, dict]:
    """Download all models. Returns {model_name: status}."""
    results = {}
    for name in MODEL_REGISTRY:
        results[name] = download_model(name, models_dir, source, timeout, workers)
    return results


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    source_label = "ModelScope (魔搭)" if args.source == "modelscope" else "HuggingFace"

    if args.check:
        print("=== Checking model assets ===")
        results = check_all_models(models_dir)
        all_ok = True
        for name, r in results.items():
            status = "OK" if r["ok"] else f"MISSING {r['missing']}"
            print(f"  {name}: {status}")
            if not r["ok"]:
                all_ok = False
        if all_ok:
            print("\nAll model assets present and ready.")
        else:
            print(f"\nRun without --check to download from {source_label}.")
        return

    to_download = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"=== Downloading model assets from {source_label} ===")
    for name in to_download:
        print(f"\n  [{name}]")
        result = download_model(name, models_dir, args.source, args.timeout, args.workers)
        if result["downloaded"]:
            print(f"    Downloaded: {result['downloaded']}")
        if result["skipped"]:
            print(f"    Existing:   {len(result['skipped'])} file(s)")
        if result["failed"]:
            print(f"    Failed:     {result['failed']}")
        print(f"    Status: {result['message']}")

    # Summary
    print(f"\n=== Summary ===")
    for name, r in check_all_models(models_dir).items():
        status = "OK" if r["ok"] else "MISSING"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
