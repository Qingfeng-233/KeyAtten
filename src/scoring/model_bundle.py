from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence



try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


_ONNX_PREFERRED_NAMES = (
    "attention_last.onnx",
    "gte_small_zh_attention.onnx",
    "model.onnx",
)


def _require_inference_dependencies() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append("torch>=2.0")
    if AutoModel is None or AutoTokenizer is None:
        missing.append("transformers>=4.30")
    if missing:
        raise ImportError(
            "Attention extraction requires optional dependencies: "
            f"{', '.join(missing)}. Install with `pip install \"keyatten[inference]\"`."
        )


def _require_lightweight_dependencies() -> None:
    missing: list[str] = []
    if ort is None:
        missing.append("onnxruntime>=1.18")
    if Tokenizer is None:
        missing.append("tokenizers>=0.15")
    if missing:
        raise ImportError(
            "Lightweight attention extraction requires optional dependencies: "
            f"{', '.join(missing)}. Install with `pip install \"keyatten[lightweight]\"`."
        )


def _discover_onnx_path(model_dir: Path, layer_index: int) -> Path | None:
    candidates: list[Path] = []
    if layer_index >= 0:
        candidates.extend(
            [
                model_dir / f"attention_layer_{layer_index}.onnx",
                model_dir / f"layer_{layer_index}.onnx",
            ]
        )
    candidates.extend(model_dir / name for name in _ONNX_PREFERRED_NAMES)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    discovered = sorted(model_dir.glob("*.onnx"))
    if len(discovered) == 1:
        return discovered[0]
    return None


def _resolve_onnx_artifacts(model_name: str, onnx_path: str | None, layer_index: int) -> tuple[Path, Path]:
    model_path = Path(model_name)
    if onnx_path is not None:
        resolved_onnx = Path(onnx_path)
        model_dir = model_path if model_path.is_dir() else resolved_onnx.parent
    elif model_path.is_file() and model_path.suffix.lower() == ".onnx":
        resolved_onnx = model_path
        model_dir = model_path.parent
    elif model_path.is_dir():
        resolved_onnx = _discover_onnx_path(model_path, layer_index)
        if resolved_onnx is None:
            raise FileNotFoundError(
                "Could not find an ONNX attention file in the model directory. "
                "Pass `onnx_path` explicitly or place the exported model at "
                "`attention_last.onnx` / `gte_small_zh_attention.onnx`."
            )
        model_dir = model_path
    else:
        raise ValueError(
            "ONNX backend requires a local model directory (for tokenizer files) "
            "and an exported ONNX attention model."
        )

    if not resolved_onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {resolved_onnx}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {model_dir}")
    return model_dir, resolved_onnx


def _load_tokenizer_metadata(model_dir: Path) -> tuple[Tokenizer, int]:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    max_length = 512
    config_path = model_dir / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        max_length = int(config.get("max_position_embeddings", max_length))
    return tokenizer, max_length


def _select_ort_providers(device: str) -> list[str]:
    available = set(ort.get_available_providers())
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _detect_is_causal(config) -> bool:
    """Detect whether a model uses causal (decoder-only) attention from its config."""
    if getattr(config, "is_decoder", False):
        return True
    architectures = getattr(config, "architectures", None) or []
    return any("CausalLM" in arch for arch in architectures)


def _detect_is_causal_from_json(config_path: Path) -> bool:
    """Detect causal model from a config.json file on disk."""
    if not config_path.is_file():
        return False
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("is_decoder", False):
        return True
    architectures = config.get("architectures", [])
    return any("CausalLM" in arch for arch in architectures)


def _detect_attention_layer_count(config) -> int | None:
    for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(config, attr_name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _detect_attention_layer_count_from_json(config_path: Path) -> int | None:
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _recommended_decoder_layer_index(layer_count: int | None) -> int | None:
    if layer_count is None or layer_count <= 0:
        return None
    return min(layer_count - 1, max(0, int(layer_count * 0.75)))


def _resolve_is_causal_override(is_causal_override: bool | None) -> bool | None:
    if is_causal_override is None or isinstance(is_causal_override, bool):
        return is_causal_override
    raise ValueError("is_causal_override must be None, True, or False.")


def build_model_bundle(
    model_name: str,
    device: str,
    backend: str = "auto",
    onnx_path: str | None = None,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    is_causal_override: bool | None = None,
    dtype: str | None = "auto",
) -> dict:
    if backend not in {"auto", "torch", "onnx"}:
        raise ValueError("backend must be one of {'auto', 'torch', 'onnx'}.")
    is_causal_override = _resolve_is_causal_override(is_causal_override)

    resolved_backend = backend
    if resolved_backend == "auto":
        model_path = Path(model_name)
        if onnx_path is not None:
            resolved_backend = "onnx"
        elif model_path.is_file() and model_path.suffix.lower() == ".onnx":
            resolved_backend = "onnx"
        elif model_path.is_dir() and _discover_onnx_path(model_path, layer_index) is not None:
            resolved_backend = "onnx"
        else:
            resolved_backend = "torch"

    if resolved_backend == "onnx":
        if layer_indices is not None:
            raise ValueError("ONNX backend currently supports only a single exported attention layer.")
        _require_lightweight_dependencies()
        model_dir, resolved_onnx = _resolve_onnx_artifacts(model_name, onnx_path, layer_index)
        tokenizer, max_length = _load_tokenizer_metadata(model_dir)
        session = ort.InferenceSession(str(resolved_onnx), providers=_select_ort_providers(device))
        config_path = model_dir / "config.json"
        detected_is_causal = _detect_is_causal_from_json(config_path)
        is_causal = detected_is_causal if is_causal_override is None else is_causal_override
        attention_layer_count = _detect_attention_layer_count_from_json(config_path)
        return {
            "backend": "onnx",
            "tokenizer": tokenizer,
            "session": session,
            "device": device,
            "onnx_path": str(resolved_onnx),
            "model_dir": str(model_dir),
            "max_length": max_length,
            "layer_index": layer_index,
            "detected_is_causal": detected_is_causal,
            "is_causal_override": is_causal_override,
            "is_causal": is_causal,
            "attention_layer_count": attention_layer_count,
            "recommended_layer_index": _recommended_decoder_layer_index(attention_layer_count) if is_causal else None,
        }

    _require_inference_dependencies()
    import torch as _torch
    _dtype_map = {"auto": None, "float32": _torch.float32, "float16": _torch.float16, "bfloat16": _torch.bfloat16}
    resolved_dtype = _dtype_map.get(dtype) if isinstance(dtype, str) else dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    load_kwargs: dict = {"output_attentions": True}
    if resolved_dtype is not None:
        load_kwargs["torch_dtype"] = resolved_dtype
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager", **load_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()
    detected_is_causal = _detect_is_causal(model.config)
    is_causal = detected_is_causal if is_causal_override is None else is_causal_override
    max_length = int(getattr(model.config, "max_position_embeddings", 512))
    attention_layer_count = _detect_attention_layer_count(model.config)
    return {
        "backend": "torch",
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "detected_is_causal": detected_is_causal,
        "is_causal_override": is_causal_override,
        "is_causal": is_causal,
        "max_length": max_length,
        "attention_layer_count": attention_layer_count,
        "recommended_layer_index": _recommended_decoder_layer_index(attention_layer_count) if is_causal else None,
    }


__all__ = [
    "build_model_bundle",
]
