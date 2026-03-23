from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

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


ATTENTION_METHODS = ("cls_attn", "received_attn", "samrank", "fusion_attn")
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


def build_model_bundle(
    model_name: str,
    device: str,
    backend: str = "auto",
    onnx_path: str | None = None,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
) -> dict:
    if backend not in {"auto", "torch", "onnx"}:
        raise ValueError("backend must be one of {'auto', 'torch', 'onnx'}.")

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
        return {
            "backend": "onnx",
            "tokenizer": tokenizer,
            "session": session,
            "device": device,
            "onnx_path": str(resolved_onnx),
            "model_dir": str(model_dir),
            "max_length": max_length,
            "layer_index": layer_index,
        }

    _require_inference_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return {"backend": "torch", "tokenizer": tokenizer, "model": model, "device": device}


def normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _aggregate_subwords_to_words(
    word_ids: list[int | None],
    token_scores: np.ndarray,
    word_count: int,
) -> np.ndarray:
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_scores[token_index])
        counts[word_id] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def _resolve_layer_index(layer_index: int, layer_count: int) -> int:
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"Layer index {layer_index} is out of range for {layer_count} attention layers.")
    return resolved


def _scores_from_attention_map(
    word_ids: list[int | None],
    attention_map: np.ndarray,
    word_count: int,
) -> dict[str, np.ndarray]:
    cls_scores = attention_map[0]
    received_scores = attention_map.sum(axis=0)

    global_scores = received_scores.copy()
    redistributed = attention_map * global_scores[None, :]
    col_sums = redistributed.sum(axis=0, keepdims=True) + 1e-10
    redistributed = np.divide(redistributed, col_sums, out=np.zeros_like(redistributed), where=col_sums > 0.0)
    proportional_scores = redistributed.sum(axis=1)
    samrank_scores = global_scores + proportional_scores
    fusion_scores = normalize_array(cls_scores) * normalize_array(received_scores)

    return {
        "cls_attn": _aggregate_subwords_to_words(word_ids, cls_scores, word_count),
        "received_attn": _aggregate_subwords_to_words(word_ids, received_scores, word_count),
        "samrank": _aggregate_subwords_to_words(word_ids, samrank_scores, word_count),
        "fusion_attn": _aggregate_subwords_to_words(word_ids, fusion_scores, word_count),
    }


def _aggregate_layer_word_scores(
    per_layer_scores: Sequence[dict[str, np.ndarray]],
    layer_weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    if not per_layer_scores:
        return {}

    weights = np.asarray(layer_weights if layer_weights is not None else [1.0] * len(per_layer_scores), dtype=np.float32)
    if weights.size != len(per_layer_scores):
        raise ValueError("layer_weights must have the same length as per_layer_scores.")
    weight_sum = float(weights.sum())
    if math.isclose(weight_sum, 0.0):
        raise ValueError("layer_weights sum to zero.")

    normalized_weights = weights / weight_sum
    aggregated: dict[str, np.ndarray] = {}
    for method_name in per_layer_scores[0]:
        stacked = np.stack([layer_scores[method_name] for layer_scores in per_layer_scores], axis=0)
        aggregated[method_name] = np.average(stacked, axis=0, weights=normalized_weights)
    return aggregated


def _encode_words_batch_onnx(
    batch_words: Sequence[Sequence[str]],
    tokenizer: Tokenizer,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, list[list[int | None]], list[int]]:
    encodings = tokenizer.encode_batch([list(words) for words in batch_words], is_pretokenized=True)
    trimmed_ids: list[np.ndarray] = []
    trimmed_masks: list[np.ndarray] = []
    word_ids_per_item: list[list[int | None]] = []
    valid_token_counts: list[int] = []

    for encoding in encodings:
        input_ids = np.asarray(encoding.ids[:max_length], dtype=np.int64)
        attention_mask = np.asarray(encoding.attention_mask[:max_length], dtype=np.int64)
        valid_token_count = int(attention_mask.sum())
        if valid_token_count <= 0:
            valid_token_count = min(len(input_ids), 1)
        trimmed_ids.append(input_ids[:valid_token_count])
        trimmed_masks.append(attention_mask[:valid_token_count])
        word_ids_per_item.append(list(encoding.word_ids[:valid_token_count]))
        valid_token_counts.append(valid_token_count)

    max_tokens = max(valid_token_counts, default=1)
    batch_input_ids = np.zeros((len(batch_words), max_tokens), dtype=np.int64)
    batch_attention_mask = np.zeros((len(batch_words), max_tokens), dtype=np.int64)

    for index, (input_ids, attention_mask) in enumerate(zip(trimmed_ids, trimmed_masks)):
        batch_input_ids[index, : input_ids.size] = input_ids
        batch_attention_mask[index, : attention_mask.size] = attention_mask

    return batch_input_ids, batch_attention_mask, word_ids_per_item, valid_token_counts


def _validate_onnx_layer_request(model_bundle: dict, layer_indices: Sequence[int]) -> None:
    exported_layer_index = int(model_bundle["layer_index"])
    if len(layer_indices) != 1 or int(layer_indices[0]) != exported_layer_index:
        raise ValueError(
            "ONNX backend supports only the exported single attention layer. "
            f"Requested {list(layer_indices)}, exported layer is {exported_layer_index}."
        )


def _batched_attention_word_scores_onnx(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
) -> list[dict[int, dict[str, np.ndarray]]]:
    _validate_onnx_layer_request(model_bundle, layer_indices)
    tokenizer = model_bundle["tokenizer"]
    session = model_bundle["session"]
    max_length = int(model_bundle["max_length"])
    output_name = session.get_outputs()[0].name

    results: list[dict[int, dict[str, np.ndarray]]] = []
    for start in range(0, len(batch_words), batch_size):
        word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        input_ids, attention_mask, word_ids_per_item, valid_token_counts = _encode_words_batch_onnx(
            word_batch,
            tokenizer,
            max_length=max_length,
        )
        outputs = session.run(
            [output_name],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        attention_batch = outputs[0]

        for item_index, words in enumerate(word_batch):
            valid_token_count = valid_token_counts[item_index]
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            attention_map = attention_batch[item_index][:valid_token_count, :valid_token_count].astype(np.float32, copy=False)
            results.append(
                {
                    layer_indices[0]: _scores_from_attention_map(valid_word_ids, attention_map, len(words)),
                }
            )

    return results


def batched_attention_word_scores(
    batch_words: Sequence[Sequence[str]],
    model_bundle: dict,
    layer_indices: Sequence[int],
    batch_size: int = 4,
) -> list[dict[int, dict[str, np.ndarray]]]:
    if not batch_words:
        return []

    if model_bundle.get("backend") == "onnx":
        return _batched_attention_word_scores_onnx(
            batch_words,
            model_bundle,
            layer_indices=layer_indices,
            batch_size=batch_size,
        )

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    results: list[dict[int, dict[str, np.ndarray]]] = []
    for start in range(0, len(batch_words), batch_size):
        word_batch = [list(words) for words in batch_words[start : start + batch_size]]
        encoded = tokenizer(
            word_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids_per_item = [encoded.word_ids(batch_index=index) for index in range(len(word_batch))]
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        layer_count = len(outputs.attentions)
        resolved_indices = {layer_index: _resolve_layer_index(layer_index, layer_count) for layer_index in layer_indices}
        attention_by_layer = {
            layer_index: outputs.attentions[resolved_index].mean(dim=1).detach().cpu().numpy()
            for layer_index, resolved_index in resolved_indices.items()
        }

        for item_index, words in enumerate(word_batch):
            per_item_scores: dict[int, dict[str, np.ndarray]] = {}
            valid_token_count = int(encoded["attention_mask"][item_index].sum().item())
            valid_word_ids = word_ids_per_item[item_index][:valid_token_count]
            for layer_index in layer_indices:
                attention_map = attention_by_layer[layer_index][item_index][:valid_token_count, :valid_token_count]
                per_item_scores[layer_index] = _scores_from_attention_map(valid_word_ids, attention_map, len(words))
            results.append(per_item_scores)

    return results


def attention_word_scores(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]
    batched_scores = batched_attention_word_scores(
        [list(words)],
        model_bundle,
        layer_indices=effective_layer_indices,
        batch_size=1,
    )
    if not batched_scores:
        return {}
    per_layer_scores = [batched_scores[0][index] for index in effective_layer_indices]
    return _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)


def attention_word_scores_with_raw(
    words: Sequence[str],
    model_bundle: dict,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_weights: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int | None]]:
    """Like attention_word_scores but also returns the token-level attention map and word_ids.

    Returns:
        (scores_by_method, attention_map, word_ids)
        - scores_by_method: same as attention_word_scores
        - attention_map: (valid_tokens, valid_tokens) averaged over heads
        - word_ids: token-to-word mapping from the tokenizer
    """
    effective_layer_indices = list(layer_indices) if layer_indices else [layer_index]

    if model_bundle.get("backend") == "onnx":
        _validate_onnx_layer_request(model_bundle, effective_layer_indices)
        tokenizer = model_bundle["tokenizer"]
        session = model_bundle["session"]
        max_length = int(model_bundle["max_length"])
        output_name = session.get_outputs()[0].name

        input_ids, attention_mask, word_ids_per_item, valid_token_counts = _encode_words_batch_onnx(
            [list(words)],
            tokenizer,
            max_length=max_length,
        )
        outputs = session.run(
            [output_name],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        valid_token_count = valid_token_counts[0]
        valid_word_ids = word_ids_per_item[0][:valid_token_count]
        raw_attention_map = outputs[0][0][:valid_token_count, :valid_token_count].astype(np.float32, copy=False)
        scores_by_method = _scores_from_attention_map(valid_word_ids, raw_attention_map, len(words))
        return scores_by_method, raw_attention_map, valid_word_ids

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    encoded = tokenizer(
        [list(words)],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    layer_count = len(outputs.attentions)
    valid_token_count = int(encoded["attention_mask"][0].sum().item())
    valid_word_ids = word_ids[:valid_token_count]

    # Use last effective layer for the raw attention map
    resolved_last = _resolve_layer_index(effective_layer_indices[-1], layer_count)
    raw_attention_map = outputs.attentions[resolved_last].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()

    # Compute word scores per layer and aggregate
    per_layer_scores = []
    for li in effective_layer_indices:
        resolved = _resolve_layer_index(li, layer_count)
        attn_map = outputs.attentions[resolved].mean(dim=1)[0][:valid_token_count, :valid_token_count].detach().cpu().numpy()
        per_layer_scores.append(_scores_from_attention_map(valid_word_ids, attn_map, len(words)))

    scores_by_method = _aggregate_layer_word_scores(per_layer_scores, layer_weights=layer_weights)
    return scores_by_method, raw_attention_map, valid_word_ids


def rescore_with_new_words(
    attention_map: np.ndarray,
    old_word_ids: list[int | None],
    old_words: Sequence[str],
    new_words: Sequence[str],
    merge_map: list[list[int]],
) -> dict[str, np.ndarray]:
    """Recompute word scores using a new word boundary mapping without re-running the model.

    Args:
        attention_map: (valid_tokens, valid_tokens) from the original forward pass
        old_word_ids: token-to-old-word mapping
        old_words: original word list
        new_words: merged word list
        merge_map: merge_map[new_idx] = [old_idx_1, old_idx_2, ...] mapping new words to old word indices
    """
    # Build new word_ids: remap old word indices to new word indices
    old_to_new = {}
    for new_idx, old_indices in enumerate(merge_map):
        for old_idx in old_indices:
            old_to_new[old_idx] = new_idx

    new_word_ids = []
    for wid in old_word_ids:
        if wid is None or wid not in old_to_new:
            new_word_ids.append(None)
        else:
            new_word_ids.append(old_to_new[wid])

    return _scores_from_attention_map(new_word_ids, attention_map, len(new_words))


__all__ = [
    "ATTENTION_METHODS",
    "normalize_array",
    "attention_word_scores",
    "attention_word_scores_with_raw",
    "rescore_with_new_words",
    "batched_attention_word_scores",
]
