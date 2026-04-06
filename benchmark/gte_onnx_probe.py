from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from keyword_bench.output_paths import resolve_output_path

try:
    import onnx  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing optional dependency 'onnx'. Install with: "
        'pip install "keyatten[inference,zh,lightweight]"'
    ) from exc

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing optional dependency 'onnxruntime'. Install with: "
        'pip install "keyatten[inference,zh,lightweight]"'
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gte-small-zh attention export with ONNX Runtime.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--words", nargs="+", required=True)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


class AttentionMeanExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layer_index: int) -> None:
        super().__init__()
        self.model = model
        self.layer_index = layer_index

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        return outputs.attentions[self.layer_index].float().mean(dim=1)


def aggregate_subwords_to_words(word_ids: list[int | None], token_scores: np.ndarray, word_count: int) -> np.ndarray:
    sums = np.zeros(word_count, dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += float(token_scores[token_index])
        counts[word_id] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def word_received_scores(attention_map: np.ndarray, word_ids: list[int | None], words: list[str]) -> np.ndarray:
    received_scores = attention_map.sum(axis=0)
    return aggregate_subwords_to_words(word_ids, received_scores, len(words))


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    try:
        model = AutoModel.from_pretrained(args.model_path, output_attentions=True, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(args.model_path, output_attentions=True)
    model.eval()

    encoded = tokenizer(
        [args.words],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    valid_token_count = int(encoded["attention_mask"][0].sum().item())
    valid_word_ids = word_ids[:valid_token_count]

    wrapper = AttentionMeanExportWrapper(model, layer_index=args.layer_index)
    with torch.no_grad():
        torch_attention = wrapper(encoded["input_ids"], encoded["attention_mask"])[0, :valid_token_count, :valid_token_count]
    torch_attention_np = torch_attention.detach().cpu().numpy().astype(np.float32, copy=False)

    export_start = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (encoded["input_ids"], encoded["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["attention_mean"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "attention_mean": {0: "batch", 1: "sequence_out", 2: "sequence_in"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - export_start

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_start = time.perf_counter()
    ort_outputs = session.run(
        ["attention_mean"],
        {
            "input_ids": encoded["input_ids"].cpu().numpy(),
            "attention_mask": encoded["attention_mask"].cpu().numpy(),
        },
    )
    ort_seconds = time.perf_counter() - ort_start

    ort_attention = ort_outputs[0][0, :valid_token_count, :valid_token_count].astype(np.float32, copy=False)
    diff = np.abs(torch_attention_np - ort_attention)

    torch_received = word_received_scores(torch_attention_np, valid_word_ids, args.words)
    ort_received = word_received_scores(ort_attention, valid_word_ids, args.words)
    received_diff = np.abs(torch_received - ort_received)

    report = {
        "model_path": args.model_path,
        "output_path": str(output_path),
        "word_count": len(args.words),
        "valid_token_count": valid_token_count,
        "attention_shape": list(torch_attention_np.shape),
        "onnx_bytes": output_path.stat().st_size,
        "export_seconds": export_seconds,
        "ort_seconds": ort_seconds,
        "attention_max_abs_diff": float(diff.max(initial=0.0)),
        "attention_mean_abs_diff": float(diff.mean() if diff.size else 0.0),
        "received_max_abs_diff": float(received_diff.max(initial=0.0)),
        "received_mean_abs_diff": float(received_diff.mean() if received_diff.size else 0.0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
