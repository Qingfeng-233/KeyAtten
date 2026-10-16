from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from .features import build_feature_vector


@dataclass(slots=True)
class PairwiseExample:
    document_text: str
    positive_candidate: str
    negative_candidate: str
    positive_features: dict[str, Any]
    negative_features: dict[str, Any]


class PairwiseCandidateDataset(Dataset):
    def __init__(
        self,
        examples: list[PairwiseExample],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, document_text: str, candidate_text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            f"文本：{document_text}\n候选：{candidate_text}",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        positive = self._encode(example.document_text, example.positive_candidate)
        negative = self._encode(example.document_text, example.negative_candidate)
        return {
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "positive_features": torch.tensor(
                build_feature_vector(example.positive_features),
                dtype=torch.float32,
            ),
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
            "negative_features": torch.tensor(
                build_feature_vector(example.negative_features),
                dtype=torch.float32,
            ),
        }
