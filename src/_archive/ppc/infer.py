from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from .features import build_feature_vector
from .model import SmallTransformerReranker, SmallTransformerRerankerConfig


class CandidateReranker:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("Inference reranker requires torch. Install with: pip install torch>=2.0")
        if AutoTokenizer is None:
            raise ImportError("Inference reranker requires transformers. Install with: pip install transformers>=4.30")

        payload = torch.load(str(checkpoint_path), map_location=device)
        config = SmallTransformerRerankerConfig(**payload["config"])
        self.model = SmallTransformerReranker(config).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(payload["tokenizer"], use_fast=True, trust_remote_code=True)
        self.device = device

    def score_candidate(
        self,
        *,
        document_text: str,
        candidate_text: str,
        numeric_features: dict[str, Any],
    ) -> float:
        encoded = self.tokenizer(
            f"文本：{document_text}\n候选：{candidate_text}",
            truncation=True,
            max_length=self.model.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        features = torch.tensor(build_feature_vector(numeric_features), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = self.model(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
                features.to(self.device),
            )
        return float(score.item())

    def rerank(
        self,
        *,
        document_text: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        scored = []
        for item in candidates:
            rerank_score = self.score_candidate(
                document_text=document_text,
                candidate_text=item["candidate_text"],
                numeric_features=item,
            )
            enriched = dict(item)
            enriched["rerank_score"] = rerank_score
            scored.append(enriched)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


def rerank_json(checkpoint_path: str | Path, input_json: str | Path, device: str = "cpu") -> str:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    reranker = CandidateReranker(checkpoint_path, device=device)
    results = reranker.rerank(
        document_text=payload["document_text"],
        candidates=payload["candidates"],
        top_k=int(payload.get("top_k", 10)),
    )
    return json.dumps(results, ensure_ascii=False, indent=2)
