from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    DataLoader = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from .dataset import PairwiseCandidateDataset, PairwiseExample
from .model import SmallTransformerReranker, SmallTransformerRerankerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small transformer reranker for keyword candidates.")
    parser.add_argument("--train-json", required=True, help="JSON file containing pairwise reranker examples.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_examples(path: str | Path) -> list[PairwiseExample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PairwiseExample(**item) for item in payload]


def main() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("Training reranker requires torch. Install with: pip install torch>=2.0")
    if AutoTokenizer is None:
        raise ImportError("Training reranker requires transformers. Install with: pip install transformers>=4.30")

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)
    examples = load_examples(args.train_json)
    dataset = PairwiseCandidateDataset(examples, tokenizer=tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = SmallTransformerRerankerConfig(
        vocab_size=int(tokenizer.vocab_size),
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = SmallTransformerReranker(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            positive_scores = model(
                batch["positive_input_ids"].to(args.device),
                batch["positive_attention_mask"].to(args.device),
                batch["positive_features"].to(args.device),
            )
            negative_scores = model(
                batch["negative_input_ids"].to(args.device),
                batch["negative_attention_mask"].to(args.device),
                batch["negative_features"].to(args.device),
            )
            targets = torch.ones_like(positive_scores)
            loss = loss_fn(positive_scores, negative_scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        mean_loss = total_loss / max(len(loader), 1)
        print(f"[epoch {epoch}] loss={mean_loss:.6f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "tokenizer": args.tokenizer,
        },
        output_dir / "reranker.pt",
    )
    print(f"[done] saved to {output_dir / 'reranker.pt'}")


if __name__ == "__main__":
    main()
