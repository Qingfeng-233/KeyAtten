from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("ppc reranker requires torch. Install with: pip install torch>=2.0")


@dataclass(slots=True)
class SmallTransformerRerankerConfig:
    vocab_size: int
    max_length: int = 512
    feature_dim: int = 10
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    ff_dim: int = 1536
    dropout: float = 0.1


class SmallTransformerReranker(nn.Module):
    def __init__(self, config: SmallTransformerRerankerConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size + config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        numeric_features: "torch.Tensor",
    ) -> "torch.Tensor":
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        projected_features = self.feature_projection(numeric_features)
        logits = self.head(torch.cat([pooled, projected_features], dim=-1))
        return logits.squeeze(-1)
