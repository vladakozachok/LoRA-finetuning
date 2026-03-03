import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        if rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))

        nn.init.normal_(self.A, std=1 / math.sqrt(rank))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * (self.dropout(x) @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )
        self.lora.to(device=linear.weight.device, dtype=linear.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)
