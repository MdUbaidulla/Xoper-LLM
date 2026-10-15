"""SwiGLU feed-forward block (gated linear unit with SiLU activation).

hidden = SiLU(xW1) * (xW3); out = hidden @ W2. The gating gives more
capacity per parameter than a plain ReLU/GELU MLP at the same width.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def swiglu_hidden_dim(dim: int, ratio: float, mult_of: int) -> int:
    hidden = int(ratio * dim)
    hidden = int(2 * hidden / 3)  # SwiGLU convention: keep params roughly matched to a dense 4x MLP
    return mult_of * ((hidden + mult_of - 1) // mult_of)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ratio: float, mult_of: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = swiglu_hidden_dim(dim, ratio, mult_of)
        self.w1 = nn.Linear(dim, hidden, bias=bias)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=bias)   # value
        self.w2 = nn.Linear(hidden, dim, bias=bias)   # down-projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
