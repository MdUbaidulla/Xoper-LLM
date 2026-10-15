"""LoRA (Low-Rank Adaptation): freeze the base weight, learn a low-rank
update W + (alpha / r) * B @ A, where A is r x in_features and B is
out_features x r. Keeps the base model's general knowledge intact
(avoids catastrophic forgetting) while specializing cheaply.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Drop-in replacement for a frozen `nn.Linear`, with a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        in_features, out_features = base.in_features, base.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # update starts at exactly zero
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out


def inject_lora(model: nn.Module, target_substrings: tuple[str, ...] = ("wq", "wv"),
                 r: int = 8, alpha: int = 16, dropout: float = 0.0) -> nn.Module:
    """Replaces attention projection Linear layers whose name contains one of
    `target_substrings` (default: query/value projections, as in the LoRA paper)
    with LoRALinear adapters, leaving everything else frozen-by-default.
    """
    for p in model.parameters():
        p.requires_grad = False

    for module_name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in attr_name for t in target_substrings):
                setattr(module, attr_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model


def lora_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
