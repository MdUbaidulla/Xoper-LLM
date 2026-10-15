"""Rotary Position Embeddings (RoPE).

Rotates query/key vectors in 2D sub-planes by an angle proportional to
absolute position, which makes the dot product QK^T depend only on the
*relative* distance between positions. Generalizes better to long
context than absolute sinusoidal encodings.
"""
from __future__ import annotations

import torch


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Returns complex-valued rotation factors of shape (max_seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64, unit magnitude


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """xq, xk: (batch, seq, n_heads, head_dim). freqs_cis: (seq, head_dim // 2)."""
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    fc = freqs_cis[: xq.shape[1]].view(1, xq.shape[1], 1, -1)

    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
