"""Grouped-Query causal self-attention with RoPE and an optional KV cache.

Q has n_heads heads; K/V have n_kv_heads heads (n_kv_heads <= n_heads),
repeated to match Q at attention time. This cuts the KV cache memory
that dominates long-context inference, trading a small amount of
representational capacity for it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, seq, n_kv_heads, head_dim) -> (batch, seq, n_kv_heads * n_rep, head_dim)."""
    if n_rep == 1:
        return x
    b, s, kvh, hd = x.shape
    return x[:, :, :, None, :].expand(b, s, kvh, n_rep, hd).reshape(b, s, kvh * n_rep, hd)


class KVCache:
    """Static pre-allocated cache so generation only computes the newest token's K/V."""

    def __init__(self, batch: int, max_seq_len: int, n_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> None:
        shape = (batch, max_seq_len, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s = k.shape[0], k.shape[1]
        self.k[:b, self.length : self.length + s] = k
        self.v[:b, self.length : self.length + s] = v
        self.length += s
        return self.k[:b, : self.length], self.v[:b, : self.length]


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int,
                 dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        if cache is not None:
            k, v = cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # (b, h, s, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = attn_mask is None and (cache is None or cache.length == k.shape[2])
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and q.shape[2] == k.shape[2],
        )
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)
