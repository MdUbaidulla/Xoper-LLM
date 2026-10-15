"""Model configuration.

Mirrors the architecture doc: decoder-only Transformer, GQA, RoPE,
RMSNorm, Pre-LN, SwiGLU MLP, tied input/output embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 512               # model (hidden) dimension
    n_layers: int = 8
    n_heads: int = 8             # query heads
    n_kv_heads: int = 2          # key/value heads (GQA); n_heads % n_kv_heads == 0
    mlp_ratio: float = 3.5       # SwiGLU hidden size = dim * mlp_ratio (rounded to mult_of)
    mult_of: int = 32            # round SwiGLU hidden dim up to a multiple of this
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    bias: bool = False           # no bias in linears (LLaMA-style)

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
