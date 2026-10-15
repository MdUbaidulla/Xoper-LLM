"""Decoder-only Transformer: Pre-LN RMSNorm, GQA + RoPE attention, SwiGLU MLP.

[Input Tokens] -> [Embedding] -> [Transformer Block x N] -> [RMSNorm] -> [Linear -> logits]
Each block: x = x + Attn(RMSNorm(x));  x = x + MLP(RMSNorm(x))
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention, KVCache
from .config import ModelConfig
from .mlp import SwiGLU
from .norm import RMSNorm
from .rope import precompute_rope_freqs


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len, cfg.dropout, cfg.bias)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_ratio, cfg.mult_of, cfg.bias, cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: KVCache | None = None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, cache, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.register_buffer(
            "freqs_cis", precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta), persistent=False
        )
        self.apply(self._init_weights)
        # scale residual-stream projections down with depth, as in GPT-2 / LLaMA
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """tokens: (batch, seq) int64. If `caches` is given, each layer reads/writes its own KVCache
        (used for incremental decoding); `tokens` may then be just the newest step(s)."""
        b, s = tokens.shape
        x = self.tok_embeddings(tokens)

        start = caches[0].length if caches is not None else 0
        freqs_cis = self.freqs_cis[start : start + s].to(x.device)

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, freqs_cis, cache)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            if loss_mask is not None:
                loss_mask = loss_mask.view(-1).float()
                loss = (loss_flat * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            else:
                loss = loss_flat.mean()
        return logits, loss

    def init_caches(self, batch: int, device: str | torch.device = "cpu") -> list[KVCache]:
        return [
            KVCache(batch, self.cfg.max_seq_len, self.cfg.n_kv_heads, self.cfg.head_dim, device=device)
            for _ in range(self.cfg.n_layers)
        ]

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """prompt_tokens: (batch, prompt_len) int64. Greedy if temperature == 0."""
        self.eval()
        b = prompt_tokens.shape[0]
        caches = self.init_caches(b, device=prompt_tokens.device)

        logits, _ = self.forward(prompt_tokens, caches=caches)
        tokens = prompt_tokens
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
            logits, _ = self.forward(next_token, caches=caches)
        return tokens
