"""Incremental generation with a pre-allocated KV cache.

Without caching, generating the N-th token recomputes K/V for all N-1
prior tokens: O(seq_len^2) total work. With a cache, each new step only
computes K/V for the one new token and looks up the rest: O(seq_len)
total. `model.Transformer.generate` (model/transformer.py) already
does this via `model/attention.py::KVCache`; this module wraps it as a
reusable, non-training-time entry point for serving code, plus a
speculative-decoding stub for pairing with a small draft model.
"""
from __future__ import annotations

import time

import torch

from model import Transformer


@torch.no_grad()
def generate_with_cache(model: Transformer, tokenizer, prompt: str, max_new_tokens: int = 128,
                         temperature: float = 0.7, top_k: int = 40, device: str = "cpu") -> dict:
    eos_id = tokenizer.token_to_id("<eos>")
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    t0 = time.time()
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, eos_id=eos_id)
    dt = time.time() - t0

    n_generated = out.shape[1] - ids.shape[1]
    text = tokenizer.decode(out[0].tolist())
    return {
        "text": text,
        "tokens_generated": n_generated,
        "seconds": dt,
        "tokens_per_second": n_generated / max(dt, 1e-6),
    }


@torch.no_grad()
def speculative_generate(target: Transformer, draft: Transformer, tokenizer, prompt: str,
                          max_new_tokens: int = 128, draft_steps: int = 4,
                          device: str = "cpu") -> dict:
    """Speculative decoding: `draft` (a small, fast model) proposes `draft_steps`
    tokens; `target` verifies them in one forward pass and accepts the longest
    correct prefix, falling back to its own sample on the first rejection.
    Both models must share a tokenizer/vocab. This is a reference
    implementation for a single sequence (batch size 1), favoring clarity
    over the batched/CUDA-graph form a production server would use.
    """
    eos_id = tokenizer.token_to_id("<eos>")
    tokens = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    target_caches = target.init_caches(1, device=device)
    draft_caches = draft.init_caches(1, device=device)

    target_logits, _ = target.forward(tokens, caches=target_caches)
    draft_logits, _ = draft.forward(tokens, caches=draft_caches)

    accepted_total = 0
    generated = 0
    while generated < max_new_tokens:
        # draft proposes `draft_steps` tokens greedily
        proposals = []
        step_logits = draft_logits
        for _ in range(draft_steps):
            next_tok = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            proposals.append(next_tok)
            step_logits, _ = draft.forward(next_tok, caches=draft_caches)

        proposed = torch.cat(proposals, dim=1)  # (1, draft_steps)

        # target verifies in one pass over the proposed continuation
        verify_logits, _ = target.forward(proposed, caches=target_caches)
        target_greedy = torch.cat(
            [target_logits[:, -1:, :].argmax(-1), verify_logits[:, :-1, :].argmax(-1)], dim=1
        )

        matches = (target_greedy == proposed).int()
        n_accept = int(matches.cumprod(dim=1).sum(dim=1).item())  # longest correct prefix

        tokens = torch.cat([tokens, proposed[:, :n_accept]], dim=1)
        accepted_total += n_accept
        generated += n_accept

        if n_accept < draft_steps or (eos_id is not None and (proposed[:, :n_accept] == eos_id).any()):
            # on rejection (or a full accept run), resync both caches' logits to the true next step
            fallback_logits, _ = target.forward(tokens[:, -1:], caches=target_caches)
            next_tok = fallback_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            generated += 1
            target_logits = fallback_logits
            draft_logits, _ = draft.forward(tokens[:, -1:], caches=draft_caches)
        else:
            target_logits = verify_logits
            draft_logits = step_logits

        if eos_id is not None and (tokens[:, -1] == eos_id).all():
            break

    text = tokenizer.decode(tokens[0].tolist())
    acceptance_rate = accepted_total / max(generated, 1)
    return {"text": text, "tokens_generated": generated, "draft_acceptance_rate": acceptance_rate}
