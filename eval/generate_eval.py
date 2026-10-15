"""Qualitative generation eval: run a fixed prompt set through a checkpoint
and print completions, for a human (or a sub-agent judge) to read.
"""
from __future__ import annotations

import argparse
import json

import torch

from model import ModelConfig, Transformer
from train.utils import load_checkpoint

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain photosynthesis in one sentence.",
    "Translate 'good morning' into Bengali:",
]


def generate_completions(checkpoint_path: str, tokenizer_path: str, prompts: list[str],
                          max_new_tokens: int = 64, temperature: float = 0.8, top_k: int = 40) -> list[dict]:
    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**payload["extra"]["model_cfg"])
    model = Transformer(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id("<eos>")

    results = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_k=top_k, eos_id=eos_id)
        text = tok.decode(out[0].tolist())
        results.append({"prompt": prompt, "completion": text})
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts-file", help="JSON list of prompts; defaults to a small built-in set")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    for r in generate_completions(args.checkpoint, args.tokenizer, prompts, args.max_new_tokens):
        print(f"\nPROMPT: {r['prompt']}\n---\n{r['completion']}\n")


if __name__ == "__main__":
    main()
