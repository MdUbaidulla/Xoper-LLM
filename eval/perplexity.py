"""Validation loss / perplexity over held-out, decontaminated shards.

perplexity = exp(mean token-level cross-entropy loss)

Run after every checkpoint so regressions between runs are visible;
`scripts/launch_pretrain.sh` can call this on a held-out shard dir on
the same cadence as checkpointing.
"""
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from data.dataset import ShardedTokenDataset
from model import ModelConfig, Transformer
from train.utils import load_checkpoint


@torch.no_grad()
def evaluate_perplexity(model: Transformer, shard_dir: str, block_size: int,
                         batch_size: int = 8, max_batches: int = 100,
                         device: str = "cpu") -> dict:
    model.eval()
    dataset = ShardedTokenDataset(shard_dir, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    mean_loss = total_loss / max(n_batches, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "batches": n_batches}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=100)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**payload["extra"]["model_cfg"])
    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    result = evaluate_perplexity(model, args.shard_dir, model_cfg.max_seq_len,
                                  args.batch_size, args.max_batches, device)
    print(f"loss={result['loss']:.4f}  perplexity={result['perplexity']:.2f}  "
          f"(over {result['batches']} batches)")


if __name__ == "__main__":
    main()
