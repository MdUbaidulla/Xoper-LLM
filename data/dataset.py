"""IterableDataset reading tokenized shards via numpy.memmap and yielding
fixed-length (input, target) blocks for next-token-prediction training.
"""
from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _read_meta(shard_dir: str) -> dict:
    meta: dict[str, str] = {}
    with open(os.path.join(shard_dir, "meta.txt"), "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split("=", 1)
            meta[k] = v
    return meta


class ShardedTokenDataset(IterableDataset):
    """Streams random `block_size`-length windows from on-disk token shards.

    Each worker opens its own memmaps and draws independent random
    offsets, which approximates shuffled sampling over a huge corpus
    without loading it into memory.
    """

    def __init__(self, shard_dir: str, block_size: int, seed: int = 0) -> None:
        super().__init__()
        self.shard_dir = shard_dir
        self.block_size = block_size
        self.seed = seed
        self.meta = _read_meta(shard_dir)
        self.dtype = np.dtype(self.meta["dtype"])
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.bin")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id)

        memmaps = [np.memmap(p, dtype=self.dtype, mode="r") for p in self.shard_paths]
        memmaps = [m for m in memmaps if len(m) > self.block_size + 1]
        if not memmaps:
            raise RuntimeError("All shards are shorter than block_size + 1")

        while True:
            m = rng.choice(memmaps)
            start = rng.randint(0, len(m) - self.block_size - 1)
            chunk = m[start : start + self.block_size + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
            yield x, y
