"""Tokenize cleaned documents and write fixed-dtype token shards to disk
as flat uint16/uint32 binary files (nanoGPT-style), so training reads
them with `numpy.memmap` instead of loading everything into RAM.

Usage:
    python -m data.shard --input cleaned/*.txt --tokenizer tokenizer.json \
        --out-dir shards/ --shard-tokens 10_000_000
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size <= 65535 else np.uint32


def shard_corpus(input_glob: str, tokenizer_path: str, out_dir: str,
                  shard_tokens: int = 10_000_000, eos_token: str = "<eos>") -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id(eos_token)
    vocab_size = tok.get_vocab_size()
    dtype = _dtype_for_vocab(vocab_size)

    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {input_glob!r}")

    buffer: list[int] = []
    shard_idx = 0
    total_tokens = 0

    def flush(buf: list[int], idx: int) -> None:
        if not buf:
            return
        arr = np.array(buf, dtype=dtype)
        path = os.path.join(out_dir, f"shard_{idx:05d}.bin")
        arr.tofile(path)
        print(f"wrote {path} ({len(arr):,} tokens)")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            continue
        ids = tok.encode(text).ids
        ids.append(eos_id)
        buffer.extend(ids)
        total_tokens += len(ids)
        while len(buffer) >= shard_tokens:
            flush(buffer[:shard_tokens], shard_idx)
            buffer = buffer[shard_tokens:]
            shard_idx += 1
    flush(buffer, shard_idx)
    shard_idx += 1

    meta_path = os.path.join(out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"dtype={np.dtype(dtype).name}\n")
        f.write(f"num_shards={shard_idx}\n")
        f.write(f"total_tokens={total_tokens}\n")
    print(f"done: {shard_idx} shard(s), {total_tokens:,} tokens total. meta -> {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="glob of cleaned .txt files")
    p.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = p.parse_args()
    shard_corpus(args.input, args.tokenizer, args.out_dir, args.shard_tokens)


if __name__ == "__main__":
    main()
