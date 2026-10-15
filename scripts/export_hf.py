"""Export a checkpoint to a plain, framework-agnostic bundle: config.json
+ pytorch_model.bin + the tokenizer file, in a directory layout close
enough to a HuggingFace model repo to be a straightforward adapter
away from `transformers`, without taking that dependency here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil

import torch

from model import ModelConfig, Transformer
from train.utils import load_checkpoint


def export(checkpoint_path: str, tokenizer_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig(**payload["extra"]["model_cfg"])
    model = Transformer(model_cfg)
    load_checkpoint(checkpoint_path, model)

    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(model_cfg), f, indent=2)

    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, os.path.join(out_dir, "tokenizer.json"))

    print(f"exported {checkpoint_path} -> {out_dir}/ "
          f"(pytorch_model.bin, config.json, tokenizer.json)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    export(args.checkpoint, args.tokenizer, args.out_dir)


if __name__ == "__main__":
    main()
