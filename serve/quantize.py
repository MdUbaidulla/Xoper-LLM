"""Post-training quantization for CPU serving.

This repo ships PyTorch dynamic INT8 quantization (weights only,
applied at load time, no calibration data needed) as a working,
dependency-light baseline. For 4-bit GPU serving at real scale, export
the checkpoint (`scripts/export_hf.py`) and run it through AutoAWQ or
GPTQ, which protect outlier dimensions and need a calibration set —
out of scope for a from-scratch CPU reference implementation.

    [FP32 (4 bytes/weight)] -> [INT8 (1 byte/weight), this module]
                             -> [4-bit AWQ/GPTQ, via an external tool]
"""
from __future__ import annotations

import argparse

import torch

from model import ModelConfig, Transformer
from train.utils import load_checkpoint


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Quantizes nn.Linear weights to INT8 dynamically (activations stay FP32,
    requantized per forward pass). CPU-only; roughly halves memory footprint
    and the arithmetic is still numerically close to FP32 weights.
    """
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="path to save the quantized state dict")
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**payload["extra"]["model_cfg"])
    model = Transformer(model_cfg)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    q_model = quantize_dynamic_int8(model)
    torch.save({"model": q_model.state_dict(), "extra": {"model_cfg": vars(model_cfg)}}, args.out)

    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"quantized {args.checkpoint} -> {args.out} "
          f"(FP32 weights were ~{orig_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
