"""Stage 1: self-supervised pre-training on next-token prediction.

    L = sum_i log P(x_i | x_<i; theta)

Supports single-process or torch.distributed (DDP) runs; see
parallel/ddp.py for the process-group setup this script calls into.
Launch with `scripts/launch_pretrain.sh` (wraps `torchrun`).
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import ShardedTokenDataset
from model import ModelConfig, Transformer
from parallel.ddp import DistributedState
from train.utils import cosine_lr_with_warmup, grad_global_norm, save_checkpoint, set_seed


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]):
    # no weight decay on norms/biases/1-D params, matching standard GPT/LLaMA practice
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", 1337))
    dist = DistributedState.init_from_env()

    device = dist.device
    model_cfg = ModelConfig(**cfg["model"])
    model = Transformer(model_cfg).to(device)
    if dist.world_size > 1:
        model = dist.wrap_ddp(model)

    raw_model = model.module if hasattr(model, "module") else model
    if dist.is_main:
        n_params = raw_model.num_params()
        print(f"model: {n_params / 1e6:.1f}M non-embedding params, {model_cfg.n_layers} layers, "
              f"dim={model_cfg.dim}, heads={model_cfg.n_heads}/{model_cfg.n_kv_heads} (q/kv)")

    opt = build_optimizer(model, cfg["train"]["max_lr"], cfg["train"]["weight_decay"],
                           tuple(cfg["train"].get("betas", (0.9, 0.95))))

    dataset = ShardedTokenDataset(cfg["data"]["shard_dir"], block_size=model_cfg.max_seq_len,
                                   seed=cfg.get("seed", 1337) + dist.rank)
    loader = DataLoader(dataset, batch_size=cfg["train"]["micro_batch_size"],
                         num_workers=cfg["train"].get("num_workers", 2))

    grad_accum = cfg["train"]["grad_accum_steps"]
    total_steps = cfg["train"]["total_steps"]
    warmup_steps = cfg["train"]["warmup_steps"]
    max_lr = cfg["train"]["max_lr"]
    min_lr = cfg["train"].get("min_lr", max_lr * 0.1)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    log_every = cfg["train"].get("log_every", 10)
    ckpt_every = cfg["train"].get("ckpt_every", 1000)
    ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints/pretrain")

    use_bf16 = cfg["train"].get("dtype", "bf16") == "bf16" and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    data_iter = iter(loader)
    model.train()
    t0 = time.time()
    for step in range(total_steps):
        lr = cosine_lr_with_warmup(step, warmup_steps=warmup_steps, total_steps=total_steps,
                                    max_lr=max_lr, min_lr=min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                 dtype=autocast_dtype, enabled=use_bf16):
                _, loss = model(x, targets=y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        opt.step()

        if dist.is_main and step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            toks_per_step = grad_accum * cfg["train"]["micro_batch_size"] * model_cfg.max_seq_len * dist.world_size
            print(f"step {step:7d} | loss {loss_accum:.4f} | lr {lr:.2e} | grad_norm {gnorm:.2f} "
                  f"| {toks_per_step / max(dt, 1e-6):,.0f} tok/s")

        if dist.is_main and step > 0 and step % ckpt_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"step_{step}.pt"), raw_model, opt, step,
                             extra={"model_cfg": vars(model_cfg)})

    if dist.is_main:
        save_checkpoint(os.path.join(ckpt_dir, "final.pt"), raw_model, opt, total_steps,
                         extra={"model_cfg": vars(model_cfg)})
    dist.destroy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
