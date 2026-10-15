"""Stage 2: Supervised Fine-Tuning. Trains the pre-trained base model on
curated instruction/response pairs, with the loss computed only over
response tokens (see data/instruction_dataset.py), teaching it the
assistant behavioral format rather than open-ended continuation.
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.instruction_dataset import SFTDataset, sft_collate
from model import ModelConfig, Transformer
from train.utils import cosine_lr_with_warmup, load_checkpoint, save_checkpoint, set_seed


def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", 1337))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = ModelConfig(**cfg["model"])
    model = Transformer(model_cfg).to(device)
    if cfg.get("init_from_pretrain"):
        load_checkpoint(cfg["init_from_pretrain"], model)
        print(f"initialized from {cfg['init_from_pretrain']}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])

    dataset = SFTDataset(cfg["data"]["sft_jsonl"], tokenizer, max_len=model_cfg.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=lambda b: sft_collate(b, pad_id=tokenizer.token_to_id("<pad>") or 0),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["max_lr"],
                             weight_decay=cfg["train"].get("weight_decay", 0.0))

    epochs = cfg["train"]["epochs"]
    total_steps = epochs * len(loader)
    warmup_steps = cfg["train"].get("warmup_steps", max(1, total_steps // 20))
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints/sft")

    model.train()
    step = 0
    for epoch in range(epochs):
        for input_ids, targets, loss_mask in loader:
            input_ids, targets, loss_mask = input_ids.to(device), targets.to(device), loss_mask.to(device)
            lr = cosine_lr_with_warmup(step, warmup_steps=warmup_steps, total_steps=total_steps,
                                        max_lr=cfg["train"]["max_lr"], min_lr=cfg["train"]["max_lr"] * 0.1)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets=targets, loss_mask=loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            if step % cfg["train"].get("log_every", 10) == 0:
                print(f"epoch {epoch} step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}")
            step += 1

        save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), model, opt, step,
                         extra={"model_cfg": vars(model_cfg)})

    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), model, opt, step,
                     extra={"model_cfg": vars(model_cfg)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
