"""Stage 3 (preference optimization): Direct Preference Optimization.

Bypasses training a separate reward model. For a prompt x with a
preferred completion y_w and a dispreferred one y_l:

    L_DPO = -log sigmoid( beta * [ (log pi_t(y_w|x) - log pi_ref(y_w|x))
                                  - (log pi_t(y_l|x) - log pi_ref(y_l|x)) ] )

`pi_ref` is a frozen copy of the SFT model, kept to prevent the policy
from drifting arbitrarily far while it chases the preference signal —
the same role the KL penalty plays in PPO/RLHF, without a reward model.
"""
from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.instruction_dataset import DPODataset, dpo_collate
from model import ModelConfig, Transformer
from train.utils import load_checkpoint, save_checkpoint, set_seed


def sequence_logp(model: Transformer, input_ids: torch.Tensor, targets: torch.Tensor,
                   loss_mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(target_t | <t) over the response span (loss_mask == 1), per batch row."""
    logits, _ = model(input_ids)
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * loss_mask).sum(dim=-1)


def dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta: float):
    policy_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
    return loss, accuracy


def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", 1337))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = ModelConfig(**cfg["model"])
    policy = Transformer(model_cfg).to(device)
    load_checkpoint(cfg["init_from_sft"], policy)

    ref_model = copy.deepcopy(policy).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])
    pad_id = tokenizer.token_to_id("<pad>") or 0

    dataset = DPODataset(cfg["data"]["dpo_jsonl"], tokenizer, max_len=model_cfg.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=lambda b: dpo_collate(b, pad_id=pad_id, max_len=model_cfg.max_seq_len),
    )

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["max_lr"])
    beta = cfg["train"].get("beta", 0.1)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints/dpo")

    policy.train()
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for (c_ids, c_tgt, c_mask), (r_ids, r_tgt, r_mask) in loader:
            c_ids, c_tgt, c_mask = c_ids.to(device), c_tgt.to(device), c_mask.to(device)
            r_ids, r_tgt, r_mask = r_ids.to(device), r_tgt.to(device), r_mask.to(device)

            policy_chosen = sequence_logp(policy, c_ids, c_tgt, c_mask)
            policy_rejected = sequence_logp(policy, r_ids, r_tgt, r_mask)
            with torch.no_grad():
                ref_chosen = sequence_logp(ref_model, c_ids, c_tgt, c_mask)
                ref_rejected = sequence_logp(ref_model, r_ids, r_tgt, r_mask)

            loss, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()

            if step % cfg["train"].get("log_every", 10) == 0:
                print(f"epoch {epoch} step {step:6d} | loss {loss.item():.4f} | "
                      f"preferred-ranked-higher {acc.item():.2%}")
            step += 1

        save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), policy, opt, step,
                         extra={"model_cfg": vars(model_cfg)})

    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), policy, opt, step,
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
