"""Datasets for Stage 2 (SFT) and Stage 3 (DPO) training.

SFT examples are JSONL: {"prompt": "...", "response": "..."}.
DPO examples are JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}.

Loss is computed only over response tokens (prompt tokens are masked
out), which is what teaches the model to answer rather than to
continue imitating arbitrary prompt text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

PAD_ID_DEFAULT = 0


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


@dataclass
class SFTExample:
    input_ids: list[int]
    loss_mask: list[int]  # 1 where loss is computed (response tokens), 0 elsewhere


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 1024,
                 bos_id: int | None = None, eos_id: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = bos_id if bos_id is not None else tokenizer.token_to_id("<bos>")
        self.eos_id = eos_id if eos_id is not None else tokenizer.token_to_id("<eos>")
        self.examples: list[SFTExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.examples.append(self._build(row["prompt"], row["response"]))

    def _build(self, prompt: str, response: str) -> SFTExample:
        prompt_ids = [self.bos_id] + _encode(self.tokenizer, prompt)
        response_ids = _encode(self.tokenizer, response) + [self.eos_id]
        ids = (prompt_ids + response_ids)[: self.max_len]
        mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[: self.max_len]
        return SFTExample(ids, mask)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.examples[idx]


def sft_collate(batch: list[SFTExample], pad_id: int = PAD_ID_DEFAULT):
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.float)

    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        input_ids[i, :n] = torch.tensor(ex.input_ids)
        # next-token targets: shift by one within the same row; last position has no target
        if n > 1:
            targets[i, : n - 1] = torch.tensor(ex.input_ids[1:])
            loss_mask[i, : n - 1] = torch.tensor(ex.loss_mask[1:], dtype=torch.float)
    return input_ids, targets, loss_mask


@dataclass
class DPOExample:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


class DPODataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 1024,
                 bos_id: int | None = None, eos_id: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = bos_id if bos_id is not None else tokenizer.token_to_id("<bos>")
        self.eos_id = eos_id if eos_id is not None else tokenizer.token_to_id("<eos>")
        self.examples: list[DPOExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt_ids = [self.bos_id] + _encode(self.tokenizer, row["prompt"])
                chosen_ids = (_encode(self.tokenizer, row["chosen"]) + [self.eos_id])[: max_len]
                rejected_ids = (_encode(self.tokenizer, row["rejected"]) + [self.eos_id])[: max_len]
                self.examples.append(DPOExample(prompt_ids, chosen_ids, rejected_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DPOExample:
        return self.examples[idx]


def _build_sequence(prompt_ids: list[int], response_ids: list[int], max_len: int):
    ids = (prompt_ids + response_ids)[:max_len]
    mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[:max_len]
    return ids, mask


def dpo_collate(batch: list[DPOExample], pad_id: int = PAD_ID_DEFAULT, max_len: int = 1024):
    """Builds four padded tensors: chosen/rejected input_ids and their response loss masks.
    Targets are the inputs shifted by one, matching the causal LM convention used in pretrain/SFT.
    """
    chosen_seqs, chosen_masks, rejected_seqs, rejected_masks = [], [], [], []
    for ex in batch:
        c_ids, c_mask = _build_sequence(ex.prompt_ids, ex.chosen_ids, max_len)
        r_ids, r_mask = _build_sequence(ex.prompt_ids, ex.rejected_ids, max_len)
        chosen_seqs.append(c_ids)
        chosen_masks.append(c_mask)
        rejected_seqs.append(r_ids)
        rejected_masks.append(r_mask)

    def pad_batch(seqs: list[list[int]], masks: list[list[int]]):
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        targets = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        loss_mask = torch.zeros((len(seqs), width), dtype=torch.float)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            n = len(s)
            input_ids[i, :n] = torch.tensor(s)
            if n > 1:
                targets[i, : n - 1] = torch.tensor(s[1:])
                loss_mask[i, : n - 1] = torch.tensor(m[1:], dtype=torch.float)
        return input_ids, targets, loss_mask

    chosen = pad_batch(chosen_seqs, chosen_masks)
    rejected = pad_batch(rejected_seqs, rejected_masks)
    return chosen, rejected
