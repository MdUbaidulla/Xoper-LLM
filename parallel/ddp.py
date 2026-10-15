"""Data-parallel training setup (the DP axis of 3D parallelism): each
worker holds a full model copy, processes a distinct data slice, and
syncs gradients with All-Reduce via torch.distributed's DDP wrapper.

Tensor and pipeline parallelism (splitting one model's weight matrices
or layers across GPUs), and ZeRO-style optimizer/gradient/parameter
sharding, are the next axes to reach for once a single model copy no
longer fits one accelerator's memory. At that point swap this module
for DeepSpeed (ZeRO stages 1-3) or FSDP, and Megatron-LM-style tensor
parallelism for attention/MLP weight splits — the training loops in
train/ are written against a plain nn.Module/DDP-module interface, so
the swap doesn't touch pretrain.py's step logic, only this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist_pkg
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def init_from_env(cls) -> "DistributedState":
        """Reads standard torchrun env vars (RANK, WORLD_SIZE, LOCAL_RANK).
        Falls back to a single-process, single-device run when they're absent.
        """
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist_pkg.init_process_group(backend=backend, rank=rank, world_size=world_size)
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            return cls(rank=rank, world_size=world_size, local_rank=local_rank, device=device)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(rank=0, world_size=1, local_rank=0, device=device)

    def wrap_ddp(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.world_size <= 1:
            return model
        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids)

    def destroy(self) -> None:
        if self.world_size > 1 and dist_pkg.is_initialized():
            dist_pkg.destroy_process_group()
