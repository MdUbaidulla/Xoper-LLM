#!/usr/bin/env bash
# Launch pre-training. Single GPU/CPU: NPROC=1 (default). Multi-GPU on one
# node: NPROC=<num_gpus>. Multi-node: wrap with torchrun's --nnodes/--node_rank
# (see https://pytorch.org/docs/stable/elastic/run.html) and set MASTER_ADDR.
set -euo pipefail

CONFIG="${1:-configs/train_pretrain.yaml}"
NPROC="${NPROC:-1}"

cd "$(dirname "$0")/.."

if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" -m train.pretrain --config "$CONFIG"
else
  python -m train.pretrain --config "$CONFIG"
fi
