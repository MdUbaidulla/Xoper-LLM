#!/usr/bin/env bash
# Resume pre-training from the latest checkpoint in a run's ckpt_dir.
# The training loops in this repo start step counting at 0 on every
# invocation; wire a `--resume-from` flag into train/pretrain.py's
# build_optimizer/train() if you need exact step-count continuation
# (load_checkpoint already restores model + optimizer state either way).
set -euo pipefail

CONFIG="${1:?usage: resume.sh <config.yaml> <ckpt_dir>}"
CKPT_DIR="${2:?usage: resume.sh <config.yaml> <ckpt_dir>}"

cd "$(dirname "$0")/.."

LATEST=$(ls -t "$CKPT_DIR"/step_*.pt 2>/dev/null | head -1 || true)
if [ -z "$LATEST" ]; then
  echo "no checkpoint found in $CKPT_DIR, starting fresh"
else
  echo "resuming from $LATEST"
fi

RESUME_FROM="$LATEST" python -m train.pretrain --config "$CONFIG"
