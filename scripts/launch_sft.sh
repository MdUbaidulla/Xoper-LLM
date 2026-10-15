#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/train_sft.yaml}"
cd "$(dirname "$0")/.."
python -m train.sft --config "$CONFIG"
