#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/train_dpo.yaml}"
cd "$(dirname "$0")/.."
python -m train.dpo --config "$CONFIG"
