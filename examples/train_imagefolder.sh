#!/usr/bin/env bash
set -euo pipefail
python scripts/train_imagefolder.py \
  --data-root /path/to/imagefolder_dataset \
  --dataset-name custom_task \
  --model vit_base_patch16_224 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --rank 8 \
  --route auto \
  --out-dir runs/custom_task_scope
