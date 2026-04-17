#!/bin/bash
set -e

PYTHON=/opt/conda/bin/python
cd /workspace/le-wm

echo "=== Step 1: Generate depth maps ==="
$PYTHON generate_depth.py --input /data/lewm_data/aigen_train.h5 --model-size Small
$PYTHON generate_depth.py --input /data/lewm_data/aigen_val.h5 --model-size Small

echo "=== Step 2: Train ViT-tiny baseline ==="
$PYTHON train.py --config-name lewm_tiny data=aigen 2>&1 | tee /tmp/lewm_tiny.log

echo "=== Step 3: Train ViT-tiny + depth regularization ==="
$PYTHON train_depth_reg.py 2>&1 | tee /tmp/lewm_tiny_depth.log

echo "=== Step 4: Evaluate all models ==="
# Find latest checkpoints
TINY_CKPT=$(ls -t ~/.stable_worldmodel/lewm_tiny_epoch_100_object.ckpt 2>/dev/null | head -1)
DEPTH_CKPT=$(ls -t ~/.stable_worldmodel/lewm_tiny_depth_epoch_100_object.ckpt 2>/dev/null | head -1)

echo "Tiny checkpoint: $TINY_CKPT"
echo "Depth checkpoint: $DEPTH_CKPT"

echo "--- VO Probe: Tiny baseline ---"
$PYTHON eval_vo.py --ckpt "$TINY_CKPT" --epochs 50

echo "--- VO Probe: Tiny + depth reg ---"
$PYTHON eval_vo.py --ckpt "$DEPTH_CKPT" --epochs 50

echo "--- Rollout: Tiny baseline ---"
$PYTHON eval_rollout.py --ckpt "$TINY_CKPT"

echo "--- Rollout: Tiny + depth reg ---"
$PYTHON eval_rollout.py --ckpt "$DEPTH_CKPT"

echo "=== All done ==="
