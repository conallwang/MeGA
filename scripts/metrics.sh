#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/path/to/checkpoints/MeGA/0728/train_306_b16_MeGA/checkpoint_latest.pth"
SPLIT="test"    # choose from ['test', 'val', 'train', 'onef']

# Command Line Arguments for metrics.py
# --checkpoint,         path of the trained models
# --split,              choose from ['test', 'val', 'train', 'onef']
# -bz, --batch_size,    batch size used during evaluation
# --skip_render,        skip rendering process, only compute metrics from previous rendering results
# --skip_metric,        skip computing metrics, only rendering images
# --time,               print processing time of each module
# -d, --debug,          debug mode, saving some extra results

python metrics.py --checkpoint $CKPT_PATH --split $SPLIT