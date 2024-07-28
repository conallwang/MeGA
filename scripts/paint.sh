#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/path/to/checkpoints/MeGA/0728/train_306_b16_MeGA/checkpoint_latest.pth"
VERSION=duola
SPLIT=duola

# Command Line Arguments for paint_opt.py
# --checkpoint,         path of the trained models
# --pti,                optimize textures using PTI-like mode
# --lr,                 learning rate to optimize textures
# --version,            the results will be saved in 'dir($CKPT_PATH)/$version'
# --split,              used to specify the json file for painting
# --lambda_s,           the weight of the skin loss

python funny_demo/paint_opt.py \
    --checkpoint $CKPT_PATH \
    --version $VERSION \
    --split $SPLIT