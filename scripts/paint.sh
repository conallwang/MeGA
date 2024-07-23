#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

CKPT_PATH="/home/exp/conallwang_works/checkpoints/MeGA/0719/train_306_b16_MeGA/checkpoint_latest.pth"
VERSION=duola
SPLIT=duola

python funny_demo/paint_opt.py \
    --checkpoint $CKPT_PATH \
    --version $VERSION \
    --split $SPLIT