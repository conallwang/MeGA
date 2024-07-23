#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

CKPT_PATH="/home/exp/conallwang_works/checkpoints/MeGA/0719/train_306_b16_MeGA/duola/checkpoint_latest.pth"
MOTIONFILE="/home/exp/conallwang_works/nersemble/preprocess/306/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json"
NAME="emo2"

python funny_demo/render_funny.py \
    --checkpoint $CKPT_PATH \
    --motionfile $MOTIONFILE \
    --name $NAME