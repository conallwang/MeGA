#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

HEAD_CKPT_PATH="/path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/duola/checkpoint_latest.pth"
MOTIONFILE="/path/to/nersemble/preprocess/306/306_EXP-3_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json"
NAME="exp3"

# Refer to ./scripts/alter_hair.sh for detailed command line arguments
python funny_demo/render_funny.py \
    --checkpoint $HEAD_CKPT_PATH \
    --motionfile $MOTIONFILE \
    --name $NAME --with_pose