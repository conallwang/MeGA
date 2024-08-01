#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

HEAD_CKPT_PATH="/path/to/checkpoints/MeGA/0801/train_218_b16_MeGA/checkpoint_latest.pth"
HAIR_CKPT_PATH="/path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/checkpoint_latest.pth"
MOTIONFILE="/path/to/nersemble/preprocess/218/218_EXP-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json"
NAME="exp2_306hair"

# Command Line Arguments for render_funny.py

# --checkpoint,         path of the trained models
# --motionfile,         specify the json file for rendering
# --hair,               path of the trained models (used to alter the original hair) 
# --name,               the results will be saved in 'dir($HEAD_CKPT_PATH)/$name_eval'
# --time,               print processing time of each module

# --fix_expr_id,        specify only one frame to render
# --fix_view_id,        specify only one view to render, default=8, forward view

# --free_view,          render images with 'spheric' or 'zoom' camera trajectory
# -n, --num_frames,     specify total frame number that need to be rendered. Must be set when specify --free_view
# --view_mode,          specify the free view mode, default 'spheric'.

# --with_pose,          using FLAME params with pose, always False when performing cross-identity rendering


python funny_demo/render_funny.py \
    --checkpoint $HEAD_CKPT_PATH \
    --motionfile $MOTIONFILE \
    --hair $HAIR_CKPT_PATH \
    --name $NAME --with_pose