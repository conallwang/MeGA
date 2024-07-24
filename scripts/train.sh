#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

WORKSPACE='/home/exp/conallwang_works/checkpoints/MeGA/0724/'
VERSION=train_306_b8_MeGA_hair

DEFAULT_PARAMS=./configs/nersemble/306/hair.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'