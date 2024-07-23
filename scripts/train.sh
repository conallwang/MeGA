#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

WORKSPACE='/home/exp/conallwang_works/checkpoints/MeGA/0719/'
VERSION=train_306_b16_MeGA

DEFAULT_PARAMS=./configs/nersemble/306/full.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'