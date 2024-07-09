#!/bin/bash

WORKSPACE='/home/share/wangcong/checkpoint/MeGA/0701/'
VERSION=train_306_b16_MeGA

DEFAULT_PARAMS=./configs/nersemble/306/full.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'