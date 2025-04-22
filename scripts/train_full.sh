#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ID=$1
DATE=0407

WORKSPACE=/path/to/checkpoints/mega/$DATE/
VERSION=train_${ID}_b16_MeGA

DEFAULT_PARAMS=./configs/nersemble/${ID}/full.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'