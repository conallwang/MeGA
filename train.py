import argparse
import datetime
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dome import DomeData
from dataset.nersemble import NeRSembleData
from joint_trainer import JointTrainer
from hair_trainer import HairTrainer
from utils import seed_everything

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./params.yaml", type=str)
parser.add_argument("--workspace", type=str, required=True)
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--extra_config", type=str, default="{}", required=False)
parser.add_argument("--time", action="store_true")
args = parser.parse_args()

# Load config yaml file and pre-process params
with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

extra_config = json.loads(args.extra_config)
for k in extra_config.keys():
    assert k in config, k
config.update(extra_config)
config["bald"] = args.config_path.split("/")[-2] in ["218"]

# Dump tmp config file
tmp_config_path = os.path.join(os.path.dirname(args.config_path), "params_tmp.yaml")
with open(tmp_config_path, "w") as f:
    print("Dumping extra config file...")
    yaml.dump(config, f)

# pre-process params
config["current_epoch"] = 0

# Config gpu
torch.cuda.set_device(0)


def get_dataset(logger, datatype="nersemble"):
    data_dict = {"nersemble": NeRSembleData, "dome": DomeData}
    assert datatype in data_dict.keys(), "Not Supported Datatype: {}".format(datatype)
    Data = data_dict[datatype]

    # get dataset
    batch_size = config["data.per_gpu_batch_size"]
    train_set = Data(config)
    if logger:
        logger.info("number of train images: {}".format(len(train_set)))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data.num_workers"],
        drop_last=False,
    )

    val_set = Data(config, split="val")
    if logger:
        logger.info("number of val images: {}".format(len(val_set)))
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader, train_set.radius, train_set.load_all_flame_params()


def train():
    seed_everything(42)

    # Enable cudnn benchmark for speed optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    config["log_file"] = os.path.join(args.workspace, args.version, "training.log")
    logger = logging.getLogger("MeGA")
    file_handler = logging.FileHandler(config["log_file"])
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.handlers = [file_handler, stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("Training config: {}".format(config))

    # tensorboard summary_writer
    config["tb_writer"] = SummaryWriter(log_dir=config["local_workspace"])
    config["logger"] = logger

    # Init data loader
    config["datatype"] = args.config_path.split("/")[-3]
    train_loader, val_loader, radius, all_flame_params = get_dataset(logger, config["datatype"])

    Trainer = HairTrainer if config["pipe.neutral_hair"] else JointTrainer
    trainer = Trainer(config, logger, radius, all_flame_params=all_flame_params)
    trainer.train(train_loader, val_loader, show_time=args.time)


def main():
    workspace = os.path.join(args.workspace, args.version)
    config["local_workspace"] = workspace
    # Create sub working dir
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    shutil.copy(tmp_config_path, os.path.join(workspace, "params.yaml"))

    # Start training
    train()


if __name__ == "__main__":
    main()
