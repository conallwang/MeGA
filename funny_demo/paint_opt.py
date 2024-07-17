import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.nersemble import NeRSembleData
from joint_trainer import Trainer
from utils import seed_everything, visimg, visPositionMap


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


parser = argparse.ArgumentParser("PAINTING")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--pti", action="store_true")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--version", type=str, default="painting")
parser.add_argument("--split", type=str, default="painting")
parser.add_argument("--ratio", type=float, default=1.0)

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["training.pretrained_checkpoint_path"] = args.checkpoint
config["local_workspace"] = dir_name
config["training.epochs"] = 2000
config["training.learning_rate"] = args.lr

# direct optimization
stages = ["painting"]
stages_epoch = [0, 10000000]

# PTI-like optimization
if args.pti:
    stages = ["painting_code", "painting_mlp"]
    stages_epoch = [0, 1200, 10000000]
    args.version = args.version + "_pti"

# Dump tmp config file
logdir = os.path.join(config["local_workspace"], args.version)
directory(logdir)
tmp_config_path = os.path.join(logdir, "params.yaml")
with open(tmp_config_path, "w") as f:
    print("Dumping extra config file...")
    yaml.dump(config, f)

seed_everything(42)


def update_stage(trainer, current_epoch):
    try:
        cur_id = next(i for i, v in enumerate(stages_epoch) if v > current_epoch)
    except:
        cur_id = -1
    new_stage = stages[cur_id - 1]
    if new_stage != trainer.stage:
        trainer.stage = new_stage
        trainer._set_stage(new_stage)


def get_dataset(logger, datatype="nersemble"):
    data_dict = {"nersemble": NeRSembleData}
    assert datatype in data_dict.keys(), "Not Supported Datatype: {}".format(datatype)
    Data = data_dict[datatype]

    # get dataset
    config["data.per_gpu_batch_size"] = 1
    batch_size = config["data.per_gpu_batch_size"]
    painting_set = Data(config, split=args.split)
    if logger:
        logger.info("number of test images: {}".format(len(painting_set)))
    painting_loader = DataLoader(
        painting_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    return painting_loader, painting_set.radius


def get_uvmask(trainer, painting_loader):
    for step, items in enumerate(painting_loader):
        if "painting_mask" in items:
            # 1. Set data for trainer
            trainer.set_data(items)
            painting_mask = items["painting_mask"].cuda()

            # 2. Run the network
            outputs, visualization = trainer.network_forward()

            painting_mask = painting_mask[0, ..., None].detach().cpu().numpy()
            face_ids = set(outputs["head_face_ids"][0].reshape(-1).tolist())
            if -1 in face_ids:
                face_ids.remove(-1)
            f_ids = list(face_ids)
            f_ids.sort()

            uv_mask = trainer.remap_tex_from_2dmask(outputs["head_verts_refine"], painting_mask, f_ids)
            return uv_mask

    print("\n\nThere is no any painting mask !!!  Please check !!!\n\n")
    exit(1)


def get_xymask(uv_mask, XY_uv):
    size = uv_mask.shape[0]
    XY_uv = XY_uv * (size - 1)
    xy_mask = cv2.remap(uv_mask.astype(np.float32), XY_uv[:, :, 0], XY_uv[:, :, 1], cv2.INTER_LINEAR)

    return torch.from_numpy(xy_mask).bool().cuda()


def get_painting_masks(trainer, painting_loader, uv_mask):
    painting_masks = {}
    for step, items in enumerate(painting_loader):
        # 1. Set data for trainer
        trainer.set_data(items)
        name = items["name"][0]

        # 2. Run the network
        outputs, visualization = trainer.network_forward()

        head_face_ids = outputs["head_face_ids"][0]  # [H, W]
        head_face_bw = outputs["head_face_bw"][0]  # [H, W, 2]
        head_face_bw = torch.cat([head_face_bw, 1 - head_face_bw.sum(-1, keepdim=True)], dim=-1)
        head_face_uvs = outputs["head_face_uvs"]  # [F, 3, 2]
        valid = head_face_ids > -1

        XY_uv = (head_face_uvs[head_face_ids] * head_face_bw[..., None]).sum(-2)  # [H, W, 2]
        painting_mask = get_xymask(uv_mask, XY_uv.detach().cpu().numpy()) * valid
        painting_masks[name] = painting_mask[None].detach()

    return painting_masks


def painting(trainer, painting_loader, logger, save_stage=None):
    uv_mask = get_uvmask(trainer, painting_loader)
    painting_masks = get_painting_masks(trainer, painting_loader, uv_mask)
    res = np.where(uv_mask)
    us, vs = list(res[1]), list(res[0])
    old_tex = trainer.models["neural_texture"].texture.data.clone()

    bar = tqdm(range(config["training.epochs"]))
    global_step = 0
    for i in range(config["training.epochs"]):
        update_stage(trainer, i)

        losses = []
        for step, items in enumerate(painting_loader):
            global_step += 1

            # 1. Set data for trainer
            trainer.set_data(items)
            name = items["name"][0]
            painting = items["painting"][0]

            # 2. Run the network
            outputs, visualization = trainer.network_forward()
            painting_mask = painting_masks[name]

            # 3. compute loss
            hair_mask = trainer.mask["hair"]
            head_mask = torch.clip(1 - hair_mask, min=0.0, max=1.0)
            painting_mask_out = (head_mask - painting_mask.int()).bool()

            render_head = outputs["render_head"]
            gt_head = trainer.img * head_mask[..., None]
            if painting:
                loss_painting = torch.linalg.norm(render_head[painting_mask] - gt_head[painting_mask], dim=-1).mean()
                loss_skin = torch.linalg.norm(
                    render_head[painting_mask_out] - gt_head[painting_mask_out], dim=-1
                ).mean()
            else:
                loss_painting = torch.tensor(0.0).cuda()
                loss_skin = torch.linalg.norm(
                    render_head[painting_mask_out] - gt_head[painting_mask_out], dim=-1
                ).mean()
            loss = loss_painting + args.ratio * loss_skin

            # 4. update params
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.optimizer.step()

            new_values = trainer.models["neural_texture"].texture.data[:, :, vs, us].clone()
            trainer.models["neural_texture"].texture.data[:] = old_tex[:]
            trainer.models["neural_texture"].texture.data[:, :, vs, us] = new_values

            losses.append(loss.detach().item())

            if (global_step == 1) or global_step % 1000 == 0:
                # Save model
                checkpoint_path = os.path.join(logdir, "checkpoint_latest.pth")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                trainer.save_ckpt(checkpoint_path, stage=save_stage)
                logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

                savepath = os.path.join(logdir, "render_head_it{}.png".format(global_step))
                visimg(savepath, outputs["render_head"])
                savepath = os.path.join(logdir, "render_basic_head_it{}.png".format(global_step))
                visimg(savepath, outputs["render_basic_head"])

            if global_step % 2000 == 0:
                # Save model
                checkpoint_path = os.path.join(logdir, "checkpoint_it{}.pth".format(global_step))
                trainer.save_ckpt(checkpoint_path, stage=save_stage)
                logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

        bar.set_description("loss {}".format(np.mean(losses)))
        bar.update()


if __name__ == "__main__":
    # Enable cudnn benchmark for speed optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("MeGA EDIT")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler(config["log_file"])
    # file_handler.setFormatter(formatter)
    # logger.handlers = [file_handler, stream_handler]
    logger.handlers = [stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    config["logger"] = logger

    logger.info("Config: {}".format(config))

    painting_loader, radius = get_dataset(logger, datatype="nersemble")

    opt_flame_params = np.load(os.path.join(dir_name, "flame_params.npz"))
    trainer = Trainer(config, logger, radius, painting=True)
    trainer.load_all_flame_params(all_flame_params=opt_flame_params)
    trainer.set_train()
    trainer._set_stage("painting")

    painting(trainer, painting_loader, logger, save_stage=trainer.stage)
