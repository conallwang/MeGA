import argparse
import math
import os
import sys

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.nersemble import NeRSembleData
from joint_trainer import Trainer
from utils import CUDA_Timer, seed_everything, ssim, visDepthMap, visimg

parser = argparse.ArgumentParser("EVALUATE")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("-bz", "--batch_size", type=int, default=6)
parser.add_argument("--skip_render", action="store_true")
parser.add_argument("--skip_metric", action="store_true")
parser.add_argument("--time", action="store_true", help="evaluate time when set true.")
parser.add_argument("-d", "--debug", action="store_true", help="debug mode")

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["training.pretrained_checkpoint_path"] = args.checkpoint
config["local_workspace"] = dir_name
config["data.load_images"] = False
config["gs.pretrain"] = None

seed_everything(42)


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def get_dataset(logger, datatype="nersemble"):
    data_dict = {"nersemble": NeRSembleData}
    assert datatype in data_dict.keys(), "Not Supported Datatype: {}".format(datatype)
    Data = data_dict[datatype]

    config["data.per_gpu_batch_size"] = args.batch_size
    batch_size = config["data.per_gpu_batch_size"]
    test_set = Data(config, split=args.split)
    if logger:
        logger.info("number of test images: {}".format(len(test_set)))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data.num_workers"],
        drop_last=False,
    )

    return test_loader, test_set.radius, test_set.load_all_flame_params()


def reshape_mask(mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)  # [H, W]
    return mask[None, None].expand(-1, 3, -1, -1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 10 * torch.log10(1.0 / mse)


def render(trainer, test_loader, logger, skip_render=False, debug=False):
    logdir = os.path.join(config["local_workspace"], "{}_eval".format(args.split))
    directory(logdir)

    gt_dir = os.path.join(logdir, "gt")
    renders_dir = os.path.join(logdir, "renders")
    rastmasks_dir = os.path.join(logdir, "rastmasks")
    head_depths_dir = os.path.join(logdir, "head_depths")
    gt_head_depths_dir = os.path.join(logdir, "gt_head_depths")
    directory(gt_dir)
    directory(renders_dir)
    directory(rastmasks_dir)
    directory(head_depths_dir)
    directory(gt_head_depths_dir)

    if debug:
        hairs_dir = os.path.join(logdir, "hairs")
        hairmasks_dir = os.path.join(logdir, "hairmasks")
        directory(hairs_dir)
        directory(hairmasks_dir)

    if skip_render:
        logger.info("Skipping rendering...")
        return gt_dir, renders_dir, rastmasks_dir, head_depths_dir, gt_head_depths_dir

    bar = tqdm(range(len(test_loader)))
    # time test
    show_time = args.time
    warmup_steps = 10
    ld_timer = CUDA_Timer("load data", logger, valid=show_time, warmup_steps=warmup_steps)
    render_timer = CUDA_Timer("render", logger, valid=show_time, warmup_steps=warmup_steps)
    ld_timer.start(0)

    fw = open(os.path.join(logdir, "parsing_path.txt"), "w")
    for step, items in enumerate(test_loader):
        step += 1

        # 1. Set data for trainer
        trainer.set_data(items)
        if show_time and step > warmup_steps:
            ld_timer.end(step - 1)

        # 2. Run the network
        render_timer.start(step)
        outputs, visualization = trainer.network_forward(is_val=True)
        render_timer.end(step)

        for i in range(trainer.img.shape[0]):
            render_path = os.path.join(renders_dir, trainer.name[i] + ".png")
            gt_path = os.path.join(gt_dir, trainer.name[i] + ".png")
            rastmask_path = os.path.join(rastmasks_dir, trainer.name[i] + ".png")
            head_depth_path = os.path.join(head_depths_dir, trainer.name[i] + ".npy")
            gt_head_depth_path = os.path.join(gt_head_depths_dir, trainer.name[i] + ".npy")
            fw.write("{} {}\n".format(trainer.name[i], items["parsing_path"][i]))

            visimg(render_path, outputs["render_fuse"][i : i + 1])
            visimg(gt_path, trainer.img[i : i + 1])
            visimg(rastmask_path, outputs["fullmask"][i : i + 1])
            np.save(head_depth_path, outputs["head_depth"][i].cpu().numpy())
            np.save(gt_head_depth_path, items["depth_map"][i].numpy())

            if debug:
                hair_path = os.path.join(hairs_dir, trainer.name[i] + ".png")
                hairmask_path = os.path.join(hairmasks_dir, trainer.name[i] + ".png")
                visimg(hair_path, outputs["render_hair"][i : i + 1])
                visimg(hairmask_path, outputs["raster_hairmask"][i : i + 1])

        bar.update()

        if show_time and step > warmup_steps:
            ld_timer.start(step)

    fw.close()
    logger.info("Rendering Finished.\n\n")

    return gt_dir, renders_dir, rastmasks_dir, head_depths_dir, gt_head_depths_dir


def metrics(gt_dir, renders_dir, rastmasks_dir, head_depths_dir, gt_head_depths_dir, logger, skip_metric=False):
    if skip_metric:
        logger.info("Skipping rendering...")
        return

    logger.info("Evaluating metrics...")

    basedir = os.path.dirname(renders_dir)
    parsing_path = os.path.join(basedir, "parsing_path.txt")
    parsing_paths = {}
    with open(parsing_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        fname, path = line.split(" ")
        parsing_paths[fname] = path.split("\n")[0]

    lpips_alex = lpips.LPIPS(net="alex")

    ssims = []
    psnrs = []
    lpipss = []
    depth_errs = []

    bar = tqdm(range(len(os.listdir(renders_dir))), desc="Metric evaluation progress")
    for fname in os.listdir(renders_dir):
        render = tf.to_tensor(Image.open(os.path.join(renders_dir, fname))).unsqueeze(0)
        gt = tf.to_tensor(Image.open(os.path.join(gt_dir, fname))).unsqueeze(0)
        rast_mask = cv2.imread(os.path.join(rastmasks_dir, fname), cv2.IMREAD_GRAYSCALE) / 255.0
        head_depth = np.load(os.path.join(head_depths_dir, fname.replace(".png", ".npy")))
        gt_head_depth = np.load(os.path.join(gt_head_depths_dir, fname.replace(".png", ".npy")))

        parsing_labels = cv2.imread(parsing_paths[fname[:-4]], cv2.IMREAD_GRAYSCALE)
        valid_regions = parsing_labels > 0
        masks = {
            "all": np.ones_like(valid_regions),
            "obj": valid_regions * (parsing_labels < 18) * (parsing_labels != 14),
            "head": valid_regions * (parsing_labels < 18) * (parsing_labels != 14) * (parsing_labels != 13),
            "face": valid_regions
            * (parsing_labels < 16)
            * (parsing_labels != 14)
            * (parsing_labels != 13)
            * (parsing_labels != 8)
            * (parsing_labels != 9)
            * (parsing_labels != 10),
            "hair": valid_regions * (parsing_labels == 13),
        }

        used_mask = masks["obj"] * rast_mask
        render *= reshape_mask(used_mask)
        gt *= reshape_mask(used_mask)

        psnrs.append(psnr(render, gt).item())
        ssims.append(ssim(render, gt).item())
        lpipss.append(lpips_alex(render, gt, normalize=True).item())

        depth_mask = masks["face"] * rast_mask
        depth_errs.append(np.sum(np.abs((head_depth - gt_head_depth) * depth_mask)) / np.sum(depth_mask))

        bar.update()

    mean_ssim = torch.tensor(ssims).mean()
    mean_psnr = torch.tensor(psnrs).mean()
    mean_lpips = torch.tensor(lpipss).mean()
    mean_depth_errs = torch.tensor(depth_errs).mean()

    logger.info("  SSIM : {}".format(mean_ssim))
    logger.info("  PSNR : {}".format(mean_psnr))
    logger.info("  LPIPS: {}".format(mean_lpips))
    logger.info("  DEPTH ERR: {}mm".format(mean_depth_errs))
    print("")

    savepath = os.path.join(basedir, "metrics.txt")
    with open(str(savepath), "w") as f:
        f.write("SSIM : {}\n".format(mean_ssim))
        f.write("PSNR : {}\n".format(mean_psnr))
        f.write("LPIPS: {}\n".format(mean_lpips))
        f.write("DEPTH ERR: {}mm\n".format(mean_depth_errs))


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = False

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("MeGA")
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

    if not args.skip_render:
        test_loader, radius, all_flame_params = get_dataset(logger, datatype="nersemble")

        opt_shape_params = np.load(os.path.join(dir_name, "flame_params.npz"))["shape"]  # [1, 300]
        trainer = Trainer(config, logger, radius)
        trainer.init_all_flame_params(all_flame_params, is_val=True)
        trainer.all_flame_params["shape"] = torch.from_numpy(opt_shape_params).float().cuda()
        trainer.set_eval()
        # trainer.stage = "joint"  # render with hair

        torch.set_grad_enabled(False)
    else:
        trainer, test_loader = None, None

    gt_dir, renders_dir, rastmasks_dir, head_depths_dir, gt_head_depths_dir = render(
        trainer, test_loader, logger, skip_render=args.skip_render, debug=args.debug
    )
    metrics(
        gt_dir, renders_dir, rastmasks_dir, head_depths_dir, gt_head_depths_dir, logger, skip_metric=args.skip_metric
    )
