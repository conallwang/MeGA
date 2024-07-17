import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from joint_trainer import Trainer
from utils import CUDA_Timer, fov2focal, seed_everything, visimg

parser = argparse.ArgumentParser("RENDER FUNNY")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--motionfile", type=str, required=True, help="path of motion")
parser.add_argument("--hair", type=str, default=None)
parser.add_argument("--fix_expr_id", type=int, default=-1)
parser.add_argument("-n", "--num_frames", type=int, default=-1)
parser.add_argument("--fix_view_id", type=int, default=8)  # default: forward view
parser.add_argument("--free_view", action="store_true")
parser.add_argument("--with_pose", action="store_true")
parser.add_argument("--view_mode", type=str, default="spheric", help="spheric or zoom. Default: spheric")
parser.add_argument("--name", type=str, default="debug")
parser.add_argument("--time", action="store_true", help="evaluate time when set true.")

args = parser.parse_args()

fix_expr = args.fix_expr_id > -1
given_num = args.num_frames > 0
assert not (fix_expr ^ given_num), "Fix expression and number of frames need to be set simultaneously"

update_list = [
    "data.canonical_flame_path",
    "gs.deform_weightnorm",
    "gs.deform_layers",
    "gs.pe.num_freqs",
    "gs.deform_lr",
    "gs.deform_attr",
]

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# make sure params.yaml in the same directory with checkpoint
if args.hair is not None:
    sourcedir_name = os.path.dirname(args.hair)
    config_path = os.path.join(sourcedir_name, "params.yaml")
    with open(config_path, "r") as f:
        src_config = yaml.safe_load(f)

    for k, v in src_config.items():
        if k in update_list:
            config[k] = v

    config["hair.shape_params"] = opt_shape_params = np.load(os.path.join(sourcedir_name, "flame_params.npz"))["shape"]
# Uncomment to disable hair non-rigid alignment.
# config["hair.shape_params"] = None

config["training.pretrained_checkpoint_path"] = args.checkpoint
config["local_workspace"] = dir_name
config["data.load_images"] = False
config["gs.pretrain"] = None
config["data.per_gpu_batch_size"] = 1

seed_everything(42)


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def get_share_parts():
    basedir = config["data.root"]
    jsonfile_path = os.path.join(basedir, "transforms_onef.json")
    with open(jsonfile_path) as fr:
        jsonfile = json.load(fr)
    frame = jsonfile["frames"][0]  # one frame for each view

    cx, cy = frame["cx"], frame["cy"]
    fx, fy = fov2focal(frame["camera_angle_x"], frame["w"]), fov2focal(frame["camera_angle_y"], frame["h"])
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    path = os.path.join(dir_name, "flame_params.npz")
    shape_param = np.load(path)["shape"]  # [1, 300]

    return intrinsics, shape_param


def get_fixposes(n_steps=120):
    jsonfile_path = os.path.join(config["data.root"], "transforms_onef.json")
    with open(jsonfile_path) as fr:
        jsonfile = json.load(fr)
    framelist = jsonfile["frames"][:16]  # one frame for each view

    fix_idx = args.fix_view_id  # can change according to your choice
    frame = framelist[fix_idx]

    c2w = np.array(frame["transform_matrix"])
    c2w[:3, 1:3] *= -1

    all_c2w = np.repeat(c2w[None], n_steps, axis=0)
    return all_c2w


def create_spheric_poses(n_steps=120):
    z = 1.2

    center = np.array([0.0, 0.02, 0.0]).astype(np.float32)
    r = 0.2
    up = np.array([0.0, 1.0, 0.0], dtype=center.dtype)

    all_c2w = []
    for theta in np.linspace(2 * math.pi, 0, n_steps):
        diff = np.stack([r * np.cos(theta), r * np.sin(theta), z])
        cam_pos = center + diff
        l = -diff / np.linalg.norm(diff)
        s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
        u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
        c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        all_c2w.append(c2w)

    all_c2w = np.stack(all_c2w, axis=0)

    return all_c2w


def create_zoom_poses(n_steps=120):
    near = 0.2
    far = 3.5

    half_steps = n_steps // 2
    trace_out = np.linspace(near, far, half_steps)
    trace_in = np.linspace(far, near, n_steps - half_steps + 1)
    trace = np.concatenate([trace_out, trace_in[1:]], axis=0)

    center = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=center.dtype)

    all_c2w = []
    for z in trace:
        diff = np.stack([0, 0, z])
        cam_pos = center + diff
        l = -diff / np.linalg.norm(diff)
        s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
        u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
        c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        all_c2w.append(c2w)

    all_c2w = np.stack(all_c2w, axis=0)
    return all_c2w


def load_motions(motionfile, with_pose=False):
    """Load motions from nersemble data"""

    basedir = os.path.dirname(motionfile)
    with open(motionfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

        motion_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not "timestep_index" in frame or frame["timestep_index"] in motion_infos:
                continue

            flame_param = dict(np.load(os.path.join(basedir, frame["flame_param_path"]), allow_pickle=True))
            if not with_pose:
                flame_param["neck_pose"][:] = 0.0
                flame_param["rotation"][:] = 0.0
                flame_param["translation"][:] = 0.0
            motion_infos[frame["timestep_index"]] = flame_param

    # make keys start from 0
    sort_keys = list(motion_infos.keys())
    sort_keys.sort()
    new_motion_infos = {}
    for i, key in enumerate(sort_keys):
        new_motion_infos[i] = motion_infos[key]

    return new_motion_infos


def load_voca(motion_path):
    """Load motions from VOCA data"""
    flame_paths = []
    for file in os.listdir(motion_path):
        if file[-4:] != ".npy":
            continue
        flame_paths.append(os.path.join(motion_path, file))
    flame_paths.sort()

    flame_params = {}
    for i, path in enumerate(flame_paths):
        flame_param = np.load(path)  # [300 + 100 + 15 + 3]
        flame_params[i] = {
            "expr": flame_param[300:400][None],
            "rotation": flame_param[400:403][None],
            "neck_pose": flame_param[403:406][None],
            "jaw_pose": flame_param[406:409][None],
            "eyes_pose": flame_param[409:415][None],
            "translation": flame_param[415:][None],
        }

        flame_params[i]["neck_pose"][:] = 0.0
        flame_params[i]["rotation"][:] = 0.0
        flame_params[i]["translation"][:] = 0.0

    return flame_params


def render(trainer, logger):

    if ".json" in args.motionfile:
        flame_params = load_motions(args.motionfile, with_pose=args.with_pose)
    else:
        flame_params = load_voca(args.motionfile)
    num_frames = len(flame_params.keys()) if args.num_frames <= 0 else args.num_frames

    bar = tqdm(range(num_frames))
    logdir = os.path.join(config["local_workspace"], "{}_eval".format(args.name))
    directory(logdir)

    # load cameras
    intrin, shape_param = get_share_parts()
    if args.free_view:
        if args.view_mode == "spheric":
            cams = create_spheric_poses(num_frames)
        elif args.view_mode == "zoom":
            cams = create_zoom_poses(num_frames)
        else:
            raise ValueError("Unknown view mode: {}".format(args.view_mode))
    else:
        cams = get_fixposes(num_frames)

    fix_flame = None
    if args.fix_expr_id >= 0:
        fix_flame = flame_params[args.fix_expr_id]

    renders_dir = os.path.join(logdir, "renders")
    rastmasks_dir = os.path.join(logdir, "rastmasks")
    directory(renders_dir)
    directory(rastmasks_dir)

    # time test
    show_time = args.time
    warmup_steps = 10
    ld_timer = CUDA_Timer("load data", logger, valid=show_time, warmup_steps=warmup_steps)
    render_timer = CUDA_Timer("render", logger, valid=show_time, warmup_steps=warmup_steps)
    ld_timer.start(0)

    for i in range(num_frames):
        name = "{:05}".format(i)
        # current flame params
        if fix_flame is None:
            flame_param = flame_params[i]
        else:
            flame_param = fix_flame
        flame_param["shape"] = shape_param

        # 1. Set data for trainer
        trainer.batch_size = 1
        trainer.load_flame_params(flame_param)
        trainer.load_cameras(cams[i : i + 1], {"intr": intrin[None], "cam": [name]})
        trainer.expand_dims()
        if show_time and (i + 1) > warmup_steps:
            ld_timer.end(i)

        # 2. Run the network
        render_timer.start(i)
        outputs, visualization = trainer.network_forward(is_val=True)
        render_timer.end(i)

        # 3. Save images
        render_path = os.path.join(renders_dir, name + ".png")
        rastmask_path = os.path.join(rastmasks_dir, name + ".png")

        visimg(render_path, outputs["render_fuse"])
        visimg(rastmask_path, outputs["fullmask"])

        bar.update()

        if show_time and (i + 1) > warmup_steps:
            ld_timer.start(i)

    logger.info("Rendering Finished.")


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

    trainer = Trainer(config, logger, spatial_lr_scale=0.0)
    trainer.stage = "head" if config["bald"] else "joint"
    if args.hair is not None:
        trainer.load_hair(args.hair)
        trainer.stage = "joint"
    trainer.set_eval()

    torch.set_grad_enabled(False)
    render(trainer, logger)
