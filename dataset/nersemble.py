import json
import os

import cv2
import numpy as np
import sparse
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import fov2focal, get_center_and_diag


class NeRSembleData(Dataset):
    def __init__(self, config, split="train"):
        super().__init__()

        self.config = config
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0

        self.basedir = config["data.root"]
        if config["pipe.neutral_hair"]:
            split = "onef"
        jsonfile_path = os.path.join(self.basedir, "transforms_{}.json".format(split))

        with open(jsonfile_path) as fr:
            jsonfile = json.load(fr)
        self.framelist = jsonfile["frames"]

        campos = []
        for frame in self.framelist:
            campos.append(np.array(frame["transform_matrix"])[:3, 3:])
        _, diagonal = get_center_and_diag(campos)
        self.radius = diagonal * 1.1

        # find max frame index
        self.max_frame_idx = -1
        for frame in self.framelist:
            self.max_frame_idx = max(self.max_frame_idx, frame["timestep_index"])

    def load_all_flame_params(self):
        params_infos = {}
        for idx, frame in tqdm(enumerate(self.framelist), total=len(self.framelist)):
            if not "timestep_index" in frame or frame["timestep_index"] in params_infos:
                continue

            flame_param = dict(np.load(os.path.join(self.basedir, frame["flame_param_path"]), allow_pickle=True))
            params_infos[frame["timestep_index"]] = flame_param
        return params_infos

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, index):
        frame = self.framelist[index]

        sample = {"frame_idx": frame["timestep_index"], "cam_idx": frame["camera_index"], "cam": frame["camera_id"]}
        sample["name"] = frame["file_path"].split("/")[-1][:-4]

        cx, cy = frame["cx"] * self.rate_w, frame["cy"] * self.rate_h
        fx, fy = fov2focal(frame["camera_angle_x"], self.img_w), fov2focal(frame["camera_angle_y"], self.img_h)
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        sample["intr"] = intrinsics

        c2w = np.array(frame["transform_matrix"])
        sample["proj_w2c"] = np.linalg.inv(c2w)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        sample["w2c"] = w2c

        # view direction
        campos = c2w[:3, 3]
        view = campos / np.linalg.norm(campos)
        sample["view"] = np.tile(view, (8, 8, 1)).transpose((2, 0, 1))

        # load images
        imgpath = os.path.join(self.basedir, frame["file_path"])
        photo = cv2.imread(imgpath)
        photo = cv2.resize(photo, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        sample["img"] = photo / 255.0

        painting = len(sample["name"]) > 8
        sample["painting"] = painting
        basename = sample["name"][8:]
        # load depth map
        path = imgpath.replace("images", "depths").replace("{}.png".format(basename) if painting else ".png", ".npz")
        depth_map = sparse.load_npz(path).todense()
        depth_map = cv2.resize(depth_map, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        depth_map = depth_map * 1000  # 转换单位为 mm
        sample["depth_map"] = depth_map

        # load parsing results
        path = imgpath.replace("images", "parsing").replace(
            "{}.png".format(basename) if painting else ".png", "_labels.png"
        )
        sample["parsing_path"] = path
        labels = cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )  # [img_h, img_w]
        valid_regions = (labels > 0) * (labels < 18)
        sample["obj_mask"] = valid_regions * (labels != 14)
        sample["hair_mask"] = valid_regions * np.logical_or(labels == 13, labels == 15)
        sample["head_mask"] = valid_regions * (labels != 14) * (labels != 13)

        kernel = np.ones((5, 5), dtype=np.uint8)
        sample["erode_hair_mask"] = cv2.erode(sample["hair_mask"].astype(np.float32), kernel, iterations=1).astype(
            np.bool_
        )

        if self.config.get("bald", False):
            sample["hair_mask"] = np.zeros_like(valid_regions).astype(np.bool_)
            sample["head_mask"] = sample["obj_mask"]

        # load painting mask
        if painting:
            path = imgpath.replace("images", "fg_masks")
            painting_mask = (
                cv2.resize(
                    cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )  # [img_h, img_w]
            sample["painting_mask"] = painting_mask

        return sample
