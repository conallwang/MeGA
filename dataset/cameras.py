import math
from copy import deepcopy

import numpy as np
import torch


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W).float()
    return Rt


def getProjectionMatrix(znear, zfar, intr, W, H):
    """refer to https://github.com/graphdeco-inria/gaussian-splatting/issues/399"""
    P = torch.zeros(4, 4)

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]

    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1
    P[1, 2] = 2 * (cy / H) - 1
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    return P


class Camera:
    def __init__(self, R, t, intr, zfar, znear, img_h, img_w, name):
        """Camera data management

        Args:
            R (_type_): world2cam rotation matrix
            t (_type_): world2cam translation vector
            FoVx (_type_): _description_
            FoVy (_type_): _description_
            image (_type_): _description_
            obj_mask (_type_): _description_
            hair_mask (_type_): _description_
            image_name (_type_): _description_
        """
        super(Camera, self).__init__()

        self.R = R
        self.t = t
        self.intr = intr
        self.name = name
        self.zfar = zfar
        self.znear = znear
        self.img_h = img_h
        self.img_w = img_w

        focal_length_x = self.intr[0, 0]
        focal_length_y = self.intr[1, 1]
        self.FovX = focal2fov(focal_length_x, self.img_w)
        self.FovY = focal2fov(focal_length_y, self.img_h)

        self.world2cam = getWorld2View2(R, t).cuda()  # world2cam

        # For 3d gaussian
        self.world_view_transform = self.world2cam.transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, intr=intr, W=self.img_w, H=self.img_h)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def apply(self, trans, scale):
        scale = max(1e-10, scale)
        self.world2cam[:3, 3] = (self.world2cam[:3, 3] - torch.from_numpy(trans).cuda()) / scale

        self.world_view_transform = self.world2cam.transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, intr=self.intr, W=self.img_w, H=self.img_h)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # For pytorch3d ndc
        s = min(self.img_h, self.img_w)
        fx_ndc = self.intr[0, 0] * 2.0 / s
        fy_ndc = self.intr[1, 1] * 2.0 / s
        cx_ndc = -(self.intr[0, 2] - self.img_w / 2.0) * 2.0 / s
        cy_ndc = -(self.intr[1, 2] - self.img_h / 2.0) * 2.0 / s

        self.intrinsics_py3d = np.zeros((4, 4))
        self.intrinsics_py3d[0, 0] = fx_ndc
        self.intrinsics_py3d[1, 1] = fy_ndc
        self.intrinsics_py3d[0, 2] = cx_ndc
        self.intrinsics_py3d[1, 2] = cy_ndc

        self.intrinsics_py3d[3, 2] = 1.0
        self.intrinsics_py3d[2, 3] = 1.0

        self.world2cam_py3d = deepcopy(self.world2cam[:3])
        if self.intrinsics_py3d[0, 0] < 0:
            self.intrinsics_py3d[:, 0] *= -1
            self.world2cam_py3d[0, :] *= -1
        self.world2cam_py3d[:3, :3] = self.world2cam_py3d[:3, :3].T
        self.world2cam_py3d[:3, :2] *= -1
        self.world2cam_py3d[:2, 3] *= -1
