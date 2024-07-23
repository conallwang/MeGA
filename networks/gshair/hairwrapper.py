import os
import cv2

import torch
import torch.nn as nn
import numpy as np

from pytorch3d.loss import chamfer_distance

from networks.gshair.gs.gaussian_model import GaussianModel
from networks.gshair.gs.deformation import DeformMLP

from utils import full_aiap_loss, params_with_lr, render, render_list, ssim, update_lambda


class GSHairWrapper(nn.Module):
    def __init__(self, cfg, init_pts, spatial_lr_scale, sh_degree=3):
        super().__init__()

        self.attr_dims = {
            "xyz": 3,
            "scaling": 3,
            "rotation": 4,
            "opacity": 1,
            "features_dc": 3,
            "features_rest": ((sh_degree + 1) ** 2 - 1) * 3,
        }
        self.cfg = cfg
        self.img_h, self.img_w = cfg["data.img_h"], cfg["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0
        self.rate = min(self.rate_h, self.rate_w)
        self.spatial_lr_scale = spatial_lr_scale
        self.buffer = {}

        self.models = {}
        self.parameters_to_train = []

        # Initialize canonical gaussians
        self.models["canonical_gs"] = GaussianModel(cfg["gs.sh_degree"])

        init_pts = np.load(cfg["gs.init_pts"])  # [N, 3]
        self.models["canonical_gs"].create_from_pts(init_pts, self.spatial_lr_scale)
        self.parameters_to_train += self.models["canonical_gs"].trainable_params(cfg)

        # Deformation MLPs
        self.models["deform_mlp"] = DeformMLP(cfg, self.attr_dims).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["deform_mlp"].named_parameters()), cfg["gs.deform_lr"], label="hair"
        )

        # pixel coords
        px, py = np.meshgrid(np.arange(self.img_w), np.arange(self.img_h))
        self.pixelcoords = torch.from_numpy(np.stack((px, py), axis=-1)).float().cuda()

        # find all lambda
        self.all_lambdas = {}
        prelen = len("training.lambda_")
        for k, v in cfg.items():
            if "lambda" not in k or "lambda_update_list" in k:
                continue
            self.all_lambdas[k[prelen:]] = v

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def get_model(self, key):
        return self.models[key]

    def update_x(self, lambda_name, step):
        return update_lambda(
            self.cfg["training.lambda_{}".format(lambda_name)],
            self.cfg["training.lambda_{}.slope".format(lambda_name)],
            self.cfg["training.lambda_{}.end".format(lambda_name)],
            step,
            self.cfg["training.lambda_{}.interval".format(lambda_name)],
        )

    def update_weights(self, step):
        update_names = self.cfg["training.lambda_update_list"]
        for k, _ in self.all_lambdas.items():
            if k in update_names:
                self.all_lambdas[k] = self.update_x(k, step)

    def update_xyz_lr(self, step, optimizer):
        self.models["canonical_gs"].update_learning_rate(step, optimizer)

    def render(self, viewpoint_cameras, bg_color=[1.0, 1.0, 1.0]):
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        batch_size = len(viewpoint_cameras)

        rasterized_hair = render_list(
            self.cfg, viewpoint_cameras, [self.models["canonical_gs"] for _ in range(batch_size)], background
        )

        # save some variables
        self.buffer["visibility_filter"] = rasterized_hair["visibility_filter"]
        self.buffer["radii"] = rasterized_hair["radii"]
        self.buffer["viewspace_points"] = rasterized_hair["viewspace_points"]
        self.buffer["gs_reg_loss"] = {}
        self.buffer["gs_aiap_xyz_loss"] = torch.tensor(0.0).cuda()
        self.buffer["gs_aiap_cov_loss"] = torch.tensor(0.0).cuda()

        return rasterized_hair

    def render_with_trans(self, viewpoint_cameras, flame_params, R=None, t=None, bg_color=[1.0, 1.0, 1.0]):
        loss_reg = {}
        aiap_xyz_loss, aiap_cov_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        batch_size = len(viewpoint_cameras)

        # Rigid
        R = torch.eye(3)[None].expand(batch_size, -1, -1).float().cuda() if R is None else R
        t = torch.zeros(batch_size, 3).float().cuda() if t is None else t
        deformed_pts = self.models["canonical_gs"].rigid_deform(R, t)

        # Non-Rigid
        pts_c = self.models["canonical_gs"].get_xyz[None].expand(batch_size, -1, -1)
        expr_params = flame_params["expr"]
        expression_expand = expr_params[:, None].expand(-1, pts_c.shape[1], -1).reshape(-1, expr_params.shape[-1])
        offsets = self.models["deform_mlp"](pts_c.reshape(-1, 3), expression_expand).reshape(
            batch_size, pts_c.shape[1], -1
        )
        offsets *= 0.001

        gs_offsets = {}
        idcs = 0
        for attr in self.cfg["gs.deform_attr"]:
            next_idcs = idcs + self.attr_dims[attr]
            offset = offsets[..., idcs:next_idcs]
            gs_offsets[attr] = offset
            loss_reg[attr] = torch.linalg.norm(offset.clone(), dim=-1).mean()
            idcs = next_idcs
        gs_offsets["xyz"] += deformed_pts - pts_c
        deformed_gaussians = self.models["canonical_gs"].update_gaussian(gs_offsets, batch_size)

        if self.cfg["gs.enable_aiap"]:
            for deform_gs in deformed_gaussians:
                xyz_loss, cov_loss = full_aiap_loss(
                    self.models["canonical_gs"], deform_gs, n_neighbors=self.cfg["gs.K"]
                )
                aiap_xyz_loss += xyz_loss
                aiap_cov_loss += cov_loss
            aiap_xyz_loss /= batch_size
            aiap_cov_loss /= batch_size
        rasterized_hair = render_list(self.cfg, viewpoint_cameras, deformed_gaussians, background)

        # save some variables
        self.buffer["visibility_filter"] = rasterized_hair["visibility_filter"]
        self.buffer["radii"] = rasterized_hair["radii"]
        self.buffer["viewspace_points"] = rasterized_hair["viewspace_points"]
        self.buffer["gs_reg_loss"] = loss_reg
        self.buffer["gs_aiap_xyz_loss"] = aiap_xyz_loss
        self.buffer["gs_aiap_cov_loss"] = aiap_cov_loss

        return rasterized_hair

    def oneupSHdegree(self):
        self.models["canonical_gs"].oneupSHdegree()

    def update_use_flags(self, step):
        if step == 1:
            self.models["canonical_gs"].reset_use_flags()
        self.models["canonical_gs"].update_use_flags()

    def track_gsstates(self, i):
        # Keep track of max radii in image-space for pruning
        radii, visibility_filter = self.buffer["radii"][i], self.buffer["visibility_filter"][i]
        viewspace_point_tensor = self.buffer["viewspace_points"][i]
        self.models["canonical_gs"].max_radii2D[visibility_filter] = torch.max(
            self.models["canonical_gs"].max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.models["canonical_gs"].add_densification_stats(viewspace_point_tensor, visibility_filter)

    def densify_n_prune(self, optimizer, step):
        size_threshold = 20 * self.rate if step > self.cfg["gs.opacity_reset_interval"] else None
        self.models["canonical_gs"].densify_and_prune(
            self.cfg["gs.densify_grad_threshold"],
            0.005,
            self.spatial_lr_scale,
            size_threshold,
            optimizer,
        )

    def reset_opacities(self, optimizer):
        self.models["canonical_gs"].reset_opacity(optimizer)
        self.models["canonical_gs"].reset_use_flags()

    def get_lambda(self, key):
        return self.all_lambdas.get(key, 0.0)

    def get_optim_params(self):
        return self.parameters_to_train

    def restore_models(self, state_dict, optimizer, global_step, logger=None):
        for key, model in self.models.items():
            if key not in state_dict:
                continue

            _state_dict = {
                k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict[key].items()
            }
            # Check if there is key mismatch:
            missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
            missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

            if logger:
                logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
                logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))
            else:
                print("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
                print("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

            try:
                if key == "canonical_gs":
                    model.load_state_dict(_state_dict, optimizer, global_step, self.cfg["gs.upSH"])
                else:
                    model.load_state_dict(_state_dict, strict=False)
            except:
                if logger:
                    logger.info("[warning] {} weights are not loaded.".format(key))
                else:
                    print("[warning] {} weights are not loaded.".format(key))

    def compute_losses(self, outputs, gt_hair, hair_mask, erode_hair_mask, step):
        render_hair = outputs["render_hair"]
        hair_silhoutte = outputs["hair_silhoutte"]
        occlussion_mask = outputs["occlussion_mask"]

        if render_hair is None:
            hair_loss = torch.tensor(0.0).cuda()
            ssim_loss = torch.tensor(0.0).cuda()
            hair_silh_loss = torch.tensor(0.0).cuda()
            solid_hair_loss = torch.tensor(0.0).cuda()
            loss_aiap = torch.tensor(0.0).cuda()
        else:
            self.update_weights(step)
            gs_render_hair = render_hair.clone()
            gs_render_hair[(1 - hair_mask).bool()] = 1.0

            # L2 Loss
            hair_loss = torch.linalg.norm((gs_render_hair - gt_hair) * hair_mask[..., None], dim=-1).mean()

            # SSIM Loss
            ssim_loss = 1.0 - ssim(gs_render_hair.permute((0, 3, 1, 2)), gt_hair.permute((0, 3, 1, 2)))

            # Silh Loss
            gs_render_mask = hair_silhoutte * occlussion_mask
            distmap = torch.zeros_like(hair_mask).float().cuda()
            for i in range(gs_render_mask.shape[0]):
                valid_pixels = gs_render_mask[i] > 0
                render_mask_pixels = self.pixelcoords[valid_pixels]
                hair_mask_pixels = self.pixelcoords[hair_mask[i] > 0]
                dist, _ = chamfer_distance(
                    render_mask_pixels[None],
                    hair_mask_pixels[None],
                    single_directional=True,
                    batch_reduction=None,
                    point_reduction=None,
                )
                distmap[i][valid_pixels] = dist[0]
            hair_silh_loss = torch.abs((gs_render_mask - hair_mask) * distmap).mean()

            # Reg
            solid_hair_loss = ((1 - hair_silhoutte) * erode_hair_mask).mean()
        loss_hair = (
            self.get_lambda("rgb.hair") * hair_loss
            + self.get_lambda("ssim") * ssim_loss
            + self.get_lambda("silh.hair") * hair_silh_loss
            + self.get_lambda("silh.solid_hair") * solid_hair_loss
        )
        if render_hair is not None:
            for k, v in self.buffer["gs_reg_loss"].items():
                loss_hair += self.get_lambda(k) * v
            loss_aiap = (
                self.get_lambda("aiap.xyz") * self.buffer["gs_aiap_xyz_loss"]
                + self.get_lambda("aiap.cov") * self.buffer["gs_aiap_cov_loss"]
            )

        loss_dict = {
            "loss_pho/rgb.hair": hair_loss,
            "loss_geo/silh.hair": hair_silh_loss,
            "loss_pho/ssim.hair": ssim_loss,
            "loss_reg/silh.solid_hair": solid_hair_loss,
        }
        loss = loss_hair + loss_aiap

        # save some variables for visualization
        self.buffer["gt_hair"] = gt_hair

        return loss, loss_dict

    def visualize(self, savedir, outputs, step):
        white_img = np.ones((self.img_h, self.img_w, 3)).astype(np.float32)

        # raster hair mask
        savepath = os.path.join(savedir, "hairmask_it{}.png".format(step))
        raster_hairmask = (
            white_img
            if outputs["raster_hairmask"] is None
            else outputs["raster_hairmask"][0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        )
        cv2.imwrite(savepath, raster_hairmask * 255)

        # gt hair
        savepath = os.path.join(savedir, "hairgt_it{}.png".format(step))
        gt_hair = self.buffer["gt_hair"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_hair * 255)

        # render hair
        savepath = os.path.join(savedir, "hair_it{}.png".format(step))
        render_hair = white_img if outputs["render_hair"] is None else outputs["render_hair"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_hair * 255)

        return {"raster_hairmask": raster_hairmask, "gt_hair": gt_hair, "render_hair": render_hair}

    def state_dict(self):
        state_dict = {}
        for k, m in self.models.items():
            model_dict = m.state_dict()
            if k == "canonical_gs":
                use_flags = m.get_use_flags
                for key, v in model_dict.items():
                    model_dict[key] = v[use_flags]
            state_dict[k] = model_dict

        return state_dict
