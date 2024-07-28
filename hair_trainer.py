import os
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, erosion

from dataset.cameras import Camera
from networks.gshair.hairwrapper import GSHairWrapper
from networks.meshface.facewrapper import MeshFaceWrapper
from utils import (
    AverageMeter,
    CUDA_Timer,
    color_mask,
    directory,
    restore_model,
    ssim,
    update_lambda,
    visimg,
    visPositionMap,
    write_obj,
)


class HairTrainer:
    def __init__(self, config, logger, spatial_lr_scale, all_flame_params=None, is_val=False):
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        self.config = config
        self.neural = config.get("training.neural_texture", True)
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0
        self.rate = min(self.rate_h, self.rate_w)
        self.nan_detect = False
        self.is_val = is_val
        self.spatial_lr_scale = spatial_lr_scale
        self.lr = config["training.learning_rate"]
        self.stages = config["training.stages"]
        config["training.stages_epoch"] = (
            [] if None in config["training.stages_epoch"] else config["training.stages_epoch"]
        )
        self.stages_epoch = [0] + config["training.stages_epoch"] + [1e10]
        assert (
            len(self.stages_epoch) - len(self.stages)
        ) >= -1, "The length of 'training.stages_epoch' should be larger than the length of 'training.stages' - 1."
        self.xyz_cond = config.get("flame.xyz_cond", False)
        self.move_eyes = config.get("flame.move_eyes", False)

        self.parameters_to_train = []
        self._init_nets()

        # set optimizer
        self.optimizer = torch.optim.Adam(self.parameters_to_train, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config["training.step"], gamma=0.1
        )
        if all_flame_params is not None:
            self.init_all_flame_params(all_flame_params)

        # Restore checkpoint
        checkpoint_path = (
            os.path.join(config["local_workspace"], "checkpoint_latest.pth")
            if config["training.pretrained_checkpoint_path"] is None
            else config["training.pretrained_checkpoint_path"]
        )

        self.current_epoch = 1
        self.global_step = 0
        self.stage = self.stages[0]
        self.stage_step = 0
        if os.path.exists(checkpoint_path):
            self.current_epoch, self.global_step, stage, stage_step = restore_model(
                checkpoint_path, self.hairwrapper, self.facewrapper, self.optimizer, logger
            )

            # load optimized flame params
            dir_name = os.path.dirname(checkpoint_path)
            opt_flame_params = np.load(os.path.join(dir_name, "flame_params.npz"))
            self.load_all_flame_params(opt_flame_params)

            if stage is not None:
                self.stage = stage
                self.stage_step = stage_step

        self.logger = logger
        self.tb_writer = config.get("tb_writer", None)

        self._init_data()
        self._init_losses()
        self._set_stage(self.stage)

        # find all lambda
        self.all_lambdas = {}
        prelen = len("training.lambda_")
        for k, v in self.config.items():
            if "lambda" not in k or "lambda_update_list" in k:
                continue
            self.all_lambdas[k[prelen:]] = v

    def _freeze(self, label):
        for group in self.optimizer.param_groups:
            if label == "all" or label in group["name"]:
                group["params"][0].requires_grad = False

    def _unfreeze(self, label):
        for group in self.optimizer.param_groups:
            if label == "all" or label in group["name"]:
                group["params"][0].requires_grad = True

    def _set_stage(self, stage):
        if stage == "joint":
            self._freeze("all")
            self._unfreeze("gs")  # train canonical hair
        elif stage == "head":
            # learn facial mesh
            self._freeze("all")
            self._unfreeze("head")
        else:
            self.logger.info("Unknown training stage: {}".format(stage))
            exit(1)

    def _init_nets(self):
        init_pts = np.load(self.config["gs.init_pts"])
        self.hairwrapper = GSHairWrapper(self.config, init_pts, self.spatial_lr_scale)
        self.facewrapper = MeshFaceWrapper(self.config, self.move_eyes, self.xyz_cond)

        self.parameters_to_train = self.hairwrapper.get_optim_params() + self.facewrapper.get_optim_params()

    def _init_data(self):
        B, H, W = (
            self.config["data.per_gpu_batch_size"],
            self.config["data.img_h"],
            self.config["data.img_w"],
        )

        self.img = torch.zeros((B, H, W, 3), dtype=torch.float32).cuda()
        self.view = torch.zeros((B, 3, 8, 8), dtype=torch.float32).cuda()

        self.mask = {}
        self.mask["full"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["hair"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["head"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["erode_hair"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()

        self.depth_map = torch.zeros((B, H, W), dtype=torch.float32).cuda()

    def _init_losses(self):
        # TODO: check unuseful losses
        self.train_losses = {
            "loss": AverageMeter("train_loss"),
            "loss_pho/rgb.obj": AverageMeter("train_rgb_obj_loss"),
            "loss_pho/rgb.hair": AverageMeter("train_rgb_hair_loss"),
            "loss_pho/rgb.head": AverageMeter("train_rgb_head_loss"),
            "loss_pho/rgb.basic_head": AverageMeter("train_rgb_basic_head_loss"),
            "loss_geo/silh.hair": AverageMeter("train_silh_hair_loss"),
            "loss_geo/depth.head": AverageMeter("train_depth_head_loss"),
            "loss_geo/normal.head": AverageMeter("train_normal_head_loss"),
            "loss_pho/ssim.obj": AverageMeter("train_ssim_obj_loss"),
            "loss_pho/ssim.hair": AverageMeter("train_ssim_hair_loss"),
            "loss_pho/ssim.head": AverageMeter("train_ssim_head_loss"),
            "loss_reg/mesh.laplacian": AverageMeter("train_mesh_laplacian_loss"),
            "loss_reg/mesh.normal": AverageMeter("train_mesh_normal_loss"),
            "loss_reg/mesh.edges": AverageMeter("train_mesh_edges_loss"),
            "loss_reg/mesh.vscale": AverageMeter("train_mesh_vscale_loss"),
            "loss_reg/silh.solid_hair": AverageMeter("train_silh_solid_hair_loss"),
        }
        self.val_losses = {
            "loss": AverageMeter("val_loss"),
            "metrics/mse": AverageMeter("metrics.mse"),
            "metrics/psnr": AverageMeter("metrics.psnr"),
            "loss_pho/rgb.obj": AverageMeter("val_rgb_obj_loss"),
            "loss_pho/rgb.hair": AverageMeter("val_rgb_hair_loss"),
            "loss_pho/rgb.head": AverageMeter("val_rgb_head_loss"),
            "loss_pho/rgb.basic_head": AverageMeter("val_rgb_basic_head_loss"),
            "loss_geo/silh.hair": AverageMeter("val_silh_hair_loss"),
            "loss_geo/depth.head": AverageMeter("val_depth_head_loss"),
            "loss_geo/normal.head": AverageMeter("val_normal_head_loss"),
            "loss_pho/ssim.obj": AverageMeter("val_ssim_obj_loss"),
            "loss_pho/ssim.hair": AverageMeter("val_ssim_hair_loss"),
            "loss_pho/ssim.head": AverageMeter("val_ssim_head_loss"),
            "loss_reg/mesh.laplacian": AverageMeter("val_mesh_laplacian_loss"),
            "loss_reg/mesh.normal": AverageMeter("val_mesh_normal_loss"),
            "loss_reg/mesh.edges": AverageMeter("val_mesh_edges_loss"),
            "loss_reg/mesh.vscale": AverageMeter("val_mesh_vscale_loss"),
            "loss_reg/silh.solid_hair": AverageMeter("val_silh_solid_hair_loss"),
        }

    def set_train(self):
        """Convert models to training mode"""
        self.hairwrapper.set_train()
        self.facewrapper.set_train()

    def set_eval(self):
        """Convert models to evaluation mode"""
        self.hairwrapper.set_eval()
        self.facewrapper.set_eval()

    def update_stage(self):
        try:
            cur_id = next(i for i, v in enumerate(self.stages_epoch) if v > self.current_epoch)
        except:
            cur_id = -1
        new_stage = self.stages[cur_id - 1]
        if new_stage != self.stage:
            # save the final ckpt of the current stage
            savepath = os.path.join(
                self.config["local_workspace"], "checkpoint_{}_it{}.pth".format(self.stage, self.global_step)
            )
            self.save_ckpt(savepath)

            self.stage = new_stage
            self.stage_step = 0
            self._set_stage(new_stage)

    def train(self, train_loader, val_loader, show_time=False):
        torch.cuda.empty_cache()
        while self.current_epoch <= self.config["training.epochs"]:
            self.update_stage()
            success = self.train_epoch(train_loader, val_loader, show_time)
            if not success:
                return

            self.scheduler.step()
            self.logger.info("Epoch finished, average losses: ")
            for v in self.train_losses.values():
                self.logger.info("    {}".format(v))
            self.current_epoch += 1

    def set_data(self, items):
        self.batch_size = items["view"].shape[0]

        self.view.resize_as_(items["view"]).copy_(items["view"])
        self.img.resize_as_(items["img"]).copy_(items["img"])

        self.mask["full"].resize_as_(items["obj_mask"]).copy_(items["obj_mask"])
        self.mask["hair"].resize_as_(items["hair_mask"]).copy_(items["hair_mask"])
        self.mask["head"].resize_as_(items["head_mask"]).copy_(items["head_mask"])
        self.mask["erode_hair"].resize_as_(items["erode_hair_mask"]).copy_(items["erode_hair_mask"])

        self.depth_map.resize_as_(items["depth_map"]).copy_(items["depth_map"])

        # driving flame params
        self.flame_params = {}
        frame_idx = items["frame_idx"]
        for key, val in self.all_flame_params.items():
            if key == "shape":
                self.flame_params[key] = val.expand(self.batch_size, -1)
            else:
                self.flame_params[key] = val[frame_idx]

        # build cameras
        self.camera = []
        for i in range(self.batch_size):
            camera = Camera(
                R=items["w2c"][i, :3, :3],
                t=items["w2c"][i, :3, 3],
                intr=items["intr"][i],
                zfar=100.0,
                znear=0.01,
                img_h=self.img_h,
                img_w=self.img_w,
                name=items["cam"][i],
            )
            self.camera.append(camera)

        self.name = items["name"]

    def init_all_flame_params(self, flame_params):
        # learnable flame params
        T = max(list(flame_params.keys())) + 1
        m_id = min(list(flame_params.keys()))
        self.all_flame_params = {
            "shape": torch.from_numpy(flame_params[m_id]["shape"])[None],
            "expr": torch.zeros([T, flame_params[m_id]["expr"].shape[1]]),
            "rotation": torch.zeros([T, 3]),
            "neck_pose": torch.zeros([T, 3]),
            "jaw_pose": torch.zeros([T, 3]),
            "eyes_pose": torch.zeros([T, 6]),
            "translation": torch.zeros([T, 3]),
        }

        for i, param in flame_params.items():
            self.all_flame_params["expr"][i] = torch.from_numpy(param["expr"])
            self.all_flame_params["rotation"][i] = torch.from_numpy(param["rotation"])
            self.all_flame_params["neck_pose"][i] = torch.from_numpy(param["neck_pose"])
            self.all_flame_params["jaw_pose"][i] = torch.from_numpy(param["jaw_pose"])
            self.all_flame_params["eyes_pose"][i] = torch.from_numpy(param["eyes_pose"])
            self.all_flame_params["translation"][i] = torch.from_numpy(param["translation"])

        for k, v in self.all_flame_params.items():
            self.all_flame_params[k] = v.float().cuda()

        optimize_params = self.config.get("flame.optimize_params", False)
        if (not self.is_val) and optimize_params:
            flame_lrs = {"shape": 1e-5, "expr": 1e-3, "pose": 1e-5, "translation": 1e-6}

            # shape
            self.all_flame_params["shape"].requires_grad = True
            param_shape = {
                "params": [self.all_flame_params["shape"]],
                "lr": flame_lrs["shape"],
                "name": "head.flame_shape",
            }
            self.optimizer.add_param_group(param_shape)

            # expression
            self.all_flame_params["expr"].requires_grad = True
            param_expr = {"params": [self.all_flame_params["expr"]], "lr": flame_lrs["expr"], "name": "head.flame_expr"}
            self.optimizer.add_param_group(param_expr)

            # pose
            self.all_flame_params["rotation"].requires_grad = True
            self.all_flame_params["neck_pose"].requires_grad = True
            self.all_flame_params["jaw_pose"].requires_grad = True
            self.all_flame_params["eyes_pose"].requires_grad = True
            params = [
                self.all_flame_params["rotation"],
                self.all_flame_params["neck_pose"],
                self.all_flame_params["jaw_pose"],
                self.all_flame_params["eyes_pose"],
            ]
            param_pose = {"params": params, "lr": flame_lrs["pose"], "name": "head.flame_pose"}
            self.optimizer.add_param_group(param_pose)

            # translation
            self.all_flame_params["translation"].requires_grad = True
            param_trans = {
                "params": [self.all_flame_params["translation"]],
                "lr": flame_lrs["translation"],
                "name": "head.flame_trans",
            }
            self.optimizer.add_param_group(param_trans)

    def load_all_flame_params(self, all_flame_params):
        self.all_flame_params = {k: torch.from_numpy(v).float().cuda() for k, v in all_flame_params.items()}

    def load_flame_params(self, flame_params):
        """Used to driving from another flame params"""
        self.flame_params = flame_params
        for key, val in flame_params.items():
            if isinstance(val, np.ndarray):
                self.flame_params[key] = torch.from_numpy(val).float().cuda()
            elif isinstance(val, torch.Tensor):
                self.flame_params[key] = val.float().cuda()

    def load_cameras(self, T_c2w, items):
        """Used to render free views"""
        for key, val in items.items():
            if isinstance(val, np.ndarray):
                items[key] = torch.from_numpy(val).cuda()

        self.camera = []
        for i in range(T_c2w.shape[0]):
            w2c = torch.from_numpy(np.linalg.inv(T_c2w[i])).float().cuda()
            camera = Camera(
                R=w2c[:3, :3],
                t=w2c[:3, 3],
                intr=items["intr"][i],
                zfar=100.0,
                znear=0.01,
                img_h=self.img_h,
                img_w=self.img_w,
                name=items["cam"][i],
            )
            self.camera.append(camera)

        views = []
        for c2w in T_c2w:
            campos = c2w[:3, 3]
            view = campos / np.linalg.norm(campos)
            views.append(np.tile(view, (8, 8, 1)).transpose((2, 0, 1)))
        self.view = torch.from_numpy(np.stack(views, axis=0)).float().cuda()

    def compare_depth(self, hair_depth, head_depth):
        hair_nz, head_nz = (
            torch.ones_like(hair_depth).cuda() * 1e10,
            torch.ones_like(head_depth).cuda() * 1e10,
        )
        valid_hair, valid_head = hair_depth > 0, head_depth > 0
        hair_nz[valid_hair] = hair_depth[valid_hair]
        head_nz[valid_head] = head_depth[valid_head]
        hair_mask = (head_nz > hair_nz).int()

        return hair_mask

    def fuse(self, rasterized_hair, rasterized_face, is_val=False):
        """To fuse hair image and head image"""
        # ablation options
        gsdepth = self.config.get("ab.gsdepth", False)
        hardblend = self.config.get("ab.hardblend", False)

        has_face = rasterized_face is not None
        has_hair = rasterized_hair is not None
        need_fusion = has_hair and has_face
        assert has_face or has_hair, "Please render face, or hair, or both of them. "

        outputs = {}
        # 1. Hair processing, [B, H, W, 3]
        if has_hair:
            rendered_hair = rasterized_hair["render"].permute((0, 2, 3, 1))

        # 2. Head processing
        if has_face:
            valid_pixels = rasterized_face["valid_nograd"]
            s_id = 3 if self.xyz_cond else 0
            rasterized_xyz, rasterized_uv = (
                rasterized_face["neural_img"][..., :s_id],
                rasterized_face["neural_img"][..., s_id : s_id + 2],
            )  # [N, 3], [N, 2], [N, tex_ch], N is the num of valid pixels.
            tex_ch = self.config["training.tex_ch"] if self.neural else 3
            neural_features = rasterized_face["neural_features"]
            rasterized_features = F.grid_sample(neural_features, rasterized_uv, mode="bilinear", align_corners=True)

            valid_xyz = rasterized_xyz[valid_pixels]
            valid_uv = rasterized_uv[valid_pixels]
            valid_features = rasterized_features.permute((0, 2, 3, 1))[valid_pixels]
            valid_tex = valid_features[..., :tex_ch]
            valid_basic_tex = valid_features[..., tex_ch:]

            # uv_pe
            if self.neural:
                rendered_face = self.facewrapper.feats2rgbs(valid_xyz, valid_uv, valid_tex, valid_pixels)
                rendered_basic_face = self.facewrapper.feats2rgbs(valid_xyz, valid_uv, valid_basic_tex, valid_pixels)
            else:
                B, H, W = valid_pixels.shape
                rendered_face = torch.ones((B, H, W, 3)).float().cuda()
                rendered_face[valid_pixels] = valid_tex

                rendered_basic_face = torch.ones((B, H, W, 3)).float().cuda()
                rendered_basic_face[valid_pixels] = valid_basic_tex

        # 3. Compute fusing mask & fuse
        if need_fusion:
            hair_depth = rasterized_hair["near_z"] if not gsdepth else rasterized_hair["depth"]
            head_depth = rasterized_face["depth"]

            hair_mask = self.compare_depth(hair_depth, head_depth)

            # TRY: different ways to use hair mask
            if is_val:
                processed_hair_mask = erosion(hair_mask.unsqueeze(1), torch.ones(3, 3).cuda())
                processed_hair_mask = dilation(processed_hair_mask, torch.ones(3, 3).cuda())
                processed_hair_mask = dilation(processed_hair_mask, torch.ones(5, 5).cuda())
                processed_hair_mask = erosion(processed_hair_mask, torch.ones(5, 5).cuda())
                processed_hair_mask = processed_hair_mask[:, 0]
            else:
                processed_hair_mask = hair_mask

            if hardblend:
                outputs["raster_hairmask"] = processed_hair_mask
            else:
                outputs["raster_hairmask"] = rasterized_hair["silhoutte"] * processed_hair_mask

            outputs["raster_headmask"] = torch.clip(
                rasterized_face["valid_nograd"].float() - outputs["raster_hairmask"], min=0.0, max=1.0
            )
            outputs["fullmask"] = outputs["raster_headmask"] + outputs["raster_hairmask"]

            head_part = outputs["raster_headmask"][..., None].expand(-1, -1, -1, 3)
            hair_part = outputs["raster_hairmask"][..., None].expand(-1, -1, -1, 3)
            render_fuse = head_part * rendered_face + hair_part * rendered_hair

            bg = torch.ones_like(render_fuse).float().cuda()
            render_fuse = (1 - outputs["fullmask"])[..., None] * bg + render_fuse
        else:
            render_fuse = rendered_face if has_face else rendered_hair

        # 4. Output results
        outputs["render_hair"] = rendered_hair if has_hair else None
        outputs["render_face"] = rendered_face if has_face else None
        outputs["render_basic_face"] = rendered_basic_face if has_face else None
        outputs["render_fuse"] = render_fuse
        outputs["hair_depth"] = rasterized_hair["depth"] if has_hair else None
        outputs["head_depth"] = rasterized_face["depth"] if has_face else None
        outputs["hair_silhoutte"] = rasterized_hair["silhoutte"] if has_hair else None
        outputs["occlussion_mask"] = hair_mask if need_fusion else None  # no gradients
        outputs["head_geomap"] = rasterized_face["rgba"][..., :3] if has_face else None

        outputs["raster_hairmask"] = None if "raster_hairmask" not in outputs else outputs["raster_hairmask"]
        outputs["raster_headmask"] = None if "raster_headmask" not in outputs else outputs["raster_headmask"]
        outputs["fullmask"] = None if "fullmask" not in outputs else outputs["fullmask"]

        #   DEBUG
        # cv2.imwrite('test_rasthair.png', outputs['raster_hairmask'][0, ..., None].detach().cpu().numpy() * 255)
        # cv2.imwrite('test_rasthead.png', outputs['raster_headmask'][0, ..., None].detach().cpu().numpy() * 255)

        return outputs

    def network_forward(self, is_val=False):
        rasterized_hair, rasterized_face = None, None
        bg_color = [1.0, 1.0, 1.0]  # white

        rasterized_face, _ = self.facewrapper.render(self.camera, self.flame_params, self.view, bg_color)
        if self.stage == "joint":
            rasterized_hair = self.hairwrapper.render(self.camera, bg_color)

        outputs = self.fuse(rasterized_hair, rasterized_face, is_val=is_val)

        return outputs

    def update_x(self, lambda_name):
        return update_lambda(
            self.config["training.lambda_{}".format(lambda_name)],
            self.config["training.lambda_{}.slope".format(lambda_name)],
            self.config["training.lambda_{}.end".format(lambda_name)],
            self.global_step,
            self.config["training.lambda_{}.interval".format(lambda_name)],
        )

    def update_lambda(self):
        update_names = self.config["training.lambda_update_list"]
        for k, _ in self.all_lambdas.items():
            if k in update_names:
                self.all_lambdas[k] = self.update_x(k)

    def get_lambda(self, key):
        return self.all_lambdas.get(key, 0.0)

    def compute_loss(self, outputs):
        render_rgb = outputs["render_fuse"]

        # update hyper-parameters
        self.update_lambda()

        # RGB Loss
        hair_mask = self.mask["hair"]
        erode_hair_mask = self.mask["erode_hair"]
        head_mask = self.mask["head"]

        gt_hair = self.img.clone()
        gt_head = self.img.clone()
        gt_hair[(1 - hair_mask).bool()] = 1.0
        gt_head[hair_mask.bool()] = 1.0

        # L2 Loss
        rgb_loss = torch.linalg.norm((render_rgb - self.img), dim=-1).mean()

        # SSIM Loss
        if self.get_lambda("ssim") > 0:
            ssim_loss = 1.0 - ssim(render_rgb.permute((0, 3, 1, 2)), self.img.permute((0, 3, 1, 2)))
        else:
            ssim_loss = torch.tensor(0.0).cuda()

        loss_head, loss_head_dict = self.facewrapper.compute_losses(
            outputs, self.img, gt_head, self.depth_map, hair_mask, head_mask, self.global_step
        )
        loss_hair, loss_hair_dict = self.hairwrapper.compute_losses(
            outputs, gt_hair, hair_mask, erode_hair_mask, self.global_step
        )
        loss_joint = self.get_lambda("rgb") * rgb_loss + self.get_lambda("ssim") * ssim_loss

        loss_all = {"head": loss_head, "hair": loss_hair, "joint": loss_joint}

        loss = 0.0
        for name in self.config["training.{}_stage_loss".format(self.stage)]:
            loss += loss_all[name]

        outputs["gt_hair"] = gt_hair
        outputs["gt_head"] = gt_head
        loss_dict = {"loss": loss, "loss_pho/rgb.obj": rgb_loss, "loss_pho/ssim.obj": ssim_loss}

        # Update with head & hair loss
        loss_dict.update(loss_head_dict)
        loss_dict.update(loss_hair_dict)

        return loss_dict

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb_obj = loss_dict["loss_pho/rgb.obj"]
        loss_rgb_hair = loss_dict["loss_pho/rgb.hair"]
        loss_rgb_head = loss_dict["loss_pho/rgb.head"]
        loss_rgb_basic_head = loss_dict["loss_pho/rgb.basic_head"]
        loss_silh_hair = loss_dict["loss_geo/silh.hair"]
        loss_depth_head = loss_dict["loss_geo/depth.head"]
        loss_normal_head = loss_dict["loss_geo/normal.head"]
        loss_ssim_obj = loss_dict["loss_pho/ssim.obj"]
        loss_ssim_hair = loss_dict["loss_pho/ssim.hair"]
        loss_ssim_head = loss_dict["loss_pho/ssim.head"]
        loss_mesh_laplacian = loss_dict["loss_reg/mesh.laplacian"]
        loss_mesh_normal = loss_dict["loss_reg/mesh.normal"]
        loss_mesh_edges = loss_dict["loss_reg/mesh.edges"]
        loss_mesh_vscale = loss_dict["loss_reg/mesh.vscale"]
        loss_silh_solid_hair = loss_dict["loss_reg/silh.solid_hair"]

        lr = self.scheduler.get_last_lr()[0]
        self.logger.info(
            "stage [%s] epoch [%.3d] step [%d/%d] global_step = %d loss = %.4f lr = %.6f\n"
            "        rgb = %.4f                     w: %.4f\n"
            "           hair = %.4f                 w: %.4f\n"
            "           head = %.4f                 w: %.4f\n"
            "           basic_head = %.4f           w: %.4f\n"
            "        silh:                                 \n"
            "           hair = %.4f                 w: %.4f\n"
            "        depth:                                \n"
            "           head = %.4f                 w: %.4f\n"
            "        normal:                               \n"
            "           head = %.4f                 w: %.4f\n"
            "        ssim = %.4f                    w: %.4f\n"
            "           hair = %.4f                 w: %.4f\n"
            "           head = %.4f                 w: %.4f\n"
            "        reg:                                  \n"
            "           mesh_laplacian = %.4f       w: %.4f\n"
            "           mesh_normal = %.4f          w: %.4f\n"
            "           mesh_edges = %.4f           w: %.4f\n"
            "           mesh_vscale = %.4f          w: %.4f\n"
            "           silh_binary = %.4f          w: %.4f\n"
            % (
                self.stage,
                epoch,
                step,
                dataset_length,
                self.global_step,
                loss.item(),
                lr,
                loss_rgb_obj.item(),
                self.get_lambda("rgb"),
                loss_rgb_hair.item(),
                self.get_lambda("rgb.hair"),
                loss_rgb_head.item(),
                self.get_lambda("rgb.head"),
                loss_rgb_basic_head.item(),
                self.get_lambda("rgb.head"),
                loss_silh_hair.item(),
                self.get_lambda("silh.hair"),
                loss_depth_head.item(),
                self.get_lambda("depth.head"),
                loss_normal_head.item(),
                self.get_lambda("normal.head"),
                loss_ssim_obj.item(),
                self.get_lambda("ssim"),
                loss_ssim_hair.item(),
                self.get_lambda("ssim"),
                loss_ssim_head.item(),
                self.get_lambda("ssim"),
                loss_mesh_laplacian.item(),
                self.get_lambda("mesh.laplacian"),
                loss_mesh_normal.item(),
                self.get_lambda("mesh.normal"),
                loss_mesh_edges.item(),
                self.get_lambda("mesh.edges"),
                loss_mesh_vscale.item(),
                self.get_lambda("mesh.verts_scale"),
                loss_silh_solid_hair.item(),
                self.get_lambda("silh.solid_hair"),
            )
        )

        # Write losses to tensorboard
        # Update avg meters
        for key, value in self.train_losses.items():
            if self.tb_writer:
                self.tb_writer.add_scalar(key, loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def run_eval(self, val_loader):
        self.logger.info("Start running evaluation on validation set:")
        self.set_eval()

        # clear train losses average meter
        for val_loss_item in self.val_losses.values():
            val_loss_item.reset()

        batch_count = 0
        with torch.no_grad():
            for step, items in enumerate(val_loader):
                batch_count += 1
                if batch_count % 20 == 0:
                    self.logger.info("    Eval progress: {}/{}".format(batch_count, len(val_loader)))

                self.set_data(items)
                outputs = self.network_forward(is_val=True)
                loss_dict = self.compute_loss(outputs)

                mse, psnr = self.compute_metrics(outputs)

                loss_dict["metrics/mse"] = mse
                loss_dict["metrics/psnr"] = psnr

                self.log_val(step, loss_dict)

            # log evaluation result
            self.logger.info("Evaluation finished, average losses: ")
            for v in self.val_losses.values():
                self.logger.info("    {}".format(v))

            # Write val losses to tensorboard
            if self.tb_writer:
                for key, value in self.val_losses.items():
                    self.tb_writer.add_scalar(key + "//val", value.avg, self.global_step)

        self.set_train()

    def log_val(self, step, loss_dict):
        B = self.batch_size
        # loss logging
        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B)

    def compute_metrics(self, outputs):
        if outputs["fullmask"] is None:
            return np.array(0.0), np.array(0.0)

        valid_mask = outputs["fullmask"] * self.mask["full"]

        gt_img = (self.img[0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        pred_img = (outputs["render_fuse"][0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        mse = ((pred_img - gt_img) ** 2).mean()
        psnr = 10 * np.log10(65025 / mse)

        return mse, psnr

    def visualization(self, outputs, step, label="log"):
        # create dirs
        logdir = os.path.join(self.config["local_workspace"], label)
        directory(logdir)
        if label == "log":
            savedir = os.path.join(logdir, "it{}".format(step))
            directory(savedir)
        elif label == "eval":
            savedir = os.path.join(logdir, self.name[0])
            directory(savedir)

        valid_mask = self.mask["full"]  # foreground mask
        hair_mask = self.mask["hair"]
        face_mask = self.mask["head"]
        valid_mask, face_mask, hair_mask = valid_mask.bool(), face_mask.bool(), hair_mask.bool()

        # gt_img
        savepath = os.path.join(savedir, "gt_it{}.png".format(step))
        gt_img = self.img[0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_img * 255)

        # image
        savepath = os.path.join(savedir, "rendering_it{}.png".format(step))
        render_fuse = outputs["render_fuse"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_fuse * 255)

        face_visuals = self.facewrapper.visualize(savedir, outputs, step)
        hair_visuals = self.hairwrapper.visualize(savedir, outputs, step)
        gt_head, gt_head_normal, render_head, raster_headmask, head_normal, head_geomap, colored_mask = (
            face_visuals["gt_head"],
            face_visuals["gt_head_normal"],
            face_visuals["render_head"],
            face_visuals["raster_headmask"],
            face_visuals["head_normal"],
            face_visuals["head_geomap"],
            face_visuals["colored_mask"],
        )
        gt_hair, raster_hairmask, render_hair = (
            hair_visuals["gt_hair"],
            hair_visuals["raster_hairmask"],
            hair_visuals["render_hair"],
        )

        # concat img
        white_img = np.ones((self.img_h, self.img_w, 3))

        savepath = os.path.join(savedir, "combined_it{}.png".format(step))
        gt = np.concatenate([gt_img, gt_head, gt_head_normal[..., ::-1], gt_hair], axis=1)

        alpha = 0.4
        color = [1.0, 0.0, 0.0]
        head_withmask = cv2.addWeighted(render_head, 1.0, color_mask(raster_hairmask, color=color), alpha, 0)
        hair_withmask = cv2.addWeighted(render_hair, 1.0, color_mask(raster_headmask, color=color), alpha, 0)
        pred = np.concatenate([render_fuse, head_withmask, head_normal[..., ::-1], hair_withmask], axis=1)

        alpha = 0.5
        color_pred = [0.0, 0.0, 0.8]
        color_gt = [0.0, 0.6, 0.3]
        headmask_gt = face_mask[0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        hairmask_gt = hair_mask[0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        headmask_withgt = cv2.addWeighted(
            color_mask(raster_headmask, color=color_pred, bg_white=True),
            1.0,
            color_mask(headmask_gt, color=color_gt),
            alpha,
            0,
        )
        hairmask_withgt = cv2.addWeighted(
            color_mask(raster_hairmask, color=color_pred, bg_white=True),
            1.0,
            color_mask(hairmask_gt, color=color_gt),
            alpha,
            0,
        )
        mask = np.concatenate([head_geomap, headmask_withgt, colored_mask, hairmask_withgt], axis=1)
        combined_img = np.concatenate([gt, pred, mask], axis=0)
        cv2.imwrite(savepath, combined_img * 255)

        self.facewrapper.visualize_textures(savedir, step)
        # fuse mask
        if outputs["fullmask"] is not None:
            savepath = os.path.join(savedir, "fullmask_it{}.png".format(step))
            fullmask = outputs["fullmask"] * self.mask["full"]
            cv2.imwrite(savepath, fullmask[0].detach().cpu().numpy() * 255)

        # metrics
        if label == "eval":
            savepath = os.path.join(savedir, "metrics.txt")
            mse, psnr = self.compute_metrics(outputs)

            with open(savepath, "w") as f:
                f.write("MSE: {}\n".format(mse))
                f.write("PSNR: {}\n".format(psnr))

            print("MSE: {}\nPSNR: {}\n".format(mse, psnr))

    def clip_grad(self, max_norm=0.01):
        for dict in self.parameters_to_train:
            torch.nn.utils.clip_grad_norm_(dict["params"], max_norm)

    def save_ckpt(self, savepath, stage=None):
        save_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "stage": self.stage if stage is None else stage,
            "stage_step": self.stage_step,
        }
        save_dict.update(self.facewrapper.state_dict())
        save_dict.update(self.hairwrapper.state_dict())

        torch.save(save_dict, savepath)

        basedir = os.path.dirname(savepath)
        npz_path = os.path.join(basedir, "flame_params.npz")
        if os.path.exists(npz_path):
            os.remove(npz_path)

        flame_params = {k: v.detach().cpu().numpy() for k, v in self.all_flame_params.items()}
        np.savez(str(npz_path), **flame_params)

    def nan_check(self, loss):
        # DEBUG: save the checkpoint before NaN Loss
        if not self.nan_detect and loss.isnan().any():
            self.nan_detect = True
            checkpoint_path = os.path.join(self.config["local_workspace"], "nan_break.pth")
            self.save_ckpt(checkpoint_path)
            print("NaN break checkpoint has saved at {}".format(checkpoint_path))
            print("Data {}".format(self.name))
            return False

    def train_epoch(self, train_loader, val_loader, show_time=False):
        # convert models to traning mode
        self.set_train()

        warmup_steps = 10
        ld_timer = CUDA_Timer("load data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        sd_timer = CUDA_Timer("set data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        f_timer = CUDA_Timer("forward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        cl_timer = CUDA_Timer("compute loss", self.logger, valid=show_time, warmup_steps=warmup_steps)
        b_timer = CUDA_Timer("backward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        pp_timer = CUDA_Timer("post process", self.logger, valid=show_time, warmup_steps=warmup_steps)
        up_timer = CUDA_Timer("update params", self.logger, valid=show_time, warmup_steps=warmup_steps)
        ld_timer.start(0)

        for step, items in enumerate(train_loader):
            step += 1
            self.global_step += 1
            self.stage_step += 1

            if show_time and self.global_step > warmup_steps:
                ld_timer.end(step - 1)

            if self.stage in ["joint"]:
                self.hairwrapper.update_xyz_lr(self.stage_step, self.optimizer)

            # 1. Set data for trainer
            sd_timer.start(step)
            self.set_data(items)
            sd_timer.end(step)

            # 2. Run the network
            f_timer.start(step)
            outputs = self.network_forward()
            f_timer.end(step)

            # 3. Compute losses
            cl_timer.start(step)
            loss_dict = self.compute_loss(outputs)
            cl_timer.end(step)
            loss = loss_dict["loss"]

            self.nan_check(loss)

            # 4. Backprop
            b_timer.start(step)
            loss.backward()
            b_timer.end(step)

            # 5. Process Gaussians
            pp_timer.start(step)
            if self.stage in ["joint"] and self.stage_step % self.config["gs.upSH"] == 0:
                self.hairwrapper.oneupSHdegree()

            if (
                self.stage in ["joint"]
                and self.config["training.enable_densify"]
                and (self.stage_step < self.config["gs.densify_until_iter"])
            ):
                self.hairwrapper.update_use_flags(self.stage_step)

                do_densify = False
                do_reset = False
                for i in range(self.batch_size):
                    actual_step = (self.stage_step - 1) * self.batch_size + i + 1

                    self.hairwrapper.track_gsstates(i)

                    if (
                        actual_step > self.config["gs.densify_from_iter"]
                        and actual_step % self.config["gs.densification_interval"] == 0
                    ):
                        do_densify = True

                    if actual_step % (self.config["gs.opacity_reset_interval"]) == 0:
                        do_reset = True

                    if i == (self.batch_size - 1) and do_densify:
                        self.hairwrapper.densify_n_prune(self.optimizer, self.stage_step)

                    if i == (self.batch_size - 1) and do_reset and self.config["gs.enable_reset"]:
                        # Save model
                        checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_reset.pth")
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        self.save_ckpt(checkpoint_path)
                        self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

                        self.hairwrapper.reset_opacities(self.optimizer)
            pp_timer.end(step)

            up_timer.start(step)
            # self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            up_timer.end(step)

            # logging
            if step > 0 and (step % 10 == 0 or step == len(train_loader)):
                self.log_training(
                    self.current_epoch,
                    step,
                    self.global_step,
                    len(train_loader),
                    loss_dict,
                )

            # Visualize
            if self.stage_step == 1 or self.global_step % self.config["training.visual_interval"] == 0:
                self.visualization(outputs, self.global_step)

            if self.global_step % 2000 == 0:
                # Save model
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                self.save_ckpt(checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

            if self.global_step > 0 and self.global_step % self.config["training.eval_interval"] == 0:
                self.run_eval(val_loader)

                # Save model
                checkpoint_path = os.path.join(
                    self.config["local_workspace"],
                    "checkpoint_%012d.pth" % self.global_step,
                )
                self.save_ckpt(checkpoint_path)

            if show_time and self.global_step > warmup_steps:
                ld_timer.start(step)

            #   DEBUG: CUDA footprint
            # free, total = torch.cuda.mem_get_info()
            # self.logger.info('free and total mem: {}GB / {}GB'.format(free/1024/1024/1024, total/1024/1024/1024))

        # log time
        # if show_time:
        #     self.logger.info(
        #         "Running Time Profile:" + ld_timer + sd_timer + f_timer + cl_timer + b_timer + pp_timer + up_timer
        #     )

        return True


#
