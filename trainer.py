import os
from copy import deepcopy

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.morphology import dilation, erosion
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes

from dataset.cameras import Camera
from networks.flame2023.flame import FlameHead
from networks.gs.deformation import DeformMLP
from networks.gs.gaussian_model import GaussianModel
from networks.gs.gaussian_utils import build_rotation
from networks.mesh_renderer.mesh_renderer import NVDiffRenderer
from networks.neural2rgb.lpe import LPE, PE
from networks.neural2rgb.pixel_decoder import PixelDecoder
from networks.textures.detail_decoder import DynamicDecoder, ViewDecoder
from networks.textures.disp_decoder import DispDecoder
from networks.textures.neural_texture import NeuralTexture
from utils import (
    AverageMeter,
    CUDA_Timer,
    color_mask,
    depth_map2normals,
    directory,
    edge_loss,
    estimate_rigid,
    full_aiap_loss,
    img2mask,
    params_with_lr,
    render_list,
    restore_model,
    ssim,
    update_lambda,
    visimg,
    visPositionMap,
    write_obj,
)


class Trainer:
    def __init__(self, config, logger, spatial_lr_scale, painting=False):
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        self.attr_dims = {
            "xyz": 3,
            "scaling": 3,
            "rotation": 4,
            "opacity": 1,
            "features_dc": 3,
            "features_rest": ((config["gs.sh_degree"] + 1) ** 2 - 1) * 3,
        }

        self.config = config
        self.neural = config.get("training.neural_texture", True)
        self.tex_ch = config["training.tex_ch"]
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0
        self.rate = min(self.rate_h, self.rate_w)
        self.nan_detect = False
        self.spatial_lr_scale = spatial_lr_scale
        self.lr = config["training.learning_rate"]
        self.gs_pretrain = config["gs.pretrain"]
        self.alter_hair = False
        self.stages = config["training.stages"]
        config["training.stages_epoch"] = (
            [] if None in config["training.stages_epoch"] else config["training.stages_epoch"]
        )
        self.stages_epoch = [0] + config["training.stages_epoch"] + [1e10]
        assert (
            len(self.stages_epoch) - len(self.stages)
        ) >= -1, "The length of 'training.stages_epoch' should be larger than the length of 'training.stages' - 1."
        self.xyz_cond = config.get("flame.xyz_cond", True)
        self.move_eyes = config.get("flame.move_eyes", True)

        self.models = {}
        self.parameters_to_train = []

        self._init_nets(painting)

        # set optimizer
        self.optimizer = torch.optim.Adam(self.parameters_to_train, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config["training.step"], gamma=0.1
        )

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
                config, checkpoint_path, self.models, self.optimizer, logger
            )
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
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = False

    def _unfreeze(self, label):
        for group in self.optimizer.param_groups:
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = True

    def _set_stage(self, stage):
        if stage == "joint":
            self._freeze("head")

            if self.gs_pretrain is None:
                # train canonical hair
                self._unfreeze("gs")
            else:
                # load canonical hair
                state_dict = torch.load(self.gs_pretrain, map_location=lambda storage, loc: storage.cpu())
                _state_dict = {
                    k.replace("module.", "") if k.startswith("module.") else k: v
                    for k, v in state_dict["canonical_gs"].items()
                }
                self.models["canonical_gs"].load_state_dict(
                    _state_dict, self.optimizer, self.global_step, self.config["gs.upSH"]
                )
                # stop learning canonical hair
                # learn deformation field & head tex
                self._freeze("gs")
                self._unfreeze("hair")
                self._unfreeze("head_tex")
        elif stage == "head":
            # learn facial mesh
            self._freeze("hair")
            self._freeze("gs")
            self._unfreeze("head")
        elif stage == "painting":
            self._freeze("all")
            self._unfreeze("head_tex_basic")
            self._unfreeze("head_tex_mlp")
        elif stage == "painting_code":
            self._freeze("all")
            self._unfreeze("head_tex_basic")
        elif stage == "painting_mlp":
            self._freeze("all")
            self._unfreeze("head_tex_mlp")
        else:
            self.logger.info("Unknown training stage: {}".format(stage))
            exit(1)

    def _init_nets(self, painting=False):
        # Initialize canonical gaussians
        self.models["canonical_gs"] = GaussianModel(self.config["gs.sh_degree"])

        init_pts = np.load(self.config["gs.init_pts"])  # [N, 3]
        self.models["canonical_gs"].create_from_pts(init_pts, self.spatial_lr_scale)
        self.parameters_to_train += self.models["canonical_gs"].trainable_params(self.config)

        # Deformation MLPs
        self.models["deform_mlp"] = DeformMLP(self.config, self.attr_dims).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["deform_mlp"].named_parameters()), self.config["gs.deform_lr"], label="hair"
        )

        self.models["neural_texture"] = NeuralTexture(self.config).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["neural_texture"].named_parameters()), self.lr, label="head_tex_basic"
        )

        self.models["dynamic_texture"] = DynamicDecoder(self.config).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["dynamic_texture"].named_parameters()), self.lr, label="head_tex_dynamic"
        )

        self.models["view_texture"] = ViewDecoder(self.config).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["view_texture"].named_parameters()), self.lr, label="head_tex_view"
        )

        # FLAME Model
        self.flame_dec = FlameHead(self.config).cuda()

        # FLAME Disp Map
        self.models["disp_decoder"] = DispDecoder(self.config, eyes=self.move_eyes).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["disp_decoder"].named_parameters()), self.lr, label="head_geo"
        )

        # Head Rasterizer
        self.head_rasterizer = NVDiffRenderer(self.config)

        # Head Renderer
        pe_cfg = {
            "input_dims": 2,
            "num_freqs": self.config["training.pe.num_freqs"],
            "periodic_fns": [torch.sin, torch.cos],
            "log_sampling": self.config["training.pe.log_sampling"],
            "include_input": self.config["training.pe.include_input"],
        }
        pe_module = globals()[self.config["training.pe"]]
        self.models["pe"] = pe_module(pe_cfg).cuda()
        pe_dim = self.models["pe"].out_dim

        self.tex_ch = self.tex_ch + 3 + pe_dim
        self.models["head_mlp"] = PixelDecoder(cin=self.tex_ch + 1, pe_dim=pe_dim, xyz_cond=self.xyz_cond).cuda()

        self.parameters_to_train += params_with_lr(
            list(self.models["pe"].named_parameters()), self.lr, label="head_tex_pe"
        )

        scale = 0.01 if painting else 1.0  # less lr for pixel decoder
        self.parameters_to_train += params_with_lr(
            list(self.models["head_mlp"].named_parameters()), self.lr * scale, label="head_tex_mlp"
        )

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

        body_parts = self.flame_dec.get_body_parts()
        ex_sets = set()
        ex_sets = set.union(ex_sets, set(body_parts["left_eyeball"].tolist()))  # eye deformation does not use offsets.
        ex_sets = set.union(ex_sets, set(body_parts["right_eyeball"].tolist()))
        # ex_sets = set.union(ex_sets, set(self.flame_dec.vid_teeth.tolist()))
        for part in self.config["flame.offsets_ignore_parts"]:
            if part is None:
                continue
            ex_sets = set.union(ex_sets, set(body_parts[part].tolist()))

        self.ex_indices = np.array(list(ex_sets))
        self.leye_idcs = body_parts["left_eyeball"]
        self.reye_idcs = body_parts["right_eyeball"]

        mouth_set = set()
        for key in body_parts.keys():
            if "mouth" in key:
                mouth_set = set.union(mouth_set, set(body_parts[key].tolist()))
        self.mouth_idcs = list(mouth_set)
        indices = np.arange(len(self.flame_dec.v_template))
        self._offset_indices = np.delete(indices, self.ex_indices)  # delete eyeballs indices

        # canonical flame mesh, used to compute rigid transform
        canonical_flame_params = np.load(self.config["data.canonical_flame_path"])
        params = {}
        for k, v in canonical_flame_params.items():
            params[k] = torch.from_numpy(v).cuda()
        params["shape"] = params["shape"][None]
        canonical_head_verts, _, _ = self.flame_forward(params)
        self.canonical_scalp = canonical_head_verts[0, body_parts["scalp"]]
        self.scalp_idcs = body_parts["scalp"]
        self.shrink_scalp_idcs = body_parts["shrink_scalp"]

        # pixel coords
        px, py = np.meshgrid(np.arange(self.img_w), np.arange(self.img_h))
        self.pixelcoords = torch.from_numpy(np.stack((px, py), axis=-1)).float().cuda()

        # uv --> mesh vertices, barycentric
        # TODO: external settings
        self.uv2verts_ids = np.load("/home/exp/conallwang_works/face-data/H3Avatar/unwrap_uv_idx_v_idx.npy")
        self.uv2verts_bw = np.load("/home/exp/conallwang_works/face-data/H3Avatar/unwrap_uv_idx_bw.npy")

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
            "loss_pho/lpips": AverageMeter("train_lpips_loss"),
            "loss_pho/ssim": AverageMeter("train_ssim_loss"),
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
            "loss_geo/silh.hair": AverageMeter("val_silh_hair_loss"),
            "loss_geo/depth.head": AverageMeter("val_depth_head_loss"),
            "loss_geo/normal.head": AverageMeter("val_normal_head_loss"),
            "loss_pho/lpips": AverageMeter("val_lpips_loss"),
            "loss_pho/ssim": AverageMeter("val_ssim_loss"),
            "loss_reg/mesh.laplacian": AverageMeter("val_mesh_laplacian_loss"),
            "loss_reg/mesh.normal": AverageMeter("val_mesh_normal_loss"),
            "loss_reg/mesh.edges": AverageMeter("val_mesh_edges_loss"),
            "loss_reg/mesh.vscale": AverageMeter("val_mesh_vscale_loss"),
        }
        self.lpips = lpips.LPIPS(net="vgg").cuda()

    def set_train(self):
        """Convert models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert models to evaluation mode"""
        for m in self.models.values():
            m.eval()

    def load_hair(self, ckpt):
        self.logger.info("\n\nLoading hairstyle ...")

        state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage.cpu())
        for key in ["canonical_gs", "deform_mlp"]:
            model = self.models[key]
            _state_dict = {
                k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict[key].items()
            }
            missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
            missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

            if self.logger:
                self.logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
                self.logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

            try:
                if key == "canonical_gs":
                    model.load_state_dict(_state_dict, self.optimizer, self.global_step, self.config["gs.upSH"])
                else:
                    model.load_state_dict(_state_dict, strict=False)
            except Exception as error:
                if self.logger:
                    self.logger.info("[warning] {} weights are not loaded.".format(key))
                else:
                    print("[warning] {} weights are not loaded.".format(key))
        self.alter_hair = True

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
        self.proj_R = items["proj_w2c"][:, :3, :3].float().cuda()
        self.proj_t = items["proj_w2c"][:, :3, 3].float().cuda()
        self.intr = items["intr"].float().cuda()
        self.intr[:, 0, 0] *= -1

        self.name = items["name"]
        self.expand_dims()

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

    def init_all_flame_params(self, flame_params, is_val=False):
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
        if (not is_val) and optimize_params:
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

    def expand_dims(self):
        self.flame_uvcoords = (2 * self.flame_dec.verts_uvs - 1)[None, None].expand(self.batch_size, -1, -1, -1)
        self.flame_split_uvcoords = (2 * self.flame_dec.split_verts_uvs - 1)[None, None].expand(
            self.batch_size, -1, -1, -1
        )
        self.canonical_scalp_batch = self.canonical_scalp[None].expand(self.batch_size, -1, -1)

    def neural2rgb(self, neural_features, valid):
        """Decode neural image to RGB image

        Args:
            neural_features: Nx15, N is the num of valid pixels.
            valid:
        """
        B, H, W = valid.shape
        colors = self.models["head_mlp"](neural_features)

        rgb = torch.ones((B, H, W, 3)).float().cuda()
        rgb[valid] = colors

        return rgb

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

    def fuse(self, rasterized_hair, rasterized_head, is_val=False):
        """To fuse hair image and head image"""
        # ablation options
        gsdepth = self.config.get("ab.gsdepth", False)
        hardblend = self.config.get("ab.hardblend", False)
        usemorph = self.config.get("training.usemorph", False)

        outputs = {}
        # 1. Hair processing, [B, H, W, 3]
        rendered_hair = rasterized_hair["render"].permute((0, 2, 3, 1))

        # 2. Head processing
        valid_pixels = rasterized_head["valid_nograd"]
        s_id = 3 if self.xyz_cond else 0
        rasterized_xyz, rasterized_uv = (
            rasterized_head["neural_img"][..., :s_id],
            rasterized_head["neural_img"][..., s_id : s_id + 2],
        )  # [N, 3], [N, 2], [N, tex_ch], N is the num of valid pixels.
        tex_ch = self.config["training.tex_ch"] if self.neural else 3
        neural_features = rasterized_head["neural_features"]
        rasterized_features = F.grid_sample(neural_features, rasterized_uv, mode="bilinear", align_corners=True)

        valid_xyz = rasterized_xyz[valid_pixels]
        valid_uv = rasterized_uv[valid_pixels]
        valid_features = rasterized_features.permute((0, 2, 3, 1))[valid_pixels]
        valid_tex = valid_features[..., :tex_ch]
        valid_basic_tex = valid_features[..., tex_ch:]

        #   DEBUG: vis uv & features
        # B, H, W, _ = rasterized_head["neural_img"].shape
        # uv_cat = torch.cat([-1 * torch.ones((B, H, W, 1)).cuda(), rasterized_head["neural_img"][..., :2]], dim=-1)
        # uv_cat = (uv_cat + 1) / 2
        # uv_cat[~valid_pixels] = 1.0
        # cv2.imwrite("uv_rast.png", uv_cat[0].detach().cpu().numpy() * 255)
        # feats = rasterized_head["neural_img"][..., 3:6].detach().cpu().numpy()
        # bg_mask = ~valid_pixels[0].detach().cpu().numpy()
        # visPositionMap("feats_rast.png", feats[0], bg_mask=bg_mask, bg_color=np.array([1.0, 1.0, 1.0]))

        # uv_pe
        if self.neural:
            uv_pe = self.models["pe"](valid_uv)  # [N, 4]
            rasterized_features = torch.cat([valid_xyz, uv_pe, valid_tex], dim=-1)
            rendered_head = self.neural2rgb(rasterized_features, valid=valid_pixels)

            rasterized_basic_features = torch.cat([valid_xyz, uv_pe, valid_basic_tex], dim=-1)
            rendered_basic_head = self.neural2rgb(rasterized_basic_features, valid=valid_pixels)
        else:
            B, H, W = valid_pixels.shape
            rendered_head = torch.ones((B, H, W, 3)).float().cuda()
            rendered_head[valid_pixels] = valid_tex

            rendered_basic_head = torch.ones((B, H, W, 3)).float().cuda()
            rendered_basic_head[valid_pixels] = valid_basic_tex

        # 3. Compute fusing mask & fuse
        hair_depth = rasterized_hair["near_z"] if not gsdepth else rasterized_hair["depth"]
        head_depth = rasterized_head["depth"]

        hair_mask = self.compare_depth(hair_depth, head_depth)

        # TRY: different ways to use hair mask
        if is_val:
            if self.alter_hair:
                processed_hair_mask = hair_mask
            else:
                processed_hair_mask = self.compare_depth(rasterized_hair["near_z2"], head_depth)
            processed_hair_mask = erosion(processed_hair_mask.unsqueeze(1), torch.ones(3, 3).cuda())
            processed_hair_mask = dilation(processed_hair_mask, torch.ones(3, 3).cuda())
            processed_hair_mask = dilation(processed_hair_mask, torch.ones(5, 5).cuda())
            processed_hair_mask = erosion(processed_hair_mask, torch.ones(5, 5).cuda())
            processed_hair_mask = processed_hair_mask[:, 0]
        elif usemorph:
            processed_hair_mask = erosion(hair_mask.unsqueeze(1), torch.ones(3, 3).cuda())
            processed_hair_mask = dilation(processed_hair_mask, torch.ones(3, 3).cuda())
            processed_hair_mask = dilation(processed_hair_mask, torch.ones(5, 5).cuda())
            processed_hair_mask = erosion(processed_hair_mask, torch.ones(5, 5).cuda())
            processed_hair_mask = processed_hair_mask[:, 0]
        else:
            processed_hair_mask = hair_mask

        # processed_hair_mask = hair_mask

        if hardblend:
            outputs["raster_hairmask"] = processed_hair_mask
        else:
            outputs["raster_hairmask"] = rasterized_hair["silhoutte"] * processed_hair_mask

        outputs["raster_headmask"] = torch.clip(
            rasterized_head["valid_nograd"].float() - outputs["raster_hairmask"], min=0.0, max=1.0
        )
        outputs["fullmask"] = outputs["raster_headmask"] + outputs["raster_hairmask"]

        head_part = outputs["raster_headmask"][..., None].expand(-1, -1, -1, 3)
        hair_part = outputs["raster_hairmask"][..., None].expand(-1, -1, -1, 3)
        render_fuse = head_part * rendered_head + hair_part * rendered_hair

        bg = torch.ones_like(render_fuse).float().cuda()
        render_fuse = (1 - outputs["fullmask"])[..., None] * bg + render_fuse

        # 4. Output results
        outputs["render_hair"] = rendered_hair
        outputs["render_head"] = rendered_head
        outputs["render_basic_head"] = rendered_basic_head
        outputs["render_fuse"] = render_fuse
        outputs["hair_depth"] = rasterized_hair["depth"]
        outputs["head_depth"] = rasterized_head["depth"]
        outputs["hair_silhoutte"] = rasterized_hair["silhoutte"]
        outputs["occlussion_mask"] = hair_mask  # no gradients
        outputs["head_geomap"] = rasterized_head["rgba"][..., :3]

        #   DEBUG
        # cv2.imwrite('test_rasthair.png', outputs['raster_hairmask'][0, ..., None].detach().cpu().numpy() * 255)
        # cv2.imwrite('test_rasthead.png', outputs['raster_headmask'][0, ..., None].detach().cpu().numpy() * 255)

        return outputs

    def render_meshes(self, verts, faces):
        verts_tensor, faces_tensor = verts, faces
        if isinstance(verts, np.ndarray):
            verts_tensor = torch.from_numpy(verts).float().cuda()
        if isinstance(faces, np.ndarray):
            faces_tensor = torch.from_numpy(faces).float().cuda()
        rasterized_mesh = self.head_rasterizer.render_from_camera(verts_tensor, faces_tensor, self.camera)
        return rasterized_mesh["rgba"][..., :3]

    def remap_tex_from_2dmask(self, verts, input_img, face_ids):
        B, N_v, _ = verts.shape
        C = input_img.shape[-1]

        verts_cam = self.transform(verts, self.proj_R, self.proj_t)
        verts_proj = verts_cam.bmm(self.intr.transpose(1, 2))
        coords_screen = verts_proj[:, :, :2] / verts_proj[:, :, 2:]  # [B, N_v, 2]

        ver_XY = coords_screen[0].unsqueeze(1).detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.uv2verts_ids.astype(np.float32)
        unwrap_uv_idx_bw = self.uv2verts_bw.astype(np.float32)

        # test ver_XY
        # input_img[ver_XY[:, 0, 1].astype(np.int64), ver_XY[:, 0, 0].astype(np.int64), :] = 0.0

        uv_ver_map_y0 = unwrap_uv_idx_v_idx[:, :, 0].astype(np.float32)
        uv_ver_map_y1 = unwrap_uv_idx_v_idx[:, :, 1].astype(np.float32)
        uv_ver_map_y2 = unwrap_uv_idx_v_idx[:, :, 2].astype(np.float32)
        uv_ver_map_x = np.zeros_like(uv_ver_map_y0).astype(np.float32)

        uv_XY_0 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
        uv_XY_1 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
        uv_XY_2 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
        uv_XY = (
            uv_XY_0 * unwrap_uv_idx_bw[:, :, 0:1]
            + uv_XY_1 * unwrap_uv_idx_bw[:, :, 1:2]
            + uv_XY_2 * unwrap_uv_idx_bw[:, :, 2:3]
        )

        remap_tex = cv2.remap(input_img.astype(np.float32), uv_XY[:, :, 0], uv_XY[:, :, 1], cv2.INTER_LINEAR)
        remap_tex = np.clip(remap_tex, 0.0, 255.0)

        ver_vis_mask = np.zeros((N_v, 1, 1)).astype(np.float32)
        vis_vert_ids = list(set(self.flame_dec.faces[face_ids].reshape(-1).tolist()))
        vis_vert_ids.sort()
        ver_vis_mask[vis_vert_ids] = 1.0
        remap_vis_mask0 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
        remap_vis_mask1 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
        remap_vis_mask2 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
        remap_vis_mask = (
            remap_vis_mask0 * unwrap_uv_idx_bw[:, :, 0]
            + remap_vis_mask1 * unwrap_uv_idx_bw[:, :, 1]
            + remap_vis_mask2 * unwrap_uv_idx_bw[:, :, 2]
        )
        thres = 0.5
        if C == 1:
            remap_vis_mask = (remap_vis_mask > thres).astype(np.float32)
        else:
            remap_vis_mask = img2mask(remap_vis_mask, thre=thres)

        remap_tex = remap_tex * remap_vis_mask

        return remap_tex

    def transform(self, pts, R, t):
        translation = deepcopy(t)
        if len(translation.shape) == 2:
            translation = translation[:, None]
        R_inv = R.transpose(1, 2)

        return pts.bmm(R_inv) + translation

    def flame_forward(self, flame_params):
        b = flame_params["shape"].shape[0]
        head_verts, head_lmks = self.flame_dec(
            shape=flame_params["shape"],
            expr=flame_params["expr"],
            rotation=flame_params["rotation"],
            neck=flame_params["neck_pose"],
            jaw=flame_params["jaw_pose"],
            eyes=flame_params["eyes_pose"],
            translation=flame_params["translation"],
            zero_centered_at_root_node=False,
            static_offset=None,
        )
        head_faces_split = self.flame_dec.split_faces[None].expand(b, -1, -1)
        head_faces = self.flame_dec.faces[None].expand(b, -1, -1)
        return head_verts, head_faces, head_faces_split

    def transform_eyes(self, verts, tranform, use_R=False):
        """transform one eyeball

        Args:
            verts [B, N, 3]:
            tranform [B, 8]: _description_
        """
        if tranform is None:
            return verts

        center = verts.mean(1).unsqueeze(1)  # [B, 1, 3]
        normalized_verts = verts - center  # move the sphere to originate.
        if use_R:
            s, R, t = tranform[:, :1], build_rotation(tranform[:, 1:5]), tranform[:, 5:]
            R_inv = R.transpose(1, 2)
        else:
            s, R, t = tranform[:, :1], torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1).cuda(), tranform[:, 1:]
            R_inv = R.transpose(1, 2)

        trans_verts = (s.unsqueeze(1) * normalized_verts).bmm(R_inv) + t.unsqueeze(1)
        return trans_verts + center  # move back

    def rescale(self, head_verts, scale=1.0):
        translation = self.flame_params["translation"].unsqueeze(1)
        normalized_verts = head_verts - translation
        return (normalized_verts * scale) + translation

    def network_forward(self, is_val=False):
        driving_sig = torch.cat(
            [
                self.flame_params["expr"],
                self.flame_params["rotation"],
                self.flame_params["neck_pose"],
                self.flame_params["jaw_pose"],
                self.flame_params["eyes_pose"],
            ],
            dim=-1,
        )

        # ablation options
        noview = self.config.get("ab.noview", False)
        nodyn = self.config.get("ab.nodyn", False)
        nodisp = self.config.get("ab.nodisp", False)
        hairshape = self.config.get("hair.shape_params", None)

        # 1. Get Neural Texture
        basic_tex = self.models["neural_texture"]().expand(self.batch_size, -1, -1, -1)
        neural_texture = basic_tex.clone()
        if not noview:
            view_tex = self.models["view_texture"](self.view)
            neural_texture += view_tex
        if not nodyn:
            dynamic_tex = self.models["dynamic_texture"](driving_sig)
            neural_texture += dynamic_tex
        neural_features = torch.cat([neural_texture, basic_tex], dim=1)

        #   DEBUG: vis RGB texture
        # basic_texture = self.vis_textures(basic_tex)
        # basic_n_view_texture = self.vis_textures(basic_tex + view_tex)
        # basic_n_dyn_texture = self.vis_textures(basic_tex + dynamic_tex)
        # texture = self.vis_textures(neural_texture)
        # visimg("basic_texture.png", basic_texture[None])
        # visimg("basic_n_view_texture.png", basic_n_view_texture[None])
        # visimg("basic_n_dyn_texture.png", basic_n_dyn_texture[None])
        # visimg("texture.png", texture[None])

        # 3. Generate FLAME mesh with offsets
        head_verts, head_faces, head_faces_split = self.flame_forward(self.flame_params)
        rigid_transform = estimate_rigid(self.canonical_scalp_batch, head_verts[:, self.scalp_idcs])
        head_verts_flame = head_verts

        if hairshape is not None:
            hair_flame_params = self.flame_params.copy()
            hair_flame_params["shape"] = torch.from_numpy(hairshape).float().cuda()
            hair_head_verts, _, _ = self.flame_forward(hair_flame_params)
            rigid_transform = estimate_rigid(self.canonical_scalp_batch, hair_head_verts[:, self.scalp_idcs])

        # add disp
        if not nodisp:
            disp_map, l_trans, r_trans = self.models["disp_decoder"](driving_sig)  # [B, 3, 256, 256], [B, 8]
            disp_map *= 0.001
            if l_trans is not None:
                l_trans[:, -3:] *= 0.001
            if r_trans is not None:
                r_trans[:, -3:] *= 0.001

            offsets = F.grid_sample(disp_map, self.flame_uvcoords, align_corners=True)[:, :, 0].permute(
                (0, 2, 1)
            )  # [B, V, 3]
            offsets[:, self.ex_indices] = 0.0
            head_verts_refine = head_verts_flame + offsets

            # eye transform
            head_verts_refine[:, self.leye_idcs] = self.transform_eyes(head_verts_refine[:, self.leye_idcs], l_trans)
            head_verts_refine[:, self.reye_idcs] = self.transform_eyes(head_verts_refine[:, self.reye_idcs], r_trans)
        else:
            head_verts_refine = head_verts_flame
        # head_verts_refine = self.rescale(head_verts_refine, self.config.get("head_scale", 1.0))
        head_verts_split = torch.cat([head_verts_refine, head_verts_refine[:, self.flame_dec.extra_verts_ids]], dim=1)

        head_latent_li = []
        if self.xyz_cond:
            head_latent_li = [head_verts_split]
        head_latent = torch.cat(head_latent_li + [self.flame_split_uvcoords[:, 0]], dim=-1)

        # 5. Render head image
        bg_color = [1.0, 1.0, 1.0]
        raster_in = [head_verts_split, head_faces_split[0], self.camera, head_latent, bg_color]
        rasterized_head = self.head_rasterizer.render_from_camera(*raster_in)
        rasterized_head["neural_features"] = neural_features

        # 6. Render hair image
        loss_reg = {}
        aiap_xyz_loss, aiap_cov_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        if self.stage in ["joint"]:
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            deformed_pts = self.models["canonical_gs"].rigid_deform(
                R=rigid_transform[:, :3, :3], t=rigid_transform[:, :3, 3]
            )
            if is_val or (self.gs_pretrain is not None):
                pts_c = self.models["canonical_gs"].get_xyz[None].expand(self.batch_size, -1, -1)
                expr_params = self.flame_params["expr"]
                expression_expand = (
                    expr_params[:, None].expand(-1, pts_c.shape[1], -1).reshape(-1, expr_params.shape[-1])
                )
                offsets = self.models["deform_mlp"](pts_c.reshape(-1, 3), expression_expand).reshape(
                    self.batch_size, pts_c.shape[1], -1
                )
                offsets *= 0.001

                gs_offsets = {}
                idcs = 0
                for attr in self.config["gs.deform_attr"]:
                    next_idcs = idcs + self.attr_dims[attr]
                    offset = offsets[..., idcs:next_idcs]
                    gs_offsets[attr] = offset
                    loss_reg[attr] = torch.linalg.norm(offset.clone(), dim=-1).mean()
                    idcs = next_idcs

                gs_offsets["xyz"] += deformed_pts - pts_c
            else:
                gs_offsets = {}

            # uncomment for testing results only using rigid transform
            # gs_offsets = {"xyz": deformed_pts - pts_c}
            deformed_gaussians = self.models["canonical_gs"].update_gaussian(gs_offsets, self.batch_size)
            if hairshape is not None:
                for i, gs in enumerate(deformed_gaussians):
                    gs.binding(hair_head_verts[i], head_faces[i])
                    # gs.transfer(head_verts_refine[i], head_faces[i])
                    gs.transfer(head_verts_flame[i], head_faces[i])

            if self.config["gs.enable_aiap"]:
                for deform_gs in deformed_gaussians:
                    xyz_loss, cov_loss = full_aiap_loss(
                        self.models["canonical_gs"], deform_gs, n_neighbors=self.config["gs.K"]
                    )
                    aiap_xyz_loss += xyz_loss
                    aiap_cov_loss += cov_loss
                aiap_xyz_loss /= self.batch_size
                aiap_cov_loss /= self.batch_size
            rasterized_hair = render_list(self.config, self.camera, deformed_gaussians, background)
        else:
            rasterized_hair = {
                "render": torch.zeros((self.batch_size, 3, self.img_h, self.img_w)).cuda(),
                "silhoutte": torch.zeros((self.batch_size, self.img_h, self.img_w)).cuda(),
                "depth": torch.zeros((self.batch_size, self.img_h, self.img_w)).cuda(),
                "near_z": torch.ones((self.batch_size, self.img_h, self.img_w)).cuda() * -1,
                "near_z2": torch.ones((self.batch_size, self.img_h, self.img_w)).cuda() * -1,
                "visibility_filter": None,
                "radii": None,
                "viewspace_points": None,
            }
            rasterized_hair["render"] = torch.ones((self.batch_size, 3, self.img_h, self.img_w)).cuda()
            deformed_pts = None

        # 7. Fuse raster_hair and raster_head
        outputs = self.fuse(rasterized_hair, rasterized_head, is_val=is_val)

        #   DEBUG
        # face_ids = set(rasterized_head["face_id"][0].reshape(-1).tolist())
        # if -1 in face_ids:
        #     face_ids.remove(-1)
        # f_ids = list(face_ids)
        # f_ids.sort()
        # input_img = self.img[0].detach().cpu().numpy()
        # texture = self.remap_tex_from_2dmask(head_verts_refine, input_img, f_ids)

        # 8. Add some output info
        outputs["head_verts_flame"] = head_verts_flame
        outputs["head_verts_refine"] = head_verts_refine
        outputs["head_faces"] = head_faces
        outputs["visibility_filter"] = rasterized_hair["visibility_filter"]
        outputs["radii"] = rasterized_hair["radii"]
        outputs["viewspace_points"] = rasterized_hair["viewspace_points"]
        outputs["head_face_ids"] = rasterized_head["face_id"]
        outputs["head_face_bw"] = rasterized_head["face_bw"]
        outputs["head_face_uvs"] = self.flame_dec.vertex_uvs[self.flame_dec.faces_uvs]

        outputs["gs_reg_loss"] = loss_reg
        outputs["gs_aiap_xyz_loss"] = aiap_xyz_loss
        outputs["gs_aiap_cov_loss"] = aiap_cov_loss
        visualization = {
            "basic_tex": basic_tex,
            "dynamic_tex": dynamic_tex if not nodyn else None,
            "view_tex": view_tex if not noview else None,
            "neural_texture": neural_texture,
            "disp_map": disp_map if not nodisp else None,
        }

        return outputs, visualization

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
        render_hair = outputs["render_hair"]
        render_head = outputs["render_head"]
        render_basic_head = outputs["render_basic_head"]
        head_verts_refine = outputs["head_verts_refine"]
        head_verts_flame = outputs["head_verts_flame"]
        head_face_ids = outputs["head_face_ids"]
        head_faces = outputs["head_faces"]
        head_depth = outputs["head_depth"]
        hair_silhoutte = outputs["hair_silhoutte"]
        occlussion_mask = outputs["occlussion_mask"]  # no grads

        # ablation options
        nodipho = self.config.get("ab.nodipho", False)

        # update hyper-parameters
        self.update_lambda()
        pred_mesh = Meshes(verts=head_verts_refine, faces=head_faces)

        # RGB Loss
        hair_mask = self.mask["hair"]
        erode_hair_mask = self.mask["erode_hair"]
        head_mask = self.mask["head"]

        gt_hair = self.img.clone()
        gt_head = self.img.clone()
        gs_render_hair = render_hair.clone()
        gt_hair[(1 - hair_mask).bool()] = 1.0
        gt_head[hair_mask.bool()] = 1.0
        gs_render_hair[(1 - hair_mask).bool()] = 1.0

        rgb_loss = torch.linalg.norm((render_rgb - self.img), dim=-1).mean()
        hair_loss = torch.linalg.norm((gs_render_hair - gt_hair) * hair_mask[..., None], dim=-1).mean()
        head_loss = torch.linalg.norm((render_head - gt_head) * (1 - hair_mask[..., None]), dim=-1).mean()
        basic_head_loss = torch.linalg.norm((render_basic_head - gt_head) * (1 - hair_mask[..., None]), dim=-1).mean()
        whole_head_loss = torch.linalg.norm((render_head - self.img), dim=-1).mean()

        # SSIM Loss
        if self.get_lambda("ssim") > 0:
            render_img = render_head if self.stage in ["head"] else render_rgb
            gt_img = gt_head if self.stage in ["head"] else self.img
            ssim_loss = 1.0 - ssim(render_img.permute((0, 3, 1, 2)), gt_img.permute((0, 3, 1, 2)))
            basic_ssim_loss = 1.0 - ssim(render_basic_head.permute((0, 3, 1, 2)), gt_head.permute((0, 3, 1, 2)))
        else:
            ssim_loss = torch.tensor(0.0).cuda()
            basic_ssim_loss = torch.tensor(0.0).cuda()
        whole_head_ssim_loss = 1.0 - ssim(render_head.permute((0, 3, 1, 2)), self.img.permute((0, 3, 1, 2)))

        # LPIPS Loss
        # TODO: 暂时忽略
        if self.get_lambda("lpips") > 0:
            if self.stage in ["head"]:
                lpips_loss = self.lpips(
                    render_head.permute(0, 3, 1, 2), gt_head.permute(0, 3, 1, 2), normalize=True
                ).mean()
            else:
                lpips_loss = self.lpips(
                    render_rgb.permute(0, 3, 1, 2), self.img.permute(0, 3, 1, 2), normalize=True
                ).mean()
        else:
            lpips_loss = torch.tensor(0.0).cuda()

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

        # Depth Loss
        head_supervise_mask = torch.zeros_like(head_mask).cuda()
        for i in range(self.batch_size):
            nonfacial_face_ids = set(head_face_ids[i, ~(head_mask[i].bool())].tolist())
            facial_face_ids = set(head_face_ids[i, head_mask[i].bool()].tolist())
            seams_face_ids = set.intersection(nonfacial_face_ids, facial_face_ids)
            if -1 in seams_face_ids:
                seams_face_ids.remove(-1)
            seams_mask = torch.isin(head_face_ids[i], torch.tensor(list(seams_face_ids)).cuda())
            head_supervise_mask[i] = head_mask[i] * ~seams_mask

        gt_head_depth = self.depth_map * head_supervise_mask
        head_depth_mask = torch.abs(head_depth - gt_head_depth) < self.config["training.depth_thres"]
        head_depth_loss = torch.abs((head_depth - gt_head_depth) * head_depth_mask).mean()

        head_normal = depth_map2normals(head_depth)
        gt_head_normal = depth_map2normals(gt_head_depth)
        head_normal_loss = torch.linalg.norm((head_normal - gt_head_normal) * head_depth_mask[..., None], dim=-1).mean()
        #   DEBUG
        # cv2.imwrite('head_normal.png', (head_normal.detach().cpu().numpy()[0, ..., ::-1] + 1) / 2 * 255)

        # Reg
        mesh_normals_loss = mesh_normal_consistency(pred_mesh)
        mesh_edge_loss = edge_loss(head_verts_flame, pred_mesh, v_filter=self.mouth_idcs)
        mesh_laplacian_loss = mesh_laplacian_smoothing(pred_mesh, method="cot")
        solid_hair_loss = ((1 - hair_silhoutte) * erode_hair_mask).mean()

        center = head_verts_flame[:, self.scalp_idcs].mean(dim=1, keepdim=True).detach()
        mesh_verts_scale = torch.tensor(0.0).cuda()
        for i in range(self.batch_size):
            scalp_face_ids = head_face_ids[i, hair_mask[i].bool()]
            valid = scalp_face_ids > -1
            if valid.shape[0] == 0:
                continue
            scalp_vert_ids = list(
                set(head_faces[i, scalp_face_ids[valid]].reshape(-1).tolist() + self.shrink_scalp_idcs.tolist())
            )
            mesh_verts_scale += torch.linalg.norm(
                head_verts_refine[i, scalp_vert_ids] - center[i], dim=-1
            ).mean()  # task: auto shrink scalp mesh.
        mesh_verts_scale /= self.batch_size

        loss_hair = (
            self.get_lambda("rgb.hair") * hair_loss
            + self.get_lambda("silh.hair") * hair_silh_loss
            + self.get_lambda("silh.solid_hair") * solid_hair_loss
        )
        for k, v in outputs["gs_reg_loss"].items():
            loss_hair += self.get_lambda(k) * v
        loss_hair += self.get_lambda("aiap.xyz") * outputs["gs_aiap_xyz_loss"]
        loss_hair += self.get_lambda("aiap.cov") * outputs["gs_aiap_cov_loss"]

        loss_head = (
            self.get_lambda("rgb.head") * head_loss
            + self.get_lambda("ssim") * ssim_loss
            + self.get_lambda("depth.head") * head_depth_loss
            + self.get_lambda("normal.head") * head_normal_loss
            + self.get_lambda("mesh.normal") * mesh_normals_loss
            + self.get_lambda("mesh.edges") * mesh_edge_loss
            + self.get_lambda("mesh.laplacian") * mesh_laplacian_loss
            + self.get_lambda("mesh.verts_scale") * mesh_verts_scale  # used to shrink scalp
        )
        loss_whole_head = self.get_lambda("rgb.head") * whole_head_loss + self.get_lambda("ssim") * whole_head_ssim_loss

        loss_dipho = 3 * self.get_lambda("rgb.head") * basic_head_loss + 3 * self.get_lambda("ssim") * basic_ssim_loss

        loss_general = self.get_lambda("rgb") * rgb_loss + self.get_lambda("ssim") * ssim_loss

        loss_all = {"head": loss_head, "hair": loss_hair, "general": loss_general, "whole_head": loss_whole_head}

        loss = 0
        for name in self.config["training.{}_stage_loss".format(self.stage)]:
            loss += loss_all[name]
        if not nodipho:
            loss += loss_dipho

        outputs["head_depth_mask"] = head_depth_mask
        outputs["gt_hair"] = gt_hair
        outputs["gt_head"] = gt_head
        loss_dict = {
            "loss": loss,
            "loss_pho/rgb.obj": rgb_loss,
            "loss_pho/rgb.hair": hair_loss,
            "loss_pho/rgb.head": head_loss,
            "loss_pho/rgb.basic_head": basic_head_loss,
            "loss_geo/silh.hair": hair_silh_loss,
            "loss_geo/depth.head": head_depth_loss,
            "loss_geo/normal.head": head_normal_loss,
            "loss_pho/ssim": ssim_loss,
            "loss_pho/lpips": lpips_loss,
            "loss_reg/mesh.laplacian": mesh_laplacian_loss,
            "loss_reg/mesh.normal": mesh_normals_loss,
            "loss_reg/mesh.edges": mesh_edge_loss,
            "loss_reg/mesh.vscale": mesh_verts_scale,
            "loss_reg/silh.solid_hair": solid_hair_loss,
        }

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
        loss_ssim = loss_dict["loss_pho/ssim"]
        loss_lpips = loss_dict["loss_pho/lpips"]
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
            "        lpips = %.4f                   w: %.4f\n"
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
                loss_ssim.item(),
                self.get_lambda("ssim"),
                loss_lpips.item(),
                self.get_lambda("lpips"),
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
                outputs, visualization = self.network_forward(is_val=True)
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

        # TODO: 如果需要，可以将一些渲染结果存在visualization_dict中，然后用tb_writer保存一些结果图用于可视化

    def compute_metrics(self, outputs):
        valid_mask = outputs["fullmask"] * self.mask["full"]

        gt_img = (self.img[0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        pred_img = (outputs["render_fuse"][0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        mse = ((pred_img - gt_img) ** 2).mean()
        psnr = 10 * np.log10(65025 / mse)

        return mse, psnr

    def vis_textures(self, in_texture, idx=0):
        texture = in_texture
        if len(in_texture.shape) == 4:
            texture = in_texture[idx].permute(1, 2, 0)

        assert len(texture.shape) == 3, "Support only texture with shape [H, W, C]."

        H, W, _ = texture.shape
        num_texels = H * W

        h, w = torch.linspace(0, 1, H), torch.linspace(0, 1, W)
        grid = torch.meshgrid(h, w)
        coords_uvs = torch.stack([grid[1], grid[0]], dim=-1).cuda()
        flat_uvs = coords_uvs.reshape(num_texels, -1)
        normalized_uvs = 2 * flat_uvs - 1
        uv_pe = self.models["pe"](normalized_uvs)  # [N, 4]

        flat_tex = texture.reshape(num_texels, -1)

        neural_features = torch.cat([uv_pe, flat_tex], dim=-1)
        rgb = self.models["head_mlp"](neural_features).reshape(H, W, -1)

        return rgb

    def visualization(self, outputs, visualization, step, label="log", bg_white=True):
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

        # all images that need to be recorded
        savepath = os.path.join(savedir, "hairmask_it{}.png".format(step))
        raster_hairmask = outputs["raster_hairmask"][0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        cv2.imwrite(savepath, raster_hairmask * 255)

        # raster head mask
        savepath = os.path.join(savedir, "headmask_it{}.png".format(step))
        raster_headmask = outputs["raster_headmask"][0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        cv2.imwrite(savepath, raster_headmask * 255)

        savepath = os.path.join(savedir, "gt_it{}.png".format(step))
        if bg_white:
            gt_img = self.img[0].detach().cpu().numpy()
        else:
            gt_img = (self.img[0] * valid_mask[0, ..., None]).detach().cpu().numpy()  # [512, 512, 3]
        cv2.imwrite(savepath, gt_img * 255)

        savepath = os.path.join(savedir, "hairgt_it{}.png".format(step))
        gt_hair = outputs["gt_hair"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_hair * 255)

        savepath = os.path.join(savedir, "headgt_it{}.png".format(step))
        gt_head = outputs["gt_head"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_head * 255)

        # image
        savepath = os.path.join(savedir, "rendering_it{}.png".format(step))
        render_fuse = outputs["render_fuse"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_fuse * 255)

        # render hair
        savepath = os.path.join(savedir, "hair_it{}.png".format(step))
        render_hair = outputs["render_hair"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_hair * 255)

        # render head
        savepath = os.path.join(savedir, "head_it{}.png".format(step))
        render_head = outputs["render_head"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_head * 255)

        # head_depth_mask
        savepath = os.path.join(savedir, "head_depthmask_it{}.png".format(step))
        head_depth_mask = outputs["head_depth_mask"][0].int()
        head_depth_mask[head_depth_mask == 0] = 2
        head_depth_mask[outputs["head_depth"][0] == 0] = 0
        head_depth_mask = head_depth_mask.detach().cpu().numpy()
        if bg_white:
            colored_mask = np.ones((head_depth_mask.shape[0], head_depth_mask.shape[1], 3), dtype=np.float32)
            colored_mask[head_depth_mask == 1] = np.array([0.0, 0.8, 0.0])
            colored_mask[head_depth_mask == 2] = np.array([0.0, 0.0, 0.8])
        else:
            colored_mask = np.zeros((head_depth_mask.shape[0], head_depth_mask.shape[1], 3), dtype=np.float32)
            colored_mask[head_depth_mask == 1] = np.array([1.0, 1.0, 1.0])
            colored_mask[head_depth_mask == 2] = np.array([0.0, 0.0, 0.8])
        cv2.imwrite(savepath, colored_mask * 255)

        # head normal
        head_depth = outputs["head_depth"]
        head_normal = (depth_map2normals(head_depth, bg_white=bg_white)[0].detach().cpu().numpy() + 1) / 2.0
        gt_head_normal = (depth_map2normals(self.depth_map, bg_white=bg_white)[0].detach().cpu().numpy() + 1) / 2.0
        savepath = os.path.join(savedir, "head_normal_it{}.png".format(step))
        cv2.imwrite(savepath, head_normal[..., ::-1] * 255)
        savepath = os.path.join(savedir, "gt_head_normal_it{}.png".format(step))
        cv2.imwrite(savepath, gt_head_normal[..., ::-1] * 255)

        # geomap
        savepath = os.path.join(savedir, "geomap_it{}.png".format(step))
        head_geomap = outputs["head_geomap"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, head_geomap * 255)

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
            color_mask(raster_headmask, color=color_pred, bg_white=bg_white),
            1.0,
            color_mask(headmask_gt, color=color_gt),
            alpha,
            0,
        )
        hairmask_withgt = cv2.addWeighted(
            color_mask(raster_hairmask, color=color_pred, bg_white=bg_white),
            1.0,
            color_mask(hairmask_gt, color=color_gt),
            alpha,
            0,
        )
        mask = np.concatenate([head_geomap, headmask_withgt, colored_mask, hairmask_withgt], axis=1)
        combined_img = np.concatenate([gt, pred, mask], axis=0)
        cv2.imwrite(savepath, combined_img * 255)

        # fuse mask
        savepath = os.path.join(savedir, "fullmask_it{}.png".format(step))
        fullmask = outputs["fullmask"] * self.mask["full"]
        cv2.imwrite(savepath, fullmask[0].detach().cpu().numpy() * 255)

        # texture maps
        savepath = os.path.join(savedir, "basictex_it{}.png".format(step))
        basic_tex = visPositionMap(
            savepath, visualization["basic_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
        )
        dynamic_tex, view_tex = np.ones((1024, 1024, 3)), np.ones((1024, 1024, 3))
        if visualization["dynamic_tex"] is not None:
            savepath = os.path.join(savedir, "dynamictex_it{}.png".format(step))
            dynamic_tex = visPositionMap(
                savepath, visualization["dynamic_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
            )
        if visualization["view_tex"] is not None:
            savepath = os.path.join(savedir, "viewtex_it{}.png".format(step))
            view_tex = visPositionMap(
                savepath, visualization["view_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
            )
        savepath = os.path.join(savedir, "neutex_it{}.png".format(step))
        neural_texture = visPositionMap(
            savepath, visualization["neural_texture"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
        )
        if visualization["disp_map"] is not None:
            savepath = os.path.join(savedir, "dispmap_it{}.png".format(step))
            visPositionMap(savepath, visualization["disp_map"][0].permute((1, 2, 0)).detach().cpu().numpy())

        savepath = os.path.join(savedir, "maps_it{}.png".format(step))
        maps = np.concatenate([basic_tex, view_tex, dynamic_tex, neural_texture], axis=1)
        cv2.imwrite(savepath, maps)

        # metrics
        if label == "eval":
            savepath = os.path.join(savedir, "metrics.txt")
            mse, psnr = self.compute_metrics(outputs)

            with open(savepath, "w") as f:
                f.write("MSE: {}\n".format(mse))
                f.write("PSNR: {}\n".format(psnr))

            # geometry
            savepath = os.path.join(savedir, "refined_flame.obj")
            write_obj(savepath, outputs["head_verts_refine"][0], outputs["head_faces"][0] + 1)

            savepath = os.path.join(savedir, "original_flame.obj")
            write_obj(savepath, outputs["head_verts_flame"][0], outputs["head_faces"][0] + 1)

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
        for k, m in self.models.items():
            model_dict = m.state_dict()
            if k == "canonical_gs":
                use_flags = m.get_use_flags
                for key, v in model_dict.items():
                    model_dict[key] = v[use_flags]
            save_dict[k] = model_dict

        torch.save(save_dict, savepath)

        basedir = os.path.dirname(savepath)
        npz_path = os.path.join(basedir, "flame_params.npz")
        if os.path.exists(npz_path):
            os.remove(npz_path)

        flame_params = {k: v.detach().cpu().numpy() for k, v in self.all_flame_params.items()}
        np.savez(str(npz_path), **flame_params)

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
                self.models["canonical_gs"].update_learning_rate(self.stage_step, self.optimizer)

            # 1. Set data for trainer
            sd_timer.start(step)
            self.set_data(items)
            sd_timer.end(step)

            # 2. Run the network
            f_timer.start(step)
            outputs, visualization = self.network_forward()
            f_timer.end(step)

            # 3. Compute losses
            cl_timer.start(step)
            loss_dict = self.compute_loss(outputs)
            cl_timer.end(step)
            loss = loss_dict["loss"]

            # DEBUG: save the checkpoint before NaN Loss
            # if loss.isnan().any() and not self.nan_detect:
            #     self.nan_detect = True
            #     checkpoint_path = os.path.join(self.config["local_workspace"], "nan_break.pth")
            #     self.save_ckpt(checkpoint_path)
            #     print("NaN break checkpoint has saved at {}".format(checkpoint_path))
            #     print("Data {}".format(self.name))
            #     return False

            # 4. Backprop
            b_timer.start(step)
            loss.backward()
            b_timer.end(step)

            # 5. Process Gaussians
            pp_timer.start(step)
            if self.stage in ["joint"] and self.stage_step % self.config["gs.upSH"] == 0:
                self.models["canonical_gs"].oneupSHdegree()

            if (
                self.stage in ["joint"]
                and self.config["pipe.neutral_hair"]
                and self.config["training.enable_densify"]
                and (self.stage_step < self.config["gs.densify_until_iter"])
            ):
                if self.stage_step == 1:
                    self.models["canonical_gs"].reset_use_flags()

                self.models["canonical_gs"].update_use_flags()

                do_densify = False
                do_reset = False
                for i in range(self.batch_size):
                    actual_step = (self.stage_step - 1) * self.batch_size + i + 1

                    # Keep track of max radii in image-space for pruning
                    radii, visibility_filter = outputs["radii"][i], outputs["visibility_filter"][i]
                    viewspace_point_tensor = outputs["viewspace_points"][i]
                    self.models["canonical_gs"].max_radii2D[visibility_filter] = torch.max(
                        self.models["canonical_gs"].max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    self.models["canonical_gs"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if (
                        actual_step > self.config["gs.densify_from_iter"]
                        and actual_step % self.config["gs.densification_interval"] == 0
                    ):
                        do_densify = True

                    if actual_step % (self.config["gs.opacity_reset_interval"]) == 0:
                        do_reset = True

                    if i == (self.batch_size - 1) and do_densify:
                        size_threshold = (
                            20 * self.rate if self.stage_step > self.config["gs.opacity_reset_interval"] else None
                        )
                        self.models["canonical_gs"].densify_and_prune(
                            self.config["gs.densify_grad_threshold"],
                            0.005,
                            self.spatial_lr_scale,
                            size_threshold,
                            self.optimizer,
                        )

                    if i == (self.batch_size - 1) and do_reset and self.config["gs.enable_reset"]:
                        # Save model
                        checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_reset.pth")
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        self.save_ckpt(checkpoint_path)
                        self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

                        self.models["canonical_gs"].reset_opacity(self.optimizer)
                        self.models["canonical_gs"].reset_use_flags()

                    # if i == (self.batch_size - 1) and do_split:
                    #     self.models["canonical_gs"].split_flatnfat(
                    #         self.config["gs.split_rate_threshold"], self.optimizer
                    #     )
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
                self.visualization(outputs, visualization, self.global_step, bg_white=True)

            if self.global_step % 2000 == 0:
                # Save model
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                self.save_ckpt(checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

            if self.global_step % 5000 == 0:
                # Save adapted mesh
                obj_path = os.path.join(
                    self.config["local_workspace"], "mesh_with_offsets_it{}.obj".format(self.global_step)
                )
                write_obj(obj_path, outputs["head_verts_refine"][0], outputs["head_faces"][0] + 1)

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
