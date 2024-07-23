import os
import cv2

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from networks.gshair.gs.gaussian_utils import build_rotation
from networks.meshface.neural2rgb.pixel_decoder import PixelDecoder
from networks.meshface.textures.neural_texture import NeuralTexture
from networks.meshface.textures.detail_decoder import DynamicDecoder
from networks.meshface.textures.detail_decoder import ViewDecoder
from networks.meshface.flame2023.flame import FlameHead
from networks.meshface.textures.disp_decoder import DispDecoder
from networks.meshface.mesh_renderer.mesh_renderer import NVDiffRenderer
from networks.meshface.neural2rgb.lpe import LPE, PE
from utils import (
    depth_map2normals,
    edge_loss,
    estimate_rigid,
    params_with_lr,
    ssim,
    update_lambda,
    visPositionMap,
    write_obj,
)


class MeshFaceWrapper:
    def __init__(self, cfg, move_eyes=False, xyz_cond=False, painting=False):
        self.cfg = cfg
        self.img_h, self.img_w = cfg["data.img_h"], cfg["data.img_w"]
        self.move_eyes = move_eyes
        self.xyz_cond = xyz_cond
        self.tex_ch = cfg["training.tex_ch"]
        self.buffer = {}

        self.models = {}
        self.parameters_to_train = []
        self.lr = cfg["training.learning_rate"]

        # textures
        self.models["neural_texture"] = NeuralTexture(cfg).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["neural_texture"].named_parameters()), self.lr, label="head_tex_basic"
        )

        self.models["dynamic_texture"] = DynamicDecoder(cfg).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["dynamic_texture"].named_parameters()), self.lr, label="head_tex_dynamic"
        )

        self.models["view_texture"] = ViewDecoder(cfg).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["view_texture"].named_parameters()), self.lr, label="head_tex_view"
        )

        # FLAME Model
        self.flame_dec = FlameHead(cfg).cuda()

        # FLAME Disp Map
        self.models["disp_decoder"] = DispDecoder(cfg, eyes=self.move_eyes).cuda()
        self.parameters_to_train += params_with_lr(
            list(self.models["disp_decoder"].named_parameters()), self.lr, label="head_geo"
        )

        # Head Rasterizer
        self.head_rasterizer = NVDiffRenderer(cfg)

        # Head Renderer
        pe_cfg = {
            "input_dims": 2,
            "num_freqs": cfg["training.pe.num_freqs"],
            "periodic_fns": [torch.sin, torch.cos],
            "log_sampling": cfg["training.pe.log_sampling"],
            "include_input": cfg["training.pe.include_input"],
        }
        pe_module = globals()[cfg["training.pe"]]
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

        self._init_data()

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

    def get_optim_params(self):
        return self.parameters_to_train

    def restore_models(self, state_dict, logger=None):
        for key, model in self.models.items():
            if key not in state_dict:
                if logger:
                    logger.info("[warning] {} weights are not loaded. SKIP.".format(key))
                else:
                    print("[warning] {} weights are not loaded. SKIP".format(key))
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
                model.load_state_dict(_state_dict, strict=False)
            except:
                if logger:
                    logger.info("[warning] {} weights are not loaded.".format(key))
                else:
                    print("[warning] {} weights are not loaded.".format(key))

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

    def _init_data(self):
        body_parts = self.flame_dec.get_body_parts()
        ex_sets = set()
        ex_sets = set.union(ex_sets, set(body_parts["left_eyeball"].tolist()))  # eye deformation does not use offsets.
        ex_sets = set.union(ex_sets, set(body_parts["right_eyeball"].tolist()))
        # ex_sets = set.union(ex_sets, set(self.flame_dec.vid_teeth.tolist()))
        for part in self.cfg["flame.offsets_ignore_parts"]:
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
        canonical_flame_params = np.load(self.cfg["data.canonical_flame_path"])
        params = {}
        for k, v in canonical_flame_params.items():
            params[k] = torch.from_numpy(v).cuda()
        params["shape"] = params["shape"][None]
        canonical_head_verts, _, _ = self.flame_forward(params)
        self.canonical_scalp = canonical_head_verts[0, body_parts["scalp"]]
        self.scalp_idcs = body_parts["scalp"]
        self.shrink_scalp_idcs = body_parts["shrink_scalp"]

    def expand_dims(self, batch_size):
        self.flame_uvcoords = (2 * self.flame_dec.verts_uvs - 1)[None, None].expand(batch_size, -1, -1, -1)
        self.flame_split_uvcoords = (2 * self.flame_dec.split_verts_uvs - 1)[None, None].expand(batch_size, -1, -1, -1)
        self.canonical_scalp_batch = self.canonical_scalp[None].expand(batch_size, -1, -1)

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

        batch_size = verts.shape[0]
        center = verts.mean(1).unsqueeze(1)  # [B, 1, 3]
        normalized_verts = verts - center  # move the sphere to originate.
        if use_R:
            s, R, t = tranform[:, :1], build_rotation(tranform[:, 1:5]), tranform[:, 5:]
            R_inv = R.transpose(1, 2)
        else:
            s, R, t = tranform[:, :1], torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).cuda(), tranform[:, 1:]
            R_inv = R.transpose(1, 2)

        trans_verts = (s.unsqueeze(1) * normalized_verts).bmm(R_inv) + t.unsqueeze(1)
        return trans_verts + center  # move back

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
        uv_pe = self.PE(normalized_uvs)  # [N, 4]

        flat_tex = texture.reshape(num_texels, -1)

        neural_features = torch.cat([uv_pe, flat_tex], dim=-1)
        rgb = self.pix_dec(neural_features).reshape(H, W, -1)

        return rgb

    def render(self, viewpoint_cameras, flame_params, view_vec, bg_color=[1.0, 1.0, 1.0]):
        driving_sig = torch.cat(
            [
                flame_params["expr"],
                flame_params["rotation"],
                flame_params["neck_pose"],
                flame_params["jaw_pose"],
                flame_params["eyes_pose"],
            ],
            dim=-1,
        )

        noview = self.cfg.get("ab.noview", False)
        nodyn = self.cfg.get("ab.nodyn", False)
        nodisp = self.cfg.get("ab.nodisp", False)
        batch_size = flame_params["shape"].shape[0]
        self.expand_dims(batch_size)

        # 1. Get Neural Texture
        basic_tex = self.models["neural_texture"]().expand(batch_size, -1, -1, -1)
        neural_texture = basic_tex.clone()
        if not noview:
            view_tex = self.models["view_texture"](view_vec)
            neural_texture += view_tex
        if not nodyn:
            dynamic_tex = self.models["dynamic_texture"](driving_sig)
            neural_texture += dynamic_tex
        neural_features = torch.cat([neural_texture, basic_tex], dim=1)

        # 2. Generate FLAME mesh with offsets
        head_verts, head_faces, head_faces_split = self.flame_forward(flame_params)
        rigid_transform = estimate_rigid(self.canonical_scalp_batch, head_verts[:, self.scalp_idcs])
        head_verts_flame = head_verts

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
        raster_in = [head_verts_split, head_faces_split[0], viewpoint_cameras, head_latent, bg_color]
        rasterized_face = self.head_rasterizer.render_from_camera(*raster_in)
        rasterized_face["neural_features"] = neural_features

        # update buffer
        self.buffer = {
            "head_verts_refine": head_verts_refine,
            "head_verts_flame": head_verts_flame,
            "head_face_ids": rasterized_face["face_id"],
            "head_face_bw": rasterized_face["face_bw"],
            "head_face_uvs": self.flame_dec.vertex_uvs[self.flame_dec.faces_uvs],
            "head_faces": head_faces,
            "basic_tex": basic_tex,
            "dynamic_tex": dynamic_tex if not nodyn else None,
            "view_tex": view_tex if not noview else None,
            "neural_texture": neural_texture,
            "disp_map": disp_map if not nodisp else None,
        }

        return rasterized_face, rigid_transform

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

    def feats2rgbs(self, valid_xyz, valid_uv, valid_tex, valid_mask):
        uv_pe = self.models["pe"](valid_uv)  # [N, 4]
        rasterized_features = torch.cat([valid_xyz, uv_pe, valid_tex], dim=-1)
        rendered_face = self.neural2rgb(rasterized_features, valid=valid_mask)

        return rendered_face

    def get_lambda(self, key):
        return self.all_lambdas.get(key, 0.0)

    def compute_losses(self, outputs, gt_img, gt_head, gt_depth, hair_mask, head_mask, step):
        batch_size = gt_img.shape[0]

        head_verts_refine = self.buffer["head_verts_refine"]
        head_verts_flame = self.buffer["head_verts_flame"]
        head_face_ids = self.buffer["head_face_ids"]
        head_faces = self.buffer["head_faces"]

        render_head = outputs["render_face"]
        render_basic_head = outputs["render_basic_face"]
        head_depth = outputs["head_depth"]

        # ablation options
        nodipho = self.cfg.get("ab.nodipho", False)

        assert render_head is not None, "Must render facial mesh. Please check and retry."

        self.update_weights(step)
        pred_mesh = Meshes(verts=head_verts_refine, faces=head_faces)

        # L2 Loss
        head_loss = torch.linalg.norm(render_head - gt_head, dim=-1).mean()
        basic_head_loss = torch.linalg.norm(render_basic_head - gt_head, dim=-1).mean()

        # SSIM Loss
        if self.get_lambda("ssim") > 0:
            head_ssim_loss = 1.0 - ssim(render_head.permute((0, 3, 1, 2)), gt_head.permute((0, 3, 1, 2)))
            basic_head_ssim_loss = 1.0 - ssim(render_basic_head.permute((0, 3, 1, 2)), gt_head.permute((0, 3, 1, 2)))
        else:
            head_ssim_loss = torch.tensor(0.0).cuda()
            basic_head_ssim_loss = torch.tensor(0.0).cuda()

        # Depth Loss
        head_supervise_mask = torch.zeros_like(head_mask).cuda()
        for i in range(batch_size):
            nonfacial_face_ids = set(head_face_ids[i, ~(head_mask[i].bool())].tolist())
            facial_face_ids = set(head_face_ids[i, head_mask[i].bool()].tolist())
            seams_face_ids = set.intersection(nonfacial_face_ids, facial_face_ids)
            if -1 in seams_face_ids:
                seams_face_ids.remove(-1)
            seams_mask = torch.isin(head_face_ids[i], torch.tensor(list(seams_face_ids)).cuda())
            head_supervise_mask[i] = head_mask[i] * ~seams_mask

        gt_head_depth = gt_depth * head_supervise_mask
        head_depth_mask = torch.abs(head_depth - gt_head_depth) < self.cfg["training.depth_thres"]
        head_depth_loss = torch.abs((head_depth - gt_head_depth) * head_depth_mask).mean()

        head_normal = depth_map2normals(head_depth)
        gt_head_normal = depth_map2normals(gt_head_depth)
        head_normal_loss = torch.linalg.norm((head_normal - gt_head_normal) * head_depth_mask[..., None], dim=-1).mean()

        # Reg
        mesh_normals_loss = mesh_normal_consistency(pred_mesh)
        mesh_edge_loss = edge_loss(head_verts_flame, pred_mesh, v_filter=self.mouth_idcs)
        mesh_laplacian_loss = mesh_laplacian_smoothing(pred_mesh, method="cot")

        # Scalp Shrink
        center = head_verts_flame[:, self.scalp_idcs].mean(dim=1, keepdim=True).detach()
        mesh_verts_scale = torch.tensor(0.0).cuda()
        for i in range(batch_size):
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
        mesh_verts_scale /= batch_size

        loss_head = (
            self.get_lambda("rgb.head") * head_loss
            + self.get_lambda("ssim") * head_ssim_loss
            + self.get_lambda("depth.head") * head_depth_loss
            + self.get_lambda("normal.head") * head_normal_loss
            + self.get_lambda("mesh.normal") * mesh_normals_loss
            + self.get_lambda("mesh.edges") * mesh_edge_loss
            + self.get_lambda("mesh.laplacian") * mesh_laplacian_loss
            + self.get_lambda("mesh.verts_scale") * mesh_verts_scale  # used to shrink scalp
        )
        loss_dipho = (
            3 * self.get_lambda("rgb.head") * basic_head_loss + 3 * self.get_lambda("ssim") * basic_head_ssim_loss
        )

        loss_dict = {
            "loss_pho/rgb.head": head_loss,
            "loss_pho/rgb.basic_head": basic_head_loss,
            "loss_geo/depth.head": head_depth_loss,
            "loss_geo/normal.head": head_normal_loss,
            "loss_pho/ssim.head": head_ssim_loss,
            "loss_reg/mesh.laplacian": mesh_laplacian_loss,
            "loss_reg/mesh.normal": mesh_normals_loss,
            "loss_reg/mesh.edges": mesh_edge_loss,
            "loss_reg/mesh.vscale": mesh_verts_scale,
        }

        loss = loss_head.clone()
        if not nodipho:
            loss += loss_dipho

        # save some variables for visualization
        self.buffer["gt_head"] = gt_head
        self.buffer["gt_depth"] = gt_depth
        self.buffer["head_depth_mask"] = head_depth_mask

        return loss, loss_dict

    def visualize(self, savedir, outputs, step):
        white_img = np.ones((self.img_h, self.img_w, 3)).astype(np.float32)

        # raster head mask
        savepath = os.path.join(savedir, "headmask_it{}.png".format(step))
        raster_headmask = (
            white_img
            if outputs["raster_headmask"] is None
            else outputs["raster_headmask"][0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        )
        cv2.imwrite(savepath, raster_headmask * 255)

        # gt head
        savepath = os.path.join(savedir, "headgt_it{}.png".format(step))
        gt_head = self.buffer["gt_head"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_head * 255)

        # render head
        savepath = os.path.join(savedir, "head_it{}.png".format(step))
        render_head = outputs["render_face"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_head * 255)

        # head_depth_mask
        savepath = os.path.join(savedir, "head_depthmask_it{}.png".format(step))
        head_depth_mask = self.buffer["head_depth_mask"][0].int()
        head_depth_mask[head_depth_mask == 0] = 2
        head_depth_mask[outputs["head_depth"][0] == 0] = 0
        head_depth_mask = head_depth_mask.detach().cpu().numpy()
        colored_mask = np.ones((head_depth_mask.shape[0], head_depth_mask.shape[1], 3), dtype=np.float32)
        colored_mask[head_depth_mask == 1] = np.array([0.0, 0.8, 0.0])
        colored_mask[head_depth_mask == 2] = np.array([0.0, 0.0, 0.8])
        cv2.imwrite(savepath, colored_mask * 255)

        # head normal
        head_depth = outputs["head_depth"]
        head_normal = (depth_map2normals(head_depth)[0].detach().cpu().numpy() + 1) / 2.0
        gt_head_normal = (depth_map2normals(self.buffer["gt_depth"])[0].detach().cpu().numpy() + 1) / 2.0
        savepath = os.path.join(savedir, "head_normal_it{}.png".format(step))
        cv2.imwrite(savepath, head_normal[..., ::-1] * 255)
        savepath = os.path.join(savedir, "gt_head_normal_it{}.png".format(step))
        cv2.imwrite(savepath, gt_head_normal[..., ::-1] * 255)

        # geomap
        savepath = os.path.join(savedir, "geomap_it{}.png".format(step))
        head_geomap = outputs["head_geomap"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, head_geomap * 255)

        if step % 5000 == 0:
            # Save adapted mesh
            obj_path = os.path.join(self.cfg["local_workspace"], "mesh_with_offsets_it{}.obj".format(step))
            write_obj(obj_path, self.buffer["head_verts_refine"][0], self.buffer["head_faces"][0] + 1)

        return {
            "raster_headmask": raster_headmask,
            "gt_head": gt_head,
            "render_head": render_head,
            "colored_mask": colored_mask,
            "head_normal": head_normal,
            "gt_head_normal": gt_head_normal,
            "head_geomap": head_geomap,
        }

    def visualize_textures(self, savedir, step):
        # texture maps
        savepath = os.path.join(savedir, "basictex_it{}.png".format(step))
        basic_tex = visPositionMap(savepath, self.buffer["basic_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy())
        dynamic_tex, view_tex = np.ones((1024, 1024, 3)), np.ones((1024, 1024, 3))
        if self.buffer["dynamic_tex"] is not None:
            savepath = os.path.join(savedir, "dynamictex_it{}.png".format(step))
            dynamic_tex = visPositionMap(
                savepath, self.buffer["dynamic_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
            )
        if self.buffer["view_tex"] is not None:
            savepath = os.path.join(savedir, "viewtex_it{}.png".format(step))
            view_tex = visPositionMap(
                savepath, self.buffer["view_tex"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
            )
        savepath = os.path.join(savedir, "neutex_it{}.png".format(step))
        neural_texture = visPositionMap(
            savepath, self.buffer["neural_texture"][0, -3:].permute((1, 2, 0)).detach().cpu().numpy()
        )
        if self.buffer["disp_map"] is not None:
            savepath = os.path.join(savedir, "dispmap_it{}.png".format(step))
            visPositionMap(savepath, self.buffer["disp_map"][0].permute((1, 2, 0)).detach().cpu().numpy())

        savepath = os.path.join(savedir, "maps_it{}.png".format(step))
        maps = np.concatenate([basic_tex, view_tex, dynamic_tex, neural_texture], axis=1)
        cv2.imwrite(savepath, maps)

    def save_mesh(self, savepath):
        write_obj(savepath, self.buffer["head_verts_refine"][0], self.buffer["head_faces"][0] + 1)

    def state_dict(self):
        state_dict = {}
        for k, m in self.models.items():
            model_dict = m.state_dict()
            state_dict[k] = model_dict

        return state_dict
