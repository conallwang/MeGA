import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import trimesh
from roma import quat_product, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rotmat_to_unitquat
from simple_knn._C import distCUDA2

from dataset.cameras import Camera

from .gaussian_utils import (
    RGB2SH,
    SH2RGB,
    build_rotation,
    build_scaling_rotation,
    compute_face_normals,
    estimate_rigid_between_flame_faces2,
    eval_sh,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)


def write_obj(filepath, verts, tris=None, log=True):
    """将mesh顶点与三角面片存储为.obj文件,方便查看

    Args:
        verts:      Vx3, vertices coordinates
        tris:       n_facex3, faces consisting of vertices id
    """
    fw = open(filepath, "w")
    # vertices
    for vert in verts:
        fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

    if not tris is None:
        for tri in tris:
            fw.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    fw.close()
    if log:
        print(f"mesh has been saved in {filepath}.")


class GaussianModel(nn.Module):
    def _setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.optimizable_names = [
            "gs.xyz",
            "gs.features_dc",
            "gs.features_rest",
            "gs.opacity",
            "gs.scaling",
            "gs.rotation",
        ]

    def __init__(self, sh_degree: int):
        super(GaussianModel, self).__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._used = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_use_flags(self):
        return self._used

    def update_use_flags(self):
        self._used = torch.logical_or(self._used, self._xyz.grad.abs().sum(-1) > 0)

    def reset_use_flags(self):
        self._used[:] = False

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pts(self, pts, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.from_numpy(pts).float().cuda()
        fused_color = RGB2SH(torch.rand((pts.shape[0], 3)).float().cuda())
        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._used = torch.ones((self.get_xyz.shape[0]), device="cuda").bool()

    def load_state_dict(self, state_dict, optimizer, global_step, upSH):
        # Update optimizer params
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] not in self.optimizable_names:
                continue

            assert len(group["params"]) == 1
            state_name = "_" + group["name"][3:]
            group["params"][0] = nn.Parameter(state_dict[state_name].float().cuda().requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors[self.optimizable_names[0]]
        self._features_dc = optimizable_tensors[self.optimizable_names[1]]
        self._features_rest = optimizable_tensors[self.optimizable_names[2]]
        self._opacity = optimizable_tensors[self.optimizable_names[3]]
        self._scaling = optimizable_tensors[self.optimizable_names[4]]
        self._rotation = optimizable_tensors[self.optimizable_names[5]]

        self.active_sh_degree = 0
        if global_step > upSH:
            self.active_sh_degree = min(int(global_step // upSH), self.max_sh_degree)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._used = torch.ones((self.get_xyz.shape[0]), device="cuda").bool()

    def trainable_params(self, cfg):
        self.percent_dense = cfg["gs.percent_dense"]

        l = [
            {
                "params": [self._xyz],
                "lr": cfg["gs.position_lr_init"] * self.spatial_lr_scale,
                "name": self.optimizable_names[0],
            },
            {"params": [self._features_dc], "lr": cfg["gs.feature_lr"], "name": self.optimizable_names[1]},
            {"params": [self._features_rest], "lr": cfg["gs.feature_lr"] / 20.0, "name": self.optimizable_names[2]},
            {"params": [self._opacity], "lr": cfg["gs.opacity_lr"], "name": self.optimizable_names[3]},
            {"params": [self._scaling], "lr": cfg["gs.scaling_lr"], "name": self.optimizable_names[4]},
            {"params": [self._rotation], "lr": cfg["gs.rotation_lr"], "name": self.optimizable_names[5]},
        ]

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg["gs.position_lr_init"] * self.spatial_lr_scale,
            lr_final=cfg["gs.position_lr_final"] * self.spatial_lr_scale,
            lr_delay_mult=cfg["gs.position_lr_delay_mult"],
            max_steps=cfg["gs.position_lr_max_steps"],
        )

        return l

    def update_learning_rate(self, iteration, optimizer):
        for param_group in optimizer.param_groups:
            if param_group["name"] == self.optimizable_names[0]:
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_gaussian(self, offsets, bs):
        """Move 3d gaussians to self._new_xyz, with property no changes"""
        deformed_gaussians = []
        for i in range(bs):
            new_attrs = {
                "_xyz": self._xyz,
                "_features_dc": self._features_dc,
                "_features_rest": self._features_rest,
                "_scaling": self._scaling,
                "_rotation": self._rotation,
                "_opacity": self._opacity,
                "max_radii2D": self.max_radii2D,
            }

            for attr in offsets:
                name = "_" + attr
                ori_v = getattr(self, name)
                new_attrs[name] = ori_v + offsets[attr][i].reshape(ori_v.shape)

            new_gaussian = GaussianModel(self.max_sh_degree)
            new_gaussian._xyz = new_attrs["_xyz"]
            new_gaussian._features_dc = new_attrs["_features_dc"]
            new_gaussian._features_rest = new_attrs["_features_rest"]
            new_gaussian._scaling = new_attrs["_scaling"]
            new_gaussian._rotation = new_attrs["_rotation"]
            new_gaussian._opacity = new_attrs["_opacity"]
            new_gaussian.max_radii2D = new_attrs["max_radii2D"]
            deformed_gaussians.append(new_gaussian)

        return deformed_gaussians

    def rigid_deform(self, R, t):
        """apply R and t to Gaussians

        Args:
            R (Bx3x3): rigid rotations, always using neck pose.
            t (Bx3)
        """
        B = R.shape[0]
        # self._xyz: Nx3
        xyz = self._xyz[None].expand(B, -1, -1)

        translation = deepcopy(t)
        if len(translation.shape) == 2:
            translation = translation[:, None]
        R_T = R.transpose(1, 2)

        return xyz.bmm(R_T) + translation

    def binding(self, verts, faces):
        """Binding to a flame mesh, required to perform a non-rigid alignment

        Args:
            verts (_type_): _description_
            faces (_type_): _description_
        """
        pts = self.get_xyz.detach().cpu().numpy()  # [N, 3]
        self.binding_mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
        # TODO: SLOW !! Try "nearby_faces" later
        closest_pts, dist, face_id = trimesh.proximity.closest_point(self.binding_mesh, pts)
        # candidates = trimesh.proximity.nearby_faces(self.binding_mesh, pts)

        self.parent = face_id

    def transfer(self, verts, faces):
        # compute affine matrices
        binding_verts = self.binding_mesh.vertices
        binding_faces = self.binding_mesh.faces
        binding_face_verts = binding_verts[binding_faces]  # [F, 3, 3]
        face_ids = np.unique(self.parent)

        src_face_verts = torch.from_numpy(binding_face_verts[face_ids]).float().cuda()  # [F_s, 3, 3]
        tgt_face_verts = verts[faces[face_ids]]  # [F_s, 3, 3]

        aff_mats = estimate_rigid_between_flame_faces2(src_face_verts, tgt_face_verts)

        aff_dict = {}
        for i, f_id in enumerate(face_ids):
            aff_dict[f_id] = aff_mats[i]

        # transfer GS
        n_pts = self.get_xyz.shape[0]
        transfer_mats = torch.zeros((n_pts, 3, 4)).float().cuda()
        for i in range(n_pts):
            transfer_mats[i] = aff_dict[self.parent[i]]
        homo_xyz = torch.cat([self.get_xyz, torch.ones((n_pts, 1)).cuda()], dim=-1)

        self._xyz.data = torch.bmm(transfer_mats, homo_xyz.unsqueeze(-1)).squeeze(-1)
        # expand along with normals
        tgt_face_normals = compute_face_normals(verts[faces[self.parent]])
        self._xyz.data += tgt_face_normals * 0.0035

        rot = self.rotation_activation(self._rotation)  # [N, 4]
        transfer_rots = self.rotation_activation(quat_xyzw_to_wxyz(rotmat_to_unitquat(transfer_mats[..., :3])))
        self._rotation.data = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(transfer_rots), quat_wxyz_to_xyzw(rot)))

    def cat_tensors_to_optimizer(self, tensors_dict, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] not in self.optimizable_names:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer
    ):
        d = {
            self.optimizable_names[0]: new_xyz,
            self.optimizable_names[1]: new_features_dc,
            self.optimizable_names[2]: new_features_rest,
            self.optimizable_names[3]: new_opacities,
            self.optimizable_names[4]: new_scaling,
            self.optimizable_names[5]: new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizer)
        self._xyz = optimizable_tensors[self.optimizable_names[0]]
        self._features_dc = optimizable_tensors[self.optimizable_names[1]]
        self._features_rest = optimizable_tensors[self.optimizable_names[2]]
        self._opacity = optimizable_tensors[self.optimizable_names[3]]
        self._scaling = optimizable_tensors[self.optimizable_names[4]]
        self._rotation = optimizable_tensors[self.optimizable_names[5]]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, optimizer, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizer
        )
        # update use flags
        new_use_flags = self._used[selected_pts_mask].repeat(N)
        self._used = torch.cat([self._used, new_use_flags])

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter, optimizer)

    def _prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] not in self.optimizable_names:
                continue

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizer):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer)

        self._xyz = optimizable_tensors[self.optimizable_names[0]]
        self._features_dc = optimizable_tensors[self.optimizable_names[1]]
        self._features_rest = optimizable_tensors[self.optimizable_names[2]]
        self._opacity = optimizable_tensors[self.optimizable_names[3]]
        self._scaling = optimizable_tensors[self.optimizable_names[4]]
        self._rotation = optimizable_tensors[self.optimizable_names[5]]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._used = self._used[valid_points_mask]

    def densify_and_clone(self, grads, grad_threshold, scene_extent, optimizer):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer
        )

        # update use flags
        new_use_flags = self._used[selected_pts_mask]
        self._used = torch.cat([self._used, new_use_flags])

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, optimizer):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, optimizer)
        self.densify_and_split(grads, max_grad, extent, optimizer)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, optimizer)

        torch.cuda.empty_cache()

    def replace_tensor_to_optimizer(self, tensor, name, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] not in self.optimizable_names:
                continue

            if group["name"] == name:
                stored_state = optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self, optimizer):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, self.optimizable_names[3], optimizer)
        self._opacity = optimizable_tensors[self.optimizable_names[3]]

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
