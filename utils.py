import math
import os
from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.ops.knn import knn_points
from torch.autograd import Variable

from networks.gshair.gs.gaussian_model import GaussianModel
from networks.gshair.gs.gaussian_utils import eval_sh


def render_list(cfg, viewpoint_cameras, pcs, bg_color, scaling_modifier=1.0, override_color=None):
    assert len(viewpoint_cameras) == len(
        pcs
    ), "utils/render_list function: viewpoint_cameras should have the same length as deformed gaussians."

    reduce_res = {
        "render": [],
        "viewspace_points": [],
        "visibility_filter": [],
        "radii": [],
        "depth": [],
        "silhoutte": [],
        "near_z": [],
        "near_z2": [],
        "near_z3": [],
        "num_gs": [],
    }
    for i in range(len(pcs)):
        res = render(cfg, viewpoint_cameras[i], pcs[i], bg_color, scaling_modifier, override_color)
        reduce_res["render"].append(res["render"])
        reduce_res["viewspace_points"].append(res["viewspace_points"])
        reduce_res["visibility_filter"].append(res["visibility_filter"])
        reduce_res["radii"].append(res["radii"])
        reduce_res["depth"].append(res["depth"][0])
        reduce_res["silhoutte"].append(res["silhoutte"][0])
        reduce_res["near_z"].append(res["near_z"][0])
        reduce_res["near_z2"].append(res["near_z2"][0])
        reduce_res["near_z3"].append(res["near_z3"][0])
        reduce_res["num_gs"].append(res["num_gs"][0])

    for k, v in reduce_res.items():
        if k != "viewspace_points":
            reduce_res[k] = torch.stack(v, dim=0)

    return reduce_res


def render(cfg, viewpoint_camera, pc: GaussianModel, bg_color, scaling_modifier=1.0, override_color=None):
    """
    Render the gaussians.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.img_h),
        image_width=int(viewpoint_camera.img_w),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=cfg["pipe.debug"],
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if cfg["pipe.compute_cov3D_python"]:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if cfg["pipe.convert_SHs_python"]:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha, near_z, near_z2, near_z3, num_gs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        depth_scale_factor=1000.0,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "silhoutte": alpha,
        "near_z": near_z,
        "near_z2": near_z2,
        "near_z3": near_z3,
        "num_gs": num_gs,
    }


def color_mask(mask, color=[1.0, 0.0, 0.0], bg_white=False):
    if len(mask.shape) == 2:
        mask = mask[..., None].repeat(3, axis=-1)
    elif len(mask.shape) == 3:
        if mask.shape[-1] == 1:
            mask = mask.repeat(3, axis=-1)
    else:
        print("[color_mask] Not supported shape. {}".format(mask.shape))
    mask = mask.astype(np.float32)

    colored_mask = np.zeros_like(mask)
    colored_mask[..., 0] = mask[..., 0] * color[0]
    colored_mask[..., 1] = mask[..., 1] * color[1]
    colored_mask[..., 2] = mask[..., 2] * color[2]

    if bg_white:
        colored_mask[~mask.astype(np.bool_)] = 1.0

    return colored_mask


def check_tensor_in_list(tensor, li):
    return any([(tensor == item).all() for item in li])


def split_verts_for_unique_uv(V, uvs, faces_uvs, faces):
    """Split mesh verts to make verts and uvs 1<->1 match.

    Args:
        uvs (_type_): uv coords, N_uvx2
        faces_uvs (_type_): uv_ids in faces, N_fx3
        faces (_type_): vert_ids in faces, N_fx3
    """
    new_faces = faces.clone()
    extra_verts_ids = []

    vert_uvs = {}
    for i, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        uv1, uv2, uv3 = int(faces_uvs[i][0]), int(faces_uvs[i][1]), int(faces_uvs[i][2])

        if v1 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v1].items():
                if (uvs[uv1] == uv).all():
                    new_faces[i][0] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][0] = extra_id
                extra_verts_ids.append(v1)
                vert_uvs[v1][extra_id] = uvs[uv1]
        else:
            vert_uvs[v1] = {v1: uvs[uv1]}

        if v2 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v2].items():
                if (uvs[uv2] == uv).all():
                    new_faces[i][1] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][1] = extra_id
                extra_verts_ids.append(v2)
                vert_uvs[v2][extra_id] = uvs[uv2]
        else:
            vert_uvs[v2] = {v2: uvs[uv2]}

        if v3 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v3].items():
                if (uvs[uv3] == uv).all():
                    new_faces[i][2] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][2] = extra_id
                extra_verts_ids.append(v3)
                vert_uvs[v3][extra_id] = uvs[uv3]
        else:
            vert_uvs[v3] = {v3: uvs[uv3]}

    return extra_verts_ids, new_faces


def vert_uvs(V, uvs, faces_uvs, faces):
    """Compute uv coords for each vertex

    Args:
        V (int): num of vertices
        uvs (_type_): uv coords, N_uvx2
        faces_uvs (_type_): uv_ids in faces, N_fx3
        faces (_type_): vert_ids in faces, N_fx3
    """
    vert_uvs = np.ones((V, 2)) * -1
    for i, face in enumerate(faces):
        v1, v2, v3 = face
        uv1, uv2, uv3 = faces_uvs[i]

        vert_uvs[v1] = uvs[uv1]
        vert_uvs[v2] = uvs[uv2]
        vert_uvs[v3] = uvs[uv3]

    if (vert_uvs.sum(-1) < 0).sum():
        print("[Waring] Function 'vert_uvs': there are some verts that have no uv coords.")

    return vert_uvs.astype(np.float32)


def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def update_lambda(lambda_start, lambda_slope, lambda_end, global_step, interval):
    res = lambda_start
    if lambda_slope > 0:
        res = min(lambda_end, global_step // interval * lambda_slope + lambda_start)
    elif lambda_slope < 0:
        res = max(lambda_end, global_step // interval * lambda_slope + lambda_start)
    return res


def visPositionMap(savepath, posMap, savepng=True, bg_mask=None, bg_color=np.array([0, 0, 0])):
    """
    Args:
        savepath:   str, path to save
        posMap:     256x256x3, postion map of mesh
    """
    H, W, _ = posMap.shape

    verts = posMap.reshape(-1, 3)
    mmin = verts.min(axis=0)
    mmax = verts.max(axis=0)

    normalized = (verts - mmin) / (mmax - mmin)
    color = normalized.reshape(H, W, 3)
    if bg_mask is not None:
        color[bg_mask] = bg_color
    if savepng:
        cv2.imwrite(savepath, color * 255)
    return color


def visDepthMap(savepath, depth_map):
    """
    Args:
        savepath (str): path to save
        depth_map (float): HxW, rendered depth map
    """
    mmin = depth_map[depth_map > 0.0].min()
    mmax = depth_map[depth_map > 0.0].max()

    normalized = (depth_map - mmin) / (mmax - mmin)
    normalized[normalized < 0] = 1
    cv2.imwrite(savepath, (1 - normalized) * 255)


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


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = True


def visimg(filepath, img):
    cv2.imwrite(filepath, img[0].detach().cpu().numpy() * 255)


def img3channel(img):
    """make the img to have 3 channels"""
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def img2mask(img, thre=128, mode="greater"):
    """mode: greater/greater-equal/less/less-equal/equal"""
    if mode == "greater":
        mask = (img > thre).astype(np.float32)
    elif mode == "greater-equal":
        mask = (img >= thre).astype(np.float32)
    elif mode == "less":
        mask = (img < thre).astype(np.float32)
    elif mode == "less-equal":
        mask = (img <= thre).astype(np.float32)
    elif mode == "equal":
        mask = (img == thre).astype(np.float32)
    else:
        raise NotImplementedError

    mask = img3channel(mask)

    return mask


def depth_map2normals(depth_map, bg_white=True):
    """convert a depth map to normal map

    Args:
        depth_map: BxHxW
    """
    zy, zx = torch.gradient(depth_map, dim=(1, 2))
    mask = (zy == 0) * (zx == 0)

    normals = torch.stack((-zx, -zy, torch.ones_like(depth_map)), dim=-1)
    res = F.normalize(normals, dim=-1)
    if bg_white:
        res[mask] = 1.0
    else:
        res[mask] = -1.0

    return res


def edge_loss(template_verts, target_mesh, v_filter=None):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    N = len(target_mesh)

    target_verts = target_mesh.verts_packed()
    target_edges = target_mesh.edges_packed()  # [sum(E_n), 2]
    target_verts_edges = target_verts[target_edges]  # [sum(E_n), 2, 3]
    target_edges_len = (target_verts_edges[:, 0] - target_verts_edges[:, 1]).norm(dim=1, p=2)  # [sum(E_n)]

    verts = template_verts.reshape(-1, 3)
    template_verts_edges = verts[target_edges]  # [sum(E_n), 2, 3]
    template_edges_len = (template_verts_edges[:, 0] - template_verts_edges[:, 1]).norm(dim=1, p=2)  # [sum(E_n)]

    if v_filter is not None:
        num_edge_per_mesh = target_edges.shape[0] // N
        first_edges = target_edges[:num_edge_per_mesh]  # [E_n, 2]
        mask = (
            (
                sum(first_edges[:, 0] == v for v in v_filter).bool()
                * sum(first_edges[:, 1] == v for v in v_filter).bool()
            )
            .repeat(N)
            .bool()
        )

        loss = torch.abs((template_edges_len - target_edges_len) * ~mask).mean()
    else:
        loss = torch.abs(template_edges_len - target_edges_len).mean()

    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def full_aiap_loss(gs_can, gs_obs, n_neighbors=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=n_neighbors, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov


def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(
            x_canonical.unsqueeze(0), x_canonical.unsqueeze(0), K=n_neighbors + 1, return_sorted=True
        )
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:, 0, 1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:, 0, 1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded. B x C x ...
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    if num_encoding_functions == 0:
        return tensor
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def params_with_lr(params_list, lr, label=""):
    """add 'label' and 'lr' to params

    Args:
        params_list (_type_): params list
        lr (_type_): either float or list (with the same length as params_list)
    """
    if label is None:
        label = ""
    if len(label) > 0:
        label = label + "."

    params_lr_list = []
    if isinstance(lr, float):
        for i, param in enumerate(params_list):
            params_lr_list.append({"params": [param[1]], "lr": lr, "name": label + param[0]})
    elif len(lr) > 1 and len(lr) == len(params_list):
        for i, param in enumerate(params_list):
            params_lr_list.append({"params": [param[1]], "lr": lr[i], "name": label + param[0]})
    return params_lr_list


def estimate_rigid(pts1, pts2):
    """estimate affine matrix from two sets of points, need correspondence.

    Args:
        pts1 (B, N, 3): _description_
        pts2 (B, N, 3): _description_
    """
    B = pts1.shape[0]
    pts1_np, pts2_np = pts1, pts2
    if isinstance(pts1, torch.Tensor):
        pts1_np = pts1_np.detach().cpu().numpy()
    if isinstance(pts2, torch.Tensor):
        pts2_np = pts2_np.detach().cpu().numpy()

    res = torch.zeros((B, 3, 4)).cuda()
    for i in range(B):
        aff_mat = cv2.estimateAffine3D(pts1_np[i], pts2_np[i])
        res[i] = torch.from_numpy(aff_mat[1]).cuda()

    return res


def restore_model(model_path, hairwrapper, facewrapper, optimizer, logger):
    """Restore checkpoint

    Args:
        model_path (str): checkpoint path
        models (dict): model dict
        optimizer (optimizer): torch optimizer
        logger (logger): logger
    """
    if model_path is None:
        if logger:
            logger.info("Not using pre-trained model...")
        return 1

    assert os.path.exists(model_path), "Model %s does not exist!"

    logger.info("Loading ckpts from {} ...".format(model_path))
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())

    current_epoch = state_dict["epoch"] if "epoch" in state_dict else 1
    global_step = state_dict["global_step"] if "global_step" in state_dict else 0
    stage = state_dict["stage"] if "stage" in state_dict else None
    stage = "joint" if stage == "hair" else stage
    stage_step = state_dict["stage_step"] if "stage_step" in state_dict else 0

    hairwrapper.restore_models(state_dict, optimizer, global_step, logger)
    facewrapper.restore_models(state_dict, logger)

    return current_epoch, global_step, stage, stage_step


class CUDA_Timer(object):
    def __init__(self, label, logger=None, valid=True, warmup_steps=10):
        self.valid = valid
        if not valid:
            return
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.label = label
        self.logger = logger
        self.counter = 0
        self.val = 0.0
        self.warmup_steps = warmup_steps

    def start(self, step):
        if self.valid and step > self.warmup_steps:
            self.starter.record()

    def end(self, step):
        if self.valid and step > self.warmup_steps:
            self.ender.record()
            self._update_val()

    def _update_val(self):
        torch.cuda.synchronize()
        time = self.starter.elapsed_time(self.ender)
        self.val = self.val * self.counter + time
        self.counter += 1
        self.val /= self.counter

        if self.logger:
            self.logger.info("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")
        else:
            print("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")

        # reset timer
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def __str__(self):
        if self.valid:
            fmtstr = "[{}] " + "{avg " + str(self.val) + "ms}"
        else:
            fmtstr = "[{}] " + "\{avg -1ms\}"
        return fmtstr.format(self.label)

    def __enter__(self):
        if self.valid:
            self.starter.record()

    def __exit__(self, exc_type, exc_value, tb):
        if self.valid:
            self.ender.record()
            torch.cuda.synchronize()
            if self.logger:
                self.logger.info(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))
            else:
                print(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
