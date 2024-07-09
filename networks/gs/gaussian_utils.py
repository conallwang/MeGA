import math

import cv2
import numpy as np
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24]
                    )
    return result


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / (norm[:, None] + 1e-10)

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def sign(x):
    if x >= 0.0:
        return 1
    else:
        return -1


def build_quaternion(R):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = (
        R[:, 0, 0],
        R[:, 0, 1],
        R[:, 0, 2],
        R[:, 1, 0],
        R[:, 1, 1],
        R[:, 1, 2],
        R[:, 2, 0],
        R[:, 2, 1],
        R[:, 2, 2],
    )

    q = torch.zeros((R.size(0), 4), device="cuda")
    q0 = torch.sqrt(max((r11 + r22 + r33 + 1.0) / 4.0, 0.0))
    q1 = torch.sqrt(max((r11 - r22 - r33 + 1.0) / 4.0, 0.0))
    q2 = torch.sqrt(max((-r11 + r22 - r33 + 1.0) / 4.0, 0.0))
    q3 = torch.sqrt(max((-r11 - r22 + r33 + 1.0) / 4.0, 0.0))
    if q0 >= q1 and q0 >= q2 and q0 >= q3:
        q0 *= 1.0
        q1 *= sign(r32 - r23)
        q2 *= sign(r13 - r31)
        q3 *= sign(r21 - r12)
    elif q1 >= q0 and q1 >= q2 and q1 >= q3:
        q0 *= sign(r32 - r23)
        q1 *= 1.0
        q2 *= sign(r21 + r12)
        q3 *= sign(r13 + r31)
    elif q2 >= q0 and q2 >= q1 and q2 >= q3:
        q0 *= sign(r13 - r31)
        q1 *= sign(r21 + r12)
        q2 *= 1.0
        q3 *= sign(r32 + r23)
    elif q3 >= q0 and q3 >= q1 and q3 >= q2:
        q0 *= sign(r21 - r12)
        q1 *= sign(r31 + r13)
        q2 *= sign(r32 + r23)
        q3 *= 1.0
    else:
        print("convert error!")

    norm = torch.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)


def dot(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20):
    return x / length(x, eps)


def gen_4pts_for_rigid(face):
    v0, v1, v2 = face[0], face[1], face[2]
    z_dir = safe_normalize(np.cross(v1 - v0, v2 - v0))
    v3 = v0 + z_dir * 0.01

    return np.stack([v0, v1, v2, v3], axis=0).astype(np.float32)


def estimate_rigid_between_flame_faces(src_faces, tgt_faces):
    """estimate affine matrix from two corresponding flame faces

    Args:
        src_face (N, 3, 3): coordinates of three source verts
        tgt_face (N, 3, 3): coordinates of three target verts
    """
    assert src_faces.shape[0] == tgt_faces.shape[0]
    N = src_faces.shape[0]

    res = np.zeros((N, 3, 4))
    for i in range(N):
        src_pts = gen_4pts_for_rigid(src_faces[i])
        tgt_pts = gen_4pts_for_rigid(tgt_faces[i])
        res[i] = cv2.estimateAffine3D(src_pts, tgt_pts)

    return res


def build_transform_matrix_for_faces(faces):
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    x = safe_normalize(v1 - v0)
    z = safe_normalize(torch.cross(x, v2 - v0, dim=-1))
    y = safe_normalize(torch.cross(z, x, dim=-1))

    trans_mat34 = torch.cat([x[..., None], y[..., None], z[..., None], v0[..., None]], dim=-1)  # [N, 3, 4]
    homo = torch.tensor([[[0, 0, 0, 1]]]).expand(trans_mat34.shape[0], -1, -1).float().cuda()

    trans_mat44 = torch.cat([trans_mat34, homo], dim=-2)
    return trans_mat44


def estimate_rigid_between_flame_faces2(src_faces, tgt_faces):
    """estimate affine matrix from two corresponding flame faces

    Args:
        src_face (N, 3, 3): coordinates of three source verts
        tgt_face (N, 3, 3): coordinates of three target verts
    """
    src_trans_mat44 = build_transform_matrix_for_faces(src_faces)
    tgt_trans_mat44 = build_transform_matrix_for_faces(tgt_faces)

    return torch.bmm(tgt_trans_mat44, torch.inverse(src_trans_mat44))[:, :3]


def compute_face_normals(faces):
    """Compute face normals

    Args:
        faces (N, 3, 3): _description_
    """
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    return face_normals
