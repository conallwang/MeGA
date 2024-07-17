# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_obj

from utils import split_verts_for_unique_uv, vert_uvs

from .lbs import blend_shapes, edge_subdivide, lbs, vertices2joints, vertices2landmarks


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class FlameHead(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super().__init__()

        self.n_shape_params = config["flame.n_shape"]
        self.n_expr_params = config["flame.n_expr"]
        add_teeth = config["flame.add_teeth"]

        with open(config["flame.model_path"], "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer("_v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        self.register_buffer(
            "_faces", torch.from_numpy(np.load("./assets/flame/flame_faces.npy")).long(), persistent=False
        )
        self.register_buffer(
            "_vertex_uvs", torch.from_numpy(np.load("./assets/flame/vertex_uvs.npy")), persistent=False
        )
        self.register_buffer(
            "_faces_uvs", torch.from_numpy(np.load("./assets/flame/faces_uvs.npy")).long(), persistent=False
        )
        if add_teeth:
            self._faces, self._faces_uvs = self._faces[:9976], self._faces_uvs[:9976]

        self.register_buffer(
            "_verts_uvs",
            torch.from_numpy(vert_uvs(self._v_template.shape[0], self._vertex_uvs, self._faces_uvs, self._faces)),
            persistent=False,
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, : self.n_shape_params], shapedirs[:, :, 300 : 300 + self.n_expr_params]],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer("J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(config["flame.lmk_embedding_path"], allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # parts
        # load vertex lists corresponding to parts of the face
        with open(config["flame.parts_path"], "rb") as f:
            parts = pickle.load(f, encoding="latin1")
            self._parts = OrderedDict()
            self._parts["backneck"] = torch.from_numpy(np.load("./assets/flame/backneck_idcs.npy")).long()
            self._parts["shrink_scalp"] = torch.from_numpy(np.load("./assets/flame/shrink_scalp_idcs.npy")).long()
            for key in sorted(parts.keys()):
                self._parts[key] = torch.tensor(parts[key])
        scalp_idcs = list(set(self._parts["scalp"].numpy()) - set(self._parts["backneck"].numpy()))
        self._parts["scalp"] = torch.tensor(scalp_idcs)

        # upsample
        self._ignore_faces = np.load(config["flame.ignore_faces"])
        upsample_regions = dict(all=config["flame.subdivision"], mouth=config["flame.subdivision.mouth"])
        self._upsample_regions(upsample_regions)

        if add_teeth:
            self.add_teeth()

        # split verts
        self.extra_verts_ids, split_faces = split_verts_for_unique_uv(
            self._v_template.shape[0], self._vertex_uvs, self._faces_uvs, self._faces
        )
        self.register_buffer("split_faces", split_faces, persistent=False)
        self.register_buffer(
            "split_verts_uvs",
            torch.from_numpy(
                vert_uvs(
                    self._v_template.shape[0] + len(self.extra_verts_ids),
                    self._vertex_uvs,
                    self._faces_uvs,
                    self.split_faces,
                )
            ),
            persistent=False,
        )

    @property
    def v_template(self):
        # return self._v_template_filtered if len(self._ignore_faces) != 0 else self._v_template
        return self._v_template

    @property
    def faces(self):
        # return self._faces_filtered if len(self._ignore_faces) != 0 else self._faces
        return self._faces

    @property
    def faces_uvs(self):
        # return self._faces_uvs_filtered if len(self._ignore_faces) != 0 else self._faces_uvs
        return self._faces_uvs

    @property
    def vertex_uvs(self):
        # return self._vertex_uvs[self._uv_filter] if len(self._ignore_faces) != 0 else self._vertex_uvs
        return self._vertex_uvs

    @property
    def verts_uvs(self):
        # return self._vertex_uvs[self._uv_filter] if len(self._ignore_faces) != 0 else self._vertex_uvs
        return self._verts_uvs

    def get_body_parts(self):
        """
        returns body part vertex index list dictionary
        - keys: body part names
        - values: list of vertex idcs

        :return:
        """
        return self._parts

    def get_body_part_vert_idcs(self, *parts):
        """
        returns vertex indices that are part of list of specified body parts (strings). If '-' is added at beginning of
        body part name (e.g. '-face'), excludes body part.

        ATTENTION: part list is not commutative if there are overlaps between the
        body parts and at least one body part has prefix '-'

        :param parts:
        :return: torch tensor with vertex idcs
        """
        part_dict = self.get_body_parts()
        ret = set()
        for p in parts:
            if p[0] == "-":
                ret = ret - set(part_dict[p[1:]].tolist())
            else:
                ret = set.union(ret, part_dict[p].tolist())
        ret = torch.tensor(sorted(ret), device=self._v_template.device)
        # sorting is most likely not necessary but done here to ensure deterministic order of idcs
        return ret

    def faces_of_verts(self, vert_idcs, return_face_idcs=False):
        """
        calculates face tensor of shape F x 3 with face spanned by vertices in flame mesh
        all vertices of the faces returned by this function contain only vertices from vert_idcs
        :param vert_idcs:
        :return_face_idcs: if True, also returns list of relevant face idcs
        :return:
        """
        all_faces = self._faces
        vert_idcs = vert_idcs.to(all_faces.device)
        vert_faces = []
        face_idcs = []
        for i, f in enumerate(all_faces):
            keep_face = True
            for idx in f:
                if not idx in vert_idcs:
                    keep_face = False
            if keep_face:
                vert_faces.append(f)
                face_idcs.append(i)
        vert_faces = torch.stack(vert_faces)

        if return_face_idcs:
            return vert_faces, face_idcs

        return vert_faces

    def _upsample_regions(self, upsample_regions: OrderedDict):
        """
        :param upsample_regions: dict
                                "face_region": upsample_factor (int)
        :return:
        """

        if len(upsample_regions) == 0:
            return

        # densify specific regions
        for key, val in upsample_regions.items():
            if key == "all":
                continue

            for i in range(val):
                vert_idcs = self.get_body_part_vert_idcs(*[part for part in self._parts if key in part])
                face_idcs = self.faces_of_verts(vert_idcs, return_face_idcs=True)[1]
                self._upsample_faces(face_idcs)

        if "all" in upsample_regions and upsample_regions["all"] > 0:
            for i in range(upsample_regions["all"]):
                upsample_ignore = [part for part in self._parts if "eyeball" in part]
                # upsample_ignore.append("scalp")

                eye_scalp_vert_idcs = self.get_body_part_vert_idcs(*upsample_ignore)
                eye_scalp_face_idcs = self.faces_of_verts(eye_scalp_vert_idcs, return_face_idcs=True)[1]
                face_idcs = list(set(range(len(self._faces))) - set(self._ignore_faces) - set(eye_scalp_face_idcs))
                self._upsample_faces(face_idcs)

    def _upsample_faces(self, face_idcs):
        """
        splits every given face into 3 subfaces and performs necessary adjustments to flame model
        :param face_idcs: index list of length F'
        :return:
        """
        face_idcs = list(np.unique(face_idcs))
        n_v = len(self._v_template)
        n_t = len(self._vertex_uvs)
        n_f = len(self._faces)

        verts, uvs, faces, uv_faces, edges, uv_edges = edge_subdivide(
            vertices=self._v_template.cpu().numpy(),
            uvs=self._vertex_uvs.cpu().numpy(),
            faces=self._faces[face_idcs].cpu().numpy(),
            uvfaces=self._faces_uvs[face_idcs].cpu().numpy(),
        )
        faces = torch.cat(
            (self._faces, torch.tensor(faces[len(face_idcs) :], dtype=self._faces.dtype, device=self._faces.device)),
            dim=0,
        )
        uv_faces = torch.cat(
            (
                self._faces_uvs,
                torch.tensor(uv_faces[len(face_idcs) :], dtype=self._faces_uvs.dtype, device=self._faces_uvs.device),
            ),
            dim=0,
        )
        n_edges = len(edges)

        self.register_buffer("_v_template", torch.tensor(verts, dtype=self._v_template.dtype), persistent=False)
        self.register_buffer("_vertex_uvs", torch.tensor(uvs, dtype=self._vertex_uvs.dtype), persistent=False)
        self.register_buffer("_faces", faces, persistent=False)
        self.register_buffer("_faces_uvs", uv_faces, persistent=False)
        self.register_buffer(
            "_verts_uvs", torch.from_numpy(vert_uvs(verts.shape[0], uvs, uv_faces, faces)), persistent=False
        )

        # calculate new blendshapes
        new_shapedirs = self.shapedirs[edges]  # n_edges x 2 x 3 x 400
        new_shapedirs = new_shapedirs.mean(dim=1)  # n_edges x 3 x 400
        new_posedirs = self.posedirs.permute(1, 0).view(n_v, 3, 36)  # V x 3 x 36
        new_posedirs = new_posedirs[edges]  # n_edges x 2 x 3 x 36
        new_posedirs = new_posedirs.mean(dim=1)  # n_edges x 3 x 36
        new_posedirs = new_posedirs.view(n_edges * 3, 36).permute(1, 0)  # 36 x n_edges * 3
        self.register_buffer("shapedirs", torch.cat((self.shapedirs, new_shapedirs), dim=0), persistent=False)
        self.register_buffer("posedirs", torch.cat((self.posedirs, new_posedirs), dim=1), persistent=False)

        # calculate new lbs components
        new_J_regressor = torch.zeros(5, n_edges).to(self.J_regressor.dtype).to(self.J_regressor.device)
        new_lbs_weights = self.lbs_weights[edges]  # n_edges x 2 x 5
        new_lbs_weights = new_lbs_weights.mean(dim=1)  # n_edges x 5
        self.register_buffer("J_regressor", torch.cat((self.J_regressor, new_J_regressor), dim=1), persistent=False)
        self.register_buffer("lbs_weights", torch.cat((self.lbs_weights, new_lbs_weights), dim=0), persistent=False)

        # update body parts
        for part, idcs in self._parts.items():
            new_vert_idcs = []
            for i in range(len(edges)):
                if edges[i, 0] in idcs and edges[i, 1] in idcs:
                    new_vert_idcs.append(i + n_v)
            if new_vert_idcs:
                new_vert_idcs = torch.tensor(new_vert_idcs, dtype=idcs.dtype, device=idcs.device)
                self._parts[part] = torch.cat((idcs, new_vert_idcs), dim=0)

        # ignore new faces if parent face was to be ignored
        ignored_upsampled_faces = list(set.intersection(set(self._ignore_faces), set(face_idcs)))
        if ignored_upsampled_faces:
            # each parent face produces 4 child faces that are stacked on top of face stack. child faces always stick
            # together (see edge_subdivide()). So idcs of child face of parent face with index i are given by:
            # c0 = n_faces + i * 4 + 0, c1 = n_faces + i*4 + 1, ...
            ignored_upsampled_faces_idcs = np.array([face_idcs.index(f) for f in ignored_upsampled_faces])
            ignored_upsampled_faces_idcs = n_f + ignored_upsampled_faces_idcs * 4
            new_ignored_faces = np.concatenate(
                (
                    ignored_upsampled_faces_idcs,
                    ignored_upsampled_faces_idcs + 1,
                    ignored_upsampled_faces_idcs + 2,
                    ignored_upsampled_faces_idcs + 3,
                ),
                axis=0,
            )
            new_ignored_faces = list(new_ignored_faces)
        else:
            new_ignored_faces = []

        # ignore old faces
        self._ignore_faces = list(set.union(set(self._ignore_faces), set(face_idcs), set(new_ignored_faces)))

    def add_teeth(self):
        # get reference vertices from lips
        vid_lip_outside_ring_upper = torch.tensor(
            [1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830]
        )
        vid_lip_outside_ring_lower = torch.tensor(
            [1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712]
        )

        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= mean_dist * 1.5  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]]) * 0.1
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]]) * 0.1
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.4  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # depth
        depth = mean_dist * 2.0
        # upper, back
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_edge_back[:, 2] -= depth  # how depth the teeth are

        # lower, back
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_edge_back[:, 2] -= depth  # how thick the teeth are

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat(
            [
                v_teeth_upper_root,  # num_verts_orig + 0-14
                v_teeth_upper_edge,  # num_verts_orig + 15-29
                v_teeth_upper_edge_back,  # num_verts_orig + 30-44
                v_teeth_lower_edge_back,  # num_verts_orig + 45-59
                v_teeth_lower_edge,  # num_verts_orig + 60-74
                v_teeth_lower_root,  # num_verts_orig + 75-89
            ],
            dim=0,
        )
        num_verts_teeth = v_teeth.shape[0]
        self._v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(45, 60) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(60, 75) + num_verts_orig
        vid_teeth_lower_root = torch.arange(75, 90) + num_verts_orig

        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_edge_back], dim=0)
        self.vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)

        # construct uv vertices for teeth
        # a rectangular area in the uv space
        u = torch.linspace(0.348, 0.085, 15)
        v = torch.linspace(0.757, 0.791, 6)
        uv = (
            torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)
        )  # (#num_teeth, 2)
        num_verts_uv_orig = self.vertex_uvs.shape[0]
        num_verts_uv_teeth = uv.shape[0]
        self._vertex_uvs = torch.cat([self._vertex_uvs, uv], dim=0)
        self._verts_uvs = torch.cat([self._verts_uvs, uv], dim=0)

        # shapedirs copy from lips
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (
            self.shapedirs[vid_lip_outside_ring_upper, :, : self.n_shape_params]
            + self.shapedirs[vid_lip_outside_ring_lower, :, : self.n_shape_params]
        ) / 2
        self.shapedirs[vid_teeth_upper_root, :, : self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, : self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, : self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, : self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back, :, : self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back, :, : self.n_shape_params] = shape_dirs_mean

        # posedirs set to zero
        posedirs = self.posedirs.reshape(len(self.parents) - 1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat(
            [posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2
        )  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape(
            (len(self.parents) - 1) * 9, (num_verts_orig + num_verts_teeth) * 3
        )  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat(
            [self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1
        )  # (5, J) -> (5, J+num_verts_teeth)

        # lbs_weights manually set
        self.lbs_weights = torch.cat(
            [self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0
        )  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw

        # add faces for teeth
        upper_list = []
        for i in range(14):
            upper_list.append([i, i + 16, i + 15])
            upper_list.append([i, i + 1, i + 16])
        for i in range(15, 29):
            upper_list.append([i, i + 16, i + 15])
            upper_list.append([i, i + 1, i + 16])
        middle_list = []
        for i in range(30, 44):
            middle_list.append([i, i + 16, i + 15])
            middle_list.append([i, i + 1, i + 16])
        lower_list = []
        for i in range(45, 59):
            lower_list.append([i, i + 16, i + 15])
            lower_list.append([i, i + 1, i + 16])
        for i in range(60, 74):
            lower_list.append([i, i + 16, i + 15])
            lower_list.append([i, i + 1, i + 16])

        f_teeth_upper = torch.tensor(upper_list)
        f_teeth_middle = torch.tensor(middle_list)
        f_teeth_lower = torch.tensor(lower_list)

        self._faces = torch.cat(
            [
                self._faces,
                f_teeth_upper + num_verts_orig,
                f_teeth_middle + num_verts_orig,
                f_teeth_lower + num_verts_orig,
            ],
            dim=0,
        )
        self._faces_uvs = torch.cat(
            [
                self._faces_uvs,
                f_teeth_upper + num_verts_uv_orig,
                f_teeth_middle + num_verts_uv_orig,
                f_teeth_lower + num_verts_uv_orig,
            ],
            dim=0,
        )

    def forward(
        self,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape contribution
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            v_shaped += static_offset

        vertices, J, mat_rot = lbs(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        if zero_centered_at_root_node:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]

        ret_vals = [vertices]

        if return_verts_cano:
            ret_vals.append(v_shaped)

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]


if __name__ == "__main__":
    flame_model = FlameHead(shape_params=300, expr_params=100)
