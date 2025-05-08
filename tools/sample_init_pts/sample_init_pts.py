import numpy as np
import point_cloud_utils as pcu


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                try:
                    faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                    faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
                except:
                    faces_vertex.append([float(x) for x in parts[1:]])
                    faces_uv.append([0, 0, 0])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def write_obj(filepath, verts, tris=None, log=True):
    """save mesh obj

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


def faces_of_verts(vert_idcs, all_faces, return_face_idcs=False):
    """
    calculates face tensor of shape F x 3 with face spanned by vertices in flame mesh
    all vertices of the faces returned by this function contain only vertices from vert_idcs
    :param vert_idcs:
    :param consider_full_flame_model:
    :return_face_idcs: if True, also returns list of relevant face idcs
    :return:
    """
    vert_faces = []
    face_idcs = []
    for i, f in enumerate(all_faces):
        keep_face = True
        for idx in f:
            if idx not in vert_idcs:
                keep_face = False
        if keep_face:
            vert_faces.append(f)
            face_idcs.append(i)
    vert_faces = np.stack(vert_faces)

    if return_face_idcs:
        return vert_faces, face_idcs

    return vert_faces


subject = "306"
outpath = f"/path/to/nersemble/{subject}/init_pts_150000.npy"

canonical_cm = load_obj(f"/root/workspace/PaSA/tools/sample_init_pts/{subject}.obj")
verts = canonical_cm["verts"]
faces = canonical_cm["vert_ids"]
num_pts = 50000

num_off_pts = 100000
sigma = 0.02

scalp_idcs = np.load("./scalp_idcs.npy")
vert_normals = pcu.estimate_mesh_vertex_normals(verts, faces)

scalp_faces = faces_of_verts(scalp_idcs, faces)
f_i, bc = pcu.sample_mesh_random(verts, scalp_faces, num_samples=num_pts)

surf_pts = pcu.interpolate_barycentric_coords(scalp_faces, f_i, bc, verts)
surf_normals = pcu.interpolate_barycentric_coords(scalp_faces, f_i, bc, vert_normals)

# write_obj("surf_pts.obj", surf_pts)

# sample off-surface pts
rnd_idx = np.random.randint(0, surf_pts.shape[0], num_off_pts)
off_pts = surf_pts[rnd_idx] + surf_normals[rnd_idx] * np.random.rand(num_off_pts, 3) * sigma

sampled_pts = np.concatenate([surf_pts, off_pts], axis=0)
write_obj(  # For test
    "./init_pts_150000.obj",
    sampled_pts,
)
np.save(outpath, sampled_pts)
