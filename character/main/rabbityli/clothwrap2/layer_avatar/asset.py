import copy

import torch
import os
import scipy
import scipy.sparse as sparse
import json
import open3d as o3d
import numpy as np
import trimesh
from pathlib import Path

import pathlib

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def Scene2Trimesh(m):
    meshes = []
    for k in m.geometry.keys():
        ms = m.geometry[k]
        meshes.append(ms)
    m = trimesh.util.concatenate(meshes)
    return m


def proxy_to_visual(visual_vert, bary_coord, face_indices, offset):
    '''
    :param visual_vert: [N,3]
    :param bary_coord: [N, 3]
    :param face_indices: [N, 3]
    :param offset: [N, 3]
    :return:
    '''


    flow_prop = (bary_coord[..., None] * offset[face_indices]).sum(axis=1)
    V = flow_prop + visual_vert
    return V


def fix_hair_mtl_names(obj_path):
    mtl_name = obj_path.split("/")[-1]

    mtl_file = os.path.join(str(pathlib.Path(obj_path).parent), "material.mtl")
    # fix material name path
    with open(mtl_file, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if line[:6] == "newmtl":
                lines[idx] = " ".join(["newmtl", mtl_name, "\n"])
    with open(mtl_file, 'w') as file:
        file.writelines(lines)

    with open(mtl_file, 'r') as file:
        lines = file.readlines()
        print(lines)


class Asset():
    def __init__(self, proxy_path, visual_paths, trns=None, timers=None, label="hair", device=None):

        self.transform = trns

        if self.transform is not None:
            self.transform = torch.from_numpy(self.transform).float().to(device)

        self.proxy_meshes = None
        self.label = label

        visual_mesh = trimesh.load(visual_paths)
        if isinstance(visual_mesh, trimesh.scene.scene.Scene):  # need to handle uv pieces separately
            visual_mesh = Scene2Trimesh(visual_mesh)
        self.visual_mesh = visual_mesh

        self.visual_verts = np.asarray(self.visual_mesh.vertices)
        self.visual_faces = np.asarray(self.visual_mesh.faces)

        if proxy_path is not None:
            self.load_proxies(proxy_path)


    def update_mesh(self):
        self.visual_mesh.vertices = self.visual_verts

    def export_mesh(self, save_path):
        mtl_name = save_path.split("/")[-1]
        self.visual_mesh.visual.material.name = mtl_name
        if self.label == "l-shoe":
            self.visual_mesh.export(save_path + "_left.obj")
        elif self.label == "r-shoe":
            self.visual_mesh.export(save_path + "_right.obj")
        else:
            self.visual_mesh.export(save_path)
            if self.label == "hair":
                fix_hair_mtl_names(save_path)

    def load_proxies(self, proxy_fld):

        voronoi_proxy = Path(proxy_fld) / "proxy"
        voronoi_proxy.mkdir(exist_ok=True)
        info_path = Path(proxy_fld) / "info.json"
        info = load_json(info_path)
        parts = info["parts"]
        n_verts = info["n_verts"]

        prox_meshes = []
        laplacians = []
        sdfs = []
        faces = []
        verts = []

        # import pdb; pdb.set_trace()

        for idx in range(parts):
            m = os.path.join(voronoi_proxy, "part-" + str(idx) + ".ply")
            m = o3d.io.read_triangle_mesh(m)
            prox_meshes.append(m)

            l = sparse.load_npz(os.path.join(voronoi_proxy, "part-" + str(idx) + "-laplacian.npz"))
            laplacians.append(l)

            faces.append(np.asarray(m.triangles))
            verts.append(np.asarray(m.vertices))


            # sdf_path = os.path.join(voronoi_proxy, "part-" + str(idx) + "-sdf.npy")
            # with open(sdf_path, 'rb') as f:
            #     sdf = np.load(f)
            #     sdfs.append(sdf)


        bary_info = os.path.join(voronoi_proxy, "barycentric.npy")
        bary_info = np.load(bary_info, allow_pickle=True).item()


        self.data = {
            "parts": parts,
            "bary_info": bary_info,
            "laplacians": laplacians,
            "prox_meshes": prox_meshes,
            "Vs": verts,
            "Fs": faces
        }






