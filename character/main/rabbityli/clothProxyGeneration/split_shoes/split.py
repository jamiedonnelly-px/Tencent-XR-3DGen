import copy
import os.path
import trimesh
import open3d as o3d
import trimesh
import torch
import numpy as np
import argparse
import json
import pathlib






def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


def Scene2Trimesh(m):
    meshes = []
    for k in m.geometry.keys():
        ms = m.geometry[k]
        meshes.append(ms)
    m = trimesh.util.concatenate(meshes)
    return m

def process(visual_path ):

    m = trimesh.load(visual_path)
    pth = pathlib.Path(visual_path).parent

    (pth/"left").mkdir(parents=True,exist_ok=True)
    (pth/"right").mkdir(parents=True,exist_ok=True)

    lft = os.path.join(pth, "left/asset.obj")
    rgt = os.path.join(pth, "right/asset.obj")


    if isinstance(m, trimesh.scene.scene.Scene):  # need to handle uv pieces separately
        m = Scene2Trimesh(m)

    if isinstance(m, trimesh.base.Trimesh):

        # split to left and right shoe
        parts = trimesh.graph.connected_components(m.vertex_adjacency_graph.edges)
        vert = np.asarray( m.vertices )
        left_indexes = []
        right_indexes = []
        for p in parts:
            is_left = vert[p][:, 0] > 0
            is_left = True if is_left.sum() > (0.5 * len(is_left)) else False
            if is_left:
                left_indexes = left_indexes + list(p)
            else:
                right_indexes = right_indexes + list(p)

        # save left and right shoes separately
        lm = copy.deepcopy(m)
        rm = copy.deepcopy(m)
        lmask = np.zeros((len(m.vertices))).astype(int)
        rmask = np.zeros((len(m.vertices))).astype(int)
        lmask[left_indexes] = 1
        rmask[right_indexes] = 1
        lfaces = np.asarray(m.faces)
        rfaces = np.asarray(m.faces)
        lface_mask = lmask[lfaces].sum(-1) == 3
        rface_mask = rmask[rfaces].sum(-1) == 3

        lm.update_faces(lface_mask)
        rm.update_faces(rface_mask)

        lm.export(lft)
        rm.export(rgt)



if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=str, required=True)

    args = parser.parse_args()

    visual_path = args.v


    process( visual_path=visual_path)

    # process( visual_path="/home/rabbityl/tboard/DR_394_F_A/DR_394_fbx2020.obj" ,vert_limit=6000)


