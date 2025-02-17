import os, glob, sys
from pathlib import Path
import json

import trimesh

import scipy
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse import vstack


import open3d as o3d

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


def concatenent_meshes(V_lst, F_lst):
    assert len(V_lst) == len(F_lst)
    assert len(V_lst) > 1
    size = [len(e) for e in V_lst[:-1]]
    id_offset = [0] + list(np.cumsum(size))
    for i in range(len(F_lst)):
        F_lst[i] = F_lst[i] + id_offset[i]
    V = np.concatenate(V_lst, axis=0)
    F = np.concatenate(F_lst, axis=0)
    return V, F


def compute_bary_coordinate(bary_vector):
    '''
    S= A(XBC) + A(XBA) + A(XAC)
    a = A(XBC) / S
    b = A(XAC) / S
    c = A(XBA) / S
    :param bary_vector: [N,3,3]:   [ ... [ XA, XB, XC] ]
    :return: [N, 3]:  [..., [a,b,c]]
    '''

    def A(v1, v2):  # compute triangle area
        return np.linalg.norm(np.cross(v1, v2), axis=-1) * 0.5

    XA = bary_vector[:, 0]
    XB = bary_vector[:, 1]
    XC = bary_vector[:, 2]

    XBC = A(XB, XC)
    XAC = A(XA, XC)
    XBA = A(XB, XA)

    S = XBC + XBA + XAC

    a = XBC / S
    b = XAC / S
    c = XBA / S

    return np.stack([a, b, c], axis=-1)


def run(visual_path, proxy_flder, VLIAD_RADIUS=0.02):
    visual = trimesh.load(visual_path)
    if isinstance(visual, trimesh.scene.scene.Scene):  # need to handle uv pieces separately
        visual = Scene2Trimesh(visual)

    voronoi_proxy = Path(proxy_flder) / "proxy"
    voronoi_proxy.mkdir(exist_ok=True)
    info_path = Path(proxy_flder) / "info.json"
    info = load_json(info_path)
    parts = info["parts"]
    n_verts = info["n_verts"]

    prox_meshes = []
    samples = []
    face_indices = []

    valid_skinning_rate = 0

    for idx, nv in enumerate(n_verts):
        m = os.path.join(voronoi_proxy, "part-" + str(idx) + "_rm_occlusion.obj")
        m = trimesh.load(m)
        prox_meshes.append(m)

        # smp, fid = trimesh.sample.sample_surface_even(m, 1000000 )
        #
        # smp_pc = o3d.geometry.PointCloud()
        # smp_pc.points = o3d.utility.Vector3dVector(np.array(smp))
        #
        # samples.append(smp_pc)
        # face_indices.append(fid)
    if len(prox_meshes) > 1:
        V, F = concatenent_meshes(V_lst=[np.array(m.vertices) for m in prox_meshes],
                                  F_lst=[np.array(m.faces) for m in prox_meshes])
    else:
        V, F = np.array(prox_meshes[0].vertices), np.array(prox_meshes[0].faces)

    proxy = trimesh.Trimesh(V, F)

    smp, fid = trimesh.sample.sample_surface_even(proxy, 1000000)

    smp_pc = o3d.geometry.PointCloud()
    smp_pc.points = o3d.utility.Vector3dVector(np.array(smp))
    pcd_tree = o3d.geometry.KDTreeFlann(smp_pc)
    # o3d.visualization.draw([smp_pc])

    visual_pc = o3d.geometry.PointCloud()
    visual_pc.points = o3d.utility.Vector3dVector(np.array(visual.vertices))
    visual_pc.paint_uniform_color([0.8, 0.1, 0.1])

    dist = []
    ind = []
    for i, point in enumerate(visual_pc.points):
        [count, vec1, vec2] = pcd_tree.search_knn_vector_3d(point, 1)
        dist.append(np.sqrt(vec2[0]))
        ind.append(vec1[0])

    dist = np.asarray(dist)

    valid_skinning_rate = 1 -  (dist > VLIAD_RADIUS).sum() / len(dist)

    print("dist.max(), dist.min(), valid_rate", dist.max(), dist.min(), valid_skinning_rate)

    ind = np.asarray(ind)

    nn_v = np.array(smp)[ind]
    nn_pc = o3d.geometry.PointCloud()
    nn_pc.points = o3d.utility.Vector3dVector(nn_v)

    face_indices = F[fid[ind]]
    bary_vector = V[face_indices] - nn_v[:, None]
    bary_coord = compute_bary_coordinate(bary_vector)

    # nn_v2 = (bary_coord[..., None] * V[ face_indices ]).sum(axis=1)
    # nn_pc2 = o3d.geometry.PointCloud()
    # nn_pc2.points = o3d.utility.Vector3dVector( nn_v2 )

    visual_o3d = o3d.geometry.TriangleMesh()
    visual_o3d.vertices = o3d.utility.Vector3dVector(np.array(visual.vertices))
    visual_o3d.triangles = o3d.utility.Vector3iVector(np.array(visual.faces))
    visual_o3d.paint_uniform_color([0.1, 0.1, 0.7])
    visual_o3d.compute_vertex_normals()

    bary_info = {"face_indices": face_indices, "bary_coord": bary_coord }
    dump_path = os.path.join(voronoi_proxy, "barycentric.npy")
    np.save(dump_path, bary_info)

    print("skinning save to", dump_path)



    info[ "valid_skinning_rate" ] = valid_skinning_rate


    json_object = json.dumps(info, indent=4)
    with open( info_path, "w") as outfile:
        outfile.write(json_object)


# if __name__ == '__main__':
#     visual_fn = "/home/rabbityl/tboard/DR_394_F_A/DR_394_fbx2020.obj"
#
#     proxy_flder = os.path.join(Path(visual_fn).parent, "proxy_mesh")
#
#     run(visual_fn, proxy_flder)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--p", type=str, required=True)
    args = parser.parse_args()

    visual_path = args.v
    proxy_flder = args.p


    try :

        run(visual_path, proxy_flder)

    except :

        cmd = "  ".join([
            "echo ",
            visual_path,
            ">>"
            "./invalid_skinning.txt"
        ])

        os.system(cmd)
