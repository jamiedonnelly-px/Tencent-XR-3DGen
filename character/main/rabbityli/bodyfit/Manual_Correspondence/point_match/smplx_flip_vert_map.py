import os.path


import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import argparse
import pathlib
import open3d as o3d
import trimesh



def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def multual_nn_correspondence(src_pcd_deformed, tgt_pcd, search_radius=0.3, knn=1):

    src_idx = np.arange(src_pcd_deformed.shape[0])

    s2t_dists, ref_tgt_idx = knn_point_np (knn, tgt_pcd, src_pcd_deformed)
    s2t_dists, ref_tgt_idx = s2t_dists[:,0], ref_tgt_idx [:, 0]
    valid_distance = s2t_dists < search_radius

    _, ref_src_idx = knn_point_np (knn, src_pcd_deformed, tgt_pcd)
    _, ref_src_idx = _, ref_src_idx [:, 0]

    cycle_src_idx = ref_src_idx [ ref_tgt_idx ]

    is_mutual_nn = cycle_src_idx == src_idx

    mutual_nn = np.logical_and( is_mutual_nn, valid_distance)
    correspondences = np.stack([src_idx [ mutual_nn ], ref_tgt_idx[mutual_nn] ] , axis=0)

    return correspondences


if __name__ == '__main__':


    tgt_scan_dir = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/smpl_mesh_1280/smpl"


    smpl_faces = np.load(os.path.join(tgt_scan_dir, "smpl_faces.npy"))
    smpl_verts = np.load(os.path.join(tgt_scan_dir, "smpl_verts.npy"))
    smplmesh = o3d.geometry.TriangleMesh()
    smplmesh.vertices = o3d.utility.Vector3dVector(smpl_verts)
    smplmesh.triangles = o3d.utility.Vector3iVector(smpl_faces)
    smplmesh.compute_vertex_normals()



    I = np.eye(4)
    I[0,0] = -1
    # mirror_mesh = copy.deepcopy(smplmesh).transform(I)
    # mirror_mesh.compute_vertex_normals()
    # mirror_mesh.paint_uniform_color([1, 0.5, 0.5])

    vert_flip = copy.deepcopy(smpl_verts)

    vert_flip[:,0]= vert_flip[:,0] * -1
    smplmesh2 = o3d.geometry.TriangleMesh()
    smplmesh2.vertices = o3d.utility.Vector3dVector(vert_flip)
    smplmesh2.triangles = o3d.utility.Vector3iVector(smpl_faces)
    smplmesh2.paint_uniform_color([1, 0.5, 0.5])

    smplmesh2.compute_vertex_normals()

    vert_map = multual_nn_correspondence(vert_flip, smpl_verts)


    with open ("smpl_left_right_symmetric_map.npy", "wb") as f :

        np.save ( f, vert_map)


