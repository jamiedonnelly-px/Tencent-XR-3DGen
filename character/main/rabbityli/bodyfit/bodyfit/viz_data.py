import importlib.util
import os, sys
import argparse
import pathlib
import shutil
from glob import glob
import trimesh
import torch
from lib.utils.util import batch_transform
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.registration import SMPLRegistration

from lib.smpl_w_scale import SMPL_with_scale
import open3d as o3d
from pytorch3d.ops.knn import knn_points
import json
import open3d as o3d
import numpy as np




def parse_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh_path", type=str, required=True)
    # parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--config", choices=["daz", "vroid", "base"], default=None)

    args = parser.parse_args()
    return args


def interp_transform(points, template_points, template_transform, K = 6):
    results = knn_points(points[None], template_points, K=K)
    dists, idxs = results.dists, results.idx
    neighbs_weight = torch.exp(-dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)
    neighbs_transform = template_transform[:, idxs[0], :, :].view(1, -1, 4, 4)
    points_K = points[:,None].repeat(1,K,1).view(-1,3)
    points_K_warpped = batch_transform( neighbs_transform, points_K ).reshape(1, -1, K, 3)
    points_merge = ( neighbs_weight[..., None] * points_K_warpped ).sum(dim=-2)
    return points_merge.squeeze(), idxs.squeeze()







def vis_old():

    from lib.configs.config_vroid import get_cfg_defaults

    cfg = get_cfg_defaults()

    smplxs = SMPL_with_scale(cfg).to(torch.device(0))

    A_path = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/mcwy_female"

    A_data = os.path.join( A_path, "smplx_and_offset.npz")




    G_trns = np.eye(4)
    G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()



    if type(A_data) == str:
        param_data = torch.load(A_data)

    faces, template, T = smplxs.forward_skinning(param_data)

    part_idx_dict = smplxs.smplx.get_part_index()
    ndp_offset = param_data["offset"].view(1, -1, 3, 1)
    # T[..., :3, 3:] = T[..., :3, 3:] + ndp_offset
    posed_verts = batch_transform( T, template)


    smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    smplmesh.vertices = o3d.utility.Vector3dVector(posed_verts[0].detach().cpu().numpy())
    smplmesh.paint_uniform_color([0.8, 0.8, 0.8])
    smplmesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
    smplmesh.compute_vertex_normals()

    o3d.visualization.draw([smplmesh])

    o3d.io.write_triangle_mesh( os.path.join( A_path, "deformed-smpl.ply" ), smplmesh )



def vis_new ( ) :

    A_path = "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/daz_example/DMC5_Dante_0_0_0/split/smplx_and_offset_smplified.npz"

    smpl_deformed = os.path.join( pathlib.Path ( A_path ).parent, "smpl_deformed.ply" )

    param_data = torch.load(A_path)
    faces = param_data['faces'].detach().cpu().numpy()
    posed_verts = param_data['posed_verts'].detach().cpu().numpy()[0]
    # self.T = param_data['T']

    smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    smplmesh.vertices = o3d.utility.Vector3dVector(posed_verts)
    smplmesh.paint_uniform_color([0.8, 0.8, 0.8])
    smplmesh.triangles = o3d.utility.Vector3iVector(faces)
    smplmesh.compute_vertex_normals()

    o3d.visualization.draw([ smplmesh])
    o3d.io.write_triangle_mesh( smpl_deformed, smplmesh )

if __name__ == '__main__':

    vis_new()
    pass
