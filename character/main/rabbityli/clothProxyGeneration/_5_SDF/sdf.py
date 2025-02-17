import os, glob, sys
from pathlib import Path
import json
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse import vstack

import igl
import open3d as o3d

def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data






def compute_signed_distance_and_closest_goemetry(query_points, scene):

    # query cloest points
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(), axis=-1)
    face_id = closest_points['primitive_ids']

    # check inside outside
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1


    distance[is_inside] *= -1
    return distance, is_inside,  closest_points['primitive_ids'].numpy()


def register_rigid_body (body_mesh):

    body_mesh.compute_triangle_normals()
    body_normals = np.array(body_mesh.triangle_normals)

    body_normals = body_normals

    # Create a scene and add the triangle mesh
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(body_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    body_scene = scene

    return body_scene, body_normals

def run(asset_body, proxy_flder):

    # load body
    param_data = torch.load(asset_body, map_location='cuda:0')
    faces = param_data['faces'].cpu().numpy()
    posed_verts = param_data['posed_verts'].detach().cpu().numpy()[0]
    # import pdb; pdb.set_trace()
    body = o3d.geometry.TriangleMesh()
    body.vertices = o3d.utility.Vector3dVector(posed_verts)
    body.triangles = o3d.utility.Vector3iVector(faces)
    body.compute_vertex_normals()
    rigid_body_scene, _ = register_rigid_body(body)
    # sdf, _, face_ids = compute_signed_distance_and_closest_goemetry(
    #     np.asarray(proxy_mesh.vertices).astype(np.float32), rigid_scene)

    G_trns = np.eye(4)
    G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()

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

    o3d.io.write_triangle_mesh( os.path.join(voronoi_proxy, "deformed_smpl.ply"), body )

    for idx, nv in enumerate(n_verts):
        m = os.path.join(voronoi_proxy, "part-" + str(idx) + ".ply")
        m = trimesh.load(m)
        prox_meshes.append(m)



        v = np.asarray(m.vertices).astype(np.float32)
        t_v =  (G_trns [:3, :3] @ v.T + G_trns[:3, 3:]).T

        sdf, _, face_ids = compute_signed_distance_and_closest_goemetry( t_v.astype(np.float32), rigid_body_scene)
        sdf_path = os.path.join(voronoi_proxy, "part-" + str(idx) + "-sdf.npy")



        with open(sdf_path, 'wb') as f:
            np.save(f, sdf)

        print( "---save----", sdf_path)





if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=str, required=True)
    parser.add_argument("--p", type=str, required=True)
    args = parser.parse_args()

    asset_body = args.b
    proxy_flder = args.p

    run(asset_body, proxy_flder)

