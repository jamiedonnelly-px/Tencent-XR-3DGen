import copy
import importlib.util
import os, sys
import argparse
import shutil
from glob import glob
import trimesh
import torch
import pathlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import cv2
import scipy
import trimesh
import time

import numpy as np
from scipy import sparse

# from lib.timer import Timers
# timers = Timers()



import json

import open3d as o3d
import numpy as np
from pathlib import Path

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(codedir, "../") )

from layer_avatar.body import Body
from layer_avatar.body_config.smpl_config import  part_scale_ref_index_map
from layer_avatar.asset import Asset, proxy_to_visual
from layer_avatar.mesh_editor.pene_solve import solve_cloth_body_pene, solve_hair_head_pene, solve_cloth_cloth_pene, register_rigid_body, compute_signed_distance_and_closest_goemetry, query_cloest_geometry
from layer_avatar.mesh_editor.laplacian import concatenent_meshes, construct_smooth_system, solve
from layer_avatar.transfer import transfer_verts, save_cloth, scale_diff
from layer_avatar.utils.utils import load_json


from matplotlib import colors
import matplotlib.colors as mcolors
template_colors = [ colors.to_rgba(key)[:-1] for key in mcolors.TABLEAU_COLORS ]
clst = []
for  i in range (30): # assume max 300 points
    clst = clst + template_colors

def smplify_mesh ( V, F, clr ):
    m1 = o3d.geometry.TriangleMesh()
    m1.vertices = o3d.utility.Vector3dVector(V)
    m1.triangles = o3d.utility.Vector3iVector(F)
    m1.paint_uniform_color(clr)
    m1.compute_vertex_normals()
    return m1



def obtain_o3d_mesh (V, F ):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector( V )
    mesh.triangles = o3d.utility.Vector3iVector( F )
    return mesh

def compute_normal( V, F ):
    mesh = obtain_o3d_mesh(V,F)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    v_nrm = np.asarray( mesh.vertex_normals )
    f_nrm = np.asarray( mesh.triangle_normals )
    return v_nrm, f_nrm, mesh


def colorize_mesh( mesh , mask, clr ):
    color = np.ones_like(np.asarray(mesh.vertices)) * 0.75
    color[mask] = np.array(clr)
    mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh

def part_wise_scale(S_body, T_body, label):
    scale_ref_index = part_scale_ref_index_map[ label ]
    _scale = scale_diff(S_body, T_body, scale_ref_index)
    return _scale


def warp_asset_with_proxies(asset, S_body, T_body, T_s2t):

    offsets = []
    proxy_vert_wrap = []
    for i in range(asset.data["parts"]):
        # V = np.asarray( asset.data["prox_meshes"][i].vertices)
        V = asset.data["Vs"][i]

        # compute rough scale difference using points on SMPL
        _scale = part_wise_scale(S_body, T_body, asset.label)
        # scales.append(float(_scale))

        # wrap with dense flow queried on smpl surface
        V_wrapped, _vert_stand = transfer_verts(S_body, T_body, copy.deepcopy(V), asset.transform, asset.label, _scale, T_s2t)

        # preserve the laplacian on the scaled verts
        _vert_stand_scaled = (_vert_stand * _scale).detach().cpu().numpy()
        L = asset.data["laplacians"][i]
        delta = L.dot(_vert_stand_scaled)
        A, b = construct_smooth_system(V_wrapped, None, L, delta, CTR_WEIGHT=1)
        V_smooth = solve(A, b)

        proxy_offset = V_smooth - V
        offsets.append(proxy_offset)
        proxy_vert_wrap.append(V_smooth)


    offset_combined = np.concatenate(offsets, axis=0)
    warp_visual_vert = proxy_to_visual(asset.visual_verts, asset.data["bary_info"]["bary_coord"], asset.data["bary_info"]["face_indices"], offset_combined)

    # warp_proxy = concatenent_meshes(proxy_vert_wrap, asset.data["Fs"])

    return offsets, proxy_vert_wrap, warp_visual_vert



def filter_inward_VF(V, F , body_scene, body_normals):
    V_normals, _, mesh = compute_normal( V, F )
    sdf, _, face_ids, clsest_pts = compute_signed_distance_and_closest_goemetry(V, body_scene)
    cosine = np.sum(V_normals * body_normals[face_ids], axis=-1)
    # valid point mask
    valid_V = np.logical_and( sdf>0, cosine>0.1)

    valid_F = valid_V [F[:, 0]] * valid_V[F[:,1]] * valid_V [F[:,2]]
    return valid_V, valid_F, face_ids


def filter_inner_primitives (V, V_normals, F, F_normals, body_scene, body_normals, outward_VF_mask = None ):
    """
    :param V:
    :param V_normals:
    :param F:
    :param valid_V:
    :return:
    """

    data = {}

    if outward_VF_mask is None:
        outward_V_mask, outward_F_mask, _ = filter_inward_VF ( V, F , body_scene, body_normals )
        data["outward_V_mask"] = outward_V_mask
    else :
        outward_V_mask, outward_F_mask = outward_VF_mask
        data["outward_V_mask"] = outward_V_mask


    trimesh_smpl = trimesh.Trimesh(V, F)

    ray_inter_mask = []
    for i in range(len(V)):
        ray_direction = V_normals[i]
        ray_origin = V[i] + 0.0001 * ray_direction
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        locations, index_ray, index_tri = trimesh_smpl.ray.intersects_location(ray_origins = ray_origin[None], ray_directions=ray_direction[None])
        # print( len( locations))

        if len(locations) < 1: # no hit
            ray_inter_mask.append(True)



        else :# hit multiple faces, iterate to check if the face is inward face


            dist_2 = np.linalg.norm(locations - ray_origin[None]).min()

            o_flag = False

            if dist_2 > 0.1 :
                o_flag = True

            else:
                for k in range(1, len(locations)):

                    fid = index_tri[k]
                    if outward_F_mask[fid]:
                        cosine = np.sum(ray_direction * F_normals[fid])
                        if cosine < -0.1:
                            o_flag = True
                            break



            if o_flag:
                ray_inter_mask.append(True)

            else:
                ray_inter_mask.append(False)

    ray_inter_mask = np.asarray(ray_inter_mask)

    data["ray_inter_mask"] = ray_inter_mask

    data["outer_mask"] = np.logical_and( outward_V_mask, ray_inter_mask)

    return data


def solve_asset_body_penetration(asset, proxy_vert_wrap, T_body, warp_visual_vert):

    offsets2 = []

    pensol_proxy_lst = []



    for i in range(asset.data["parts"]):

        V = proxy_vert_wrap[i]
        L = asset.data["laplacians"][i]
        # sdf = asset.data["sdfs"][i]
        delta = L.dot(V)
        V_normals, _ , _= compute_normal(V, asset.data['Fs'][i])



        # mesh = obtain_o3d_mesh(V, asset.data["Fs"][i])


        V_ =  copy.deepcopy( V)
        for itr in range(3):

            if asset.label == "hair":
                data = solve_hair_head_pene(T_body.head, V_, V_normals, L, delta, PEN_THRESH=0.001, PUSH_WEIGHT=100, HOLD_WEIGHT=1, PUSH_DIST=0.01)

            else:
                data = solve_cloth_body_pene(T_body.body_manifold, V_, V_normals, L, delta, PEN_THRESH=0.001, PUSH_WEIGHT=100, HOLD_WEIGHT=1, PUSH_DIST=0.015)
            V_ = data["V_pensol"]


        offsets2.append(data["V_pensol"] - V)
        pensol_proxy_lst.append(data["V_pensol"])


        # sdf_mesh = colorize_mesh(copy.deepcopy(mesh), sdf < 0, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/sdfmesh-"+str(i)+".ply", sdf_mesh)
        # cosine = data["cosine"]
        # cosine_gt_0 = colorize_mesh(copy.deepcopy(mesh), cosine > 0, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/cosin-gt-0"+str(i)+".ply", cosine_gt_0)
        # cosine_gt_0d2 = colorize_mesh(copy.deepcopy(mesh), cosine > 0.2, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/cosin-gt-0.2"+str(i)+".ply", cosine_gt_0d2)
        # pen_mask = data["pen_mask"]
        # pen_mask_mesh = colorize_mesh(copy.deepcopy(mesh), pen_mask, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/pen_mask_mesh"+str(i)+".ply", pen_mask_mesh)

    offset2_combined = np.concatenate(offsets2, axis=0)
    pensol_visual_vert = proxy_to_visual(warp_visual_vert, asset.data["bary_info"]["bary_coord"],
                                         asset.data["bary_info"]["face_indices"], offset2_combined)
    pensol_visual = [pensol_visual_vert, asset.visual_faces]
    pensol_proxy = concatenent_meshes(pensol_proxy_lst,  asset.data["Fs"] )

    return pensol_visual, pensol_proxy, pensol_proxy_lst


def solve_asset_asset_penetration(S_asset, T_asset, T_body, Mode="above"):

    body_scene, body_normals = register_rigid_body(T_body.body_manifold)
    body_data = { "body_scene": body_scene, "body_normals": body_normals }



    # rigid_body_mesh = T_asset
    T_V, T_F = concatenent_meshes(T_asset.data["Vs"], T_asset.data["Fs"])
    TMesh = obtain_o3d_mesh( T_V, T_F )
    TMesh_scene, TMesh_normals = register_rigid_body(TMesh)
    TMesh_data = { "TMesh_scene": TMesh_scene, "TMesh_normals": TMesh_normals }


    offsets2 = []
    pensol_proxy_lst = []
    for i in range(S_asset.data["parts"]):
        V = S_asset.data["Vs"][i]
        L = S_asset.data["laplacians"][i]
        delta = L.dot(V)
        V_normals, S_F_normals, smesh = compute_normal(V, S_asset.data['Fs'][i] )

        s_msk_data = filter_inner_primitives(V, V_normals, S_asset.data['Fs'][i], S_F_normals, body_data["body_scene"], body_data["body_normals"])

        SMesh_data = {
            "V": V,
            "L": L,
            "delta": delta,
            "V_normals": V_normals,
            "outer_mask": s_msk_data["outer_mask"]
        }


        mesh = obtain_o3d_mesh(V, S_asset.data["Fs"][i])


        data = solve_cloth_cloth_pene(TMesh_data,  SMesh_data, PUSH_WEIGHT=100, HOLD_WEIGHT=1, MODE=Mode, PUSH_DIST=0.007)#, beneath=beneath)
            # SMesh_data["V"] = data["V_pensol"]

        # sdf_mesh = colorize_mesh(copy.deepcopy(mesh), sdf < 0, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/sdfmesh-"+str(i)+".ply", sdf_mesh)
        # cosine = data["cosine"]
        # cosine_gt_0 = colorize_mesh(copy.deepcopy(mesh), cosine > 0, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/cosin-gt-0"+str(i)+".ply", cosine_gt_0)
        # cosine_gt_0d2 = colorize_mesh(copy.deepcopy(mesh), cosine > 0.2, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/cosin-gt-0.2"+str(i)+".ply", cosine_gt_0d2)
        # pen_mask = data["pene_mask"]
        # pen_mask_mesh = colorize_mesh(copy.deepcopy(mesh), pen_mask, [0.8, 0, 0])
        # o3d.io.write_triangle_mesh( "./output/pen_mask_mesh"+str(i)+".ply", pen_mask_mesh)

        offsets2.append(data["V_pensol"] - V)
        pensol_proxy_lst.append(data["V_pensol"])



    offset2_combined = np.concatenate(offsets2, axis=0)
    pensol_visual_vert = proxy_to_visual(
        S_asset.visual_verts, S_asset.data["bary_info"]["bary_coord"], S_asset.data["bary_info"]["face_indices"], offset2_combined)
    pensol_visual = [pensol_visual_vert, S_asset.visual_faces]
    pensol_proxy = concatenent_meshes(pensol_proxy_lst, S_asset.data["Fs"])

    return pensol_visual, pensol_proxy, pensol_proxy_lst






