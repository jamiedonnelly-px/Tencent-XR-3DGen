import copy
import sys
import  trimesh
import open3d as o3d
import numpy as np
import scipy.sparse as sparse
import torch

from .laplacian import solve
from scipy.sparse import vstack


def obtain_o3d_mesh (V, F ):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector( V )
    mesh.triangles = o3d.utility.Vector3iVector( F )
    return mesh

def compute_normal( V, F ):
    mesh = obtain_o3d_mesh(V,F)
    mesh.compute_vertex_normals()
    nrm = np.asarray( mesh.vertex_normals )
    return nrm, mesh



def query_cloest_geometry( query_points, scene ):
    # query cloest points
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(), axis=-1)
    face_id = closest_points['primitive_ids'].numpy()
    return distance,    closest_points['points'].numpy() , face_id


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
    return distance, is_inside,  closest_points['primitive_ids'].numpy(), closest_points['points'].numpy()





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


def construct_control_block(ctr_index, ctrl_pts, num_UKN, WEIGHT):
    row = np.arange(len(ctrl_pts))
    col = ctr_index
    val = np.array([WEIGHT] * len(ctrl_pts))
    control_block = sparse.csc_matrix((val, (row, col)), shape=( len(ctrl_pts), num_UKN))
    b = ctrl_pts * WEIGHT
    return control_block, b





def solve_cloth_body_pene(body_mesh, V, V_normals, L, delta,    PEN_THRESH, PUSH_WEIGHT, HOLD_WEIGHT, PUSH_DIST = 0.01):

    data={ }

    rigid_scene, body_normals = register_rigid_body(body_mesh)
    sdf, _, face_ids, clsest_pts = compute_signed_distance_and_closest_goemetry(V, rigid_scene)


    # direction = V - clsest_pts
    # direction = direction / np.linalg.norm(direction)
    cosine = np.sum(V_normals * body_normals[face_ids], axis=-1)

    data["cosine"] = cosine

    # if cano_sdf is not None:
    #     pene_mask = np.logical_and(sdf < PEN_THRESH, cano_sdf > PEN_THRESH)
    #     hold_mask = sdf > PEN_THRESH
    # else:


    # pene_mask = sdf < 0
    pene_mask = np.logical_and(sdf<PEN_THRESH, cosine > 0.2)

    hold_mask = ~pene_mask

    face_normals = body_normals[face_ids][pene_mask]
    distance = np.abs(sdf[pene_mask] - PEN_THRESH) + PUSH_DIST
    offset = distance[..., None] * face_normals
    V_fix = copy.deepcopy(V)
    V_fix[pene_mask] = V_fix[pene_mask] + offset
    num_inside = pene_mask.sum()
    pts_index = np.arange(len(V_fix))

    data["pen_mask"] = pene_mask


    if num_inside > 0: # penetration occur

        control_block, b = construct_control_block(pts_index[pene_mask], V_fix[pene_mask], len(V_fix), PUSH_WEIGHT)

        if len(V_fix) - num_inside > 0:

            hold_block, b2 = construct_control_block(pts_index[hold_mask], V_fix[hold_mask], len(V_fix), HOLD_WEIGHT)

            A = vstack([L, control_block, hold_block])
            b = np.concatenate([delta, b, b2], axis=0)

        else:

            A = vstack([L, control_block])
            b = np.concatenate([delta, b], axis=0)

        x = solve(A, b)

        data["V_pensol"] = x

    else:
        data["V_pensol"] = V


    return  data


def solve_hair_head_pene( head_mesh, V, V_normals, L, delta,  PEN_THRESH, PUSH_WEIGHT, HOLD_WEIGHT, PUSH_DIST = 0.01):

    CHECK_RANGE = 0.02

    data={ }

    rigid_scene, head_normals = register_rigid_body(head_mesh)
    distance, clsest_pts, face_ids = query_cloest_geometry(V, rigid_scene)
    clsest_head_normals = head_normals[face_ids]


    outer_mask = np.sum(V_normals * clsest_head_normals, axis=-1) > 0.1



    direction = V - clsest_pts
    direction = direction / np.linalg.norm(direction)
    cosine = np.sum(direction * clsest_head_normals, axis=-1)

    data["cosine"] = cosine

    pene_mask = np.logical_and( np.logical_and( cosine < 0, distance < CHECK_RANGE), outer_mask )

    hold_mask = ~pene_mask

    face_normals = head_normals[face_ids][pene_mask]
    distance = np.abs(distance[pene_mask] ) + PUSH_DIST
    offset = distance[..., None] * face_normals


    V_fix = copy.deepcopy(V)
    V_fix[pene_mask] = V_fix[pene_mask] + offset
    num_inside = pene_mask.sum()
    pts_index = np.arange(len(V_fix))

    data["pen_mask"] = pene_mask


    if num_inside > 0: # penetration occur

        control_block, b = construct_control_block(pts_index[pene_mask], V_fix[pene_mask], len(V_fix), PUSH_WEIGHT)

        if len(V_fix) - num_inside > 0:

            hold_block, b2 = construct_control_block(pts_index[hold_mask], V_fix[hold_mask], len(V_fix), HOLD_WEIGHT)

            A = vstack([L, control_block, hold_block])
            b = np.concatenate([delta, b, b2], axis=0)

        else:

            A = vstack([L, control_block])
            b = np.concatenate([delta, b], axis=0)

        x = solve(A, b)

        data["V_pensol"] = x

    else:
        data["V_pensol"] = V


    return  data


def solve_cloth_cloth_pene(TMesh_data,  SMesh_data, PUSH_WEIGHT, HOLD_WEIGHT, PUSH_DIST = 0.001, MODE="above" ):
    '''
    :param TMesh_data:
    :param SMesh_data:
    :param PUSH_WEIGHT:
    :param HOLD_WEIGHT:
    :param PUSH_DIST:
    :param MODE: [ 'above', 'below' ]
    :return:
    '''



    V = SMesh_data ["V"]
    V_normals = SMesh_data ["V_normals"]
    L = SMesh_data ["L"]
    delta = SMesh_data ["delta"]
    outer_mask = SMesh_data ["outer_mask"]

    data = {}

    CHECK_RANGE = 0.02 # Meters


    distance, clsest_pts, face_ids = query_cloest_geometry(V, TMesh_data["TMesh_scene"])
    clsest_TMesh_normals = TMesh_data["TMesh_normals"][face_ids]

    direction = V - clsest_pts
    direction = direction / np.linalg.norm(direction)
    cosine = np.sum(direction * clsest_TMesh_normals, axis=-1)

    if MODE == "adaptive":

        overlap_mask = np.logical_and( distance < CHECK_RANGE,  outer_mask )
        below_mask =  np.logical_and(cosine < 0, overlap_mask)
        below_ratio = below_mask.sum() * 1. /  overlap_mask.sum()
        if below_ratio > 0.45:
            MODE = 'below'
        else:
            MODE = 'above'


        print("-------Adaptive MODE-------", MODE)



    if MODE == 'above':

        pene_mask = np.logical_and( np.logical_and( cosine < 0, distance < CHECK_RANGE ), outer_mask)
        hold_mask = ~pene_mask
        face_normals = clsest_TMesh_normals[pene_mask]
        distance = np.abs(distance[pene_mask] ) + PUSH_DIST
        offset = distance[..., None] * face_normals
        # if beneath: offset = -1 * offset


    elif MODE == 'below' :

        pene_mask = np.logical_and( cosine > 0, distance < CHECK_RANGE )
        hold_mask = ~ pene_mask
        face_normals = clsest_TMesh_normals[pene_mask]
        distance = np.abs(distance[pene_mask] ) + PUSH_DIST
        offset = -1 * distance[..., None] * face_normals



    else :
        raise NotImplementedError()


    V_fix = copy.deepcopy(V)
    V_fix[pene_mask] = V_fix[pene_mask] + offset
    num_inside = pene_mask.sum()
    pts_index = np.arange(len(V_fix))



    data["pene_mask"] = pene_mask



    if num_inside > 0: # penetration occur

        control_block, b = construct_control_block(pts_index[pene_mask], V_fix[pene_mask], len(V_fix), PUSH_WEIGHT)

        if len(V_fix) - num_inside > 0:
            hold_block, b2 = construct_control_block(pts_index[hold_mask], V_fix[hold_mask], len(V_fix), HOLD_WEIGHT)
            A = vstack([L, control_block, hold_block])
            b = np.concatenate([delta, b, b2], axis=0)

        else:
            A = vstack([L, control_block])
            b = np.concatenate([delta, b], axis=0)
        x = solve(A, b)
        data["V_pensol"] = x

    else:
        data["V_pensol"] = V

    return data

