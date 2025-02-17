import argparse
import math
import os
import sys
import time

import bmesh
import bpy
import miniball
import numpy as np
import trimesh


def read_mesh_to_ndarray(mesh, mode="Edit"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in ["edit", "object"]

    if mode == "object":
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
    elif mode == "edit":
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, None


def compute_mesh_size(meshes):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        vert, _ = read_mesh_to_ndarray(mesh, mode="edit")
        mat = np.asarray(mesh.matrix_world)
        R, t = mat[:3, :3], mat[:3, 3:]  # Apply World Scale
        verts.append((R @ vert.T + t).T)
    verts = np.concatenate(verts, axis=0)

    min_0 = verts[:, 0].min(axis=0)
    max_0 = verts[:, 0].max(axis=0)
    min_1 = verts[:, 1].min(axis=0)
    max_1 = verts[:, 1].max(axis=0)
    min_2 = verts[:, 2].min(axis=0)
    max_2 = verts[:, 2].max(axis=0)

    min_ = np.array([min_0, min_1, min_2])
    max_ = np.array([max_0, max_1, max_2])

    obj_center = (min_ + max_) / 2

    # use max len of xyz, instead of z
    length = max(max_ - min_)
    diagonal = np.linalg.norm(max_ - min_)

    return obj_center, length, diagonal, min_, max_, verts


def calculate_scale_matrix(mesh_verts: np.array, standard_height: float = 1.98):
    original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
    hull_vertices = original_mesh.convex_hull.vertices
    bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
        hull_vertices)

    obj_center = bounding_sphere_C
    length = 2 * math.sqrt(bounding_sphere_r2)
    scale = standard_height / length
    translation = -1 * obj_center
    transformation = np.array(
        [[scale, 0, 0, scale * translation[0]],
         [0, scale, 0, scale * translation[1]],
         [0, 0, scale, scale * translation[2]],
         [0, 0, 0, 1]]
    )
    return transformation


def join_list_of_mesh(mesh_list):
    assert len(mesh_list) > 0
    if len(mesh_list) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(mesh_list):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
        joint_mesh = bpy.context.object
    else:
        joint_mesh = mesh_list[0]
    return joint_mesh


def load_mesh(mesh_path: str):
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_import(filepath=mesh_path,
                              forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(
            filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    return meshes


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Scale calculation starts. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='input path of a mesh')
    parser.add_argument('--output_transformation_path', type=str,
                        help='path of output transfomation txt')
    parser.add_argument('--standard_height', type=float, default=1.98,
                        help='standard lenght of mesh, defined by mesh\'s bounding sphere')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    output_transformation_path = args.output_transformation_path
    standard_height = args.standard_height

    meshes = load_mesh(mesh_path)
    joint_mesh = join_list_of_mesh(meshes)
    obj_center, length, diagonal, max_point, min_point, mesh_verts = compute_mesh_size(
        meshes)
    print(obj_center, max_point, min_point)

    try:
        scale_matrix = calculate_scale_matrix(
            mesh_verts, standard_height=standard_height)
        print("Try first time works, ", scale_matrix)
        np.savetxt(output_transformation_path, scale_matrix)
    except:
        pass

    time.sleep(0.1)
    if not os.path.exists(output_transformation_path):
        try:
            scale_matrix = calculate_scale_matrix(
                mesh_verts, standard_height=standard_height)
            print("Try second time works, ", scale_matrix)
            np.savetxt(output_transformation_path, scale_matrix)
        except:
            pass

        time.sleep(0.1)
        if not os.path.exists(output_transformation_path):
            try:
                scale_matrix = calculate_scale_matrix(
                    mesh_verts, standard_height=standard_height)
                print("Try third time works, ", scale_matrix)
                np.savetxt(output_transformation_path, scale_matrix)
            except:
                pass

            time.sleep(0.1)
