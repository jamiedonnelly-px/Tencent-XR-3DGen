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
from mathutils import Matrix


def load_mesh(mesh_path: str, XY_Axis=False):
    version_info = bpy.app.version
    if version_info[0] > 2:
        if XY_Axis:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='X', up_axis='Y')
        else:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        if XY_Axis:
            bpy.ops.import_scene.obj(
                filepath=mesh_path, axis_forward='X', axis_up='Y')
        else:
            bpy.ops.import_scene.obj(
                filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    return meshes


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    version_info = bpy.app.version
    if version_info[0] > 2:
        if z_up:
            bpy.ops.wm.obj_export(filepath=mesh_path,
                                  path_mode=path_mode,
                                  forward_axis='Y', up_axis='Z',
                                  global_scale=global_scale,
                                  export_selected_objects=True)
        else:
            bpy.ops.wm.obj_export(filepath=mesh_path,
                                  path_mode=path_mode,
                                  forward_axis='NEGATIVE_Z', up_axis='Y',
                                  global_scale=global_scale,
                                  export_selected_objects=True)
    else:
        if z_up:
            bpy.ops.export_scene.obj(filepath=mesh_path,
                                     use_selection=True,
                                     path_mode=path_mode,
                                     axis_forward='Y', axis_up='Z',
                                     global_scale=global_scale)
        else:
            bpy.ops.export_scene.obj(filepath=mesh_path,
                                     use_selection=True,
                                     path_mode=path_mode,
                                     axis_forward='-Z', axis_up='Y',
                                     global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


def read_mesh_to_ndarray(mesh, mode="edit"):
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
        bm.free()
    elif mode == "edit":
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co


def compute_mesh_size(meshes):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        vert = read_mesh_to_ndarray(mesh, mode="edit")
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


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Resize op start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--resize_mesh_path', type=str, default="",
                        help='path of resized mesh')
    parser.add_argument('--standard_height', type=float, default=1.92,
                        help='length of longest edge of bbox of the object (unit meter)')
    parser.add_argument('--input_transformation_txt_path', type=str, default="",
                        help='input transformation txt path, leave empty if do not want to use')
    parser.add_argument('--output_transformation_txt_path', type=str, default="",
                        help='output transformation txt path, leave empty if do not want to use')
    parser.add_argument('--calculate_transformation_only', action='store_true',
                        help='only calculate transformation txt')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy texture image of the object')

    args = parser.parse_args(raw_argv)

    copy_texture = args.copy_texture

    resize_mesh_path = args.resize_mesh_path
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    calculate_transformation_only = args.calculate_transformation_only
    # standard_height = 1.98  # meter
    standard_height = args.standard_height
    input_transformation_txt_path = args.input_transformation_txt_path
    output_transformation_txt_path = args.output_transformation_txt_path

    mesh_path = args.mesh_path
    original_meshes = load_mesh(mesh_path)
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_name)[0]

    obj_center, length, diagonal, _, _, mesh_verts = compute_mesh_size(
        original_meshes)
    print("Mesh center is %s, length is %f....." % (str(obj_center), length))

    original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
    hull_vertices = original_mesh.convex_hull.vertices

    if len(input_transformation_txt_path) > 1 and os.path.exists(input_transformation_txt_path):
        T = np.loadtxt(input_transformation_txt_path)
        print("outside ", T)
        for mesh in original_meshes:
            mesh.matrix_world = Matrix(T) @ mesh.matrix_world
    else:
        try:
            bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
                hull_vertices)
        except:
            time.sleep(0.1)
            print("Miniball failed. Retry once...............")
            try:
                bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
                    hull_vertices)
            except:
                time.sleep(0.1)

        obj_center = bounding_sphere_C
        length = 2 * math.sqrt(bounding_sphere_r2)
        scale = standard_height / length

        for mesh in original_meshes:
            trn = -1 * obj_center[..., np.newaxis]
            T = np.eye(4)
            T[:3, 3:] = scale * trn
            T[:3, :3] = scale * T[:3, :3]
            print(T)
            mesh.matrix_world = Matrix(T) @ mesh.matrix_world

    if len(output_transformation_txt_path) > 1:
        np.savetxt(output_transformation_txt_path, T)

    if not calculate_transformation_only:
        resize_mesh = join_list_of_mesh(original_meshes)
        if copy_texture:
            export_mesh_obj(resize_mesh, resize_mesh_path, 'COPY', z_up=False)
        else:
            export_mesh_obj(resize_mesh, resize_mesh_path, 'AUTO', z_up=False)
    time.sleep(0.1)
