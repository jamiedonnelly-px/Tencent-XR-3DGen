import argparse
import os
import sys
import time

import bmesh
import bpy
import mathutils
import numpy as np


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


def export_glb(mesh, mesh_path: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.export_scene.gltf(filepath=mesh_path, use_selection=True)
    time.sleep(0.1)


def process_mesh(input_mesh_path: str, output_mesh_path: str, mesh_extension: str):
    original_meshes = []
    if mesh_extension == ".glb":
        bpy.ops.import_scene.gltf(filepath=input_mesh_path)
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                original_meshes.append(obj)
    else:
        original_meshes = load_mesh(input_mesh_path)

    obj_center, length, diagonal, _, _, mesh_verts = compute_mesh_size(
        original_meshes)
    print("Mesh center is %s, length is %f....." % (str(obj_center), length))

    move_x = obj_center[0]
    move_y = obj_center[1]
    move_z = 0
    move_vector = np.array([move_x, move_y, move_z])

    scale = 1.0
    for mesh in original_meshes:
        trn = -1 * move_vector[..., np.newaxis]
        T = np.eye(4)
        T[:3, 3:] = scale * trn
        T[:3, :3] = scale * T[:3, :3]
        print(T)
        mesh.matrix_world = mathutils.Matrix(T) @ mesh.matrix_world
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        bpy.ops.object.select_all(action='DESELECT')

    resize_mesh = join_list_of_mesh(original_meshes)

    export_glb(resize_mesh, output_mesh_path)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()

    time.sleep(0.1)


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', nargs='+',
                        help='path list to mesh to be rendered')
    parser.add_argument('--resize_mesh_path', nargs='+', default=[],
                        help='path list of output resized mesh')
    args = parser.parse_args(raw_argv)

    mesh_path_list = args.mesh_path
    resize_mesh_path_list = args.resize_mesh_path

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for index in range(len(mesh_path_list)):
        mesh_path = mesh_path_list[index]
        mesh_folder = os.path.split(mesh_path)[0]
        mesh_name = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]
        mesh_extension = os.path.splitext(mesh_name)[1]
        if len(resize_mesh_path_list) <= index:
            resize_mesh_name = mesh_basename + "_resize" + mesh_extension
            resize_fullpath = os.path.join(mesh_folder, resize_mesh_name)
            resize_mesh_path_list.append(resize_fullpath)
        else:
            resize_fullpath = resize_mesh_path_list[index]

        process_mesh(mesh_path, resize_fullpath, mesh_extension)

    time.sleep(0.1)
