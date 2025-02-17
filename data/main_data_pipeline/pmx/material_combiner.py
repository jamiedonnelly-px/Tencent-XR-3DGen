import os
import numpy as np
import gc
import argparse
import time
import math
import json
import multiprocessing
import bpy
import bmesh
import sys


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='NEGATIVE_Z', up_axis='Y',
                              global_scale=global_scale)
    else:
        bpy.ops.export_scene.obj(filepath=mesh_path,
                                 use_selection=True,
                                 path_mode=path_mode,
                                 axis_forward='-Z', axis_up='Y',
                                 global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Uncompress a compressed .blend file.')
    parser.add_argument('--input_mesh_path', type=str,
                        help='path to input mesh to be converted')
    parser.add_argument('--output_mesh_folder', type=str,
                        help='folder of output obj mesh file')
    args = parser.parse_args(raw_argv)

    bpy.ops.preferences.addon_enable(module="material-combiner-addon-master")

    input_mesh_path = args.input_mesh_path
    output_mesh_folder = args.output_mesh_folder
    if not os.path.exists(output_mesh_folder):
        os.mkdir(output_mesh_folder)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    load_mesh(mesh_path=input_mesh_path)

    obj_mesh_file = os.path.join(output_mesh_folder, "combined.obj")
    atlas_folder = os.path.join(output_mesh_folder, "atlas")
    if not os.path.exists(atlas_folder):
        os.mkdir(atlas_folder)

    try:
        bpy.ops.smc.refresh_ob_data()
        bpy.ops.smc.combiner(directory=atlas_folder)
    except RuntimeError:
        pass

    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    joint_mesh = join_list_of_mesh(meshes)

    time.sleep(0.1)

    export_mesh_obj(joint_mesh, obj_mesh_file, path_mode="COPY")
    print("Combine obj file from %s and export combined obj to %s" %
          (input_mesh_path, obj_mesh_file))
