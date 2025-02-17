from mathutils import Matrix, Vector, Quaternion, Euler
import bpy
import bmesh
import os
import time
import json
import math
import argparse
import random
import sys
import gc
import numpy as np
from numpy import arange, sin, cos, arccos
import shutil

weapon = ["weapon", "Weapon"]


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, YZ_Axis=False, ZY_Axis=False):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    if YZ_Axis:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='Y', up_axis='Z',
                              global_scale=global_scale)
    elif ZY_Axis:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='Z', up_axis='Y',
                              global_scale=global_scale)
    else:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='NEGATIVE_Z', up_axis='Y',
                              global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


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

    return obj_center, length, diagonal, min_, max_


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


def copy_image_textures(object, new_image_folder):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        image_path = node.image.filepath
                        image_filename = os.path.split(image_path)[1]
                        new_image_path = os.path.join(
                            new_image_folder, image_filename)
                        shutil.copyfile(image_path, new_image_path)


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_fullpath', type=str,
                        default="", help='render result output folder')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better fbx as import plugin')
    parser.add_argument('--apply_pose_toggle', action='store_true',
                        help='force toggle pose mode of the model')
    parser.add_argument('--apply_center_bias', action='store_true',
                        help='force move the model to center')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--force_zy', action='store_true',
                        help='force use z/y axis in obj exporting')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath
    output_folder = os.path.split(output_fullpath)[0]
    os.makedirs(output_folder, exist_ok=True)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    if args.force_better_fbx:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # print(bpy.ops.preferences.addon_expand(module="better_fbx"))

        addon_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'addons/better_fbx.zip')
        bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
                                          filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
        bpy.ops.preferences.addon_enable(module="better_fbx")

        bpy.ops.better_import.fbx(filepath=mesh_path, use_optimize_for_blender=False,
                                  use_auto_bone_orientation=True,
                                  use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                  use_detect_deform_bone=True, use_auto_smooth=True,
                                  use_animation=True)

    else:
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)
        if args.apply_pose_toggle:
            try:
                # switch character to rest mode, i.e. A-pose in most case
                bpy.ops.object.posemode_toggle()
                bpy.ops.pose.select_all(action='SELECT')
                bpy.ops.pose.loc_clear()
                bpy.ops.pose.rot_clear()
                bpy.ops.pose.scale_clear()
                bpy.ops.object.posemode_toggle()
            except:
                print('posemode_toggle failed')

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            if any(wp in obj.name for wp in weapon):
                obj.select_set(state=True)
                bpy.ops.object.delete()
            else:
                meshes.append(obj)

    if len(meshes) < 1:
        print("No avatar found in model.....")
        exit(-1)

    for mesh in meshes:
        if mesh.scale[0] < 0.99:
            mesh.scale[0] = 1.0
        if mesh.scale[1] < 0.99:
            mesh.scale[1] = 1.0
        if mesh.scale[2] < 0.99:
            mesh.scale[2] = 1.0

    time.sleep(0.1)
    obj_center, length, diagonal, _, _ = compute_mesh_size(meshes)

    print("Mesh center is %s, lenght is %f....." % (str(obj_center), length))
    standard_height = 1.98  # meter
    scale = standard_height / length
    if args.apply_center_bias:
        for mesh in meshes:
            trn = -1 * obj_center[..., np.newaxis]
            T = np.eye(4)
            T[:3, 3:] = scale * trn
            T[:3, :3] = scale * T[:3, :3]
            print(T)
            mesh.matrix_world = Matrix(T) @ mesh.matrix_world

    joint_mesh = join_list_of_mesh(meshes)
    print("Export fbx mesh with from %s to %s" % (mesh_path, output_fullpath))

    export_mesh_obj(joint_mesh, output_fullpath,
                    path_mode='RELATIVE', ZY_Axis=args.force_zy)
    if args.copy_texture:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='COPY', ZY_Axis=args.force_zy)
    else:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='RELATIVE', ZY_Axis=args.force_zy)