import bpy
import bmesh
import os
import time
import math
import argparse
import random
import sys
import numpy as np
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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    if z_up:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='Y', up_axis='Z',
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
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--force_z_up', action='store_true',
                        help='force use z/y axis in obj exporting')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath
    output_folder = os.path.split(output_fullpath)[0]

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.wm.stl_import(filepath=mesh_path)

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

    time.sleep(0.1)

    joint_mesh = join_list_of_mesh(meshes)
    print("Export stl mesh with from %s to %s" % (mesh_path, output_fullpath))

    if args.copy_texture:
        export_mesh_obj(joint_mesh, output_fullpath, path_mode='COPY', z_up=args.force_z_up)
    else:
        export_mesh_obj(joint_mesh, output_fullpath, path_mode='RELATIVE', z_up=args.force_z_up)
