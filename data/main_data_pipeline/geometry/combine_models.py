import argparse
import os
import shutil
import sys
import time

import bpy


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              global_scale=global_scale,
                              export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=mesh_path,
                                 use_selection=True,
                                 path_mode=path_mode,
                                 global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


def load_mesh(mesh_path: str, z_up=False):
    bpy.ops.object.select_all(action='DESELECT')
    version_info = bpy.app.version
    if version_info[0] > 2:
        if z_up:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(
            filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
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

    parser = argparse.ArgumentParser(
        description='Combine on model data script.')
    parser.add_argument('--mesh_path_list', nargs='+',
                        help='LIST of meshes to be combined')
    parser.add_argument('--joint_mesh_folder', type=str, default="",
                        help='output joint mesh results')
    parser.add_argument('--joint_mesh_name', type=str, default="",
                        help='output mesh name')
    args = parser.parse_args(raw_argv)

    mesh_path_list = args.mesh_path_list
    joint_mesh_folder = args.joint_mesh_folder
    joint_mesh_name = args.joint_mesh_name
    if not os.path.exists(joint_mesh_folder):
        os.mkdir(joint_mesh_folder)
    joint_mesh_path = os.path.join(joint_mesh_folder, joint_mesh_name + ".obj")
    mesh_origin_filepath = os.path.join(joint_mesh_folder, "origin.txt")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for mesh_path in mesh_path_list:
        if not os.path.exists(mesh_path):
            print("No mesh at %s!!! EXIT!!!" % (mesh_path))
            exit(-1)
        load_mesh(mesh_path)

    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    joint_mesh = join_list_of_mesh(meshes)
    export_mesh_obj(mesh=joint_mesh, mesh_path=joint_mesh_path,
                    path_mode="COPY")

    time.sleep(0.1)

    print("Join mesh from %s; output path is %s" %
          (str(mesh_path_list), joint_mesh_path))

    write_list(mesh_origin_filepath, mesh_path_list)
