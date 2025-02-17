import argparse
import os
import sys
import time

import bpy


def load_mesh(mesh_path: str, XY_Axis=False):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]
    mesh_extension = mesh_extension.lower()
    if mesh_extension == ".obj":
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
    elif mesh_extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)
        bpy.ops.object.select_all(action='DESELECT')
        meshes = []
        for ind, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                meshes.append(obj)
        return meshes


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


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Parse mesh info start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path_list', nargs='+',
                        help='a group of paths to mesh to be parsed')
    parser.add_argument('--output_mesh_path', type=str, default="",
                        help='path of output mesh file')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')

    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path_list = args.mesh_path_list
    output_mesh_path = args.output_mesh_path

    full_mesh_list = []
    for mesh_path in mesh_path_list:
        original_mesh = load_mesh(mesh_path)
        full_mesh_list.extend(original_mesh)

    joint_mesh = join_list_of_mesh(full_mesh_list)

    if args.copy_texture:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='COPY')
    else:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='RELATIVE')
