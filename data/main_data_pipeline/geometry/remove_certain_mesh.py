import argparse
import os
import sys
import time

import bpy


def load_mesh(mesh_path: str, forward: str = 'NEGATIVE_Z', up: str = 'Y'):
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_import(filepath=mesh_path,
                              forward_axis=forward, up_axis=up)
    else:
        if forward == "NEGATIVE_Z":
            forward = "-Z"
        if forward == "NEGATIVE_Y":
            forward = "-Y"
        if forward == "NEGATIVE_X":
            forward = "-X"
        if up == "NEGATIVE_Z":
            up = "-Z"
        if up == "NEGATIVE_Y":
            up = "-Y"
        if up == "NEGATIVE_X":
            up = "-X"
        bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward=forward, axis_up=up)
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
                              export_selected_objects=True,
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


def check_mesh_with_certain_material(object, material_name: str = 'skin'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            if slot.material.name == material_name:
                return True
    return False


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Mesh direction correction process start. Local time is %s" %
          (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be corrected')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path of output mesh')
    parser.add_argument('--remove_name_list_str', type=str, default="",
                        help='remove all meshes with these material')

    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_mesh_path = args.output_mesh_path
    remove_name_list_str = args.remove_name_list_str
    to_remove_material_name_list = remove_name_list_str.split('+')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    original_meshes = load_mesh(mesh_path)
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_name)[0]
    combine_mesh = join_list_of_mesh(original_meshes)

    bpy.ops.object.mode_set(mode='EDIT')
    # Seperate by material
    bpy.ops.mesh.separate(type='MATERIAL')
    # Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    time.sleep(0.1)

    correct_mesh_list = []
    skin_mesh_list = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            obj_erased = False
            for material_name in to_remove_material_name_list:
                if check_mesh_with_certain_material(obj, material_name=material_name):
                    skin_mesh_list.append(obj)
                    obj_erased = True
                    break
            if not obj_erased:
                correct_mesh_list.append(obj)

    combine_correct_mesh = join_list_of_mesh(correct_mesh_list)
    export_mesh_obj(combine_correct_mesh, output_mesh_path, 'COPY')
    time.sleep(0.1)
