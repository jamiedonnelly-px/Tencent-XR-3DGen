import argparse
import json
import os
import sys
import time

import bpy


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
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


def join_list_of_mesh(mesh_list: list):
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


def check_face_number(the_mesh):
    return len(the_mesh.data.polygons)


def check_mesh_with_certain_material(object, material_name_element: str = 'skin'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            slot_material_name = slot.material.name.lower()
            print(slot_material_name)
            if material_name_element in slot_material_name:
                return True
    return False


def parse_material_name(material_name: str):
    if material_name.startswith("M"):
        return "Male"
    else:
        return "Female"


def formulate_mesh_info(the_mesh, split_info, mesh_name):
    if mesh_name not in split_info.keys():
        split_info[mesh_name] = {}
    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            node_tree = slot.material.node_tree
            slot_material_name = slot.material.name.lower()
            split_info[mesh_name]["material"] = slot_material_name
            split_info[mesh_name]["gender"] = parse_material_name(
                slot_material_name)
            split_info[mesh_name]["face_number"] = check_face_number(the_mesh)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Merge same mesh start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Daz asset processor')
    parser.add_argument('--source_mesh_folder', type=str,
                        help='split mesh folder input path')
    parser.add_argument('--output_json_path', type=str,
                        help='a json of output mesh info json')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_folder = args.source_mesh_folder
    output_json_path = args.output_json_path

    output_struct = {}

    top_obj_name = os.path.join(source_mesh_folder, "top/top.obj")
    bottom_obj_name = os.path.join(source_mesh_folder, "bottom/bottom.obj")
    outfit_obj_name = os.path.join(source_mesh_folder, "outfit/outfit.obj")
    shoe_obj_name = os.path.join(source_mesh_folder, "shoe/shoe.obj")
    hair_obj_name = os.path.join(source_mesh_folder, "hair/hair.obj")

    if os.path.exists(top_obj_name):
        top_mesh_list = load_mesh(top_obj_name)
        top_mesh = join_list_of_mesh(top_mesh_list)
        formulate_mesh_info(top_mesh, output_struct, "top")

    if os.path.exists(bottom_obj_name):
        bottom_mesh_list = load_mesh(bottom_obj_name)
        bottom_mesh = join_list_of_mesh(bottom_mesh_list)
        formulate_mesh_info(bottom_mesh, output_struct, "bottom")

    if os.path.exists(outfit_obj_name):
        outfit_mesh_list = load_mesh(outfit_obj_name)
        outfit_mesh = join_list_of_mesh(outfit_mesh_list)
        formulate_mesh_info(outfit_mesh, output_struct, "outfit")

    if os.path.exists(shoe_obj_name):
        shoe_mesh_list = load_mesh(shoe_obj_name)
        shoe_mesh = join_list_of_mesh(shoe_mesh_list)
        formulate_mesh_info(shoe_mesh, output_struct, "shoe")

    if os.path.exists(hair_obj_name):
        hair_mesh_list = load_mesh(hair_obj_name)
        hair_mesh = join_list_of_mesh(hair_mesh_list)
        formulate_mesh_info(hair_mesh, output_struct, "hair")

    write_json(output_json_path, output_struct)
