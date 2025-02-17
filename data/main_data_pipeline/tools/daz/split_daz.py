import argparse
import os
import sys
import time

import bpy

body_name_list = ["genesis8"]
body_exclude_list = ["eyelash"]
top_name_list = ["top", "shirt", "vest", "coat"]
shoe_name_list = ["shoe", "boot"]
bottom_name_list = ["pant", "bottom"]
outfit_name_list = ["outfit", "robe", "coat"]
hair_name_list = ["hair"]


def check_mesh_with_certain_material(object, material_name: str = 'skin'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            if slot.material.name.lower() == material_name.lower():
                return True
    return False


def remove_material_with_name(object, material_name: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = object
    object.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    # Seperate by material
    bpy.ops.mesh.separate(type='MATERIAL')
    # Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    selected_objects = [o for o in bpy.context.scene.objects if o.select]
    correct_mesh_list = []
    to_delete_mesh_list = []
    for ind, obj in enumerate(selected_objects):
        if obj.type == 'MESH':
            to_delete_result = check_mesh_with_certain_material(
                obj, material_name=material_name)
            if to_delete_result:
                to_delete_mesh_list.append(obj)
            else:
                correct_mesh_list.append(obj)

    print(correct_mesh_list)
    correct_mesh = join_list_of_mesh(correct_mesh_list)
    for skin_mesh in to_delete_mesh_list:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = skin_mesh
        skin_mesh.select_set(True)
        bpy.ops.object.delete(use_global=False)

    return correct_mesh


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


def check_name_in_which_list(mesh_name: str):
    flag_list = [False] * 6

    for hair_element in hair_name_list:
        if hair_element in mesh_name:
            flag_list[0] = True
            break

    for top_element in top_name_list:
        if top_element in mesh_name:
            flag_list[1] = True
            break

    for bottom_element in bottom_name_list:
        if bottom_element in mesh_name:
            flag_list[2] = True
            break

    for outfit_element in outfit_name_list:
        if outfit_element in mesh_name:
            flag_list[3] = True
            break

    for shoe_element in shoe_name_list:
        if shoe_element in mesh_name:
            flag_list[4] = True
            break

    for body_element in body_name_list:
        if body_element in mesh_name:
            exclude_part = False
            for body_exclude_element in body_exclude_list:
                if body_exclude_element in mesh_name:
                    exclude_part = True
                    break
            if exclude_part:
                continue
            flag_list[5] = True
            break

    return flag_list


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, YZ_Axis=False):
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


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Split daz assets start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Daz asset processor')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--output_mesh_folder', type=str, default="",
                        help='path to a folder containing output mesh')
    parser.add_argument('--body_name', type=str, default="",
                        help='body mesh name')
    parser.add_argument('--split_clothes', action='store_true',
                        help='split clothes from full model')
    parser.add_argument('--clothes_name', type=str, default="",
                        help='clothes mesh name')
    parser.add_argument('--split_hair', action='store_true',
                        help='split hair from full model')
    parser.add_argument('--hair_name', type=str, default="",
                        help='hair mesh name')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    output_mesh_folder = args.output_mesh_folder
    body_name = args.body_name
    split_clothes = args.split_clothes
    clothes_name = args.clothes_name
    split_hair = args.split_hair
    hair_name = args.hair_name

    if not os.path.exists(output_mesh_folder):
        os.mkdir(output_mesh_folder)

    hair_mesh_list = []
    if split_hair:
        hair_output_folder = os.path.join(output_mesh_folder, "hair")
        if not os.path.exists(hair_output_folder):
            os.mkdir(hair_output_folder)
        hair_mesh_path = os.path.join(
            hair_output_folder, hair_name + "_hair.obj")

    top_mesh_list = []
    bottom_mesh_list = []
    outfit_mesh_list = []
    shoe_mesh_list = []

    if split_clothes:
        top_output_folder = os.path.join(output_mesh_folder, "top")
        if not os.path.exists(top_output_folder):
            os.mkdir(top_output_folder)
        top_mesh_path = os.path.join(
            top_output_folder, clothes_name + "_top.obj")

        bottom_output_folder = os.path.join(output_mesh_folder, "bottom")
        if not os.path.exists(bottom_output_folder):
            os.mkdir(bottom_output_folder)
        bottom_mesh_path = os.path.join(
            bottom_output_folder, clothes_name + "_bottom.obj")

        outfit_output_folder = os.path.join(output_mesh_folder, "outfit")
        if not os.path.exists(outfit_output_folder):
            os.mkdir(outfit_output_folder)
        outfit_mesh_path = os.path.join(
            outfit_output_folder, clothes_name + "_outfit.obj")

        shoe_output_folder = os.path.join(output_mesh_folder, "shoe")
        if not os.path.exists(shoe_output_folder):
            os.mkdir(shoe_output_folder)
        shoe_mesh_path = os.path.join(
            shoe_output_folder, clothes_name + "_shoe.obj")

    body_mesh_list = []
    body_output_folder = os.path.join(output_mesh_folder, "body")
    if not os.path.exists(body_output_folder):
        os.mkdir(body_output_folder)
    body_mesh_path = os.path.join(body_output_folder, body_name + "_body.obj")

    meshes = load_mesh(source_mesh_path)
    for mesh in meshes:
        mesh_name = mesh.name.lower()
        flag_list = check_name_in_which_list(mesh_name)
        print(mesh_name, flag_list)
        if flag_list[0]:
            hair_mesh_list.append(mesh)
        elif flag_list[1]:
            top_mesh_list.append(mesh)
        elif flag_list[2]:
            bottom_mesh_list.append(mesh)
        elif flag_list[3]:
            outfit_mesh_list.append(mesh)
        elif flag_list[4]:
            shoe_mesh_list.append(mesh)
        elif flag_list[5]:
            body_mesh_list.append(mesh)

    if split_hair:
        if len(hair_mesh_list) > 0:
            hair_mesh = join_list_of_mesh(hair_mesh_list)
            export_mesh_obj(hair_mesh, hair_mesh_path, path_mode='COPY')

    if split_clothes:
        if len(top_mesh_list) > 0:
            top_mesh = join_list_of_mesh(top_mesh_list)
            export_mesh_obj(top_mesh, top_mesh_path, path_mode='COPY')

        if len(bottom_mesh_list) > 0:
            bottom_mesh = join_list_of_mesh(bottom_mesh_list)
            export_mesh_obj(bottom_mesh, bottom_mesh_path, path_mode='COPY')

        if len(outfit_mesh_list) > 0:
            outfit_mesh = join_list_of_mesh(outfit_mesh_list)
            export_mesh_obj(outfit_mesh, outfit_mesh_path, path_mode='COPY')

        if len(shoe_mesh_list) > 0:
            shoe_mesh = join_list_of_mesh(shoe_mesh_list)
            export_mesh_obj(shoe_mesh, shoe_mesh_path, path_mode='COPY')

    if len(body_mesh_list) > 0:
        body_mesh = join_list_of_mesh(body_mesh_list)
        correct_body_mesh = remove_material_with_name(body_mesh, "Cornea")
        export_mesh_obj(correct_body_mesh, body_mesh_path, path_mode='COPY')

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Split daz objects done. Local time is %s" % (local_time_str))
