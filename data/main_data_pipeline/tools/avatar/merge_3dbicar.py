import argparse
import hashlib
import json
import os
import time


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt):
    str_list = []
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        return str_list

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


def calculate_hash_folder_name(input_list_str: list):
    input_str = "_".join(input_list_str)
    hash_obj = hashlib.sha1(input_str.encode('utf-8'))
    hash_str = str(hash_obj.hexdigest())
    return hash_str


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    total_number = 0
    for category_name in category:
        mesh_number = len(json_struct["data"][category_name])
        number_info[category_name] = mesh_number
        total_number = total_number + mesh_number
    number_info["all"] = total_number
    return number_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove mesh with certain material name.')
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--output_data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.1-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    output_data_json_path = args.output_data_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    temporal_folder = os.path.join(output_folder, "temporal")
    if not os.path.exists(temporal_folder):
        os.mkdir(temporal_folder)
    category_folder = os.path.join(output_folder, "3DBiCar")
    if not os.path.exists(category_folder):
        os.mkdir(category_folder)

    mesh_path_list = read_list(in_mesh_list_txt)

    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    mesh_info_struct["data"]["3DBiCar"] = {}

    mesh_key_mesh_parts_info = {}

    merge_mesh_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../manifold/merge_mesh.py")
    texture_name_mesh_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modify_3dbicar_texture_file.py")

    for mesh_path in mesh_path_list:
        mesh_elements = mesh_path.split("/")
        mesh_key = mesh_elements[-3]
        if mesh_key not in mesh_key_mesh_parts_info.keys():
            mesh_key_mesh_parts_info[mesh_key] = []
        if mesh_elements[-2] == "pose":
            mesh_key_mesh_parts_info[mesh_key].append(mesh_path)

    for mesh_key in mesh_key_mesh_parts_info.keys():
        new_mesh_folder_name = calculate_hash_folder_name(mesh_key_mesh_parts_info[mesh_key])
        new_mesh_folder_fullpath = os.path.join(category_folder, new_mesh_folder_name)
        if not os.path.exists(new_mesh_folder_fullpath):
            os.mkdir(new_mesh_folder_fullpath)
        new_mesh_filename = new_mesh_folder_name + ".obj"
        new_mesh_file_fullpath = os.path.join(new_mesh_folder_fullpath, new_mesh_filename)

        texture_correct_mesh_path_list = []
        for mesh_path in mesh_key_mesh_parts_info[mesh_key]:
            mesh_filename = os.path.split(mesh_path)[1]
            temporal_mesh_folder = os.path.join(temporal_folder, new_mesh_folder_name)
            if not os.path.exists(temporal_mesh_folder):
                os.mkdir(temporal_mesh_folder)

            temporal_mesh_filename = os.path.join(temporal_mesh_folder, mesh_filename)

            modify_texture_cmd = "{} -b -P {} -- ".format(blender_root, texture_name_mesh_op)
            modify_texture_cmd = modify_texture_cmd + " --mesh_path \'{}\' ".format(mesh_path)
            modify_texture_cmd = modify_texture_cmd + " --output_mesh_path \'{}\' ".format(temporal_mesh_filename)

            print(modify_texture_cmd)
            os.system(modify_texture_cmd)
            time.sleep(0.1)

            texture_correct_mesh_path_list.append(temporal_mesh_filename)

        merge_mesh_cmd = "{} -b -P {} -- ".format(blender_root, merge_mesh_op)
        merge_mesh_cmd = merge_mesh_cmd + " --copy_texture --mesh_path_list "
        for mesh_path in texture_correct_mesh_path_list:
            merge_mesh_cmd = merge_mesh_cmd + " \'{}\' ".format(mesh_path)
        merge_mesh_cmd = merge_mesh_cmd + " --output_mesh_path \'{}\' ".format(new_mesh_file_fullpath)

        print(merge_mesh_cmd)
        os.system(merge_mesh_cmd)
        time.sleep(0.1)

        mesh_info_struct["data"]["3DBiCar"][new_mesh_folder_name] = {}
        mesh_info_struct["data"]["3DBiCar"][new_mesh_folder_name]["Mesh"] = new_mesh_file_fullpath

    print(check_individual_number(mesh_info_struct))
    write_json(output_data_json_path, mesh_info_struct)
