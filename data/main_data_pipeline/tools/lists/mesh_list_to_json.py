import argparse
import hashlib
import json
import os


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


def calculate_hash_folder_name(input_str: str):
    hash_obj = hashlib.sha1(input_str.encode('utf-8'))
    hash_str = str(hash_obj.hexdigest())
    return hash_str


def remove_space(input_str: str):
    result_str = input_str.replace(" ", "_")
    return result_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mesh list to json')
    parser.add_argument('--mesh_list_txt_path', type=str,
                        help='path to mesh list txt file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated json file')
    args = parser.parse_args()

    mesh_list_txt_path = args.mesh_list_txt_path
    output_json_path = args.output_json_path

    data_info = {}
    data_info["data"] = {}

    mesh_path_list = read_list(mesh_list_txt_path)
    for mesh_path in mesh_path_list:
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]
        mesh_extension = os.path.splitext(mesh_filename)[1]
        mesh_elements = mesh_path.split('/')
        data_name = mesh_elements[-3]
        mesh_name = mesh_elements[-2]
        # mesh_name = mesh_name.replace("-", "_")
        if data_name not in data_info["data"].keys():
            data_info["data"][data_name] = {}
        data_info["data"][data_name][mesh_name] = {}
        data_info["data"][data_name][mesh_name]["Mesh"] = mesh_path

    write_json(output_json_path, data_info)
