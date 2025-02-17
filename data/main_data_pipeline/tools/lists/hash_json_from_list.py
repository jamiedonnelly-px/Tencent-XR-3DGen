import argparse
import hashlib
import json
import os

import base62


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
    hash_obj = hashlib.shake_128(input_str.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(20), 16)
    hash_str = base62.encode(hash_int)
    return hash_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mesh list to json')
    parser.add_argument('--mesh_list_txt_path', type=str,
                        help='path to mesh list txt file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated json file')
    parser.add_argument('--data_name', type=str,
                        help='input data name string')
    args = parser.parse_args()

    mesh_list_txt_path = args.mesh_list_txt_path
    output_json_path = args.output_json_path
    data_name = args.data_name

    data_info = {}
    data_info["data"] = {}

    mesh_path_list = read_list(mesh_list_txt_path)
    for mesh_path in mesh_path_list:
        mesh_elements = mesh_path.split('/')
        mesh_name = calculate_hash_folder_name(mesh_path)
        if data_name not in data_info["data"].keys():
            data_info["data"][data_name] = {}
        data_info["data"][data_name][mesh_name] = {}
        data_info["data"][data_name][mesh_name]["Mesh"] = mesh_path

    write_json(output_json_path, data_info)
