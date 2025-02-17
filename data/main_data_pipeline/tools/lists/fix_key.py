import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--original_json', type=str,
                        help='original merged json path')
    parser.add_argument('--new_json', type=str,
                        help='new json path')

    args = parser.parse_args()

    original_json_data = read_json(args.original_json)
    new_json_data = {}
    new_json_data["data"] = {}

    original_key_list = []
    new_key_list = []

    for data_name in original_json_data["data"].keys():
        if data_name not in new_json_data["data"].keys():
            new_json_data["data"][data_name] = {}
        for mesh_name in original_json_data["data"][data_name].keys():
            mesh_path_elements = original_json_data["data"][data_name][mesh_name]["Mesh"].split(
                "/")
            new_mesh_name = mesh_path_elements[-3]
            if "_output_512_MightyWSB" in new_mesh_name:
                new_mesh_name = new_mesh_name.replace("_output_512_MightyWSB", "")
            else:
                new_mesh_name = new_mesh_name
            new_json_data["data"][data_name][new_mesh_name] = original_json_data["data"][data_name][mesh_name]

    write_json(args.new_json, new_json_data)
