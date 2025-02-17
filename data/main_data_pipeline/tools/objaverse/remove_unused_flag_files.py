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


def delete_file_single_dir(folder_path: str):
    taks_valid_file = os.path.join(folder_path, "task.valid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--json_file_path', type=str,
                        help='abspath of data json files')
    args = parser.parse_args()

    json_file_path = args.json_file_path

    data_struct = read_json(json_file_path)
    if "data" not in data_struct.keys():
        exit(-1)

    for data_name in data_struct["data"].keys():
        single_data_struct = data_struct["data"][data_name]
        for mesh_name in single_data_struct.keys():
            image_dir = single_data_struct[mesh_name]["ImgDir"]
