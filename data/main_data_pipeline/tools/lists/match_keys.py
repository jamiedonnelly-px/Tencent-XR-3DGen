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
    parser.add_argument('--new_json', type=str, default='',
                        help='new json path')
    parser.add_argument('--output_key_match_path', type=str,
                        help='key match json path')

    args = parser.parse_args()

    original_json_data = read_json(args.original_json)
    new_json_data = read_json(args.new_json)

    original_key_list = []
    new_key_list = []

    for data_name in original_json_data["data"].keys():
        for mesh_name in original_json_data["data"][data_name].keys():
            original_key_list.append(mesh_name)

    for data_name in new_json_data["data"].keys():
        for mesh_name in new_json_data["data"][data_name].keys():
            new_key_list.append(mesh_name)

    new_original_map = {}
    for new_key in new_key_list:
        for old_key in original_key_list:
            new_key_low = new_key.lower()
            old_key_low = old_key.lower()
            if new_key_low in old_key_low:
                new_original_map[new_key] = old_key
                break

    write_json(args.output_key_match_path, new_original_map)
