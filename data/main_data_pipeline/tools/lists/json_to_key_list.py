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


def write_list(path: str, write_list: list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='json to mesh list')
    parser.add_argument('--input_json_path', type=str,
                        help='path to mesh data json file')
    parser.add_argument('--output_txt_path', type=str,
                        help='path to generated output list file')
    args = parser.parse_args()

    input_json_path = args.input_json_path
    output_txt_path = args.output_txt_path

    data_struct = read_json(input_json_path)
    mesh_key_list = []

    mesh_struct = data_struct["data"]
    for data_name in mesh_struct.keys():
        print(data_name)
        for mesh_name in mesh_struct[data_name].keys():
            print(mesh_name)
            mesh_key_list.append(mesh_name)

    write_list(output_txt_path, mesh_key_list)
