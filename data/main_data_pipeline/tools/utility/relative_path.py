import argparse
import json
import os


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


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--input_folder', type=str,
                        help='input folder containing files want to be included in txt')
    parser.add_argument('--target_folder', type=str,
                        help='input list txt path')
    parser.add_argument('--output_list_path', type=str,
                        help='output relative path list file path')
    args = parser.parse_args()

    input_folder = args.input_folder
    target_folder = args.target_folder
    output_list_path = args.output_list_path

    relative_path_list = []
    input_file_list = os.listdir(input_folder)
    for input_file_name in input_file_list:
        full_input_path = os.path.join(input_folder, input_file_name)
        relative_path = os.path.relpath(full_input_path, target_folder)
        relative_path_list.append(relative_path)

    list_file_format = os.path.splitext(output_list_path)[1].lower()
    if list_file_format == '.txt':
        write_list(output_list_path, relative_path_list)
    elif list_file_format == '.json':
        write_json(output_list_path, relative_path_list)
