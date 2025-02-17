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
        description='mesh list to json')
    parser.add_argument('--input_lists', nargs='+',
                        help='a list of mesh list files')
    parser.add_argument('--output_list_path', type=str,
                        help='path to merged list file with all lists')
    args = parser.parse_args()

    input_lists = args.input_lists
    output_list_path = args.output_list_path

    file_basename_path_map = {}
    for input_list_name in input_lists:
        list_1 = read_list(input_list_name)
        for file_path in list_1:
            file_basename = os.path.splitext(file_path)[0]
            file_extension = os.path.splitext(file_path)[1]
            if file_basename not in file_basename_path_map.keys():
                file_basename_path_map[file_basename] = file_extension
            else:
                if file_basename_path_map[file_basename] == ".dae":
                    if file_extension == ".fbx":
                        file_basename_path_map[file_basename] = file_extension
                if file_basename_path_map[file_basename] == ".obj":
                    file_basename_path_map[file_basename] = file_extension

    final_file_list = []
    for file_basename in file_basename_path_map.keys():
        final_file_list.append(file_basename + file_basename_path_map[file_basename])

    write_list(output_list_path, final_file_list)
