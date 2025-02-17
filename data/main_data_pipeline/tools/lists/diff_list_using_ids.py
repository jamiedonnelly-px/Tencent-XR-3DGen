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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--txt_path', type=str,
                        help='file path list path')
    parser.add_argument('--mesh_id_list_path', type=str,
                        help='path of mesh file hash ids')
    parser.add_argument('--output_txt', type=str,
                        help='output txt file with path of selected ids')
    parser.add_argument('--output_hash_rest_meshes', type=str, default="",
                        help='output txt file with hash id of NOT selected ids')
    parser.add_argument('--output_txt_rest_meshes', type=str, default="",
                        help='output txt file with path of NOT selected ids')
    args = parser.parse_args()

    txt_path = args.txt_path
    mesh_id_list_path = args.mesh_id_list_path
    output_txt = args.output_txt
    output_hash_rest_meshes = args.output_hash_rest_meshes
    output_txt_rest_meshes = args.output_txt_rest_meshes

    mesh_list_extention = os.path.splitext(mesh_id_list_path)[1]
    if mesh_list_extention == '.json':
        hash_list = read_json(mesh_id_list_path)
    else:
        hash_list = read_list(mesh_id_list_path)

    path_list = read_list(txt_path)

    hash_mesh_path_map = {}
    output_mesh_path = []
    rest_mesh_list = []
    rest_hash_list = []
    for mesh_path in path_list:
        mesh_folder = os.path.split(mesh_path)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        hash_mesh_path_map[mesh_folder_name] = mesh_path

    for hash_id in hash_list:
        if hash_id not in hash_mesh_path_map.keys():
            continue
        if "white_" in hash_mesh_path_map[hash_id]:
            rest_mesh_list.append(hash_mesh_path_map[hash_id])
            rest_hash_list.append(hash_id)
        else:
            output_mesh_path.append(hash_mesh_path_map[hash_id])

    print(len(output_mesh_path))
    print(len(rest_mesh_list), len(rest_hash_list))
    write_list(output_txt, output_mesh_path)
    if len(output_txt_rest_meshes) > 0:
        write_list(output_hash_rest_meshes, rest_hash_list)
        write_list(output_txt_rest_meshes, rest_mesh_list)
