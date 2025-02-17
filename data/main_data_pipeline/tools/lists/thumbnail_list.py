import argparse
import json
import os
import shutil
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


def write_map(path, index_list, write_map):
    with open(path, 'w') as f:
        for index in range(len(index_list)):
            if index_list[index] in write_map:
                f.write(write_map[index_list[index]] + "\n")


def find_thumbnail_picture_in_folder(folder_path: str, move_folder: str, thumbnail_file_struct: dict):
    parent_folder = os.path.split(folder_path)[0]
    parent_foldername = os.path.split(parent_folder)[1]
    thumbnail_file_path = os.path.join(folder_path, "thumbnail.png")
    if os.path.exists(thumbnail_file_path):
        new_thumbnail_path = os.path.join(
            move_folder, parent_foldername + ".png")
        thumbnail_file_struct[parent_foldername] = {}
        thumbnail_file_struct[parent_foldername]["old"] = thumbnail_file_path
        thumbnail_file_struct[parent_foldername]["new"] = new_thumbnail_path


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--thumbnail_folder_list', nargs='+',
                        help='folder containing rendered thumbnail')
    parser.add_argument('--mesh_txt_list', nargs='+',
                        help='a list of mesh list txt files')
    parser.add_argument('--output_folder', type=str, default="",
                        help='move thumbnail to this folder')

    args = parser.parse_args()
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    thumbnail_move_folder = os.path.join(output_folder, 'thumbnail')
    if not os.path.exists(thumbnail_move_folder):
        os.mkdir(thumbnail_move_folder)
    thumbnail_list_file = os.path.join(output_folder, "mesh_list.txt")

    mesh_list = []
    thumbnail_file_struct = {}
    thumbnail_folder_list = args.thumbnail_folder_list
    mesh_txt_list = args.mesh_txt_list
    print(thumbnail_folder_list)
    for index in range(len(thumbnail_folder_list)):
        single_list = thumbnail_folder_list[index]
        mesh_list_path = mesh_txt_list[index]
        thumbnail_folders = read_list(single_list)
        mesh_list = read_list(mesh_list_path)
        for index2 in range(len(thumbnail_folders)):
            thumbnail_folder_name = thumbnail_folders[index2]
            mesh_list.append(mesh_list[index2])
            find_thumbnail_picture_in_folder(thumbnail_folder_name,
                                             thumbnail_move_folder,
                                             thumbnail_file_struct=thumbnail_file_struct)

    for data_name in thumbnail_file_struct.keys():
        shutil.copyfile(
            thumbnail_file_struct[data_name]["old"], thumbnail_file_struct[data_name]["new"])
    write_list(thumbnail_list_file, mesh_list)
