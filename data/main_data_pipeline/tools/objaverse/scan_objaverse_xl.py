import argparse
import json
import os
import time


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    total_number = 0
    for category_name in category:
        mesh_number = len(json_struct["data"][category_name])
        number_info[category_name] = mesh_number
        total_number = total_number + mesh_number
    number_info["all"] = total_number
    return number_info


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
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate objaverse xl mesh json start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Renders multi-gpu with pool.')
    parser.add_argument('--oxl_folders', nargs='+',
                        help='folder taht could contain oxl mesh')
    parser.add_argument('--output_list_path', type=str, default="",
                        help='output objaverse xl mesh info json')
    args = parser.parse_args()

    oxl_folders = args.oxl_folders
    output_list_path = args.output_list_path

    for objaverse_xl_folder in oxl_folders:
        if not os.path.exists(objaverse_xl_folder):
            print("Folder %s do not exists..." % (objaverse_xl_folder))
            exit(-1)

    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    mesh_info_struct["data"]["oxl"] = {}

    for objaverse_xl_folder in oxl_folders:
        first_layer_folders = os.listdir(objaverse_xl_folder)
        for folder_name in first_layer_folders:
            folder_fullpath = os.path.join(objaverse_xl_folder, folder_name)
            if not os.path.isdir(folder_fullpath):
                continue
            info_json_path = os.path.join(folder_fullpath, ".objaverse-file-hashes.json")
            if not os.path.exists(info_json_path):
                continue
            repo_info = read_json(info_json_path)
            for mesh_info in repo_info:
                mesh_key = mesh_info["sha256"]
                mesh_url = str(mesh_info["fileIdentifier"])
                mesh_url_elements = mesh_url.split("/")
                split_index = mesh_url_elements.index("blob")
                relative_path_start_index = split_index + 2
                relative_path = "/".join(mesh_url_elements[relative_path_start_index:])
                mesh_full_path = os.path.join(folder_fullpath, relative_path)
                if not os.path.exists(mesh_full_path):
                    print("Mesh do not exist at %s, url is %s" % (mesh_full_path, mesh_url))
                    continue
                mesh_info_struct["data"]["oxl"][mesh_key] = {}
                mesh_info_struct["data"]["oxl"][mesh_key]["Mesh"] = mesh_full_path
                print("Add mesh at %s, url is %s" % (mesh_full_path, mesh_url))

    print(check_individual_number(mesh_info_struct))
    write_json(output_list_path, mesh_info_struct)
