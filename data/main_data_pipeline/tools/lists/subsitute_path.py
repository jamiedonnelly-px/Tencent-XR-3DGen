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


def scan_source_log_folder(source_log_folder: str):
    log_folder_list = []
    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            log_folder_list.append(folder_fullpath)
    log_folder_list.sort()
    return log_folder_list


def check_data_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    total_data_number = 0
    for category_name in category:
        total_data_number = total_data_number + \
                            len(json_struct["data"][category_name])
    return total_data_number


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        number_info[category_name] = len(json_struct["data"][category_name])
    return number_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--json_path', type=str,
                        help='path of data json')
    parser.add_argument('--new_json_path', type=str,
                        help='path of output data json')
    parser.add_argument('--mesh_info_tags', nargs='+',
                        help='a group of mesh info entries to be substituted')
    parser.add_argument('--incoming_data_json', type=str,
                        help='data json file containing incoming data')
    args = parser.parse_args()

    json_path = args.json_path
    new_json_path = args.new_json_path
    # data_name = args.data_name
    mesh_info_tags = args.mesh_info_tags
    incoming_data_json = args.incoming_data_json

    original_data = read_json(json_path=json_path)
    incoming_data = read_json(json_path=incoming_data_json)

    new_data_struct = {}
    new_data_struct["data"] = {}

    original_mesh_data = original_data["data"]
    incoming_mesh_data = incoming_data["data"]

    for data_name in original_mesh_data.keys():
        if data_name not in incoming_mesh_data.keys():
            new_data_struct["data"][data_name] = original_mesh_data[data_name]
        else:
            if data_name not in new_data_struct["data"].keys():
                new_data_struct["data"][data_name] = {}
            for mesh_name in incoming_mesh_data[data_name].keys():
                if mesh_name not in original_mesh_data[data_name].keys():
                    continue
                if mesh_name not in new_data_struct["data"][data_name].keys():
                    new_data_struct["data"][data_name][mesh_name] = original_mesh_data[data_name][mesh_name]
                for tag in mesh_info_tags:
                    new_data_struct["data"][data_name][mesh_name][tag] = incoming_mesh_data[data_name][mesh_name][tag]
                    if tag == "Mesh":
                        if "Obj_Mesh" in original_mesh_data[data_name][mesh_name].keys():
                            new_data_struct["data"][data_name][mesh_name]["Obj_Mesh"] = \
                                incoming_mesh_data[data_name][mesh_name]["Mesh"]

    print(check_data_number(new_data_struct))
    write_json(new_json_path, new_data_struct)
