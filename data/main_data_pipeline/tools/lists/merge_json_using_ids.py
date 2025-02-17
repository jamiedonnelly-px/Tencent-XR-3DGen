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


def read_list(in_list_txt: str):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--input_json_file', type=str,
                        help='path of first json file')
    parser.add_argument('--mesh_id_list_path', type=str,
                        help='path of object id list')
    parser.add_argument('--output_json_file', type=str,
                        help='path of output json file')
    args = parser.parse_args()

    input_json_file = args.input_json_file
    mesh_id_list_path = args.mesh_id_list_path
    output_json_file = args.output_json_file

    original_json_data = read_json(input_json_file)
    mesh_list_extention = os.path.splitext(mesh_id_list_path)[1]
    select_id_data_name_map = {}
    if mesh_list_extention == '.json':
        mesh_id_struct = read_json(mesh_id_list_path)
        if isinstance(mesh_id_struct, list):
            select_id_data_name_map = {key: "objaverse" for key in mesh_id_struct}
        else:
            for data_name in mesh_id_struct["data"].keys():
                for mesh_name in mesh_id_struct["data"][data_name].keys():
                    select_id_data_name_map[mesh_name] = data_name
    else:
        mesh_id_list = read_list(mesh_id_list_path)
        select_id_data_name_map = {key: "" for key in mesh_id_list}

    mesh_name_data_name_map = {}
    for data_name in original_json_data["data"].keys():
        for mesh_name in original_json_data["data"][data_name].keys():
            mesh_name_data_name_map[mesh_name] = data_name

    new_json_data = {}
    new_json_data["data"] = {}
    for mesh_id in select_id_data_name_map.keys():
        if mesh_id in mesh_name_data_name_map.keys():
            current_data_name = mesh_name_data_name_map[mesh_id]
            if not current_data_name == select_id_data_name_map[mesh_id]:
                continue
            if current_data_name not in new_json_data["data"].keys():
                new_json_data["data"][current_data_name] = {}
            new_json_data["data"][current_data_name][mesh_id] = original_json_data["data"][current_data_name][mesh_id]

    write_json(output_json_file, new_json_data)
    print(check_individual_number(new_json_data))
