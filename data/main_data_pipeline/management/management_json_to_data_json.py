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
    parser = argparse.ArgumentParser(description='Management json file to data json file')
    parser.add_argument('--management_json_path', type=str, default='',
                        help='abspath of management json files')
    parser.add_argument('--data_json_path', type=str, default='',
                        help='output data json path')
    args = parser.parse_args()

    management_json_path = args.management_json_path
    data_json_path = args.data_json_path

    management_struct = read_json(management_json_path)
    output_struct = {}
    output_struct["data"] = {}

    for management_data_entry in management_struct["data"]:
        data_name = management_data_entry["Category"]
        mesh_key = management_data_entry["Key"]
        mesh_path = management_data_entry["SavePaths"]["MeshFilename"]

        if data_name not in output_struct["data"].keys():
            output_struct["data"][data_name] = {}

        if mesh_key not in output_struct["data"][data_name].keys():
            output_struct["data"][data_name][mesh_key] = {}

        output_struct["data"][data_name][mesh_key]["Mesh"] = mesh_path

    print(check_individual_number(output_struct))
    write_json(data_json_path, output_struct)
