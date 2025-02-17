import argparse
import hashlib
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


def calculate_hash_folder_name(input_str: str):
    hash_obj = hashlib.sha1(input_str.encode('utf-8'))
    hash_str = str(hash_obj.hexdigest())
    return hash_str


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
        description='mesh list to json')
    parser.add_argument('--data_json_path', type=str,
                        help='path of input data json file')
    parser.add_argument('--management_json_path', type=str,
                        help='path to management json path')
    parser.add_argument('--output_json_path', type=str,
                        help='path to output data json with management info')
    args = parser.parse_args()

    data_json_path = args.data_json_path
    management_json_path = args.management_json_path
    output_json_path = args.output_json_path

    input_mesh_data = read_json(data_json_path)
    management_data = read_json(management_json_path)

    output_data_struct = {}
    output_data_struct["data"] = {}

    for data_entry in management_data["data"]:
        if "Specific" in data_entry.keys():
            data_category = data_entry["Category"]
            data_mesh_name = data_entry["Key"]
            if data_category not in input_mesh_data["data"].keys():
                continue
            if data_mesh_name not in input_mesh_data["data"][data_category].keys():
                continue
            if data_category not in output_data_struct["data"].keys():
                output_data_struct["data"][data_category] = {}
            output_data_struct["data"][data_category][data_mesh_name] = input_mesh_data["data"][data_category][
                data_mesh_name]
            output_data_struct["data"][data_category][data_mesh_name]["Specific"] = data_entry["Specific"]

    print(check_individual_number(output_data_struct))
    write_json(output_json_path, output_data_struct)
