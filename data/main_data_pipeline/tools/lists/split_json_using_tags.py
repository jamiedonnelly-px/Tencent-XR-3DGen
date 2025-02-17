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


def check_str_in_list(check_str: str, str_list: list):
    for str_element in str_list:
        if str_element in check_str:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--input_json_file', type=str,
                        help='path of first json file')
    parser.add_argument('--mesh_tags_str', type=str,
                        help='path of object id list')
    parser.add_argument('--output_json_file', type=str,
                        help='path of output json file')
    args = parser.parse_args()

    input_json_file = args.input_json_file
    mesh_tags_str = args.mesh_tags_str
    output_json_file = args.output_json_file

    original_json_data = read_json(input_json_file)
    if len(mesh_tags_str) < 1:
        print("No mesh tag input................")
        exit(-1)

    mesh_tag_list = mesh_tags_str.split("+")
    # mesh_tag_set=set(mesh_tag_list)
    data_struct = original_json_data["data"]

    new_json_data = {}
    new_json_data["data"] = {}

    for data_name in data_struct.keys():
        if not check_str_in_list(data_name, mesh_tag_list):
            continue
        new_json_data["data"][data_name] = data_struct[data_name]

    write_json(output_json_file, new_json_data)
    print(check_individual_number(new_json_data))
