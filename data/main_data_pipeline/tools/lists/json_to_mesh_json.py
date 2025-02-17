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


def write_list(path: str, write_list: list):
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
        description='json to mesh list')
    parser.add_argument('--json_files', nargs='+',
                        help='abspath of json files')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated output json file with only mesh')
    args = parser.parse_args()

    json_files = args.json_files
    output_json_path = args.output_json_path
    output_struct = {}
    output_struct["data"] = {}

    for json_filepath in json_files:
        data_struct = read_json(json_filepath)
        print(check_individual_number(data_struct))
        mesh_struct = data_struct["data"]
        for data_name in mesh_struct.keys():
            if data_name not in output_struct["data"].keys():
                output_struct["data"][data_name] = {}
            for mesh_name in mesh_struct[data_name].keys():
                if "Mesh" in mesh_struct[data_name][mesh_name].keys():
                    if mesh_struct[data_name][mesh_name]["Mesh"] is not None:
                        output_struct["data"][data_name][mesh_name] = {}
                        output_struct["data"][data_name][mesh_name]["Mesh"] = mesh_struct[data_name][mesh_name]["Mesh"]

    print("Total number is:")
    print(check_individual_number(output_struct))
    write_json(output_json_path, output_struct)
