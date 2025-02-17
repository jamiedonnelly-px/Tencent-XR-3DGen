import argparse
import json
import os
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


def check_individual_category_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        category_number = 0
        if category_name not in number_info.keys():
            number_info[category_name] = {}
        for data_name in json_struct["data"][category_name].keys():
            number_info[category_name][data_name] = len(
                json_struct["data"][category_name][data_name])
            category_number = category_number + \
                              len(json_struct["data"][category_name][data_name])
        number_info[category_name]["Total"] = category_number
    return number_info


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--original_json', type=str, default="",
                        help='json file containing old data info')
    parser.add_argument('--output_json', type=str, default="",
                        help='output json file path of only input files')

    args = parser.parse_args()

    original_json = args.original_json
    output_json = args.output_json

    category_struct = {}
    category_struct["data"] = {}
    data_struct = read_json(original_json)

    for data_name in data_struct["data"].keys():
        for mesh_name in data_struct["data"][data_name].keys():
            category = data_struct["data"][data_name][mesh_name]["Category"]
            if category is None:
                continue
            if category not in category_struct["data"].keys():
                category_struct["data"][category] = {}
            if data_name not in category_struct["data"][category].keys():
                category_struct["data"][category][data_name] = {}
            category_struct["data"][category][data_name][mesh_name] = data_struct["data"][data_name][mesh_name]
    category_output_json = output_json
    write_json(category_output_json, category_struct)
    print("conditions of each category data is %s" %
          (str(check_individual_category_number(category_struct))))
