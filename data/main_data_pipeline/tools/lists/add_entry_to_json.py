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
        description='Merge json using json file lists')
    parser.add_argument('--json_files', nargs='+',
                        help='abspath of json files containing entries to add')
    parser.add_argument('--original_json_path', type=str,
                        help='original json path')
    parser.add_argument('--output_json_path', type=str,
                        help='merged json output path')
    args = parser.parse_args()

    json_path_list = args.json_files
    original_json_data = read_json(args.original_json_path)
    print("Original data struct length is %s" %
          (str(check_individual_number(original_json_data))))

    for json_path in json_path_list:
        to_add_struct = read_json(json_path)
        for data_name in to_add_struct["data"]:
            if data_name not in original_json_data["data"].keys():
                continue
            for mesh_name in to_add_struct["data"][data_name]:
                if mesh_name not in original_json_data["data"][data_name].keys():
                    continue
                original_json_data["data"][data_name][mesh_name]["GLB_Mesh"] = \
                to_add_struct["data"][data_name][mesh_name]["Mesh"]

    write_json(args.output_json_path, original_json_data)
    print("Merged data struct length is %s" %
          (str(check_individual_number(original_json_data))))
