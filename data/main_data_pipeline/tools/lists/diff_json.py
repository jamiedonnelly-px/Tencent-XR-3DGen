import argparse
import json
import os


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='data in original but not in candidate (output = original - candidate)')
    parser.add_argument('--original', type=str,
                        help='first path of json')
    parser.add_argument('--candidate', type=str,
                        help='second path of json')
    parser.add_argument('--output_json_path', type=str,
                        help='path of generated diff json')
    args = parser.parse_args()

    json_path_original = args.original
    json_path_candidate = args.candidate
    output_json_path = args.output_json_path

    original_data = read_json(json_path_original)
    candidate_data = read_json(json_path_candidate)

    print(check_individual_number(original_data))
    print(check_individual_number(candidate_data))

    diff_data = {}
    diff_data["data"] = {}

    for data_name in original_data["data"].keys():
        if data_name not in candidate_data["data"].keys():
            diff_data["data"][data_name] = candidate_data["data"][data_name]
        else:
            diff_data["data"][data_name] = {}
            for mesh_name in original_data["data"][data_name].keys():
                if mesh_name not in candidate_data["data"][data_name].keys():
                    diff_data["data"][data_name][mesh_name] = original_data["data"][data_name][mesh_name]

    write_json(output_json_path, diff_data)
    print(check_data_number(diff_data))
    print(check_individual_number(diff_data))
