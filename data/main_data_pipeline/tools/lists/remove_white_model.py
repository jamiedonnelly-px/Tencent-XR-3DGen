import argparse
import json
import os


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


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


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
    parser = argparse.ArgumentParser(description='mesh list to json')
    parser.add_argument('--mesh_info_json', type=str,
                        help='path to original model info json file')
    parser.add_argument('--white_model_json', type=str,
                        help='path to white model info json file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to mesh json file with no white model')
    parser.add_argument('--test_stage_name', type=str,
                        help='test stage name used for removing')
    args = parser.parse_args()

    mesh_info_json = args.mesh_info_json
    white_model_json = args.white_model_json
    output_json_path = args.output_json_path
    test_stage_name = args.test_stage_name

    mesh_info_struct = read_json(mesh_info_json)
    white_model_info_struct = read_json(white_model_json)

    output_data_struct = {}
    output_data_struct["data"] = {}

    for data_name in mesh_info_struct["data"].keys():
        if data_name not in white_model_info_struct["data"].keys():
            continue
        if data_name not in output_data_struct["data"].keys():
            output_data_struct["data"][data_name] = {}
        for mesh_name in mesh_info_struct["data"][data_name].keys():
            if mesh_name not in white_model_info_struct["data"][data_name].keys():
                continue
            if white_model_info_struct["data"][data_name][mesh_name][test_stage_name]:
                continue
            output_data_struct["data"][data_name][mesh_name] = mesh_info_struct["data"][data_name][mesh_name]

    print(check_individual_number(output_data_struct))
    write_json(output_json_path, output_data_struct)
