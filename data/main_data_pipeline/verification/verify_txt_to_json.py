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
    parser.add_argument('--verify_info_txt_path', type=str,
                        help='path to verify info txt file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated json file')
    args = parser.parse_args()

    verify_info_txt_path = args.verify_info_txt_path
    output_json_path = args.output_json_path

    verification_json_str_list = read_list(verify_info_txt_path)
    verified_data_info_struct = {}
    verified_data_info_struct["data"] = {}
    for verify_info in verification_json_str_list:
        current_json_object = json.loads(verify_info)
        data_category = next(iter(current_json_object))
        if data_category == "data":
            data_category = next(iter(current_json_object["data"]))
            data_key = next(iter(current_json_object["data"][data_category]))
            if data_category not in verified_data_info_struct["data"].keys():
                verified_data_info_struct["data"][data_category] = {}
            verified_data_info_struct["data"][data_category][data_key] = current_json_object["data"][data_category][
                data_key]
        else:
            data_category = next(iter(current_json_object))
            data_key = next(iter(current_json_object[data_category]))
            if data_category not in verified_data_info_struct["data"].keys():
                verified_data_info_struct["data"][data_category] = {}
            verified_data_info_struct["data"][data_category][data_key] = current_json_object[data_category][data_key]

    write_json(output_json_path, verified_data_info_struct)
    print(check_individual_number(verified_data_info_struct))
