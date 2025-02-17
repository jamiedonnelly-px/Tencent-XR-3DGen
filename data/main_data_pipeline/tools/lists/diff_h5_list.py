import argparse
import json
import os


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


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--h5_txt_path', type=str,
                        help='h5 file path list path')
    parser.add_argument('--data_json_path', type=str,
                        help='path of full data json')
    parser.add_argument('--output_json_path', type=str,
                        help='output json file with all ids not converted to h5')
    args = parser.parse_args()

    h5_txt_path = args.h5_txt_path
    data_json_path = args.data_json_path
    output_json_path = args.output_json_path

    h5_list = read_list(h5_txt_path)
    data_struct = read_json(data_json_path)

    print(len(h5_list))
    print(check_individual_number(data_struct))

    id_h5_convert_status_set = set()

    for h5_path in h5_list:
        h5_filename = os.path.split(h5_path)[1]
        h5_basename = os.path.splitext(h5_filename)[0]
        id_h5_convert_status_set.add(h5_basename)

    output_data_struct = {}
    output_data_struct['data'] = {}
    for data_name in data_struct["data"].keys():
        for mesh_name in data_struct["data"][data_name].keys():
            if mesh_name in id_h5_convert_status_set:
                continue
            if data_name not in output_data_struct["data"].keys():
                output_data_struct["data"][data_name] = {}
            output_data_struct["data"][data_name][mesh_name] = data_struct["data"][data_name][mesh_name]

    write_json(output_json_path, output_data_struct)
    print(check_individual_number(output_data_struct))
