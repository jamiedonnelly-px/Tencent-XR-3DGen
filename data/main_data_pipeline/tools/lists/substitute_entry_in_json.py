import argparse
import json
import os
import time


def check_data_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    total_data_number = 0
    for category_name in category:
        total_data_number = total_data_number + \
                            len(json_struct["data"][category_name])
    return total_data_number


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


def correct_disk_name(path_name: str):
    if 'apdcephfs_data_cq3' in path_name:
        new_path_name = path_name.replace(
            'apdcephfs_data_cq3', 'apdcephfs_cq3')
        return new_path_name
    else:
        return path_name


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh json start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--original_json', type=str, default="",
                        help='json file containing old data info')
    parser.add_argument('--incoming_json', type=str, default="",
                        help='json file containing modified render folder')
    parser.add_argument('--new_json', type=str, default="",
                        help='json file which substitutes data_name with datas in incoming_json')

    args = parser.parse_args()

    original_json = args.original_json
    incoming_json = args.incoming_json
    new_json = args.new_json
