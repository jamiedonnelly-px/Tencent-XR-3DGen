import argparse
import json
import os


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
    parser = argparse.ArgumentParser(description='Batch download files to mesh folder')
    parser.add_argument('--json_folder', type=str,
                        help='folder of json')
    parser.add_argument('--output_json_path', type=str,
                        help='merged json output path')
    args = parser.parse_args()

    json_folder = args.json_folder
    output_json_path = args.output_json_path
    json_merger_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_json_file_list.py")

    pod_folder_list = []
    json_fullpath_list = []

    candidate_folder_list = os.listdir(json_folder)
    for candidate_folder in candidate_folder_list:
        pod_folder_fullpath = os.path.join(json_folder, candidate_folder)
        if os.path.isdir(pod_folder_fullpath):
            pod_folder_list.append(pod_folder_fullpath)

    for pod_folder in pod_folder_list:
        json_files = os.listdir(pod_folder)
        for json_filename in json_files:
            json_extesion = os.path.splitext(json_filename)[1]
            if json_extesion == ".json":
                json_fullpath = os.path.join(pod_folder, json_filename)
                json_fullpath_list.append(json_fullpath)

    merge_cmd = "python {} --output_json_path \'{}\' --json_files ".format(json_merger_fullpath, output_json_path)

    for individual_json in json_fullpath_list:
        individual_str = " \'{}\' ".format(individual_json)
        merge_cmd = merge_cmd + individual_str

    print(merge_cmd)
    os.system(merge_cmd)
