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
        description='resolv high heel string in mcwy datas')
    parser.add_argument('--mesh_json_path', type=str,
                        help='path to mcwy mesh json file file')
    parser.add_argument('--management_json_path', type=str,
                        help='path to management data base file file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated json file with gender information')
    args = parser.parse_args()

    mesh_json_path = args.mesh_json_path
    management_json_path = args.management_json_path
    output_json_path = args.output_json_path

    management_struct = read_json(management_json_path)
    mesh_path_training_set = set()
    for data_struct in management_struct["data"]:
        if data_struct["DataStatus"] == "Training":
            mesh_name = data_struct["MeshName"]
            mesh_basename = os.path.splitext(mesh_name)[0]
            mesh_path_training_set.add(mesh_basename)

    data_struct = read_json(mesh_json_path)
    mesh_data_struct = data_struct["data"]
    new_data_struct = {}
    new_data_struct["data"] = {}

    print(str(check_individual_number(data_struct)))

    for data_name in mesh_data_struct.keys():
        if "VRoid_" not in data_name:
            new_data_struct["data"][data_name] = mesh_data_struct[data_name]
        else:
            if data_name not in new_data_struct["data"].keys():
                new_data_struct["data"][data_name] = {}
            for mesh_name in mesh_data_struct[data_name].keys():
                mesh_path = mesh_data_struct[data_name][mesh_name]["Mesh"]
                # print(mesh_name)
                if mesh_name in mesh_path_training_set:
                    new_data_struct["data"][data_name][mesh_name] = mesh_data_struct[data_name][mesh_name]

    print(str(check_individual_number(new_data_struct)))
    write_json(output_json_path, new_data_struct)
