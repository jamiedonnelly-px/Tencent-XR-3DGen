import argparse
import json
import os
import time


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
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate daz data json start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--input_list_txt', type=str, default="",
                        help='input mesh list txt file')
    parser.add_argument('--output_json_path', type=str, default="",
                        help='output daz json file path')

    args = parser.parse_args()
    input_list_txt = args.input_list_txt
    output_json_path = args.output_json_path

    mesh_list = read_list(input_list_txt)
    output_struct = {}
    output_struct["data"] = {}
    output_struct["data"]["VRoid_Top"] = {}
    output_struct["data"]["VRoid_Bottom"] = {}
    output_struct["data"]["VRoid_Hair"] = {}
    output_struct["data"]["VRoid_Outfit"] = {}
    output_struct["data"]["VRoid_Shoe"] = {}
    output_struct["data"]["VRoid_Body"] = {}

    body_cloth_names_map = {}

    for mesh_path in mesh_list:
        mesh_folder = os.path.split(mesh_path)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        split_folder = os.path.join(mesh_folder, "split")

        body_folder = os.path.join(split_folder, "body")
        body_obj_name = os.path.join(body_folder, "body.obj")
        if not os.path.exists(body_obj_name):
            print("Cannot find body obj file %s" % (body_obj_name))
            continue

        data_name = "VRoid_" + mesh_folder_name + "_Body"
        if data_name not in output_struct["data"]["VRoid_Body"].keys():
            output_struct["data"]["VRoid_Body"][data_name] = {}
            output_struct["data"]["VRoid_Body"][data_name]["Mesh"] = body_obj_name
            print(body_obj_name)

        top_folder = os.path.join(split_folder, "top")
        top_obj_name = os.path.join(top_folder, "top.obj")
        if os.path.exists(top_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Top"
            output_struct["data"]["VRoid_Top"][data_name] = {}
            output_struct["data"]["VRoid_Top"][data_name]["Mesh"] = top_obj_name
            print(top_obj_name)

        bottom_folder = os.path.join(split_folder, "bottom")
        bottom_obj_name = os.path.join(bottom_folder, "bottom.obj")
        if os.path.exists(bottom_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Bottom"
            output_struct["data"]["VRoid_Bottom"][data_name] = {}
            output_struct["data"]["VRoid_Bottom"][data_name]["Mesh"] = bottom_obj_name
            print(bottom_obj_name)

        hair_folder = os.path.join(split_folder, "hair")
        hair_obj_name = os.path.join(hair_folder, "hair.obj")
        if os.path.exists(hair_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Hair"
            output_struct["data"]["VRoid_Hair"][data_name] = {}
            output_struct["data"]["VRoid_Hair"][data_name]["Mesh"] = hair_obj_name
            print(hair_obj_name)

        outfit_folder = os.path.join(split_folder, "outfit")
        outfit_obj_name = os.path.join(outfit_folder, "outfit.obj")
        if os.path.exists(outfit_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Outfit"
            output_struct["data"]["VRoid_Outfit"][data_name] = {}
            output_struct["data"]["VRoid_Outfit"][data_name]["Mesh"] = outfit_obj_name
            print(outfit_obj_name)

        shoe_folder = os.path.join(split_folder, "shoe")
        shoe_obj_name = os.path.join(shoe_folder, "shoe.obj")
        if os.path.exists(shoe_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Shoe"
            output_struct["data"]["VRoid_Shoe"][data_name] = {}
            output_struct["data"]["VRoid_Shoe"][data_name]["Mesh"] = shoe_obj_name
            print(shoe_obj_name)

    write_json(output_json_path, output_struct)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All daz split processes done. Local time is %s" % (local_time_str))
