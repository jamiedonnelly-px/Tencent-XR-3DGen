import argparse
import hashlib
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


def calculate_hash_folder_name(input_str: str):
    hash_obj = hashlib.sha1(input_str.encode('utf-8'))
    hash_str = str(hash_obj.hexdigest())
    return hash_str


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        number_info[category_name] = len(json_struct["data"][category_name])
    return number_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mesh list to json')
    parser.add_argument('--mesh_list_txt_path', type=str,
                        help='path to mesh list txt file')
    parser.add_argument('--category_list_txt_path', type=str,
                        help='path to category list txt file')
    parser.add_argument('--output_json_path', type=str,
                        help='path to generated json file')
    args = parser.parse_args()

    mesh_list_txt_path = args.mesh_list_txt_path
    category_list_txt_path = args.category_list_txt_path
    output_json_path = args.output_json_path

    data_struct = {}
    data_struct["data"] = {}
    result_folder_list = read_list(mesh_list_txt_path)
    category_list = read_list(category_list_txt_path)

    for index in range(len(result_folder_list)):
        mesh_folder = result_folder_list[index]
        mesh_parent_folder = os.path.split(mesh_folder)[0]
        mesh_instance_name = os.path.split(mesh_folder)[1]
        mesh_category = category_list[index]
        if mesh_category not in data_struct["data"].keys():
            data_struct["data"][mesh_category] = {}

        original_mesh_name = os.path.join(mesh_folder, "resize/resize.obj")
        manifold_mesh_name = os.path.join(mesh_folder, "manifold/manifold.obj")
        texture_mesh_name = os.path.join(mesh_folder, "texture/texture.obj")
        unify_mesh_name = os.path.join(mesh_folder, "texture/bake/baked_texture.obj")
        transformation_txt = os.path.join(mesh_folder, "transformation.txt")

        instance_struct = {}
        if not os.path.exists(original_mesh_name):
            continue
        if not os.path.exists(texture_mesh_name):
            continue
        if not os.path.exists(transformation_txt):
            continue

        instance_struct["Mesh"] = texture_mesh_name
        instance_struct["Obj_Mesh"] = texture_mesh_name
        instance_struct["Original"] = original_mesh_name

        instance_struct["Original_Transformation"] = transformation_txt
        instance_struct["Manifold"] = manifold_mesh_name
        instance_struct["Unify"] = unify_mesh_name

        data_struct["data"][mesh_category][mesh_instance_name] = instance_struct

    write_json(output_json_path, data_struct)
    print(check_individual_number(data_struct))
