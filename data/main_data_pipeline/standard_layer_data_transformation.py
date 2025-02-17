import argparse
import json
import os
import time

import numpy as np


def calculate_average_transformation(transformation_list: list):
    trans_number = len(transformation_list)
    total_trans = np.zeros((4, 4), np.float32)
    for trans in transformation_list:
        total_trans = total_trans + trans
    final_trans = total_trans / float(trans_number)
    return final_trans


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


def read_mesh_list_from_data_json(json_path: str):
    print("Parse data json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        data_config = json.load(f)

    obj_path_struct = {}
    object_category_list = []
    object_name_list = []
    obj_path_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = data_config["data"]
    data_path_name_list = list(data_path_struct.keys())

    for data_name in data_path_name_list:
        all_instance_path_struct = data_path_struct[data_name]
        for instance_name in all_instance_path_struct.keys():
            instance_paths = all_instance_path_struct[instance_name]
            if "Mesh" not in instance_paths.keys():
                continue
            mesh_path = instance_paths["Mesh"]
            if mesh_path is None:
                continue
            obj_path_list.append(mesh_path)
            object_name_list.append(instance_name)
            object_category_list.append(data_name)

            if "Manifold" in instance_paths.keys():
                if instance_paths["Manifold"] is not None:
                    manifold_path = instance_paths["Manifold"]
                else:
                    manifold_path = None
                manifold_path_list.append(manifold_path)

            if "TexPcd" in instance_paths.keys():
                if instance_paths["TexPcd"] is not None:
                    tex_pcd_path = instance_paths["TexPcd"]
                    proc_data_folder = os.path.split(tex_pcd_path)[0]
                else:
                    proc_data_folder = None
                proc_data_folder_list.append(proc_data_folder)

            if "ImgDir" in instance_paths.keys():
                if instance_paths["ImgDir"] is not None:
                    image_dir_path = instance_paths["ImgDir"]
                    image_folder = os.path.split(image_dir_path)[0]
                else:
                    image_folder = None
                render_data_list.append(image_folder)
    return obj_path_list, manifold_path_list, proc_data_folder_list, render_data_list, object_category_list, object_name_list


def generate_unrepeated_render_folder_name(path: str, connection_starts='', hash_value: bool = False):
    folder = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    file_basename = os.path.splitext(filename)[0]
    if hash_value:
        return file_basename
    else:
        original_elements = folder.split("/")
        original_elements.append(file_basename)
        elements = []
        for element in original_elements:
            new_element = element.replace(' ', '__')
            elements.append(new_element)
        if len(connection_starts) > 1 and connection_starts in elements:
            last_model_index = len(elements) - \
                               elements[::-1].index(connection_starts)
            return "_".join(elements[last_model_index:])
        else:
            if 'save' in elements:
                last_model_index = len(elements) - elements[::-1].index('save')
                return "_".join(elements[last_model_index:])
            elif 'model' in elements:
                last_model_index = len(elements) - elements[::-1].index('model')
                return "_".join(elements[last_model_index:])
            elif '3dAsset_artcenter' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('3dAsset_artcenter')
                return "_".join(elements[last_model_index:])
            elif 'share_2909871' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('share_2909871')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_1' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('aigc_bucket_1')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_2' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('aigc_bucket_2')
                return "_".join(elements[last_model_index:])
            elif 'model_denoise' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('model_denoise')
                return "_".join(elements[last_model_index:])
            elif 'game_character_obj' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('game_character_obj')
                return "_".join(elements[last_model_index:])
            else:
                return "__".join(elements)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Standard layer data processing start. Local time is %s" %
          (local_time_str))

    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--category_transtxt_json_path', type=str, default="",
                        help='output category trans txt json path')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.2-linux-x64/blender', help='path for blender executable file')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    category_transtxt_json_path = args.category_transtxt_json_path
    data_json_path = args.data_json_path
    blender_root = args.blender_root

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    resize_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "manifold/resize_object.py")

    mesh_categories = []
    mesh_instance_names = []
    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    if len(in_mesh_list_txt) > 1:
        mesh_paths = read_list(in_mesh_list_txt)
    if len(data_json_path) > 1:
        mesh_paths, _, _, _, mesh_categories, mesh_instance_names = read_mesh_list_from_data_json(
            data_json_path)

    category_transformation_map = {}
    for index in range(len(mesh_paths)):
        mesh_path = mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_filename = os.path.split(mesh_path)[1]
        if index < len(mesh_categories):
            mesh_category = mesh_categories[index].split("_")[-1]
        else:
            mesh_parent_folder = os.path.split(mesh_folder)[0]
            mesh_category = os.path.split(mesh_parent_folder)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]

        if index > len(mesh_instance_names):
            unique_mesh_folder_name = generate_unrepeated_render_folder_name(
                mesh_path, connection_starts=args.connection_starts)
        else:
            unique_mesh_folder_name = mesh_instance_names[index]
        mesh_category_folder = os.path.join(output_folder, mesh_category)
        if not os.path.exists(mesh_category_folder):
            os.mkdir(mesh_category_folder)
        new_mesh_folder = os.path.join(
            mesh_category_folder, unique_mesh_folder_name)
        if not os.path.exists(new_mesh_folder):
            os.mkdir(new_mesh_folder)

        if mesh_category not in category_transformation_map.keys():
            category_transformation_map[mesh_category] = []

        command_list = []
        transformation_txt = os.path.join(
            new_mesh_folder, "futility_nonentity_transformation.txt")
        resize_mesh_folder = os.path.join(new_mesh_folder, "resize")
        if not os.path.exists(resize_mesh_folder):
            os.mkdir(resize_mesh_folder)
        resize_mesh_path = os.path.join(resize_mesh_folder, "resize.obj")

        resize_cmd = "{} -b -P {} -- --mesh_path '{}' --output_transformation_txt_path '{}' --calculate_transformation_only ".format(
            blender_root, resize_op_fullpath, mesh_path, transformation_txt)

        print("Calculate resize transformation for mesh %s" % (mesh_path))
        if not os.path.exists(transformation_txt):
            os.system(resize_cmd)
        time.sleep(0.1)

        if os.path.exists(transformation_txt):
            T = np.loadtxt(transformation_txt)
            print("outside ", T)
            category_transformation_map[mesh_category].append(T)
        else:
            os.system(resize_cmd)
            time.sleep(0.1)
            if os.path.exists(transformation_txt):
                T = np.loadtxt(transformation_txt)
                print("outside ", T)
                category_transformation_map[mesh_category].append(T)

    category_transtxt_map = {}
    for category_name in category_transformation_map.keys():
        average_trans = calculate_average_transformation(
            category_transformation_map[category_name])
        mesh_category_folder = os.path.join(output_folder, category_name)
        overall_txt = os.path.join(
            mesh_category_folder, "overall_transformation.txt")
        print(category_name, len(category_transformation_map[category_name]))
        print(category_name, " average transformation ", average_trans)
        np.savetxt(overall_txt, average_trans)
        category_transtxt_map[category_name] = overall_txt

    write_json(category_transtxt_json_path, category_transtxt_map)
