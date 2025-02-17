import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


def key_auto_increase(original_key_set: set, new_key: str):
    key_counter = 0
    if new_key in original_key_set:
        new_key_auto_increase_candidate = new_key + "_repete_"
        for original_key in original_key_set:
            if new_key_auto_increase_candidate in original_key:
                key_counter = key_counter + 1
        increased_key_counter = key_counter + 1
        increased_new_key = new_key_auto_increase_candidate + \
                            str(increased_key_counter)
        return increased_new_key
    else:
        return None


def write_valid(path: str):
    file_name = "task.valid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("valid")


def examine_file_number(folder_path: str):
    if not os.path.exists(folder_path):
        return 0
    if not os.path.isdir(folder_path):
        return 0
    file_names = os.listdir(folder_path)
    return len(file_names)


def examine_sample_folder(folder_path: str, minimal_point_diff_number: int = 50000):
    file_number = examine_file_number(folder_path=folder_path)
    if file_number < 1:
        return False

    file_names = os.listdir(folder_path)
    for file_name in file_names:
        marker_point_number = 0
        file_name_elements = file_name.split("_")
        for file_name_element in file_name_elements:
            if file_name_element.isdigit():
                marker_point_number = int(file_name_element)

        point_cloud_filename = os.path.join(folder_path, file_name)
        print(point_cloud_filename)
        point_cloud = np.load(point_cloud_filename)
        point_number = point_cloud.shape[0]
        if abs(point_number - marker_point_number) > minimal_point_diff_number:
            return False

    return True


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
                last_model_index = len(elements) - elements[::-1].index('3dAsset_artcenter')
                return "_".join(elements[last_model_index:])
            elif 'share_2909871' in elements:
                last_model_index = len(elements) - elements[::-1].index('share_2909871')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_1' in elements:
                last_model_index = len(elements) - elements[::-1].index('aigc_bucket_1')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_2' in elements:
                last_model_index = len(elements) - elements[::-1].index('aigc_bucket_2')
                return "_".join(elements[last_model_index:])
            elif 'model_denoise' in elements:
                last_model_index = len(elements) - elements[::-1].index('model_denoise')
                return "_".join(elements[last_model_index:])
            elif 'game_character_obj' in elements:
                last_model_index = len(elements) - elements[::-1].index('game_character_obj')
                return "_".join(elements[last_model_index:])
            else:
                return "__".join(elements)


def verify_point_cloud_in_folder(point_cloud_folder, minimal_point_number=90000):
    if os.path.exists(point_cloud_folder):
        point_cloud_files = os.listdir(point_cloud_folder)
        if len(point_cloud_files) < 2:
            return False
        for filename in point_cloud_files:
            point_cloud_file_extension = os.path.splitext(filename)[1]
            if point_cloud_file_extension == ".npy":
                point_cloud_filename = os.path.join(
                    point_cloud_folder, filename)
                print(point_cloud_filename)
                point_cloud = np.load(point_cloud_filename)
                point_number = point_cloud.shape[0]
                # print(point_number)
                if point_number < minimal_point_number:
                    return False
        return True
    return False


def verify_point_cloud_folders(proc_data_folder: str,
                               sample_folder_map,
                               prefer_full_folder: bool = False,
                               check_texture_filenumber: bool = False,
                               check_geometry_filenumber: bool = False):
    print("Verify point cloud folder %s ..." % (proc_data_folder))
    proc_data_valid_filename = os.path.join(proc_data_folder, "task.valid")
    sample_folder_map["texture"] = None
    sample_folder_map["geometry"] = None

    full_folder = os.path.join(proc_data_folder, "full")
    new_full_folder = os.path.join(proc_data_folder, "new_full")
    trans_txt_path = os.path.join(proc_data_folder, "trans.txt")

    texture_folder = os.path.join(proc_data_folder, "texture")
    geometry_folder = os.path.join(proc_data_folder, "geometry")
    trans_txt_path2 = os.path.join(proc_data_folder, "transformation.txt")
    if not os.path.exists(proc_data_valid_filename):
        if not (check_texture_filenumber or check_geometry_filenumber):
            return False
        if prefer_full_folder:
            if check_texture_filenumber:
                if not verify_point_cloud_in_folder(full_folder):
                    return False
                sample_folder_map["texture"] = full_folder
            if check_geometry_filenumber:
                if not verify_point_cloud_in_folder(new_full_folder):
                    return False
                sample_folder_map["geometry"] = new_full_folder
        else:
            if check_texture_filenumber:
                if not verify_point_cloud_in_folder(texture_folder):
                    return False
                sample_folder_map["texture"] = texture_folder
            if check_geometry_filenumber:
                if not verify_point_cloud_in_folder(geometry_folder):
                    return False
                sample_folder_map["geometry"] = geometry_folder
    else:
        if prefer_full_folder:
            sample_folder_map["texture"] = full_folder
            sample_folder_map["geometry"] = new_full_folder
        else:
            sample_folder_map["texture"] = texture_folder
            sample_folder_map["geometry"] = geometry_folder
        print("Proc data for %s already checked..." % (proc_data_folder))

    # if not os.path.exists(proc_data_valid_filename):
    #     print("re-render invalid sample %s" % (proc_folder))
    #     time.sleep(0.1)
    #     return False

    write_valid(proc_data_folder)
    return True


def verify_mesh(mesh_path: str, render_daz: bool = False):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_valid_filename = os.path.join(
        mesh_folder, mesh_basename + "_task.valid")
    mesh_validity_script = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "tools/verify/verify_bounding_box2.py")
    if not os.path.exists(mesh_valid_filename):
        check_mesh_cmd = "python \'{}\' --mesh_path \'{}\'".format(
            mesh_validity_script, mesh_path)
        if render_daz:
            check_mesh_cmd = check_mesh_cmd + " --render_daz "

        print(check_mesh_cmd)
        os.system(check_mesh_cmd)
    else:
        print("Mesh %s already checked...." % (mesh_path))

    if not os.path.exists(mesh_valid_filename):
        print("invalid mesh original file %s" % (mesh_path))
        time.sleep(0.1)
        return False
    return True


def verify_render(folder_path: str, read_verify: bool, size_verify: bool, verify_color_only: bool):
    render_valid_filename = os.path.join(folder_path, "task.valid")
    if not os.path.exists(render_valid_filename):
        check_render_cmd = "python \'{}\' --render_folder \'{}\'".format(
            check_validity_script, folder_path)
        if read_verify:
            check_render_cmd = check_render_cmd + " --read_verify "
        if size_verify:
            check_render_cmd = check_render_cmd + " --size_verify "
        if verify_color_only:
            check_render_cmd = check_render_cmd + " --check_color_only "

        print(check_render_cmd)
        os.system(check_render_cmd)
    else:
        print("Folder %s already checked...." % (folder_path))

    if not os.path.exists(render_valid_filename):
        print("re-render invalid render %s" % (folder_path))
        time.sleep(0.1)
        return False
    return True


def check_once(mesh_path: str, render_folder_path: str, proc_data_folder: str, mesh_name: str,
               read_verify: bool = False,
               size_verify: bool = False,
               verify_color_only: bool = False,
               no_mesh_check: bool = False,
               no_proc_data: bool = False,
               check_texture_filenumber: bool = False,
               check_geometry_filenumber: bool = False,
               render_daz: bool = False):
    sample_folder_map = {}
    if not no_mesh_check:
        if mesh_path is not None:
            if not verify_mesh(mesh_path=mesh_path, render_daz=render_daz):
                return False
    if not verify_render(render_folder_path,
                         read_verify=read_verify,
                         size_verify=size_verify,
                         verify_color_only=verify_color_only):
        return False
    if not no_proc_data:
        if proc_data_folder is not None:
            if not verify_point_cloud_folders(proc_data_folder,
                                              sample_folder_map,
                                              prefer_full_folder=False,
                                              check_texture_filenumber=check_texture_filenumber,
                                              check_geometry_filenumber=check_geometry_filenumber):
                return False

    return True


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


def check_str_in_set(input_str: str, str_set: set):
    for string in str_set:
        if input_str in string:
            return True
    return False


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--data_json', type=str,
                        help='input data json file')
    parser.add_argument('--output_json', type=str, default="",
                        help='output json file after verification')
    parser.add_argument('--invalid_mesh_txt', type=str, default="",
                        help='txt file of invalid mesh as input')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--prefer_full_folder', action='store_true',
                        help='using full and new_full folder instead of texture and geometry folder; useful when dealing with old data')
    parser.add_argument('--read_verify', action='store_true',
                        help='verify image deeply by imread function')
    parser.add_argument('--size_verify', action='store_true',
                        help='verify image by checking image size')
    parser.add_argument('--verify_color_only', action='store_true',
                        help='verify only rendered RGB images, do not verify depth images and normal images')
    parser.add_argument('--verify_point_cloud_only', action='store_true',
                        help='verify only point cloud file, do not verify render results')
    parser.add_argument('--check_texture_filenumber', action='store_true',
                        help='verify proc data file number in texture/full folder')
    parser.add_argument('--check_geometry_filenumber', action='store_true',
                        help='verify proc data file number in geometry/new_full folder')
    parser.add_argument('--force_replace', action='store_true',
                        help='force to replace ')
    parser.add_argument('--render_daz', action='store_true',
                        help='have changed -ZY axis to YZ axis during blender process')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')
    parser.add_argument('--hash_value', action='store_true',
                        help='only use this when the mesh file name is a hash value. will override --connection_starts')

    args = parser.parse_args()
    data_json = args.data_json
    data_struct = read_json(data_json)
    invalid_mesh_txt = args.invalid_mesh_txt
    output_json = args.output_json
    pool_cnt = args.pool_cnt

    invalid_mesh_list = read_list(invalid_mesh_txt)
    invalid_key_name = []
    for invalid_mesh_name in invalid_mesh_list:
        render_key_name = generate_unrepeated_render_folder_name(invalid_mesh_name,
                                                                 connection_starts=args.connection_starts,
                                                                 hash_value=args.hash_value)
        invalid_key_name.append(render_key_name)

    pool = ThreadPoolExecutor(max_workers=pool_cnt,
                              thread_name_prefix='verify_json')

    data_verification_struct = {}
    data_verification_struct["data"] = {}
    for data_name in data_struct["data"].keys():
        data_verification_struct["data"][data_name] = {}
        for mesh_name in data_struct["data"][data_name].keys():
            data_verification_struct["data"][data_name][mesh_name] = False

    check_validity_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_render_folder.py")
    mesh_validity_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_bounding_box.py")

    all_data_map = {}
    mesh_verify_status_map = {}
    for data_name in data_struct["data"].keys():
        mesh_data_struct = data_struct["data"][data_name]
        for mesh_name in mesh_data_struct.keys():
            if check_str_in_set(mesh_name, invalid_key_name):
                print("Invalid mesh is %s" % (mesh_name))
                continue
            mesh_path = mesh_data_struct[mesh_name]["Mesh"]
            if mesh_path is not None:
                no_mesh_check = False
                if mesh_path in invalid_mesh_txt:
                    continue
            else:
                no_mesh_check = True
            tex_folder = mesh_data_struct[mesh_name]["TexPcd"]
            geo_folder = mesh_data_struct[mesh_name]["GeoPcd"]
            if tex_folder is None or geo_folder is None:
                no_proc_data = True
                proc_data_folder = None
            else:
                no_proc_data = False
                proc_data_folder = os.path.split(tex_folder)[0]
                if not os.path.exists(tex_folder):
                    continue
                if not os.path.exists(geo_folder):
                    continue
            render_folder = mesh_data_struct[mesh_name]["ImgDir"]
            if render_folder is None:
                continue
            if not os.path.exists(render_folder):
                continue

            if pool.submit(check_once,
                           mesh_path,
                           render_folder,
                           proc_data_folder,
                           mesh_name,
                           read_verify=args.read_verify,
                           size_verify=args.size_verify,
                           verify_color_only=args.verify_color_only,
                           no_mesh_check=no_mesh_check,
                           no_proc_data=no_proc_data,
                           check_texture_filenumber=args.check_texture_filenumber,
                           check_geometry_filenumber=args.check_geometry_filenumber,
                           render_daz=args.render_daz):
                data_verification_struct["data"][data_name][mesh_name] = True

    pool.shutdown()
    time.sleep(0.1)

    new_data_struct = {}
    new_data_struct["data"] = {}
    for data_name in data_struct["data"].keys():
        new_data_struct["data"][data_name] = {}
        for mesh_name in mesh_data_struct.keys():
            if data_verification_struct["data"][data_name][mesh_name]:
                new_data_struct["data"][data_name][mesh_name] = data_struct["data"][data_name][mesh_name]

    write_json(output_json, new_data_struct)

    print("after verification, merged data struct now has %i models....." %
          (int(check_data_number(new_data_struct))))
    print("after verification, conditions of current data is %s" %
          (str(check_individual_number(new_data_struct))))
    print("before verification, data struct now has %i models....." %
          (int(check_data_number(data_struct))))
    print("before verification, conditions of data struct is %s" %
          (str(check_individual_number(data_struct))))
