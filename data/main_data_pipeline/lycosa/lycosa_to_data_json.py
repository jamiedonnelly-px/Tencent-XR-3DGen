import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def convert_once(convert_struct: dict, mesh_path: str, output_path: str,
                 stat_txt: str, time_txt: str, folder_txt: str):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for convert file (at %s) format cmd is %s....' %
          (mesh_path, str(start_time_str)))

    for cmd_name in convert_struct.keys():
        convert_cmd = convert_struct[cmd_name]
        print(cmd_name, convert_cmd)
        stat = os.system(convert_cmd)
        t_end = time.time()
        end_time = time.localtime(t_end)
        end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After convert format command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(output_path))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' %
                (mesh_path, start_time_str, end_time_str))


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

    with open(in_list_txt, 'r', encoding='UTF-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='UTF-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Lycosa data to standard data process start. Local time is %s" %
          (local_time_str))

    parser = argparse.ArgumentParser(
        description='Lycosa processing pipeline.')
    parser.add_argument('--lycosa_data_json', type=str, default="",
                        help='lycosa data folder full path')
    parser.add_argument('--output_data_folder', type=str, default="",
                        help='output folder for newly converted data')
    parser.add_argument('--output_data_json_file', type=str, default="",
                        help='output folder for newly generated data json')
    parser.add_argument('--data_name', type=str, default="",
                        help='data name of this group of data')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender',
                        help='path for blender binary exe')
    parser.add_argument('--pool_cnt', type=int, default=24,
                        help='multiprocessing pool cnt')
    args = parser.parse_args()

    lycosa_data_json = args.lycosa_data_json
    output_data_folder = args.output_data_folder
    output_data_json_file = args.output_data_json_file
    data_name = args.data_name
    log_folder = args.log_folder
    blender_root = args.blender_root
    pool_cnt = args.pool_cnt

    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    # fbx_data_json_path = os.path.join(output_data_json_folder, "fbx_info.json")
    # obj_data_json_path = os.path.join(output_data_json_folder, "obj_info.json")

    lycosa_data_input_struct = read_json(lycosa_data_json)

    pool = ThreadPoolExecutor(max_workers=pool_cnt,
                              thread_name_prefix='wash_lycosa_obj')

    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    fbx_obj_converter_op = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "../../conversion/fbx_obj_converter.py")
    object_direction_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry/object_direction.py")
    dds_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../texture/dds.py")

    for index in range(len(lycosa_data_input_struct["data"])):
        lycosa_data = lycosa_data_input_struct["data"][index]
        to_hash_str = str(lycosa_data)
        hash_obj = hashlib.sha1(to_hash_str.encode('utf-8'))
        mesh_unique_name = str(hash_obj.hexdigest())

        if lycosa_data["DataStatus"] != "Training":
            continue

        mesh_name = lycosa_data["MeshName"]
        mesh_fullpath = lycosa_data["SavePaths"]["MeshFilename"]
        if mesh_fullpath.startswith("Asset"):
            mesh_fullpath = os.path.join("/mnt/aigc_bucket_1/", mesh_fullpath)
        mesh_extenstion = os.path.splitext(mesh_name)[1]

        mesh_category_folder = os.path.join(output_data_folder, data_name)
        if not os.path.exists(mesh_category_folder):
            os.mkdir(mesh_category_folder)
        mesh_instance_folder = os.path.join(mesh_category_folder, mesh_unique_name)
        if not os.path.exists(mesh_instance_folder):
            os.mkdir(mesh_instance_folder)

        final_output_path = os.path.join(mesh_instance_folder, mesh_unique_name + ".obj")

        convert_cmd_struct = {}

        if mesh_extenstion == ".fbx":
            final_middle_folder = os.path.join(mesh_instance_folder, "middleware")
            if not os.path.exists(final_middle_folder):
                os.mkdir(final_middle_folder)
            middle_mesh_path = os.path.join(final_middle_folder, mesh_unique_name + ".obj")

            print("[%i] Convert fbx file from %s to %s" %
                  (index, mesh_fullpath, final_output_path))

            convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' ".format(blender_root, fbx_obj_converter_op, mesh_fullpath)
            convert_cmd = convert_cmd + " --output_fullpath \'{}\' ".format(middle_mesh_path)
            convert_cmd = convert_cmd + " --copy_texture   > /dev/null"
            convert_cmd_struct["convert"] = convert_cmd

            dds_cmd = "{} -b -P {} -- --source_mesh_path \'{}\' ".format(blender_root, dds_op, middle_mesh_path)
            dds_cmd = dds_cmd + " --output_mesh_path \'{}\' > /dev/null".format(final_output_path)
            convert_cmd_struct["dds"] = dds_cmd

        elif mesh_extenstion == ".obj":
            print("[%i] Convert obj file from %s to %s" %
                  (index, mesh_fullpath, final_output_path))
            dds_cmd = "{} -b -P {} -- --source_mesh_path \'{}\' ".format(
                blender_root, dds_op, mesh_fullpath)
            dds_cmd = dds_cmd + \
                      " --output_mesh_path \'{}\' > /dev/null".format(
                          final_output_path)
            convert_cmd_struct["dds"] = dds_cmd

        pool.submit(convert_once, convert_cmd_struct, mesh_fullpath,
                    final_output_path, stat_txt, time_txt, folder_txt)

        with open(cmds_txt, 'a') as f:
            f.write(str(convert_cmd_struct) + '\n')

    pool.shutdown()
    time.sleep(0.1)
    cmds_file.close()
    stat_file.close()
    time_file.close()
    folder_file.close()
    time.sleep(0.1)

    final_data_struct = {}
    final_data_struct["data"] = {}
    output_data_struct = final_data_struct["data"]

    result_mesh_list = read_list(folder_txt)
    success_mesh_list = read_list(stat_txt)
    for index in range(len(result_mesh_list)):
        result_mesh_path = result_mesh_list[index]
        success_mesh_path = success_mesh_list[index]

        mesh_path_elements = result_mesh_path.split("/")
        mesh_category = mesh_path_elements[-3]
        mesh_instance_name = mesh_path_elements[-2]
        if mesh_category not in output_data_struct.keys():
            output_data_struct[mesh_category] = {}
        output_data_struct[mesh_category][mesh_instance_name] = {}
        output_data_struct[mesh_category][mesh_instance_name]["Mesh"] = result_mesh_path
        output_data_struct[mesh_category][mesh_instance_name]["Arche"] = success_mesh_path

    print(check_individual_number(final_data_struct))
    write_json(output_data_json_file, final_data_struct)
