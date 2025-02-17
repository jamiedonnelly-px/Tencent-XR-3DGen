import argparse
import json
import os
import time


def convert_once(convert_cmd_list, mesh_path, output_folder, stat_txt, time_txt, folder_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for converting cmd is %s....' % (str(start_time_str)))

    for convert_cmd in convert_cmd_list:
        stat = os.system(convert_cmd)
        time.sleep(0.1)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After converting command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (mesh_path, start_time_str, end_time_str))


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


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--data_mesh_json', type=str, default="",
                        help='input mesh json path')
    parser.add_argument('--output_folder', type=str, default="",
                        help='output converted mesh file folder')
    parser.add_argument('--blender_root', type=str,
                        default="/root/blender-3.6.2-linux-x64/blender", help='blend version 3.5.0 or higher')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    args = parser.parse_args()

    blender_root = args.blender_root
    log_folder = args.log_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    mesh_data_struct = read_json(args.data_mesh_json)
    data_categories = []
    data_instance_names = []
    mesh_paths = []
    for data_name in mesh_data_struct["data"].keys():
        for mesh_name in mesh_data_struct["data"][data_name].keys():
            data_categories.append(data_name)
            data_instance_names.append(mesh_name)
            mesh_paths.append(mesh_data_struct["data"][data_name][mesh_name]["Mesh"])

    blend_converter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pmx/pmx_blend_multi_converter.py")

    for index in range(len(mesh_paths)):
        mesh_path = mesh_paths[index]
        data_category = data_categories[index]
        mesh_instance_name = data_instance_names[index]

        data_category_folder = os.path.join(output_folder, data_category)
        if not os.path.exists(data_category_folder):
            os.mkdir(data_category_folder)

        mesh_instance_folder = os.path.join(data_category_folder, mesh_instance_name)
        if not os.path.exists(mesh_instance_folder):
            os.mkdir(mesh_instance_folder)

        convert_cmd_list = []
        blend_convert_cmd = "{} -b -P {} -- ".format(blender_root, blend_converter_path)
        blend_convert_cmd = blend_convert_cmd + " --pmx_blend_path \'{}\' ".format(mesh_path)
        blend_convert_cmd = blend_convert_cmd + " --output_folder \'{}\' ".format(mesh_instance_folder)
        convert_cmd_list.append(blend_convert_cmd)

        print(convert_cmd_list)
        convert_once(convert_cmd_list, mesh_path, mesh_instance_folder, stat_txt, time_txt, folder_txt)

        convert_cmd = " ".join(convert_cmd_list)
        with open(cmds_txt, 'a') as f:
            f.write(convert_cmd + '\n')

    # pool.close()
    # pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("PMX converting done. Local time is %s" % (local_time_str))
