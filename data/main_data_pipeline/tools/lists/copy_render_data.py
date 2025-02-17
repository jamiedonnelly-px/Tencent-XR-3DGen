import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def copy_once(mesh_original_render_folder,
              render_parent_folder, actual_render_folder,
              stat_txt, folder_txt, time_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    if not os.path.exists(render_parent_folder):
        os.mkdir(render_parent_folder)
    if not os.path.exists(actual_render_folder):
        os.mkdir(actual_render_folder)
    copy_cmd = "rclone copy \'{}\' \'{}\'".format(
        mesh_original_render_folder, actual_render_folder)
    copy_cmd = copy_cmd + " --transfers=32 -P --stats-one-line --checksum --checkers=4 "

    print("Copy render data from %s to %s" %
          (mesh_original_render_folder, actual_render_folder))

    print(copy_cmd)
    os.system(copy_cmd)
    time.sleep(0.1)
    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After copy command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_original_render_folder))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(actual_render_folder))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' %
                (mesh_original_render_folder, start_time_str, end_time_str))


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
        description='copy files from a list')
    parser.add_argument('--input_json_path', type=str, default="",
                        help='input mesh json file path')
    parser.add_argument('--new_render_data_folder', type=str, default="",
                        help='folder of copied render_data')
    parser.add_argument('--output_json_path', type=str, default="",
                        help='output mesh json file path containing converted paths')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--pool_cnt', type=int, default=24,
                        help='parallel number of one thread, no parallel between several datas')
    args = parser.parse_args()

    input_json_path = args.input_json_path
    new_render_data_folder = args.new_render_data_folder
    output_json_path = args.output_json_path
    log_folder = args.log_folder
    pool_cnt = args.pool_cnt

    if not os.path.exists(new_render_data_folder):
        os.mkdir(new_render_data_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    pool = ThreadPoolExecutor(max_workers=pool_cnt, thread_name_prefix='wash_obj')

    input_data_struct = read_json(input_json_path)
    mesh_data_struct = input_data_struct["data"]

    final_output_struct = {}
    final_output_struct["data"] = {}
    output_struct = final_output_struct["data"]

    object_direction_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry/object_direction.py")

    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    for data_name in mesh_data_struct.keys():
        if data_name not in output_struct.keys():
            output_struct[data_name] = {}
        for mesh_name in mesh_data_struct[data_name].keys():
            if mesh_name not in output_struct[data_name].keys():
                output_struct[data_name][mesh_name] = mesh_data_struct[data_name][mesh_name]

            mesh_category = data_name
            mesh_original_render_folder = mesh_data_struct[data_name][mesh_name]["ImgDir"]
            preview_path = mesh_data_struct[data_name][mesh_name]["Preview"]
            preview_filename = os.path.split(preview_path)[1]

            category_folder = os.path.join(new_render_data_folder, mesh_category)
            if not os.path.exists(category_folder):
                os.mkdir(category_folder)
            render_parent_folder = os.path.join(category_folder, mesh_name)
            actual_render_folder = os.path.join(render_parent_folder, "render_512_Valour")

            output_struct[data_name][mesh_name]["ImgDir"] = actual_render_folder
            new_preview_filepath = os.path.join(actual_render_folder, preview_filename)
            output_struct[data_name][mesh_name]["Preview"] = new_preview_filepath

            pool.submit(copy_once, mesh_original_render_folder, render_parent_folder, actual_render_folder,
                        stat_txt, folder_txt, time_txt)

    pool.shutdown()
    time.sleep(0.1)

    print(check_individual_number(input_data_struct))
    print(check_individual_number(final_output_struct))
    write_json(output_json_path, final_output_struct)
