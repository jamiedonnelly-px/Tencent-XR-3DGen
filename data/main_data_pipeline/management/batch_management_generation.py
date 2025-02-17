import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


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


def scan_source_log_folder(source_log_folder: str):
    log_folder_list = []
    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            log_folder_list.append(folder_fullpath)
    log_folder_list.sort()
    return log_folder_list


def run_command(cmd):
    print(cmd)
    time.sleep(0.1)
    os.system(cmd)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Management generation start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Generate management system json')
    parser.add_argument('--management_info_json', type=str, default="",
                        help='json file containing info for management system')
    parser.add_argument('--log_folder', type=str, default="",
                        help='list txt file containing mesh abs path')
    parser.add_argument('--temp_folder', type=str, default="",
                        help='folder containing temp data during management generation')
    parser.add_argument('--secret_key', type=str, default="",
                        help='secret key for object storage')
    parser.add_argument('--secret_id', type=str, default="",
                        help='secret id for object storage')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt for parallel operation')

    args = parser.parse_args()
    log_folder = args.log_folder
    management_info_json = args.management_info_json
    temp_folder = args.temp_folder
    secret_key=args.secret_key
    secret_id=args.secret_id
    pool_cnt = args.pool_cnt
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    management_info = read_json(management_info_json)

    log_folder_list = scan_source_log_folder(log_folder)

    render_data_name_folder_map = {}
    render_data_name_success_mesh_map = {}

    pool = ThreadPoolExecutor(max_workers=pool_cnt,
                              thread_name_prefix='management_system_batch')

    for log_folder_path in log_folder_list:
        folder_txt = os.path.join(log_folder_path, "folder.txt")
        success_txt = os.path.join(log_folder_path, "success.txt")
        if not os.path.exists(folder_txt):
            continue

        render_folder_list = read_list(folder_txt)
        success_mesh_list = read_list(success_txt)

        for index in range(len(render_folder_list)):
            render_folder_path = render_folder_list[index]
            success_mesh_path = success_mesh_list[index]

            render_folder_elements = render_folder_path.split("/")
            render_data_name = render_folder_elements[-3]

            mesh_id = render_folder_elements[-2]
            if render_data_name not in render_data_name_folder_map.keys():
                render_data_name_folder_map[render_data_name] = []
            if render_data_name not in render_data_name_success_mesh_map.keys():
                render_data_name_success_mesh_map[render_data_name] = []

            render_data_name_folder_map[render_data_name].append(
                render_folder_path)
            render_data_name_success_mesh_map[render_data_name].append(
                success_mesh_path)

    generate_management_script = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "generate_management_json.py")
    management_json_list = []

    for render_data_name in render_data_name_folder_map.keys():
        render_data_folder = os.path.join(temp_folder, render_data_name)
        if not os.path.exists(render_data_folder):
            os.mkdir(render_data_folder)

        new_folder_txt = os.path.join(render_data_folder, "folder.txt")
        new_success_txt = os.path.join(render_data_folder, "success.txt")
        thumbnail_json = os.path.join(render_data_folder, "management_thumbnail.json")
        management_json = os.path.join(render_data_folder, "management.json")

        write_list(new_folder_txt,
                   render_data_name_folder_map[render_data_name])
        write_list(new_success_txt,
                   render_data_name_success_mesh_map[render_data_name])

        time.sleep(0.1)

        management_cmd_str = "python {} --secret_id \'{}\' --secret_key \'{}\' ".format(generate_management_script,
                                                                                        secret_id,
                                                                                        secret_key)
        management_cmd_str = management_cmd_str + \
                             " --mesh_list '{}' ".format(new_success_txt)
        management_cmd_str = management_cmd_str + \
                             " --folder_list '{}' ".format(new_folder_txt)
        management_cmd_str = management_cmd_str + \
                             " --output_json '{}' ".format(management_json)
        management_cmd_str = management_cmd_str + \
                             " --output_thumbnail_json '{}' ".format(thumbnail_json)
        management_cmd_str = management_cmd_str + \
                             " --data_name '{}' ".format(render_data_name)

        if render_data_name in management_info.keys():
            data_style = management_info[render_data_name]["Style"]
            data_origin = management_info[render_data_name]["Origin"]
            management_cmd_str = management_cmd_str + \
                                 " --data_style '{}' ".format(data_style)
            management_cmd_str = management_cmd_str + \
                                 " --data_origin '{}' ".format(data_origin)
        else:
            management_cmd_str = management_cmd_str + " --data_style 'None' "
            management_cmd_str = management_cmd_str + " --data_origin 'None' "

        management_json_list.append(management_json)

        print(management_cmd_str)
        pool.submit(run_command, management_cmd_str)

    pool.shutdown()

    print("Merge all generated json.....")
    merge_json_path = os.path.join(temp_folder, local_time_str + ".json")
    merge_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../tools/lists/merge_json_file_list.py")
    merge_cmd = "python {} --output_json_path \'{}\' --json_files ".format(
        merge_op_fullpath, merge_json_path)

    for individual_json in management_json_list:
        individual_str = " \'{}\' ".format(individual_json)
        merge_cmd = merge_cmd + individual_str

    print(merge_cmd)
    os.system(merge_cmd)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Parallel operation management generation end. Start time is %s; end time is %s" %
          (local_time_str, end_time_str))
