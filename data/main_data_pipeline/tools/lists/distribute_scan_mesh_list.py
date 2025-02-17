import argparse
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor


def scan_once(cmd_str: str):
    print(cmd_str)
    os.system(cmd_str)
    time.sleep(0.1)


def decorate_subfolder_name(subfolder_name: str):
    new_subfolder_name = copy.deepcopy(subfolder_name)
    if "/" in new_subfolder_name:
        new_subfolder_name = new_subfolder_name.replace("/", "_")
    if " " in new_subfolder_name:
        new_subfolder_name = new_subfolder_name.replace(" ", "_")
    return new_subfolder_name


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Generate mesh list from a group of organized mesh folder')
    parser.add_argument('--mesh_folder', type=str,
                        help='mesh folder full path')
    parser.add_argument('--subfolder_names', nargs='+',
                        help='mesh folder full path')
    parser.add_argument('--output_list_folder', type=str, default="",
                        help='output mesh list folder path')
    parser.add_argument('--file_type', type=str, default=".obj",
                        help='file type of mesh file to searched in folder')
    parser.add_argument('--specific_name', type=str, default="",
                        help='only write mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    parser.add_argument('--exclude_name', type=str, default="",
                        help='exclude mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    args = parser.parse_args()

    mesh_folder = args.mesh_folder
    output_list_folder = args.output_list_folder

    pool = ThreadPoolExecutor(max_workers=8,
                              thread_name_prefix='generate_mesh_list')

    if not os.path.exists(output_list_folder):
        os.mkdir(output_list_folder)

    list_generate_script_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../generate_mesh_list.py")

    folder_path_map = {}
    for subfolder_name in args.subfolder_names:
        subfolder_full_path = os.path.join(mesh_folder, subfolder_name)
        if os.path.exists(subfolder_full_path):
            if os.path.isdir(subfolder_full_path):
                folder_path_map[subfolder_name] = subfolder_full_path

    for subfolder_name in folder_path_map.keys():
        list_file_name = decorate_subfolder_name(subfolder_name)
        list_name = list_file_name + "_list.txt"
        list_fullpath = os.path.join(output_list_folder, list_name)
        sub_mesh_folder = folder_path_map[subfolder_name]
        generate_cmd = "python {} --render_folders \'{}\' --output_list \'{}\' ".format(
            list_generate_script_path, sub_mesh_folder, list_fullpath)
        generate_cmd = generate_cmd + " --file_type \'{}\' --specific_name \'{}\' --exclude_name \'{}\' ".format(
            args.file_type, args.specific_name, args.exclude_name)
        pool.submit(scan_once, generate_cmd)

    pool.shutdown()
    time.sleep(0.1)
    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print("Generate mesh list start. Start time is %s; end time is %s" %
          (local_time_str, end_time_str))
