import argparse
import os
import sys
import time


def to_valid_path(string: str):
    new_string = string.replace("\\", "\\\\")
    return new_string


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='utf-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def convert_once(convert_cmd, mesh_path, convert_mesh_path,
                 success_txt, time_txt, folder_txt):
    stat = 1
    preprocess_stat = 1
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for this cmd is %s....' % (str(start_time_str)))

    print(convert_cmd)
    preprocess_stat = os.system(convert_cmd)
    time.sleep(0.1)

    t_middle = time.time()
    middle_time = time.localtime(t_middle)
    middle_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', middle_time)

    print('After max -> fbx command status is %s; time for this status is %s....' % (
        str(preprocess_stat), str(middle_time_str)))

    with open(success_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(convert_mesh_path))

    with open(time_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (mesh_path, start_time_str, middle_time_str))


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("3dsmax file to fbx conversion started. Local time is %s" %
          (start_time_str))

    parser = argparse.ArgumentParser(description='3dsmax file to fbx conversion')
    parser.add_argument('--max_file_list', type=str, default="",
                        help='mesh list file')
    parser.add_argument('--max_folder_list', type=str, default="",
                        help='folder of mesh file list')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    args = parser.parse_args()

    max_file_list_path = args.max_file_list
    max_folder_list_path = args.max_folder_list
    log_folder = args.log_folder

    if not sys.platform.startswith('win'):
        print("The os is not windows-based and cannot run 3dsmax....aborting.....")
        exit(-1)

    default_3dsmax_install = 'C:\\Program Files\\Autodesk\\3ds Max 2025'
    if not os.path.exists(default_3dsmax_install):
        print("The we  is not windows-based and cannot run 3dsmax....aborting.....")
        exit(-1)

    max_file_list = read_list(max_file_list_path)
    preprocess_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "3dsmax_convert.py")

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

    for index in range(len(max_file_list)):
        mesh_path = max_file_list[index]
        mesh_folder = os.path.split(mesh_path)[0]
        mesh_name = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]
        mesh_type = os.path.splitext(mesh_name)[1]
        fbx_path = os.path.join(mesh_folder, mesh_basename + ".fbx")

        raw_script_path = to_valid_path(preprocess_script_fullpath)
        raw_max_path = to_valid_path(mesh_path)
        raw_fbx_path = to_valid_path(fbx_path)

        max_cmd = "3dsmaxbatch.exe \"{}\" -mxsString max_file_path:'{}' -mxsString output_file_path:'{}'".format(
            raw_script_path, raw_max_path, raw_fbx_path)
        ps_max_cmd = "powershell.exe  -NoProfile -ExecutionPolicy Bypass \"{}\"".format(
            max_cmd)

        # we do not recommend parallel running here
        convert_once(ps_max_cmd, raw_max_path, raw_fbx_path, stat_txt, time_txt, folder_txt)

    time.sleep(0.1)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All max -> fbx tasks DONE. These tasks start at time %s and end at time %s" %
          (start_time_str, local_time_str))
