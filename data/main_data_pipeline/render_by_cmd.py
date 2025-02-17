import argparse
import json
import multiprocessing
import os
import time

from easydict import EasyDict as edict


def parse_config_json(json_path: str):
    print("Parse config json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        config = json.load(f)

    config = edict(config)
    return config


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


def render_once(render_cmd, mesh_path, stat_txt, time_txt):
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for this cmd is %s....' % (str(start_time_str)))

    stat = os.system(render_cmd)
    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After rendering command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(mesh_path))

    with open(time_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (mesh_path, start_time_str, end_time_str))


def parse_render_cmd(cmd: str):
    position_mesh_path = cmd.find("--mesh_path")
    position_out_put_folder = cmd.find("--output_folder")
    mesh_path_start = position_mesh_path + 13
    mesh_path_end = position_out_put_folder - 2
    return cmd[mesh_path_start:mesh_path_end]


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Rendering using cmd list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool using cmd list.')
    parser.add_argument('--cmd_list', type=str,
                        help='render mesh commands list file')
    parser.add_argument(
        '--pool_cnt', type=int, default=8,
        help='multiprocessing pool cnt, use 800M GPU memory each cmd')
    parser.add_argument(
        '--log_folder', type=str, default='./log',
        help='path for saving rendered image')
    args = parser.parse_args()

    cmd_list_path = args.cmd_list
    log_folder = args.log_folder

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    stat_file = open(time_txt, 'w')

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in rendering......'.format(
        cpu_cnt, args.pool_cnt))

    cmd_list = []
    if os.path.exists(cmd_list_path) and len(cmd_list_path) > 1:
        cmd_list = read_list(cmd_list_path)

    cnt = 0
    for index in range(len(cmd_list)):
        render_cmd = cmd_list[index]
        mesh_path = parse_render_cmd(render_cmd)

        pool.apply_async(func=render_once, args=(render_cmd, mesh_path, stat_txt, time_txt))
        cnt += 1
        with open(cmds_txt, 'a') as f:
            f.write(render_cmd + '\n')

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Rendering done. Local time is %s" % (local_time_str))
