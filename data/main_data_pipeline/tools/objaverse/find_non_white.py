import argparse
import multiprocessing
import os
import time


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


def find_once(mesh_folder: str, non_white_txt: str, white_txt: str):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for convert file (at %s) format cmd is %s....' %
          (mesh_folder, str(start_time_str)))

    folder_name = os.path.split(mesh_folder)[1]
    nonwhite_mesh_name = os.path.join(mesh_folder, folder_name + ".obj")
    white_mesh_name = os.path.join(mesh_folder, "white_" + folder_name + ".obj")

    if os.path.exists(nonwhite_mesh_name):
        with open(non_white_txt, 'a') as f:
            f.write('{}\n'.format(nonwhite_mesh_name))
    elif os.path.exists(white_mesh_name):
        with open(white_txt, 'a') as f:
            f.write('{}\n'.format(white_mesh_name))

    time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--in_mesh_list_txt', type=str, default="",
                        help='input mesh (format: obj) list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='output white/nonwhite mesh folder')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    args = parser.parse_args()

    output_folder = args.output_folder
    mesh_paths = read_list(args.in_mesh_list_txt)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    nonwhite_txt = os.path.join(output_folder, 'nonwhite.txt')
    white_txt = os.path.join(output_folder, 'white.txt')
    nonwhite_file = open(nonwhite_txt, 'w')
    white_file = open(white_txt, 'w')

    for mesh_path in mesh_paths:
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        pool.apply_async(func=find_once, args=(mesh_path, nonwhite_txt, white_txt))

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Rendering done. Local time is %s" % (local_time_str))
