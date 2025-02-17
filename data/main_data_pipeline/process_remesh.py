import argparse
import json
import multiprocessing
import os
import time


def convert_once(convert_cmd_struct, destination_mesh_path, output_folder, stat_txt, time_txt, folder_txt):
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for manifold converting command is %s....' %
          (str(start_time_str)))

    time.sleep(0.1)
    for cmd_name in convert_cmd_struct.keys():
        current_cmd = convert_cmd_struct[cmd_name]
        print("Start command %s: %s" % (cmd_name, current_cmd))
        exec_result = os.system(current_cmd)
        time.sleep(0.1)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After all mesh process command time is %s....' % (str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(destination_mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' % (output_folder, start_time_str, end_time_str))


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


def read_mesh_list_from_data_json(json_path: str, data_tag=""):
    print("Parse data json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        data_config = json.load(f)

    obj_path_struct = {}
    object_category_list = []
    src_obj_path_list = []
    instance_name_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = data_config["data"]
    data_path_name_list = list(data_path_struct.keys())

    for data_name in data_path_name_list:
        if len(data_tag) > 1:
            if data_name != data_tag:
                continue

        all_instance_path_struct = data_path_struct[data_name]
        for instance_name in all_instance_path_struct.keys():
            instance_paths = all_instance_path_struct[instance_name]
            if "Mesh" not in instance_paths.keys():
                continue

            src_mesh_path = instance_paths["Mesh"]
            if src_mesh_path is None:
                continue

            # if "TexPcd" in instance_paths.keys():
            #     if instance_paths["TexPcd"] is not None:
            #         tex_pcd_path = instance_paths["TexPcd"]
            #         proc_data_folder = os.path.split(tex_pcd_path)[0]
            #     else:
            #         proc_data_folder = None
            # else:
            #     proc_data_folder = None

            category_name = data_name
            src_obj_path_list.append(src_mesh_path)
            object_category_list.append(category_name)
            instance_name_list.append(instance_name)

    return src_obj_path_list, object_category_list, instance_name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove mesh with certain material name.')
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.1-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--process_stages', type=str,
                        default='decimate',
                        help='process stages for remesh process')
    parser.add_argument('--decimate_faces_num', type=int, default=50000,
                        help='face number used in mesh simplification')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    data_json_path = args.data_json_path
    process_stages = args.process_stages
    decimate_faces_num = args.decimate_faces_num
    blender_root = args.blender_root
    log_folder = args.log_folder

    data_start = args.data_start
    data_end = args.data_end

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    mesh_categories = []
    if len(in_mesh_list_txt) > 1:
        source_mesh_paths = read_list(in_mesh_list_txt)
    if len(data_json_path) > 1:
        source_mesh_paths, mesh_categories, instance_name_list = read_mesh_list_from_data_json(data_json_path)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(source_mesh_paths):
                data_end = len(source_mesh_paths)
            source_mesh_paths = source_mesh_paths[data_start:data_end]

    if args.pod_num >= 0 and args.pod_id >= 0:
        mesh_path_len = len(source_mesh_paths)
        per_pod_len = mesh_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = mesh_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        source_mesh_paths = source_mesh_paths[idx_start:idx_end]
        instance_name_list = instance_name_list[idx_start:idx_end]
        mesh_categories = mesh_categories[idx_start:idx_end]

        output_folder = os.path.join(output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        log_folder = os.path.join(log_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        print("Number of mesh on this pod: %i, (start: %i, end: %i)" %
              (len(source_mesh_paths), idx_start, idx_end))

    print("Number of mesh on this pod: %i" % (len(source_mesh_paths)))

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')

    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    mesh_fix_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifold/remesh.py")

    instance_name_folder_map = {}

    for index in range(len(source_mesh_paths)):
        source_mesh_path = source_mesh_paths[index]
        if not os.path.exists(source_mesh_path):
            print('Cannot find source mesh file ', source_mesh_path)
            continue

        mesh_folder = os.path.split(source_mesh_path)[0]
        mesh_name = os.path.split(source_mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]

        instance_name = instance_name_list[index]
        category = mesh_categories[index]
        category_output_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_output_folder):
            os.mkdir(category_output_folder)
        instance_output_folder = os.path.join(category_output_folder, instance_name)
        if not os.path.exists(instance_output_folder):
            os.mkdir(instance_output_folder)

        final_output_path = os.path.join(instance_output_folder, mesh_name)

        fix_cmd_struct = {}

        mesh_fix_cmd = "{} -b -P {} --".format(blender_root, mesh_fix_fullpath)
        mesh_fix_cmd = mesh_fix_cmd + " --mesh_path \"{}\" ".format(source_mesh_path)
        mesh_fix_cmd = mesh_fix_cmd + " --output_mesh_path \"{}\" ".format(final_output_path)
        mesh_fix_cmd = mesh_fix_cmd + " --process_stages \"{}\" ".format(process_stages)
        mesh_fix_cmd = mesh_fix_cmd + " --decimate_faces_num {} ".format(decimate_faces_num)

        fix_cmd_struct[process_stages] = mesh_fix_cmd

        process_stage_list_str = str(process_stages.split("+"))
        print("Fix mesh in mesh %s using stages %s" % (source_mesh_path, process_stage_list_str))

        pool.apply_async(func=convert_once,
                         args=(fix_cmd_struct, source_mesh_path, final_output_path, stat_txt, time_txt, folder_txt))

        with open(cmds_txt, 'a') as f:
            f.write(mesh_fix_cmd + '\n')

    pool.close()
    pool.join()

    time.sleep(0.1)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All mesh remove processes done. Local time is %s" % (local_time_str))
