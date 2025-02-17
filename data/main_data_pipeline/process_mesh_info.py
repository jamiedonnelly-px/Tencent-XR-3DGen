import argparse
import json
import multiprocessing
import os
import time

from tqdm import tqdm


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


def convert_once(command: str,
                 mesh_path: str, mesh_category: str, mesh_instance_name: str,
                 info_stages: list, mesh_info_json_path: str, force_rerun: bool,
                 stat_txt: str, time_txt: str, folder_txt: str):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for manifold converting command is %s....' %
          (str(start_time_str)))

    mesh_folder = os.path.split(mesh_path)[0]
    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    mesh_info_struct["data"][mesh_category] = {}
    mesh_info_struct["data"][mesh_category][mesh_instance_name] = {}

    if force_rerun and os.path.exists(mesh_info_json_path):
        os.remove(mesh_info_json_path)

    if not os.path.exists(mesh_info_json_path):
        print("Start command: %s" % str(command))
        stat = os.system(command)
        time.sleep(0.1)

    if os.path.exists(mesh_info_json_path):
        with open(mesh_info_json_path, encoding='utf-8') as f:
            mesh_info = json.load(f)
            for stage_name in info_stages:
                mesh_info_struct["data"][mesh_category][mesh_instance_name][stage_name] = mesh_info[stage_name]

    info_result_json_str = json.dumps(mesh_info_struct)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After manifold command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(info_result_json_str))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' % (mesh_path, start_time_str, end_time_str))


def read_mesh_list_from_data_json(json_path: str):
    print("Parse data json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        data_config = json.load(f)

    obj_path_struct = {}
    object_category_list = []
    object_name_list = []
    obj_path_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = data_config["data"]
    data_path_name_list = list(data_path_struct.keys())

    for data_name in data_path_name_list:
        all_instance_path_struct = data_path_struct[data_name]
        for instance_name in all_instance_path_struct.keys():
            instance_paths = all_instance_path_struct[instance_name]
            if "Mesh" not in instance_paths.keys():
                continue
            mesh_path = instance_paths["Mesh"]
            if mesh_path is None:
                continue
            obj_path_list.append(mesh_path)
            object_name_list.append(instance_name)
            object_category_list.append(data_name)

            if "Manifold" in instance_paths.keys():
                if instance_paths["Manifold"] is not None:
                    manifold_path = instance_paths["Manifold"]
                else:
                    manifold_path = None
                manifold_path_list.append(manifold_path)

            if "TexPcd" in instance_paths.keys():
                if instance_paths["TexPcd"] is not None:
                    tex_pcd_path = instance_paths["TexPcd"]
                    proc_data_folder = os.path.split(tex_pcd_path)[0]
                else:
                    proc_data_folder = None
                proc_data_folder_list.append(proc_data_folder)

            if "ImgDir" in instance_paths.keys():
                if instance_paths["ImgDir"] is not None:
                    image_dir_path = instance_paths["ImgDir"]
                    image_folder = os.path.split(image_dir_path)[0]
                else:
                    image_folder = None
                render_data_list.append(image_folder)
    return obj_path_list, manifold_path_list, proc_data_folder_list, render_data_list, object_category_list, object_name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--info_stage_str', type=str, default="texture",
                        help='stage of info generation to be used')
    parser.add_argument("--force_rerun", action='store_true',
                        help="force to re-run all verification process even if valid files exists")
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    parser.add_argument("--minimal_face_number", type=int, default=500,
                        help="minimal face number required for mesh used in training")

    args = parser.parse_args()

    output_folder = args.output_folder
    data_json_path = args.data_json_path
    info_stage_str = args.info_stage_str
    force_rerun = args.force_rerun
    blender_root = args.blender_root
    pool_cnt = args.pool_cnt
    data_start = args.data_start
    data_end = args.data_end
    minimal_face_number = args.minimal_face_number

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in rendering......'.format(cpu_cnt, args.pool_cnt))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    mesh_paths = []
    mesh_categories = []
    mesh_instance_names = []
    original_mesh_data = read_json(data_json_path)
    if len(data_json_path) > 1:
        mesh_paths, _, _, _, mesh_categories, mesh_instance_names = read_mesh_list_from_data_json(data_json_path)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(mesh_paths):
                data_end = len(mesh_paths)
            mesh_paths = mesh_paths[data_start:data_end]
            if len(mesh_categories) > 0:
                mesh_categories = mesh_categories[data_start:data_end]
            if len(mesh_instance_names) > 0:
                mesh_instance_names = mesh_instance_names[data_start:data_end]

    if args.pod_num >= 0 and args.pod_id >= 0:
        mesh_path_len = len(mesh_paths)
        per_pod_len = mesh_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = mesh_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        mesh_paths = mesh_paths[idx_start:idx_end]
        mesh_categories = mesh_categories[idx_start:idx_end]
        mesh_instance_names = mesh_instance_names[idx_start:idx_end]

        output_folder = os.path.join(output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        print("Number of mesh on this pod: %i, (start: %i, end: %i)" % (len(mesh_paths), idx_start, idx_end))

    cmds_txt = os.path.join(output_folder, 'cmds.txt')
    stat_txt = os.path.join(output_folder, 'info_success.txt')
    folder_txt = os.path.join(output_folder, 'info_folder.txt')
    time_txt = os.path.join(output_folder, 'info_time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    info_stages = info_stage_str.split("+")

    mesh_basic_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier/mesh_basic.py")

    output_json_folder = os.path.join(output_folder, "json")
    if not os.path.exists(output_json_folder):
        os.mkdir(output_json_folder)

    for index in tqdm(range(len(mesh_paths))):
        mesh_path = mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_category = mesh_categories[index]
        mesh_instance_name = mesh_instance_names[index]
        mesh_info_json_filename = mesh_category + "_" + mesh_instance_name + "_info.json"
        mesh_info_json_path = os.path.join(output_json_folder, mesh_info_json_filename)

        model_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_mesh_json_info \'{}\'".format(blender_root,
                                                                                              mesh_basic_op,
                                                                                              mesh_path,
                                                                                              mesh_info_json_path)
        model_cmd = model_cmd + " --info_stage_str \'{}\' ".format(info_stage_str)

        pool.apply_async(func=convert_once, args=(model_cmd, mesh_path,
                                                  mesh_category, mesh_instance_name,
                                                  info_stages, mesh_info_json_path, force_rerun,
                                                  stat_txt, time_txt, folder_txt))

        with open(cmds_txt, 'a') as f:
            f.write(model_cmd + '\n')

    pool.close()
    time.sleep(0.1)
    pool.join()
    time.sleep(0.1)
    folder_file.close()
    time.sleep(0.1)

    info_json_str_list = read_list(folder_txt)
    info_data_info_struct = {}
    info_data_info_struct["data"] = {}

    filtered_mesh_data = {}
    filtered_mesh_data["data"] = {}

    for mesh_info in info_json_str_list:
        current_json_object = json.loads(mesh_info)
        data_category = next(iter(current_json_object["data"]))
        data_key = next(iter(current_json_object["data"][data_category]))

        if len(current_json_object["data"][data_category][data_key]) < 1:
            continue

        if data_category not in info_data_info_struct["data"].keys():
            info_data_info_struct["data"][data_category] = {}
        info_data_info_struct["data"][data_category][data_key] = current_json_object["data"][data_category][data_key]

        property_status_list = []
        for property_name in info_data_info_struct["data"][data_category][data_key].keys():
            if property_name == "White":
                true_status = not info_data_info_struct["data"][data_category][data_key][property_name]
                property_status_list.append(true_status)
            elif property_name == "FaceNum":
                if info_data_info_struct["data"][data_category][data_key][property_name] < minimal_face_number:
                    property_status_list.append(False)
                else:
                    property_status_list.append(True)
            else:
                true_status = info_data_info_struct["data"][data_category][data_key][property_name]
                property_status_list.append(true_status)

        if all(value == True for value in property_status_list):
            if data_category not in filtered_mesh_data["data"].keys():
                filtered_mesh_data["data"][data_category] = {}
            if data_category not in original_mesh_data["data"].keys():
                continue
            if data_key not in original_mesh_data["data"][data_category].keys():
                continue
            filtered_mesh_data["data"][data_category][data_key] = original_mesh_data["data"][data_category][data_key]

    output_json_path = os.path.join(output_folder, "mesh_internal_info.json")
    write_json(output_json_path, info_data_info_struct)
    print(check_individual_number(info_data_info_struct))

    filtered_mesh_json_path = os.path.join(output_folder, "filtered_mesh_info.json")
    write_json(filtered_mesh_json_path, filtered_mesh_data)
    print(check_individual_number(filtered_mesh_data))

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All mesh info processes done. Local time is %s" % (local_time_str))
