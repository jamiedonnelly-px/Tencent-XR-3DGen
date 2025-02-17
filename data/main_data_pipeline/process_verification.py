import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def is_valid_path(path):
    if not os.path.isabs(path):
        return False
    normalized_path = os.path.normpath(path)
    if normalized_path != path:
        return False
    return normalized_path.startswith('/') and '..' not in normalized_path.split('/')


def verify_once(verify_sdf_flag: bool,
                verify_render_flag: bool,
                success_mesh_path: str,
                proc_data_folder: str,
                render_data_folder: str,
                sdf_format: str,
                read_verify: bool,
                size_verify: bool,
                verify_color_only: bool,
                force_rerun: bool,
                cmds_txt: str,
                stat_txt: str,
                time_txt: str,
                folder_txt: str):
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for verification command is %s....' %
          (str(start_time_str)))

    result_struct = {}
    # result_struct[data_category] = {}
    # result_struct[data_category][data_key] = {}
    mesh_folder = os.path.split(success_mesh_path)[0]
    mesh_parent_folder = os.path.split(mesh_folder)[0]
    manifold_path = os.path.join(mesh_parent_folder, "manifold/manifold.obj")

    verify_command_struct = {}

    if verify_sdf_flag:
        proc_parent_folder = os.path.split(proc_data_folder)[0]
        proc_category_folder = os.path.split(proc_parent_folder)[0]
        data_key = os.path.split(proc_parent_folder)[1]
        data_category = os.path.split(proc_category_folder)[1]

        if data_category not in result_struct.keys():
            result_struct[data_category] = {}
        if data_key not in result_struct[data_category].keys():
            result_struct[data_category][data_key] = {}

        sdf_valid_filename = os.path.join(proc_data_folder, "mesh.valid")
        if force_rerun and os.path.exists(sdf_valid_filename):
            os.remove(sdf_valid_filename)

        verify_proc_cmd = "python {} ".format(verify_sdf_op)
        verify_proc_cmd = verify_proc_cmd + "  --proc_folder \'{}\' ".format(proc_data_folder)
        if verify_render_data:
            verify_proc_cmd = verify_proc_cmd + "  --render_folder \'{}\' ".format(render_data_folder)
        verify_proc_cmd = verify_proc_cmd + "  --check_format \'{}\' ".format(sdf_format)
        verify_command_struct["sdf"] = verify_proc_cmd

        if not os.path.exists(sdf_valid_filename):
            print("Start sdf verification command: %s" % (verify_proc_cmd))
            exec_result = os.system(verify_proc_cmd)
            time.sleep(0.01)

        if os.path.exists(sdf_valid_filename):
            result_struct[data_category][data_key]["GeoPcd"] = str(os.path.join(proc_data_folder, "geometry"))
            result_struct[data_category][data_key]["Manifold"] = manifold_path
            transformation_txt_path = os.path.join(proc_data_folder, "transformation.txt")
            z_txt_path = os.path.join(proc_data_folder, "z.txt")
            if os.path.exists(transformation_txt_path):
                result_struct[data_category][data_key]["Transformation"] = transformation_txt_path
            if os.path.exists(z_txt_path):
                result_struct[data_category][data_key]["Z_Transformation"] = z_txt_path
        else:
            print("SDF sample results is wrong....")
            return

    if verify_render_flag:
        verify_command_struct["render"] = {}

        render_parent_folder = os.path.split(render_data_folder)[0]
        render_category_folder = os.path.split(render_parent_folder)[0]
        data_key = os.path.split(render_parent_folder)[1]
        data_category = os.path.split(render_category_folder)[1]

        if data_category not in result_struct.keys():
            result_struct[data_category] = {}
        if data_key not in result_struct[data_category].keys():
            result_struct[data_category][data_key] = {}

        camera_parameters_json = os.path.join(render_data_folder, "cam_parameters.json")
        render_config_file = os.path.join(render_data_folder, "config.json")

        if not os.path.exists(camera_parameters_json):
            print("Cannot find camera parameters json in render data folder %s..." % (camera_parameters_json))
            return

        if not os.path.exists(render_config_file):
            print("Cannot find render config json in render data folder %s..." % (render_config_file))
            return

        render_config_struct = read_json(render_config_file)
        render_stages = list(render_config_struct["stages"].keys())

        for stage_name in render_stages:
            stage_folder = render_config_struct["stages"][stage_name]
            render_valid_filename = os.path.join(stage_folder, "mesh.valid")
            if force_rerun and os.path.exists(render_valid_filename):
                os.remove(render_valid_filename)

            if stage_name == "common" or stage_name == "no_smooth":
                verify_render_command = render_verify_cmd_generation(verify_render_op=verify_render_op,
                                                                     render_data_folder=stage_folder,
                                                                     camera_parameters_json=camera_parameters_json,
                                                                     render_config_file=render_config_file,
                                                                     read_verify=read_verify,
                                                                     size_verify=size_verify,
                                                                     verify_color_only=verify_color_only,
                                                                     image_random_number=image_random_number)
            else:
                verify_render_command = render_verify_cmd_generation(verify_render_op=verify_render_op,
                                                                     render_data_folder=stage_folder,
                                                                     camera_parameters_json=camera_parameters_json,
                                                                     render_config_file=render_config_file,
                                                                     read_verify=read_verify,
                                                                     size_verify=size_verify,
                                                                     verify_color_only=True,
                                                                     image_random_number=image_random_number)

            verify_command_struct["render"][stage_name] = verify_render_command

            if not os.path.exists(render_valid_filename):
                print("Start render verification command for stage %s: %s" % (stage_name, verify_render_command))
                exec_result = os.system(verify_render_command)
                time.sleep(0.01)

            # check render results here
            if not os.path.exists(render_valid_filename):
                print("One stage %s at location %s is wrong..." % (stage_name, stage_folder))
                return

        result_struct[data_category][data_key]["ImgDir"] = render_data_folder
        result_struct[data_category][data_key]["Mesh"] = success_mesh_path
        z_txt_path = os.path.join(render_data_folder, "transformation.txt")
        if os.path.exists(z_txt_path):
            result_struct[data_category][data_key]["Z_Transformation"] = z_txt_path

    verify_result_json_str = json.dumps(result_struct)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After verification command time is %s....' % (str(end_time_str)))

    time.sleep(0.05)

    with open(cmds_txt, 'a') as f:
        if verify_proc_data:
            f.write(verify_command_struct["sdf"] + '\n')
        if verify_render_data:
            for stage_name in verify_command_struct["render"].keys():
                f.write(verify_command_struct["render"][stage_name] + '\n')

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(success_mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(verify_result_json_str))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' % (success_mesh_path, start_time_str, end_time_str))


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


def scan_source_log_folder(source_log_folder: str):
    log_folder_list = []
    first_layer_cmd_txt_file = os.path.join(source_log_folder, "cmds.txt")
    first_layer_proc_cmd_txt_file = os.path.join(source_log_folder, "proc_cmds.txt")
    if os.path.exists(first_layer_cmd_txt_file) and os.path.exists(first_layer_proc_cmd_txt_file):
        log_folder_list.append(source_log_folder)

    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        if "verify" in folder_name:
            continue

        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            cmd_txt_file = os.path.join(folder_fullpath, "cmds.txt")
            proc_cmd_txt_file = os.path.join(folder_fullpath, "proc_cmds.txt")
            if os.path.exists(cmd_txt_file) and os.path.exists(proc_cmd_txt_file):
                log_folder_list.append(folder_fullpath)
    log_folder_list.sort()
    return log_folder_list


def render_verify_cmd_generation(verify_render_op: str,
                                 render_data_folder: str,
                                 camera_parameters_json: str,
                                 render_config_file: str,
                                 read_verify: bool,
                                 size_verify: bool,
                                 verify_color_only: bool,
                                 image_random_number: int):
    verify_render_cmd = "python {} ".format(verify_render_op)
    verify_render_cmd = verify_render_cmd + " --render_folder \'{}\' ".format(render_data_folder)
    verify_render_cmd = verify_render_cmd + " --camera_parameters_json \'{}\' ".format(camera_parameters_json)
    verify_render_cmd = verify_render_cmd + " --render_config_json \'{}\' ".format(render_config_file)
    if read_verify:
        verify_render_cmd = verify_render_cmd + " --read_verify "
    if size_verify:
        verify_render_cmd = verify_render_cmd + " --size_verify "
    if verify_color_only:
        verify_render_cmd = verify_render_cmd + " --check_color_only "
    verify_render_cmd = verify_render_cmd + " --random_number {} ".format(image_random_number)
    return verify_render_cmd


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Verification starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Remove mesh with certain material name.')
    parser.add_argument('--log_folder', type=str, default="",
                        help='input render log folder')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing output verify logs')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--verify_render_data', action='store_true',
                        help='verify render data folder')
    parser.add_argument('--verify_proc_data', action='store_true',
                        help='verify proc data folder')
    parser.add_argument('--read_verify', action='store_true',
                        help='verify render images by reading these images')
    parser.add_argument('--size_verify', action='store_true',
                        help='verify render images only by calculating image file size')
    parser.add_argument('--verify_color_only', action='store_true',
                        help='only verify render rgb images')
    parser.add_argument("--sdf_format", type=str, default="h5",
                        help="sdf sample data format, choose between h5 or npy")
    parser.add_argument("--image_random_number", type=int, default=5,
                        help="random number of image selected from rendered images")
    parser.add_argument("--force_rerun", action='store_true',
                        help="force to re-run all verification process even if valid files exists")
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')

    args = parser.parse_args()
    log_folder = args.log_folder
    output_folder = args.output_folder
    pool_cnt = args.pool_cnt
    verify_render_data = args.verify_render_data
    verify_proc_data = args.verify_proc_data
    read_verify = args.read_verify
    size_verify = args.size_verify
    verify_color_only = args.verify_color_only
    sdf_format = args.sdf_format
    image_random_number = args.image_random_number
    force_rerun = args.force_rerun
    data_start = args.data_start
    data_end = args.data_end

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    pool = ThreadPoolExecutor(max_workers=pool_cnt, thread_name_prefix='verify_')
    print('Use {} threads in converting......'.format(args.pool_cnt))

    source_log_folder_list = scan_source_log_folder(log_folder)

    if args.pod_num >= 0 and args.pod_id >= 0:
        log_path_len = len(source_log_folder_list)
        per_pod_len = log_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = log_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        source_log_folder_list = source_log_folder_list[idx_start:idx_end]

        output_folder = os.path.join(output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        print(
            "Number of mesh on this pod: %i, (start: %i, end: %i)" % (len(source_log_folder_list), idx_start, idx_end))

    print("Number of mesh on this pod: %i" % (len(source_log_folder_list)))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cmds_txt = os.path.join(output_folder, 'verify_cmds.txt')
    stat_txt = os.path.join(output_folder, 'verify_success.txt')
    folder_txt = os.path.join(output_folder, 'verify_folder.txt')
    time_txt = os.path.join(output_folder, 'verify_time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    verify_sdf_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification/verify_sample.py")
    verify_render_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification/verify_image_folder.py")

    instance_name_folder_map = {}

    for index in range(len(source_log_folder_list)):
        source_log_folder = source_log_folder_list[index]
        print("Verify folder at %s" % (source_log_folder))

        data_mesh_txt = os.path.join(source_log_folder, "success.txt")
        mesh_data_list = read_list(data_mesh_txt)
        mesh_data_number = len(mesh_data_list)

        if verify_render_data:
            data_folder_txt = os.path.join(source_log_folder, "folder.txt")
            render_folder_list = read_list(data_folder_txt)
            render_folder_number = len(render_folder_list)

        if verify_proc_data:
            data_proc_txt = os.path.join(source_log_folder, "proc.txt")
            proc_data_folder_list = read_list(data_proc_txt)
            proc_data_folder_number = len(proc_data_folder_list)

        if verify_render_data and verify_proc_data:
            total_folder_number = min(render_folder_number, proc_data_folder_number, mesh_data_number)
        elif verify_render_data and not verify_proc_data:
            total_folder_number = min(render_folder_number, mesh_data_number)
        elif verify_proc_data and not verify_render_data:
            total_folder_number = min(proc_data_folder_number, mesh_data_number)
        else:
            continue

        data_key_render_folder_map = {}
        data_key_proc_folder_map = {}
        data_key_mesh_path_map = {}
        for folder_index in range(total_folder_number):
            verify_command_struct = {}
            if verify_render_data:
                render_data_folder = render_folder_list[folder_index]
                if not is_valid_path(render_data_folder):
                    continue
                render_folder_elements = render_data_folder.split("/")
                data_key_render = render_folder_elements[-2]
                data_key_render_folder_map[data_key_render] = render_data_folder
            else:
                render_data_folder = ""
            if verify_proc_data:
                proc_data_folder = proc_data_folder_list[folder_index]
                if not is_valid_path(render_data_folder):
                    continue
                proc_folder_elements = proc_data_folder.split("/")
                data_key_proc = proc_folder_elements[-2]
                data_key_proc_folder_map[data_key_proc] = proc_data_folder
            else:
                proc_data_folder = ""
            success_mesh = mesh_data_list[folder_index]
            mesh_folder_elements = success_mesh.split("/")
            data_key_mesh = mesh_folder_elements[-3]
            data_key_mesh_path_map[data_key_mesh] = success_mesh

        for data_key in data_key_mesh_path_map.keys():
            if verify_render_data:
                if data_key not in data_key_render_folder_map.keys():
                    continue
                render_data_folder = data_key_render_folder_map[data_key]
            else:
                render_data_folder = ""
            if verify_proc_data:
                if data_key not in data_key_proc_folder_map.keys():
                    continue
                proc_data_folder = data_key_proc_folder_map[data_key]
            else:
                proc_data_folder = ""
            success_mesh = data_key_mesh_path_map[data_key]

            pool.submit(verify_once, verify_proc_data, verify_render_data, success_mesh, proc_data_folder,
                        render_data_folder, sdf_format, read_verify, size_verify, verify_color_only, force_rerun,
                        cmds_txt, stat_txt, time_txt, folder_txt)

    pool.shutdown()
    time.sleep(0.1)
    folder_file.close()
    time.sleep(0.1)

    verification_json_str_list = read_list(folder_txt)
    verified_data_info_struct = {}
    verified_data_info_struct["data"] = {}

    for verify_info in verification_json_str_list:
        current_json_object = json.loads(verify_info)
        data_category = next(iter(current_json_object))
        data_key = next(iter(current_json_object[data_category]))

        if data_category not in verified_data_info_struct["data"].keys():
            verified_data_info_struct["data"][data_category] = {}
        verified_data_info_struct["data"][data_category][data_key] = current_json_object[data_category][data_key]

    mode_map = {}
    for data_name in verified_data_info_struct["data"].keys():
        for mesh_name in verified_data_info_struct["data"][data_name].keys():
            render_folder = verified_data_info_struct["data"][data_name][mesh_name]["ImgDir"]
            render_category_folder = os.path.dirname(os.path.dirname(os.path.dirname(render_folder)))
            mode_txt = os.path.join(render_category_folder, "mode.txt")
            if os.path.exists(mode_txt):
                with open(mode_txt, 'r') as fin:
                    lines = fin.readlines()
                    mode_string = lines[0]
                    mode_map[data_name] = mode_string

    verified_data_info_struct["modus"] = mode_map

    output_json_path = os.path.join(output_folder, "verify.json")
    write_json(output_json_path, verified_data_info_struct)
    print(check_individual_number(verified_data_info_struct))

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All verification processes done. Local time is %s" % (local_time_str))
