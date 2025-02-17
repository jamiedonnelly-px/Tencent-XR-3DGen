import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def verify_once(verify_cmd_struct: dict,
                data_key: str,
                data_category: str,
                verify_sdf_flag: bool,
                verify_render_flag: bool,
                glb_conversion_flag: bool,
                success_mesh_path: str,
                proc_data_folder: str,
                render_data_folder: str,
                stat_txt: str,
                time_txt: str,
                folder_txt: str,
                glb_folder_path: str):
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    glb_converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "../conversion/mesh_glb_converter.py")

    print('Start time for verification command is %s....' %
          (str(start_time_str)))

    result_struct = {}
    result_struct[data_category] = {}
    result_struct[data_category][data_key] = {}
    mesh_folder = os.path.split(success_mesh_path)[0]
    mesh_parent_folder = os.path.split(mesh_folder)[0]
    manifold_path = os.path.join(mesh_parent_folder, "manifold/manifold.obj")

    if verify_sdf_flag:
        sdf_valid_filename = os.path.join(proc_data_folder, "mesh.valid")
        if not os.path.exists(sdf_valid_filename):
            current_cmd = verify_cmd_struct["sdf"]
            print("Start sdf verification command: %s" % (current_cmd))
            exec_result = os.system(current_cmd)
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
        for stage_name in verify_cmd_struct["render"].keys():
            render_valid_filename = os.path.join(verify_cmd_struct["render"][stage_name]["folder"], "mesh.valid")
            print(render_valid_filename)
            if not os.path.exists(render_valid_filename):
                current_cmd = verify_cmd_struct["render"][stage_name]["cmd"]
                print("Start render verification command for stage %s: %s" % (stage_name, current_cmd))
                exec_result = os.system(current_cmd)
                time.sleep(0.01)

            # check render results here
            if not os.path.exists(render_valid_filename):
                print("One stage %s at location %s is wrong..." % (
                    stage_name, verify_cmd_struct["render"][stage_name]["folder"]))
                return

        result_struct[data_category][data_key]["ImgDir"] = render_data_folder
        result_struct[data_category][data_key]["Mesh"] = success_mesh_path
        z_txt_path = os.path.join(render_data_folder, "transformation.txt")
        if os.path.exists(z_txt_path):
            result_struct[data_category][data_key]["Z_Transformation"] = z_txt_path

    if glb_conversion_flag:
        glb_category_folder = os.path.join(glb_folder_path, data_category)
        if not os.path.exists(glb_category_folder):
            os.mkdir(glb_category_folder)
        glb_mesh_folder = os.path.join(glb_category_folder, data_key)
        if not os.path.exists(glb_mesh_folder):
            os.mkdir(glb_mesh_folder)
        mesh_path = result_struct[data_category][data_key]["Mesh"]
        new_glb_fullpath = os.path.join(glb_mesh_folder, data_key + ".glb")
        convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_fullpath \'{}\' ".format(
            blender_root, glb_converter_fullpath, mesh_path, new_glb_fullpath)
        print("Convert from obj file %s to glb file %s" % (mesh_path, new_glb_fullpath))
        os.system(convert_cmd)

        if not os.path.exists(new_glb_fullpath):
            print("GLB conversion of mesh %s failed" % (mesh_path))
            return
        result_struct[data_category][data_key]["GLB_Mesh"] = new_glb_fullpath

    verify_result_json_str = json.dumps(result_struct)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After verification command time is %s....' % (str(end_time_str)))

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
    first_layer_success_txt_file = os.path.join(source_log_folder, "success.txt")
    if os.path.exists(first_layer_cmd_txt_file) and os.path.exists(first_layer_success_txt_file):
        log_folder_list.append(source_log_folder)

    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            cmd_txt_file = os.path.join(folder_fullpath, "cmds.txt")
            success_txt_file = os.path.join(folder_fullpath, "success.txt")
            if os.path.exists(cmd_txt_file) and os.path.exists(success_txt_file):
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


def find_unite_category(category_name: str):
    unite_category_list = ["TOP", "BOTTOM", "OUTFIT",
                           "DRESS", "SHOE", "HAIR", "HAT", "GLOVE", "SOCK"]
    # hair, top, trousers, shoe, outfit, others
    pure_category_name = category_name.split("_")[-1].upper()
    if pure_category_name == "TOP":
        return "top"
    elif pure_category_name == "BOTTOM":
        return "trousers"
    elif pure_category_name == "OUTFIT" or pure_category_name == "DRESS":
        return "outfit"
    elif pure_category_name == "SHOE" or pure_category_name == "FOOTWEAR":
        return "shoe"
    elif pure_category_name == "HAIR":
        return "hair"
    elif pure_category_name == "GLOVE" or pure_category_name == "SOCK" or pure_category_name == "HAT":
        return "others"
    else:
        return None


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
    parser.add_argument('--glb_conversion', action='store_true',
                        help='convert obj mesh to glb mesh')
    parser.add_argument('--glb_folder_path', type=str, default="",
                        help='path of output folder in glb conversion')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender', help='path for blender binary exe')
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
    glb_conversion = args.glb_conversion
    glb_folder_path = args.glb_folder_path
    blender_root = args.blender_root
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

    verify_sdf_op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../verification/verify_sample.py")
    verify_render_op = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../verification/verify_image_folder.py")

    instance_name_folder_map = {}

    if glb_conversion:
        if not os.path.exists(glb_folder_path):
            os.mkdir(glb_folder_path)

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

        for folder_index in range(total_folder_number):
            verify_command_struct = {}
            render_data_folder = ""
            proc_data_folder = ""
            data_key = ""
            data_category = ""
            success_mesh = mesh_data_list[folder_index]
            if verify_proc_data:
                proc_data_folder = proc_data_folder_list[folder_index]
                proc_parent_folder = os.path.split(proc_data_folder)[0]
                proc_category_folder = os.path.split(proc_parent_folder)[0]
                data_key = os.path.split(proc_parent_folder)[1]
                data_category = os.path.split(proc_category_folder)[1]

                valid_filename = os.path.join(proc_data_folder, "mesh.valid")
                if force_rerun and os.path.exists(valid_filename):
                    os.remove(valid_filename)

                verify_proc_cmd = "python {} ".format(verify_sdf_op)
                verify_proc_cmd = verify_proc_cmd + "  --proc_folder \'{}\' ".format(proc_data_folder)
                if verify_render_data:
                    render_data_folder = render_folder_list[folder_index]
                    verify_proc_cmd = verify_proc_cmd + "  --render_folder \'{}\' ".format(render_data_folder)
                verify_proc_cmd = verify_proc_cmd + "  --check_format \'{}\' ".format(sdf_format)
                verify_command_struct["sdf"] = verify_proc_cmd

            if verify_render_data:
                verify_command_struct["render"] = {}
                render_data_folder = render_folder_list[folder_index]
                render_parent_folder = os.path.split(render_data_folder)[0]
                render_category_folder = os.path.split(render_parent_folder)[0]
                data_key = os.path.split(render_parent_folder)[1]
                data_category = os.path.split(render_category_folder)[1]
                camera_parameters_json = os.path.join(render_data_folder, "cam_parameters.json")
                if not os.path.exists(camera_parameters_json):
                    print("Cannot find camera paramters json in render data folder %s..." % (camera_parameters_json))
                    continue

                render_config_file = os.path.join(render_data_folder, "config.json")
                render_config_struct = read_json(render_config_file)
                render_stages = list(render_config_struct["stages"].keys())
                for stage_name in render_stages:
                    stage_folder = render_config_struct["stages"][stage_name]
                    valid_filename = os.path.join(stage_folder, "mesh.valid")
                    if force_rerun and os.path.exists(valid_filename):
                        os.remove(valid_filename)
                    if stage_name == "common" or stage_name == "no_smooth":
                        verify_command = render_verify_cmd_generation(verify_render_op=verify_render_op,
                                                                      render_data_folder=stage_folder,
                                                                      camera_parameters_json=camera_parameters_json,
                                                                      render_config_file=render_config_file,
                                                                      read_verify=read_verify,
                                                                      size_verify=size_verify,
                                                                      verify_color_only=verify_color_only,
                                                                      image_random_number=image_random_number)

                    else:
                        verify_command = render_verify_cmd_generation(verify_render_op=verify_render_op,
                                                                      render_data_folder=stage_folder,
                                                                      camera_parameters_json=camera_parameters_json,
                                                                      render_config_file=render_config_file,
                                                                      read_verify=read_verify,
                                                                      size_verify=size_verify,
                                                                      verify_color_only=True,
                                                                      image_random_number=image_random_number)

                    verify_command_struct["render"][stage_name] = {}
                    verify_command_struct["render"][stage_name]["cmd"] = verify_command
                    verify_command_struct["render"][stage_name]["folder"] = stage_folder

            pool.submit(verify_once, verify_command_struct,
                        data_key, data_category,
                        verify_proc_data, verify_render_data, glb_conversion,
                        success_mesh, proc_data_folder, render_data_folder,
                        stat_txt, time_txt, folder_txt, glb_folder_path)

            with open(cmds_txt, 'a') as f:
                if verify_proc_data:
                    f.write(verify_command_struct["sdf"] + '\n')
                if verify_render_data:
                    for stage_name in verify_command_struct["render"].keys():
                        f.write(verify_command_struct["render"][stage_name]["cmd"] + '\n')

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
        verified_data_info_struct["data"][data_category][data_key]["Manifold"] = None
        verified_data_info_struct["data"][data_category][data_key]["GeoTri"] = None
        verified_data_info_struct["data"][data_category][data_key]["TexTri"] = None

        original_mesh_path = verified_data_info_struct["data"][data_category][data_key]["Mesh"]
        render_folder = verified_data_info_struct["data"][data_category][data_key]["ImgDir"]

        layer_category = find_unite_category(data_category)
        verified_data_info_struct["data"][data_category][data_key]["Category"] = layer_category
        verified_data_info_struct["data"][data_category][data_key]["body_key"] = None
        verified_data_info_struct["data"][data_category][data_key]["Gender"] = "Asexual"
        verified_data_info_struct["data"][data_category][data_key]["Obj_Mesh"] = original_mesh_path

        preview_image_path = os.path.join(render_folder, "color/cam-0022.png")
        if not os.path.exists(preview_image_path):
            continue
        verified_data_info_struct["data"][data_category][data_key]["Preview"] = preview_image_path

        if layer_category == 'shoe':
            verified_data_info_struct["data"][data_category][data_key]["HighHeel"] = False

    output_json_path = os.path.join(output_folder, "verify.json")
    write_json(output_json_path, verified_data_info_struct)
    print(check_individual_number(verified_data_info_struct))

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All verification processes done. Local time is %s" % (local_time_str))
