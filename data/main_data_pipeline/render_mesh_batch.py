#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import random
import shlex
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')


def read_list(in_list_txt: str):
    """
    Read a list of contents from a txt file.
    :param in_list_txt: path to the list txt file
    :return: contents in the list
    """
    str_list = []
    if not os.path.exists(in_list_txt):
        logging.error('Cannot find input list txt file ', in_list_txt)
        return str_list
    try:
        with open(in_list_txt, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                one_line_content = line.strip()
                if len(one_line_content) > 1:
                    str_list.append(one_line_content)
    except (IOError, FileNotFoundError):
        logging.error("Cannot read list file %s" % in_list_txt)
    return str_list


def write_list(list_path: str, write_list: list):
    """
    Write a list of contents to a txt file.
    :param list_path: path to the list txt file
    :param write_list: list to write
    """
    try:
        with open(list_path, 'w') as f:
            for index in range(len(write_list)):
                f.write(write_list[index] + "\n")
    except (IOError, FileNotFoundError):
        logging.error("Cannot write list file %s" % list_path)


def read_json(json_path: str):
    """
    Read a json file to a json struct.
    :param json_path: path of the json file
    :return: result json struct
    """
    try:
        with open(json_path, encoding='utf-8') as f:
            json_struct = json.load(f)
            return json_struct
    except (IOError, FileNotFoundError):
        logging.error("Cannot read json file from %s" % json_path)


def write_json(json_path: str, json_struct):
    """
    Write a json struct to a json file.
    :param json_path: path of the json file
    :param json_struct: json struct to write
    """
    try:
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(json_struct, f, indent=4, ensure_ascii=False)
    except (IOError, FileNotFoundError):
        logging.error("Cannot write json file %s" % json_path)


def current_time():
    """
    Get current time string.
    :return: current time string
    """
    t_current = time.time()
    local_time = time.localtime(t_current)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    return local_time_str


def calculate_hash(input_str: str):
    """
    Calculate sha3_224 hash value of a string.
    :param input_str: input string
    :return: sha3_224 hash value
    """
    hash_obj = hashlib.sha3_224(input_str.encode('utf-8'))
    hash_str = str(hash_obj.hexdigest())
    return hash_str


def run_cmd(cmd_str: str, silent: bool = False):
    """
    Run a shell command using subprocess
    :param cmd_str: shell command
    :return: True if success, False otherwise
    """
    cmd_elements = shlex.split(cmd_str)
    try:
        if silent:
            subprocess.run(cmd_elements, check=True, text=True, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(cmd_elements, check=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error("Error in running cmd %s" % cmd_str)
        logging.error("Error code %s" % e.returncode)
        logging.error("Error msg %s" % e.stderr)
        return False
    return True


def render_once(render_cmd_struct, mesh_cmd_struct,
                mesh_path: str,
                output_folder: str,
                proc_data_folder: str,
                stat_txt: str,
                time_txt: str,
                folder_txt: str,
                proc_txt: str,
                slient: bool,
                apply_render_cmd: bool,
                apply_preprocess_mesh: bool,
                render_stages: list):
    # since we use multiprocessing to run commands in parallel,
    # we cannot re-use functions defined outside,
    # so we need to define duplicate codes here
    def run_cmd(cmd_str: str, slient: bool = False):
        """
        Run a shell command using subprocess
        :param cmd_str: shell command
        :return: True if success, False otherwise
        """
        cmd_elements = shlex.split(cmd_str)
        try:
            if slient:
                subprocess.run(cmd_elements, check=True, text=True, stdout=subprocess.DEVNULL)
            else:
                subprocess.run(cmd_elements, check=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error("Error in running cmd %s" % cmd_str)
            logging.error("Error code %s" % e.returncode)
            logging.error("Error msg %s" % e.stderr)
            return False
        return True

    def current_time():
        """
        Get current time string.
        :return: current time string
        """
        t_current = time.time()
        local_time = time.localtime(t_current)
        local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
        return local_time_str

    start_time_str = current_time()

    logging.info('Start time for render and sample cmd is %s....' % (str(start_time_str)))

    if apply_preprocess_mesh:
        for mesh_cmd_name in mesh_cmd_struct.keys():
            logging.info("Name is %s; cmd is %s......" % (mesh_cmd_name, mesh_cmd_struct[mesh_cmd_name]))
            if not run_cmd(mesh_cmd_struct[mesh_cmd_name], slient):
                logging.error(f"Command in {mesh_cmd_name} raise an error....")
                return

    middle_time_str = current_time()

    logging.info('After point sampling command time is %s....' % (str(middle_time_str)))

    if apply_render_cmd:
        for render_cmd_name in render_stages:
            if render_cmd_name in render_cmd_struct.keys():
                logging.info("Name is %s; cmd is %s......" % (render_cmd_name, render_cmd_struct[render_cmd_name]))
                if not run_cmd(render_cmd_struct[render_cmd_name], slient):
                    logging.error(f"Command in {render_cmd_name} raise an error....")
                    return

    end_time_str = current_time()

    logging.info('After rendering command time is %s....' % (str(end_time_str)))

    with open(stat_txt, 'a', encoding='UTF-8') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a', encoding='UTF-8') as f:
        f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a', encoding='UTF-8') as f:
        f.write('%s starts at %s, finish at %s....\n' % (mesh_path, start_time_str, end_time_str))

    if len(proc_txt) > 1:
        with open(proc_txt, 'a', encoding='UTF-8') as f:
            f.write('{}\n'.format(proc_data_folder))


def read_mesh_list_from_data_json(json_path: str, sample_on_before: bool = False):
    logging.info("Parse data json at path %s" % (json_path))
    mesh_data_struct = read_json(json_path)

    object_category_list = []
    object_name_list = []
    obj_path_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = mesh_data_struct["data"]
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

            if sample_on_before and "Fine" in instance_paths.keys():
                if instance_paths["Fine"] is not None:
                    fine_mesh_path = instance_paths["Fine"]
                else:
                    fine_mesh_path = None
                manifold_path_list.append(fine_mesh_path)
            else:
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
    mesh_data_tuple = (
        obj_path_list, manifold_path_list, proc_data_folder_list, render_data_list, object_category_list,
        object_name_list)
    return mesh_data_tuple


def generate_unrepeated_render_folder_name(path: str, connection_starts='', hash_value: bool = False):
    """
    Generate a unique name for rendering results
    :param path: path to the mesh
    :param connection_starts: starting string to connect path elements, 
                              we split the path of mesh into lists,
                              and connect all elements after connection_starts,
                              ensures a unique mesh name
    :param hash_value: if True, use hash value to generate unique name
    :return: unique name
    """
    folder = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    file_basename = os.path.splitext(filename)[0]
    if hash_value:
        return file_basename
    else:
        original_elements = folder.split("/")
        original_elements.append(file_basename)
        elements = []
        for element in original_elements:
            new_element = element.replace(' ', '__')
            elements.append(new_element)
        if len(connection_starts) > 1 and connection_starts in elements:
            last_model_index = len(elements) - elements[::-1].index(connection_starts)
            return "_".join(elements[last_model_index:])
        else:
            # use only last four path elements
            if len(elements) > 4:
                return "__".join(elements[-4:])
            else:
                return "__".join(elements)


if __name__ == '__main__':
    local_time_str = current_time()
    logging.info("Rendering start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Render and sample mesh in parallel')
    parser.add_argument('--in_mesh_list_txt', type=str, default="",
                        help='input mesh list txt file')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json path (database json), choose between this or mesh list')
    parser.add_argument('--data_json_tag_str', type=str, default="",
                        help='only render tag in data json')
    parser.add_argument('--generate_pose_config_json_path', type=str, default="",
                        help='pose generation config json file path')
    parser.add_argument('--pose_json_path', type=str, default="",
                        help='cam_parameters.json including poses json path')
    parser.add_argument('--config_json_path', type=str, default="",
                        help='render config json path')
    parser.add_argument('--output_folder', type=str,
                        help='output folder for render results')
    parser.add_argument('--proc_data_output_folder', type=str, default="",
                        help='output folder for proc data sample results')
    parser.add_argument('--success_list', type=str, default="",
                        help='mesh already rendered in mesh list')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str,
                        help='log folder to store information')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.5.0-linux-x64/blender', help='path for blender binary exe')
    parser.add_argument('--silent', action='store_true',
                        help='no log output of rendering scripts')
    parser.add_argument('--apply_preprocess_mesh', action='store_true',
                        help='preprocess mesh using mesh_proprocess.py, before rendering')
    parser.add_argument('--apply_render', action='store_true',
                        help='start the rendering process')
    parser.add_argument('--donot_apply_any_process', action='store_true',
                        help='only generate cmd lists; do not start render or mesh process')
    parser.add_argument('--force_outside_trans', action='store_true',
                        help='use outside transformation txt in all processes')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better fbx as import plugin')
    parser.add_argument('--force_sample_on_before', action='store_true',
                        help='force to sample points on before.txt if this file exists')
    parser.add_argument('--parse_exr', action='store_true',
                        help='set this will parse exr to color/depth/...')
    parser.add_argument('--only_render_png', action='store_true',
                        help='set this will only render color files; effective when not set parse_exr')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')
    parser.add_argument('--hash_value', action='store_true',
                        help='only use this when the mesh file name is a hash value. will override --connection_starts')
    parser.add_argument('--scaled_obj', action='store_true',
                        help='will output scaled obj file of rendered mesh')
    parser.add_argument('--preprocess_scale_mesh', action='store_true',
                        help='scale mesh in preprocessing process')
    parser.add_argument('--render_stage_string', type=str, default="common+emission+no_smooth",
                        help='stages of render to be used')
    parser.add_argument('--pose_generation_mode', type=str, default="RIGID_RANDOM",
                        help='generation mode of random poses')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    args = parser.parse_args()

    data_start = args.data_start
    data_end = args.data_end
    mesh_list_path = args.in_mesh_list_txt
    data_json_path = args.data_json_path
    data_json_tag_str = args.data_json_tag_str
    pose_json_path = args.pose_json_path
    generate_pose_config_json_path = args.generate_pose_config_json_path
    log_folder = args.log_folder
    blender_root = args.blender_root
    output_folder = args.output_folder
    proc_data_output_folder = args.proc_data_output_folder
    success_list_path = args.success_list

    apply_render = args.apply_render
    apply_preprocess_mesh = args.apply_preprocess_mesh
    donot_apply_any_process = args.donot_apply_any_process
    force_outside_trans = args.force_outside_trans
    force_sample_on_before = args.force_sample_on_before
    hash_value = args.hash_value
    scaled_obj = args.scaled_obj
    preprocess_scale_mesh = args.preprocess_scale_mesh

    render_stages = args.render_stage_string.split("+")
    pose_generation_mode = args.pose_generation_mode

    success_mesh_list = []
    if os.path.exists(success_list_path) and len(success_list_path) > 1:
        success_mesh_list = read_list(success_list_path)

    config_json_path = args.config_json_path
    if len(config_json_path) < 1 or not os.path.exists(config_json_path):
        config_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/default.json")
    config_struct = read_json(config_json_path)
    logging.info(f"Render config is: {str(config_struct)} \n\n")

    mesh_path_list = []
    manifold_mesh_path_list = []
    output_folder_list = []
    trans_txt_list = []
    z_txt_list = []
    proc_data_folder_list = []
    input_proc_data_folder_list = []
    mesh_category_map = {}
    category_list = []
    instance_name_list = []

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if len(proc_data_output_folder) > 1:
        if not os.path.exists(proc_data_output_folder):
            os.mkdir(proc_data_output_folder)

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if len(data_json_path) > 1:
        mesh_data_tuple = read_mesh_list_from_data_json(data_json_path, sample_on_before=force_sample_on_before)
    elif len(mesh_list_path) > 1:
        mesh_path_list = read_list(mesh_list_path)

    mesh_path_list = mesh_data_tuple[0]
    manifold_mesh_path_list = mesh_data_tuple[1]
    input_proc_data_folder_list = mesh_data_tuple[2]
    category_list = mesh_data_tuple[4]
    instance_name_list = mesh_data_tuple[5]

    # slice data by data_start and data_end
    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(mesh_path_list):
                data_end = len(mesh_path_list)
            mesh_path_list = mesh_path_list[data_start:data_end]
            manifold_mesh_path_list = manifold_mesh_path_list[data_start:data_end]
            input_proc_data_folder_list = input_proc_data_folder_list[data_start:data_end]
            category_list = category_list[data_start:data_end]
            instance_name_list = instance_name_list[data_start:data_end]

    # preprocess mesh will calculate transformation implicitly
    # so we need to disable this if we force to use outside transformation.txt
    if force_outside_trans:
        apply_preprocess_mesh = False

    mesh_number = len(mesh_path_list)
    for mesh_index in range(mesh_number):
        current_output_folder = output_folder
        mesh_path = mesh_path_list[mesh_index]

        mesh_filename = os.path.split(mesh_path)[1]
        mesh_folder_fullpath = os.path.split(mesh_path)[0]
        mesh_parent_folder = os.path.split(mesh_folder_fullpath)[0]
        if len(category_list) == mesh_number:
            mesh_category = category_list[mesh_index]
            mesh_folder_name = instance_name_list[mesh_index]
            render_category_folder = os.path.join(current_output_folder, mesh_category)
            if apply_render:
                if not os.path.exists(render_category_folder):
                    os.mkdir(render_category_folder)
            mesh_render_folder = os.path.join(render_category_folder, mesh_folder_name)
            if apply_render:
                if not os.path.exists(mesh_render_folder):
                    logging.info("Creating render data folder on %s" % (mesh_render_folder))
                    os.mkdir(mesh_render_folder)
            if len(proc_data_output_folder) > 1:
                proc_category_folder = os.path.join(proc_data_output_folder, mesh_category)
                if apply_preprocess_mesh:
                    if not os.path.exists(proc_category_folder):
                        os.mkdir(proc_category_folder)
                mesh_proc_folder = os.path.join(proc_category_folder, mesh_folder_name)
                if apply_preprocess_mesh:
                    if not os.path.exists(mesh_proc_folder):
                        logging.info("Creating proc data folder on %s" % (mesh_proc_folder))
                        os.mkdir(mesh_proc_folder)

        else:
            mesh_folder_name = generate_unrepeated_render_folder_name(mesh_path,
                                                                      connection_starts=args.connection_starts,
                                                                      hash_value=hash_value)
            mesh_render_folder = os.path.join(current_output_folder, mesh_folder_name)
            if apply_render:
                if not os.path.exists(mesh_render_folder):
                    logging.info("Creating render data folder on %s" % (mesh_render_folder))
                    os.mkdir(mesh_render_folder)
            if len(proc_data_output_folder) > 1:
                mesh_proc_folder = os.path.join(proc_data_output_folder, mesh_folder_name)
                if apply_preprocess_mesh:
                    if not os.path.exists(mesh_proc_folder):
                        logging.info("Creating proc data folder on %s" % (mesh_proc_folder))
                        os.mkdir(mesh_proc_folder)

        if len(manifold_mesh_path_list) <= mesh_index:
            mesh_new_fullpath = os.path.join(mesh_parent_folder, "manifold/manifold.obj")
            if not os.path.exists(mesh_new_fullpath):
                potential_new_full_name = mesh_filename.replace("_full", "_new_full")
                mesh_folder_fullpath = os.path.join(mesh_folder_fullpath, potential_new_full_name)
            manifold_mesh_path_list.append(mesh_new_fullpath)

        final_output_folder = os.path.join(mesh_render_folder,
                                           "render_{}_Valour".format(config_struct["render_height"]))
        output_folder_list.append(final_output_folder)

        if force_outside_trans:
            final_proc_folder = input_proc_data_folder_list[mesh_index]
            proc_data_folder_list.append(final_proc_folder)
        else:
            if len(proc_data_output_folder) > 1:
                final_proc_folder = os.path.join(mesh_proc_folder, "proc_data")
                proc_data_folder_list.append(final_proc_folder)
            else:
                final_proc_folder = None
                proc_data_folder_list.append(None)

        trans_txt = os.path.join(final_output_folder, "transformation.txt")
        z_txt = os.path.join(final_output_folder, "z.txt")

        if config_struct["outside_trans"]:
            if apply_preprocess_mesh:
                z_txt = os.path.join(final_proc_folder, "z.txt")
                trans_txt = os.path.join(final_proc_folder, "transformation.txt")
            else:
                if final_proc_folder is not None:
                    if os.path.exists(final_proc_folder):
                        if os.path.isdir(final_proc_folder):
                            z_txt = os.path.join(final_proc_folder, "z.txt")
                            trans_txt = os.path.join(final_proc_folder, "transformation.txt")
        z_txt_list.append(z_txt)
        trans_txt_list.append(trans_txt)

    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    proc_txt = os.path.join(log_folder, 'proc.txt')
    scale_cmds_txt = os.path.join(log_folder, 'scale_cmds.txt')
    proc_cmds_txt = os.path.join(log_folder, 'proc_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    scale_cmd_file = open(scale_cmds_txt, 'w')
    proc_cmd_file = open(proc_cmds_txt, 'w')
    proc_file = open(proc_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    logging.info('Find {} cpus, use {} threads in rendering......'.format(cpu_cnt, args.pool_cnt))

    render_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "render_color_depth_normal_helper.py")
    sdf_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry/sdf_sample.py")
    curvature_sdf_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "geometry/curvature_sdf_sample.py")
    texture_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry/texture_sample.py")
    scale_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry/scale_obj.py")

    generate_pose_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             config_struct["pose_script"][pose_generation_mode])

    # generate data format hash for each rendering task
    data_format_hash = "Doloris_"
    if len(generate_pose_config_json_path) > 0:
        data_format_hash = data_format_hash + "_GENERATION_" + pose_generation_mode + "_"
        generate_pose_info = read_json(generate_pose_config_json_path)
        current_camera_mode_info = generate_pose_info[pose_generation_mode]
        generate_hash_str = calculate_hash(str(current_camera_mode_info))
        data_format_hash = data_format_hash + "_" + generate_hash_str + "_"
    else:
        data_format_hash = data_format_hash + "_STATIC_"
    data_format_hash = data_format_hash + "_RENDER_"
    render_config_hash_str = calculate_hash(str(config_struct))
    data_format_hash = data_format_hash + "_" + render_config_hash_str + "___" + local_time_str
    data_format_mark_txt = os.path.join(output_folder, "mode.txt")
    write_list(data_format_mark_txt, [data_format_hash])

    for index in range(len(mesh_path_list)):
        mesh_path = mesh_path_list[index]
        manifold_mesh_path = manifold_mesh_path_list[index]
        proc_data_folder = proc_data_folder_list[index]

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_name = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]
        mesh_type = os.path.splitext(mesh_name)[1]
        final_output_folder = output_folder_list[index]
        if apply_render:
            if not os.path.exists(final_output_folder):
                os.mkdir(final_output_folder)
        final_trans_txt = trans_txt_list[index]
        final_z_txt = z_txt_list[index]

        if len(generate_pose_config_json_path) > 0 and os.path.exists(generate_pose_config_json_path):
            generate_camera_cmd = 'python \"{}\" '.format(generate_pose_script_path)
            generate_camera_cmd = "{} --output_folder \"{}\" ".format(generate_camera_cmd, final_output_folder)
            generate_camera_cmd = "{} --config_json \"{}\" ".format(generate_camera_cmd, generate_pose_config_json_path)
            generate_camera_cmd = "{} --image_size_x {} ".format(generate_camera_cmd, config_struct["render_width"])
            generate_camera_cmd = "{} --image_size_y {} ".format(generate_camera_cmd, config_struct["render_height"])
            if apply_render:
                if not run_cmd(generate_camera_cmd):
                    logging.error(f"Cannot run pose generation command {generate_camera_cmd}")
                    continue

            pose_json_path = os.path.join(final_output_folder, "internal_cam_parameters.json")

        config_struct["transformation_txt"] = final_trans_txt
        config_struct["z_txt"] = final_z_txt
        config_struct["pose_json"] = pose_json_path
        config_struct["mesh_json_path"] = data_json_path
        config_struct["mesh_list_path"] = mesh_list_path

        mesh_cmd_struct = {}

        sdf_sample_cmd = "python \"{}\" --mesh_path \"{}\" ".format(sdf_script_fullpath, manifold_mesh_path)
        sdf_sample_cmd = "{} --output_folder \"{}\" ".format(sdf_sample_cmd, proc_data_folder)
        sdf_sample_cmd = "{} --transform_path \"{}\" ".format(sdf_sample_cmd, final_trans_txt)
        sdf_sample_cmd = "{} --z_transform_path \"{}\" ".format(sdf_sample_cmd, final_z_txt)
        sdf_sample_cmd = "{} --standard_height {} ".format(sdf_sample_cmd, config_struct["standard_height"])
        sdf_sample_cmd = "{} --space_sample_number {} ".format(sdf_sample_cmd,
                                                               config_struct["geometry_space_sample_number"])

        sdf_sample_cmd = "{} --near_surface_sample_number {} ".format(sdf_sample_cmd, config_struct[
            "geometry_near_surface_sample_number"])

        sdf_sample_cmd = "{} --surface_sample_number {} ".format(sdf_sample_cmd,
                                                                 config_struct["geometry_surface_sample_number"])

        sdf_sample_cmd = "{} --sample_format \'{}\' ".format(sdf_sample_cmd, config_struct["sample_format"])
        sdf_sample_cmd = "{} --chunk_size {} ".format(sdf_sample_cmd, config_struct["h5_chunk_size"])
        if config_struct["shuffle_sample"]:
            sdf_sample_cmd = "{} --shuffle ".format(sdf_sample_cmd)

        mesh_cmd_struct["sdf"] = sdf_sample_cmd

        render_cmd = "{} -b -P {} -- --mesh_path \"{}\" ".format(blender_root, render_script_fullpath, mesh_path)
        render_stage_map = {}
        render_stage_map["cmd"] = {}

        if apply_render:
            camera_pose_json_data = read_json(pose_json_path)
            camera_pose_data = camera_pose_json_data["poses"]

            render_cmd = "{} --transform_path \"{}\" --pose_json_path \"{}\" ".format(render_cmd,
                                                                                      final_z_txt,
                                                                                      pose_json_path)

            if args.force_better_fbx:
                render_cmd = "{} --use_better_fbx ".format(render_cmd)

            if scaled_obj:
                render_cmd = "{} --export_scaled_obj ".format(render_cmd)

            if config_struct["outside_trans"]:
                render_cmd = "{} --use_outside_transform ".format(render_cmd)

            if "rotate_object" in config_struct.keys():
                if config_struct["rotate_object"]:
                    render_cmd = "{} --rotate_object ".format(render_cmd)

            if "color_background" in config_struct.keys():
                if config_struct["color_background"]:
                    render_cmd = "{} --color_background ".format(render_cmd)

            if "unit_trans" in config_struct.keys():
                if config_struct["unit_trans"]:
                    render_cmd = "{} --use_unit_transform ".format(render_cmd)

            if config_struct["daz"]:
                render_cmd = "{} --render_daz ".format(render_cmd)

            if config_struct["solidify"]:
                render_cmd = "{} --solidify ".format(render_cmd)

            if "aux_format" in config_struct.keys():
                render_cmd = "{} --aux_image_type \"{}\" ".format(render_cmd, config_struct["aux_format"])

            render_cmd = "{} --engine \"{}\" --render_height {} --render_width {} ".format(render_cmd,
                                                                                           config_struct["engine"],
                                                                                           config_struct[
                                                                                               "render_height"],
                                                                                           config_struct[
                                                                                               "render_width"])

            camera_number = len(camera_pose_data.keys())

            if config_struct["light_source"] == "hdr":
                hdr_usage = None
                if "hdr_usage" in config_struct.keys():
                    hdr_usage = config_struct["hdr_usage"]

                if hdr_usage == 'random':
                    hdr_map_path_list = config_struct["hdr_path"]
                    hdr_map_number = len(hdr_map_path_list)
                    config_struct["hdr"] = {}
                    config_struct["hdr"]["paths"] = {}

                    # configure hdr map usage for each rendering mode
                    if pose_generation_mode == "RTriVC":
                        online_config = camera_pose_json_data['config'][pose_generation_mode]
                        equator_number = online_config["equator_number"]
                        all_direction_number = online_config["all_direction_number"]

                        camera_system_camera_number = len(online_config["azimuth_list"])
                        y_only_number = online_config["persp_y_only_number"] + online_config["ortho_y_only_number"]
                        y_aux_camera_number = online_config["y_aux_camera_number"]

                        full_augmentation_number = y_only_number * (1 + y_aux_camera_number)
                        total_camera_number = camera_system_camera_number + full_augmentation_number

                        object_rotation_number = equator_number + all_direction_number
                        if online_config["first_identity"]:
                            object_rotation_number = object_rotation_number + 1

                        random_hdr_path_list = []
                        for index_light in range(y_only_number * object_rotation_number):
                            random_index = random.randint(0, hdr_map_number - 1)
                            random_map_path = hdr_map_path_list[random_index]
                            hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                            for index_same_light in range(1 + y_aux_camera_number):
                                random_hdr_path_list.append(hdr_map_path)

                        for index_x in range(object_rotation_number):
                            for index_y in range(total_camera_number):
                                camera_index = index_y + index_x * total_camera_number
                                camera_name = "cam-%04d" % (camera_index)
                                if index_y < camera_system_camera_number:
                                    relative_equilibrium_path = config_struct["approximate_equilibrium_path"]
                                    abs_equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        relative_equilibrium_path)
                                    config_struct["hdr"]["paths"][camera_name] = abs_equilibrium_path
                                else:
                                    index_light = index_y - camera_system_camera_number + index_x * full_augmentation_number
                                    config_struct["hdr"]["paths"][camera_name] = random_hdr_path_list[index_light]

                    elif pose_generation_mode == 'RSVC':
                        rsvc_config = camera_pose_json_data['config']["RSVC"]
                        target_number = len(rsvc_config["azimuth_list"])
                        target_3view_number = rsvc_config["3view_target_number"]
                        total_target_3view_number = target_3view_number * 3
                        condition_number = rsvc_config["condition_camera_number"]

                        all_camera_number = target_number + total_target_3view_number + condition_number
                        target_3view_hdr_list = []
                        for group_index in range(target_3view_number):
                            random_index = random.randint(0, hdr_map_number - 1)
                            random_map_path = hdr_map_path_list[random_index]
                            hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                            target_3view_hdr_list.append(hdr_map_path)

                        for camera_index in range(all_camera_number):
                            camera_name = "cam-%04d" % (camera_index)
                            if camera_index < target_number:

                                relative_equilibrium_path = config_struct["approximate_equilibrium_path"]
                                abs_equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    relative_equilibrium_path)
                                config_struct["hdr"]["paths"][camera_name] = abs_equilibrium_path
                            elif camera_index < (target_number + total_target_3view_number):
                                group_index = int((camera_index - target_number) / 3)
                                config_struct["hdr"]["paths"][camera_name] = target_3view_hdr_list[group_index]
                            else:
                                random_index = random.randint(0, hdr_map_number - 1)
                                random_map_path = hdr_map_path_list[random_index]
                                hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                                config_struct["hdr"]["paths"][camera_name] = hdr_map_path

                    elif pose_generation_mode == 'RC':
                        rsvc_config = camera_pose_json_data['config']["RC"]
                        target_number = len(rsvc_config["azimuth_list"])
                        condition_number_list = rsvc_config["partition_camera_number_list"]
                        condition_number = sum(i for i in condition_number_list if isinstance(i, int))
                        all_camera_number = target_number + condition_number
                        for camera_index in range(all_camera_number):
                            camera_name = "cam-%04d" % (camera_index)
                            if camera_index < target_number:

                                relative_equilibrium_path = config_struct["approximate_equilibrium_path"]
                                abs_equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    relative_equilibrium_path)
                                config_struct["hdr"]["paths"][camera_name] = abs_equilibrium_path
                            else:
                                random_index = random.randint(0, hdr_map_number - 1)
                                random_map_path = hdr_map_path_list[random_index]
                                hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                                config_struct["hdr"]["paths"][camera_name] = hdr_map_path

                    else:
                        random_index = random.randint(0, hdr_map_number - 1)
                        random_map_path = hdr_map_path_list[random_index]
                        hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                        for camera_name in camera_pose_data.keys():
                            config_struct["hdr"]["paths"][camera_name] = hdr_map_path

                elif hdr_usage == 'static' or hdr_usage is None:
                    config_struct["hdr"] = {}
                    config_struct["hdr"]["paths"] = {}
                    if "hdr_path" in config_struct:
                        hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    config_struct["hdr_path"])
                        if not os.path.exists(hdr_map_path):
                            hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "irrmaps/mud_road_puresky_4k.hdr")
                    else:
                        hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "irrmaps/mud_road_puresky_4k.hdr")
                    for camera_name in camera_pose_data.keys():
                        config_struct["hdr"]["paths"][camera_name] = hdr_map_path
            elif config_struct["light_source"] == "point":
                render_cmd = "{} --use_point_light ".format(render_cmd)

            if "equilibrium_path" in config_struct.keys():
                relative_equilibrium_path = config_struct["equilibrium_path"]
                abs_equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    relative_equilibrium_path)
                config_struct["equilibrium_path"] = abs_equilibrium_path

            render_stage_map["output_folder"] = {}
            for render_stage in render_stages:
                if render_stage != "common":
                    render_stage_map["output_folder"][render_stage] = os.path.join(final_output_folder, render_stage)
                else:
                    render_stage_map["output_folder"][render_stage] = final_output_folder
            config_struct["stages"] = render_stage_map["output_folder"]

            if apply_render:
                config_json_output_path = os.path.join(final_output_folder, "config.json")
                write_json(config_json_output_path, config_struct)
                render_cmd = "{} --camera_config_path \"{}\" ".format(render_cmd, config_json_output_path)

            for render_stage in render_stages:
                if render_stage == "emission":
                    render_stage_map["cmd"][render_stage] = "{} --only_emission ".format(render_cmd)
                if render_stage == "equilibrium":
                    render_stage_map["cmd"][render_stage] = "{} --render_equilibrium ".format(render_cmd)
                if render_stage == "PBR":
                    render_stage_map["cmd"][
                        render_stage] = "{} --render_material --material_type \"PBR\" ".format(render_cmd)
                if render_stage == "bump":
                    render_stage_map["cmd"][
                        render_stage] = "{} --render_material --material_type \"bump\" ".format(render_cmd)
                if render_stage == "common":
                    render_stage_map["cmd"][render_stage] = render_cmd
                if render_stage == "no_smooth":
                    render_stage_map["cmd"][render_stage] = render_cmd

                render_stage_map["cmd"][render_stage] = "{} --output_folder \"{}\" ".format(
                    render_stage_map["cmd"][render_stage],
                    render_stage_map["output_folder"][render_stage])

                if args.parse_exr:
                    if render_stage == "common" or render_stage == "no_smooth":
                        render_stage_map["cmd"][render_stage] = "{} --parse_exr ".format(
                            render_stage_map["cmd"][render_stage])
                else:
                    if args.only_render_png:
                        if render_stage == "common" or render_stage == "no_smooth":
                            render_stage_map["cmd"][render_stage] = "{} --only_render_png ".format(
                                render_stage_map["cmd"][render_stage])

                if render_stage == "emission" or render_stage == "equilibrium":
                    render_stage_map["cmd"][render_stage] = "{} --only_render_png --smooth --no_camera_export ".format(
                        render_stage_map["cmd"][render_stage])
                if render_stage == "PBR" or render_stage == "bump":
                    render_stage_map["cmd"][render_stage] = "{} --only_render_png --smooth --no_camera_export ".format(
                        render_stage_map["cmd"][render_stage])

                if render_stage == "no_smooth":
                    render_stage_map["cmd"][render_stage] = "{} --no_camera_export ".format(
                        render_stage_map["cmd"][render_stage])

                if render_stage == "common":
                    render_stage_map["cmd"][render_stage] = "{} --smooth ".format(render_stage_map["cmd"][render_stage])

        if not donot_apply_any_process:
            pool.apply_async(func=render_once, args=(render_stage_map["cmd"], mesh_cmd_struct,
                                                     mesh_path, final_output_folder, proc_data_folder,
                                                     stat_txt, time_txt, folder_txt, proc_txt,
                                                     args.silent, apply_render, apply_preprocess_mesh,
                                                     render_stages))

        if apply_render:
            with open(cmds_txt, 'a', encoding='UTF-8') as f:
                for render_stage in render_stages:
                    if render_stage in render_stage_map["cmd"].keys():
                        f.write(render_stage_map["cmd"][render_stage] + '\n')
        if apply_preprocess_mesh:
            with open(proc_cmds_txt, 'a', encoding='UTF-8') as f:
                for mesh_stage in mesh_cmd_struct.keys():
                    f.write(mesh_cmd_struct[mesh_stage] + '\n')

    pool.close()
    pool.join()

    local_time_str = current_time()
    logging.info("All rendering tasks DONE. These tasks end at time %s" %
                 (local_time_str))
