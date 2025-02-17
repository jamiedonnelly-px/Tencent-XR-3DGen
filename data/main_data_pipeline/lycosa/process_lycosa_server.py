import os
import numpy as np
import gc
import argparse
import time
import math
import json
import sys
import hashlib
from threading import Thread


def read_list(in_list_txt):
    str_list = []
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        return str_list

    with open(in_list_txt, 'r', encoding='UTF-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='UTF-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def check_list_txt_length(txt_path: str):
    mesh_list = read_list(txt_path)
    return len(mesh_list)


def render_one_type_data(render_data_folder: str, file_list_path: str, pose_json: str, render_config_json: str,
                         data_name: str, blender_root: str):
    batch_render_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "render_mesh_lycosa.py")
    rendering_cmd_basic = "python {} --only_render_png --pool_cnt 12 --thumbnail --apply_render --render_stage_string \"common\" --connection_starts \"{}\" ".format(
        batch_render_fullpath, data_name)
    if os.path.exists(file_list_path):
        mesh_number = check_list_txt_length(file_list_path)
        if mesh_number > 0:
            if not os.path.exists(render_data_folder):
                os.mkdir(render_data_folder)
            fbx_render_log_folder = os.path.join(render_data_folder, "log")
            if not os.path.exists(fbx_render_log_folder):
                os.mkdir(fbx_render_log_folder)
            fbx_render_data = os.path.join(render_data_folder, "render_data")
            if not os.path.exists(fbx_render_data):
                os.mkdir(fbx_render_data)
            rendering_cmd = rendering_cmd_basic + \
                            " --in_mesh_list_txt \"{}\" ".format(file_list_path)
            rendering_cmd = rendering_cmd + \
                            " --pose_json_path \"{}\" ".format(pose_json)
            rendering_cmd = rendering_cmd + \
                            " --output_folder \"{}\" ".format(fbx_render_data)
            rendering_cmd = rendering_cmd + \
                            " --config_json_path \"{}\" ".format(render_config_json)
            rendering_cmd = rendering_cmd + \
                            " --log_folder  \"{}\" ".format(fbx_render_log_folder)
            rendering_cmd = rendering_cmd + \
                            " --blender_root  \"{}\" ".format(blender_root)

            if sys.platform.startswith('win'):
                rendering_cmd = rendering_cmd + " > NUL "
            else:
                rendering_cmd = rendering_cmd + " --silent "

            print("Rendering %i files; list locates at %s" %
                  (mesh_number, file_list_path))

            print(rendering_cmd)
            os.system(rendering_cmd)
            time.sleep(0.1)


def scan_files_of_format(data_folder: str, list_filepath: str, folder_list_filepath: str, file_type: str):
    list_generation_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "generate_mesh_list.py")
    scan_max_file_list_cmd = 'python {} --render_folders {} --output_list {} --output_folder_list {} --file_type {}'.format(
        list_generation_fullpath, data_folder, list_filepath, folder_list_filepath, file_type)
    print("Scan for %s files in %s" % (file_type, lycosa_data_folder))
    os.system(scan_max_file_list_cmd)
    time.sleep(0.1)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Server side lycosa process start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Lycosa processing pipeline.')
    parser.add_argument('--lycosa_data_folder', type=str, default="",
                        help='lycosa data folder full path')
    parser.add_argument('--lycosa_render_folder', type=str, default="",
                        help='lycosa render results folder full path')
    parser.add_argument('--v_pose_json', type=str, default="",
                        help='pose json file of v-render 300 poses')
    parser.add_argument('--render_config_json', type=str, default="",
                        help='rendering config json file')
    parser.add_argument('--data_name', type=str, default="",
                        help='lycosa data name')
    parser.add_argument('--secret_id', type=str,
                        help='secret id of cos account')
    parser.add_argument('--secret_key', type=str,
                        help='secret key of cos account')
    parser.add_argument('--blender_root', type=str,
                        default="/root/blender-3.5.0-linux-x64/blender",
                        help='blender exec file absolute path')
    parser.add_argument('--upload_pool_cnt', type=int, default=8,
                        help='upload thread pool ')
    args = parser.parse_args()

    lycosa_data_folder = args.lycosa_data_folder
    lycosa_render_folder = args.lycosa_render_folder
    v_pose_json = args.v_pose_json
    render_config_json = args.render_config_json
    data_name = args.data_name
    blender_root = args.blender_root

    if not os.path.exists(lycosa_render_folder):
        os.mkdir(lycosa_render_folder)
    fbx_render = os.path.join(lycosa_render_folder, "fbx")
    if not os.path.exists(fbx_render):
        os.mkdir(fbx_render)

    obj_render = os.path.join(lycosa_render_folder, "obj")
    if not os.path.exists(obj_render):
        os.mkdir(obj_render)

    fbx_list_filepath = os.path.join(lycosa_data_folder, "fbx_list.txt")
    fbx_folder_list_filepath = os.path.join(
        lycosa_data_folder, "fbx_folder_list.txt")
    obj_list_filepath = os.path.join(lycosa_data_folder, "obj_list.txt")
    obj_folder_list_filepath = os.path.join(
        lycosa_data_folder, "obj_folder_list.txt")
    glb_list_filepath = os.path.join(lycosa_data_folder, "glb_list.txt")
    glb_folder_list_filepath = os.path.join(
        lycosa_data_folder, "glb_folder_list.txt")

    scan_files_of_format(data_folder=lycosa_data_folder,
                         list_filepath=fbx_list_filepath,
                         folder_list_filepath=fbx_folder_list_filepath,
                         file_type=".fbx")

    render_one_type_data(render_data_folder=fbx_render,
                         file_list_path=fbx_list_filepath,
                         pose_json=v_pose_json,
                         render_config_json=render_config_json,
                         data_name=data_name,
                         blender_root=blender_root)

    scan_files_of_format(data_folder=lycosa_data_folder,
                         list_filepath=obj_list_filepath,
                         folder_list_filepath=obj_folder_list_filepath,
                         file_type=".obj")

    render_one_type_data(render_data_folder=obj_render,
                         file_list_path=obj_list_filepath,
                         pose_json=v_pose_json,
                         render_config_json=render_config_json,
                         data_name=data_name,
                         blender_root=blender_root)

    total_success_list = []
    total_success_txt = os.path.join(lycosa_render_folder, "total_success.txt")
    total_folder_list = []
    total_folder_txt = os.path.join(lycosa_render_folder, "total_folder.txt")

    fbx_render_log = os.path.join(fbx_render, "log")
    obj_render_log = os.path.join(obj_render, "log")

    fbx_success_txt = os.path.join(fbx_render_log, "success.txt")
    fbx_folder_txt = os.path.join(fbx_render_log, "folder.txt")
    fbx_success_list = read_list(fbx_success_txt)
    fbx_folder_list = read_list(fbx_folder_txt)
    total_success_list = total_success_list + fbx_success_list
    total_folder_list = total_folder_list + fbx_folder_list

    obj_success_txt = os.path.join(obj_render_log, "success.txt")
    obj_folder_txt = os.path.join(obj_render_log, "folder.txt")
    obj_success_list = read_list(obj_success_txt)
    obj_folder_list = read_list(obj_folder_txt)
    total_success_list = total_success_list + obj_success_list
    total_folder_list = total_folder_list + obj_folder_list

    write_list(total_success_txt, total_success_list)
    write_list(total_folder_txt, total_folder_list)

    management_op = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "../../management/generate_management_json.py")
    management_json = os.path.join(
        lycosa_render_folder, data_name + "_management.json")
    management_thumbnail_json = os.path.join(
        lycosa_render_folder, data_name + "_management_thumbnail.json")

    management_cmd_str = "python \"{}\" --data_name \"{}\"  --data_style \"PR\"  --data_origin \"Lycosa\" ".format(
        management_op, data_name)
    management_cmd_str = management_cmd_str + \
                         "--secret_id \"{}\" --secret_key \"{}\" ".format(
                             args.secret_id, args.secret_key)
    management_cmd_str = management_cmd_str + \
                         " --output_json \"{}\" ".format(management_json)
    management_cmd_str = management_cmd_str + \
                         " --output_thumbnail_json \"{}\" ".format(management_thumbnail_json)
    management_cmd_str = management_cmd_str + \
                         " --mesh_list \"{}\" ".format(total_success_txt)
    management_cmd_str = management_cmd_str + \
                         " --folder_list \"{}\" ".format(total_folder_txt)
    management_cmd_str = management_cmd_str + \
                         "--blender_root \"{}\" ".format(blender_root)

    print(management_cmd_str)
    os.system(management_cmd_str)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Server side lycosa process end. Start time is %s; end time is %s" %
          (local_time_str, end_time_str))
