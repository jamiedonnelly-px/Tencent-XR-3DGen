import argparse
import copy
import json
import os
import shutil
import time


def check_data_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    total_data_number = 0
    for category_name in category:
        total_data_number = total_data_number + \
                            len(json_struct["data"][category_name])
    return total_data_number


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


def correct_disk_name(path_name: str):
    if 'apdcephfs_data_cq3' in path_name:
        new_path_name = path_name.replace(
            'apdcephfs_data_cq3', 'apdcephfs_cq3')
        return new_path_name
    else:
        return path_name


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh json start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--original_json', type=str, default="",
                        help='json file containing old data info')
    # parser.add_argument('--output_path_json', type=str, default="",
    #                     help='output path of data_name <-> mesh name list json')
    parser.add_argument('--data_info_json', type=str, default="",
                        help='json containing data names, whether to rerender/move, and if rerender/move, contains path info')
    parser.add_argument('--apply_render', action='store_true',
                        help='re-render all data in data_info_json')
    parser.add_argument('--render_pool_cnt', type=int, default=24,
                        help='parallel number of rendering one single data, no parallel between several datas')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')

    args = parser.parse_args()

    original_json = args.original_json
    # output_path_json = args.output_path_json
    data_info_json = args.data_info_json
    apply_render = args.apply_render
    log_folder = args.log_folder

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')

    old_data_struct = {}
    if os.path.exists(original_json):
        old_data_struct = read_json(original_json)
        new_data_struct = copy.deepcopy(old_data_struct)

    raw_all_data_struct = old_data_struct["data"]

    data_info = {}
    if os.path.exists(data_info_json):
        data_info = read_json(data_info_json)

    for data_name in data_info.keys():
        move_folder = data_info[data_name]["folder"]
        if not os.path.exists(move_folder):
            os.mkdir(move_folder)

        mesh_path_list = []
        if data_name not in raw_all_data_struct.keys():
            continue

        single_data_struct = raw_all_data_struct[data_name]
        for key, value in single_data_struct.items():
            if "Obj_Mesh" in value.keys():
                mesh_path = value["Obj_Mesh"]
            else:
                mesh_path = value["Mesh"]
            if mesh_path is not None:
                mesh_path_list.append(mesh_path)
            else:
                print(key)

        mesh_list_fullpath = os.path.join(move_folder, "obj_list.txt")
        write_list(mesh_list_fullpath, mesh_path_list)
        print("Data length is %i" % (len(mesh_path_list)))

        time.sleep(0.1)

        render_batch_script_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../render_mesh_batch.py")

        if apply_render:
            data_config_json = data_info[data_name]["config"]
            connection = data_info[data_name]["connection"]
            data_pose_json = data_info[data_name]["pose"]

            local_default_json = os.path.join(move_folder, "default.json")
            if not os.path.exists(local_default_json):
                shutil.copyfile(data_config_json, local_default_json)

            cam_filename = os.path.split(data_pose_json)[1]
            local_cam_json = os.path.join(move_folder, cam_filename)
            if not os.path.exists(local_cam_json):
                shutil.copyfile(data_pose_json, local_cam_json)

            render_folder = os.path.join(move_folder, "render_data")
            if not os.path.exists(render_folder):
                os.mkdir(render_folder)
            proc_data_folder = os.path.join(move_folder, "proc_data")
            if not os.path.exists(proc_data_folder):
                os.mkdir(proc_data_folder)
            log_folder = os.path.join(move_folder, "log")
            if not os.path.exists(log_folder):
                os.mkdir(log_folder)

            render_cmd_str = "python \'{}\' ".format(render_batch_script_path)

            input_list_str = ' --in_mesh_list_txt  \'{}\' '.format(
                mesh_list_fullpath)
            render_cmd_str = render_cmd_str + input_list_str

            pose_json_str = ' --pose_json_path \'{}\' '.format(local_cam_json)
            render_cmd_str = render_cmd_str + pose_json_str

            output_folder_str = ' --output_folder  \'{}\' '.format(
                render_folder)
            render_cmd_str = render_cmd_str + output_folder_str

            proc_data_folder_str = ' --proc_data_output_folder \'{}\' '.format(
                proc_data_folder)
            render_cmd_str = render_cmd_str + proc_data_folder_str

            log_str = ' --log_folder \'{}\' '.format(log_folder)
            render_cmd_str = render_cmd_str + log_str

            config_json_str = ' --config_json_path \'{}\' '.format(
                local_default_json)
            render_cmd_str = render_cmd_str + config_json_str

            connection_str = " --connection_starts \'{}\' ".format(connection)
            render_cmd_str = render_cmd_str + connection_str

            control_str = ' --silent --parse_exr --pool_cnt {} '.format(
                args.render_pool_cnt)
            render_cmd_str = render_cmd_str + control_str

            if "blender" in data_info[data_name]:
                blender_str = ' --blender_root \'{}\' '.format(
                    data_info[data_name]["blender"])
                render_cmd_str = render_cmd_str + blender_str

            time.sleep(0.1)
            print(render_cmd_str)

            with open(cmds_txt, 'a') as f:
                f.write(render_cmd_str + '\n')
            time.sleep(0.1)
            os.system(render_cmd_str)
            time.sleep(0.1)
        else:
            print("Finish exporting mesh lists of data %s...." % (data_name))
