import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def convert_once(convert_cmd, obj_mesh_folder, glb_mesh_folder, input_glb_path, target_glb_path,
                 mesh_path, output_folder, stat_txt, obj_txt, glb_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for convert file (at %s) format cmd is %s....' %
          (mesh_path, str(start_time_str)))

    if not os.path.exists(obj_mesh_folder):
        os.mkdir(obj_mesh_folder)
    if not os.path.exists(glb_mesh_folder):
        os.mkdir(glb_mesh_folder)

    print("Copy glb from %s to %s" % (input_glb_path, target_glb_path))

    shutil.copyfile(input_glb_path, target_glb_path)

    print(convert_cmd)
    stat = os.system(convert_cmd)
    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After convert format command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_path))

    with open(obj_txt, 'a') as f:
        f.write('{}\n'.format(output_folder))

    with open(glb_txt, 'a') as f:
        f.write('{}\n'.format(target_glb_path))


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
    str_list = []
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        return str_list

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


def check_str_in_list(check_str: str, str_list: list):
    for str_element in str_list:
        if check_str in str_element:
            return True
    return False


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Wash mesh name start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--input_json_path', type=str, default="",
                        help='input mesh json file path')
    parser.add_argument('--new_mesh_folder', type=str, default="",
                        help='folder of copied meshes')
    parser.add_argument('--output_json_path', type=str, default="",
                        help='output mesh json file path containing converted paths')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--pool_cnt', type=int, default=24,
                        help='parallel number of one thread, no parallel between several datas')
    parser.add_argument('--blender_root', type=str, default='/root/blender-3.6.2-linux-x64/blender',
                        help='blender execution file abspath')
    parser.add_argument('--data_tags', type=str, default='',
                        help='tags for data_name to be washed')

    args = parser.parse_args()
    input_json_path = args.input_json_path
    new_mesh_folder = args.new_mesh_folder
    output_json_path = args.output_json_path
    log_folder = args.log_folder
    pool_cnt = args.pool_cnt
    blender_root = args.blender_root

    obj_folder = os.path.join(new_mesh_folder, "objmesh")
    if not os.path.exists(obj_folder):
        os.mkdir(obj_folder)

    glb_folder = os.path.join(new_mesh_folder, "glbmesh")
    if not os.path.exists(glb_folder):
        os.mkdir(glb_folder)

    data_tags_str = args.data_tags
    data_tags = []
    if len(data_tags_str) > 0:
        data_tags = data_tags_str.split("+")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    pool = ThreadPoolExecutor(max_workers=pool_cnt,
                              thread_name_prefix='wash_obj')

    input_data_struct = read_json(input_json_path)
    mesh_data_struct = input_data_struct["data"]

    final_output_struct = {}
    final_output_struct["data"] = {}
    output_struct = final_output_struct["data"]

    object_direction_op = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../geometry/object_direction.py")

    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    obj_txt = os.path.join(log_folder, 'folder.txt')
    glb_txt = os.path.join(log_folder, 'glb.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(glb_txt, 'w')
    folder_file = open(obj_txt, 'w')

    mesh_folder_copied = set()

    for data_name in mesh_data_struct.keys():
        if "VRoid_" in data_name or "DAZ_" in data_name:
            if data_name not in output_struct.keys():
                output_struct[data_name] = {}
            for mesh_name in mesh_data_struct[data_name].keys():
                if mesh_name not in output_struct[data_name].keys():
                    output_struct[data_name][mesh_name] = mesh_data_struct[data_name][mesh_name]

                mesh_original_path = mesh_data_struct[data_name][mesh_name]["Mesh"]
                glb_original_path = mesh_data_struct[data_name][mesh_name]["GLB_Mesh"]

                mesh_filename_with_split = mesh_original_path.split(
                    "split")[-1]
                mesh_filename_with_split = "split" + mesh_filename_with_split

                mesh_partname_folder = os.path.split(mesh_original_path)[0]
                mesh_split_folder = os.path.split(mesh_partname_folder)[0]
                mesh_meshname_folder = os.path.split(mesh_split_folder)[0]
                mesh_category_folder = os.path.split(mesh_meshname_folder)[0]

                folder_mesh_category = os.path.split(mesh_category_folder)[1]
                folder_mesh_name = os.path.split(mesh_meshname_folder)[1]
                obj_mesh_category_folder = os.path.join(
                    obj_folder, folder_mesh_category)
                if not os.path.exists(obj_mesh_category_folder):
                    os.mkdir(obj_mesh_category_folder)
                obj_mesh_folder = os.path.join(
                    obj_mesh_category_folder, folder_mesh_name)

                glb_mesh_category_folder = os.path.join(glb_folder, data_name)
                if not os.path.exists(glb_mesh_category_folder):
                    os.mkdir(glb_mesh_category_folder)
                glb_mesh_folder = os.path.join(
                    glb_mesh_category_folder, mesh_name)
                new_glb_mesh_fullpath = os.path.join(
                    glb_mesh_folder, mesh_name + ".glb")

                if mesh_meshname_folder in mesh_folder_copied:
                    copy_cmd = "echo 1"
                else:
                    copy_cmd = "rclone copy \'{}\' \'{}\'".format(
                        mesh_meshname_folder, obj_mesh_folder)
                    copy_cmd = copy_cmd + " --transfers=32 -P --stats-one-line --checksum --checkers=4 "

                mesh_folder_copied.add(mesh_meshname_folder)
                new_mesh_fullpath = os.path.join(obj_mesh_folder, mesh_filename_with_split)
                output_struct[data_name][mesh_name]["Mesh"] = new_mesh_fullpath
                output_struct[data_name][mesh_name]["Obj_Mesh"] = new_mesh_fullpath
                output_struct[data_name][mesh_name]["GLB_Mesh"] = new_glb_mesh_fullpath

                pool.submit(convert_once, copy_cmd, obj_mesh_folder,
                            glb_mesh_folder, glb_original_path, new_glb_mesh_fullpath,
                            mesh_original_path, new_mesh_fullpath,
                            stat_txt, obj_txt, glb_txt)

        else:
            if data_name not in output_struct.keys():
                output_struct[data_name] = {}
            # if not check_str_in_list(data_name, data_tags):
            #     output_struct[data_name] = mesh_data_struct[data_name]
            #     continue
            for mesh_name in mesh_data_struct[data_name].keys():
                if mesh_name not in output_struct[data_name].keys():
                    output_struct[data_name][mesh_name] = mesh_data_struct[data_name][mesh_name]

                mesh_category = data_name
                mesh_original_path = mesh_data_struct[data_name][mesh_name]["Mesh"]
                glb_original_path = mesh_data_struct[data_name][mesh_name]["GLB_Mesh"]

                obj_mesh_category_folder = os.path.join(
                    obj_folder, mesh_category)
                if not os.path.exists(obj_mesh_category_folder):
                    os.mkdir(obj_mesh_category_folder)
                obj_mesh_folder = os.path.join(
                    obj_mesh_category_folder, mesh_name)
                new_mesh_fullpath = os.path.join(
                    obj_mesh_folder, mesh_name + ".obj")

                glb_mesh_category_folder = os.path.join(
                    glb_folder, mesh_category)
                if not os.path.exists(glb_mesh_category_folder):
                    os.mkdir(glb_mesh_category_folder)
                glb_mesh_folder = os.path.join(
                    glb_mesh_category_folder, mesh_name)
                new_glb_mesh_fullpath = os.path.join(
                    glb_mesh_folder, mesh_name + ".glb")

                direction_correct_cmd = "{} -b -P {} -- ".format(
                    blender_root, object_direction_op)
                direction_correct_cmd = direction_correct_cmd + \
                                        " --mesh_path '{}' --direction_corrected_mesh_path '{}'  > /dev/null".format(
                                            mesh_original_path, new_mesh_fullpath)

                output_struct[data_name][mesh_name]["Mesh"] = new_mesh_fullpath
                output_struct[data_name][mesh_name]["Obj_Mesh"] = new_mesh_fullpath
                output_struct[data_name][mesh_name]["GLB_Mesh"] = new_glb_mesh_fullpath

                pool.submit(convert_once, direction_correct_cmd, obj_mesh_folder,
                            glb_mesh_folder, glb_original_path, new_glb_mesh_fullpath,
                            mesh_original_path, new_mesh_fullpath,
                            stat_txt, obj_txt, glb_txt)

    pool.shutdown()
    time.sleep(0.1)

    print(check_individual_number(input_data_struct))
    print(check_individual_number(final_output_struct))
    write_json(output_json_path, final_output_struct)
