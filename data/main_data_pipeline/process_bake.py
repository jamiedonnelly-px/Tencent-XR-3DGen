import argparse
import json
import multiprocessing
import os
import time


def convert_once(cmd_list, destination_mesh_path, output_folder, category, instance_name,
                 stat_txt, time_txt, folder_txt, category_txt, instance_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for baking command is %s....' %
          (str(start_time_str)))

    for cmd_struct in cmd_list:
        time.sleep(0.1)
        print("Start command %s -----> %s" %
              (str(cmd_struct["name"]), str(cmd_struct["cmd"])))

        convert_cmd = str(cmd_struct["cmd"])
        stat = os.system(convert_cmd)

        time.sleep(0.1)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After bake command status is %s; time for this status is %s....' %
          (str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        if stat == 0:
            f.write('{}\n'.format(destination_mesh_path))

    with open(folder_txt, 'a') as f:
        if stat == 0:
            f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (output_folder, start_time_str, end_time_str))

    with open(category_txt, 'a') as f:
        if stat == 0:
            f.write('{}\n'.format(category))

    with open(instance_txt, 'a') as f:
        if stat == 0:
            f.write('{}\n'.format(instance_name))


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
    new_path_name = path_name
    if path_name is None:
        return new_path_name
    if '/apdcephfs_data_cq3/share_2909871' in path_name:
        new_path_name = path_name.replace(
            '/apdcephfs_data_cq3/share_2909871', '/apdcephfs_cq8/share_2909871/jp_cq3_cephfs')
    return new_path_name


def read_mesh_list_from_data_json(json_path: str, dest_json_path: str, data_tag=""):
    print("Parse data json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        data_config = json.load(f)

    with open(dest_json_path, encoding='utf-8') as f:
        dest_data_config = json.load(f)

    obj_path_struct = {}
    object_category_list = []
    src_obj_path_list = []
    dst_obj_path_list = []
    instance_name_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = data_config["data"]
    dest_data_path_struct = dest_data_config["data"]
    data_path_name_list = list(data_path_struct.keys())
    dest_data_path_name_list = list(dest_data_path_struct.keys())

    for data_name in data_path_name_list:

        # src_data_name=dest_data_name
        # for current_src_data_name in data_path_name_list:
        #     if dest_data_name in current_src_data_name:
        #         src_data_name=current_src_data_name
        #         break

        if len(data_tag) > 1:
            if data_name != data_tag:
                continue
        if data_name not in dest_data_path_struct.keys():
            continue

        all_instance_path_struct = data_path_struct[data_name]
        dest_instance_path_struct = dest_data_path_struct[data_name]
        for instance_name in all_instance_path_struct.keys():
            instance_paths = all_instance_path_struct[instance_name]
            if instance_name not in dest_instance_path_struct.keys():
                continue
            # dest_instance_name = instance_name
            # for current_dest_name in dest_instance_path_struct.keys():
            #     if current_dest_name in instance_name:
            #         dest_instance_name = current_dest_name
            #         break
            # if dest_instance_name is None:
            #     continue
            dest_instance_paths = dest_instance_path_struct[instance_name]

            if "Mesh" not in instance_paths.keys():
                continue
            if "Mesh" not in dest_instance_paths.keys():
                continue
            src_mesh_path = instance_paths["Mesh"]
            dst_mesh_path = dest_instance_paths["Mesh"]
            if src_mesh_path is None:
                continue
            if dst_mesh_path is None:
                continue

            if "TexPcd" in instance_paths.keys():
                if instance_paths["TexPcd"] is not None:
                    tex_pcd_path = instance_paths["TexPcd"]
                    proc_data_folder = os.path.split(tex_pcd_path)[0]
                else:
                    proc_data_folder = None
            else:
                proc_data_folder = None

            src_obj_path_list.append(correct_disk_name(src_mesh_path))
            dst_obj_path_list.append(correct_disk_name(dst_mesh_path))
            object_category_list.append(data_name)
            instance_name_list.append(instance_name)
            proc_data_folder_list.append(correct_disk_name(proc_data_folder))

            # if "Manifold" in instance_paths.keys():
            #     if instance_paths["Manifold"] is not None:
            #         manifold_path = instance_paths["Manifold"]
            #     else:
            #         manifold_path = None
            #     manifold_path_list.append(manifold_path)

            # if "ImgDir" in instance_paths.keys():
            #     if instance_paths["ImgDir"] is not None:
            #         image_dir_path = instance_paths["ImgDir"]
            #         image_folder = os.path.split(image_dir_path)[0]
            #     else:
            #         image_folder = None
            #     render_data_list.append(image_folder)
    return src_obj_path_list, dst_obj_path_list, object_category_list, instance_name_list, proc_data_folder_list


def generate_unrepeated_render_folder_name(path: str, connection_starts='', hash_value: bool = False):
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
            last_model_index = len(elements) - \
                               elements[::-1].index(connection_starts)
            return "_".join(elements[last_model_index:])
        else:
            if 'save' in elements:
                last_model_index = len(elements) - elements[::-1].index('save')
                return "_".join(elements[last_model_index:])
            elif 'model' in elements:
                last_model_index = len(elements) - elements[::-1].index('model')
                return "_".join(elements[last_model_index:])
            elif '3dAsset_artcenter' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('3dAsset_artcenter')
                return "_".join(elements[last_model_index:])
            elif 'share_2909871' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('share_2909871')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_1' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('aigc_bucket_1')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_2' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('aigc_bucket_2')
                return "_".join(elements[last_model_index:])
            elif 'model_denoise' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('model_denoise')
                return "_".join(elements[last_model_index:])
            elif 'game_character_obj' in elements:
                last_model_index = len(elements) - \
                                   elements[::-1].index('game_character_obj')
                return "_".join(elements[last_model_index:])
            else:
                return "__".join(elements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bake meshes with pool.')
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--config_json_path', type=str, default="",
                        help='baking config json file path')
    parser.add_argument('--to_bake_data_json_path', type=str, default="",
                        help='data json file path contains models for baking')
    parser.add_argument('--already_baked_data_json_path', type=str, default="",
                        help='baking results in json data info format')
    parser.add_argument('--internal_uv', action='store_true',
                        help='calculate uv internally for destination mesh')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.1-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
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
    config_json_path = args.config_json_path
    to_bake_data_json_path = args.to_bake_data_json_path
    already_baked_data_json_path = args.already_baked_data_json_path
    internal_uv = args.internal_uv
    blender_root = args.blender_root
    log_folder = args.log_folder

    data_start = args.data_start
    data_end = args.data_end

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if not os.path.exists(config_json_path):
        config_json_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "config/bake.json")

    config_data_struct = read_json(config_json_path)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    mesh_categories = []
    if len(in_mesh_list_txt) > 1:
        destination_mesh_paths = read_list(in_mesh_list_txt)
    if len(data_json_path) > 1 and len(to_bake_data_json_path) > 1:
        source_mesh_paths, destination_mesh_paths, mesh_categories, instance_name_list, proc_data_folder_list = read_mesh_list_from_data_json(
            data_json_path, to_bake_data_json_path)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(source_mesh_paths):
                data_end = len(source_mesh_paths)
            source_mesh_paths = source_mesh_paths[data_start:data_end]
            destination_mesh_paths = destination_mesh_paths[data_start:data_end]
            proc_data_folder_list = proc_data_folder_list[data_start:data_end]
            instance_name_list = instance_name_list[data_start:data_end]
            mesh_categories = mesh_categories[data_start:data_end]

    if args.pod_num >= 0 and args.pod_id >= 0:
        mesh_path_len = len(source_mesh_paths)
        per_pod_len = mesh_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = mesh_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        source_mesh_paths = source_mesh_paths[idx_start:idx_end]
        destination_mesh_paths = destination_mesh_paths[idx_start:idx_end]
        proc_data_folder_list = proc_data_folder_list[idx_start:idx_end]
        instance_name_list = instance_name_list[idx_start:idx_end]
        mesh_categories = mesh_categories[idx_start:idx_end]

        output_folder = os.path.join(
            output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        log_folder = os.path.join(
            log_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        print("Number of mesh on this pod: %i/%i, (start: %i, end: %i)" %
              (len(source_mesh_paths), len(destination_mesh_paths), idx_start, idx_end))

    print("Number of mesh on this pod: %i/%i" %
          (len(source_mesh_paths), len(destination_mesh_paths)))

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    bake_cmds_txt = os.path.join(log_folder, 'bake_cmds.txt')
    uv_cmds_txt = os.path.join(log_folder, 'uv_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    category_txt = os.path.join(log_folder, 'category.txt')
    instance_txt = os.path.join(log_folder, 'instance.txt')

    bake_cmds_file = open(bake_cmds_txt, 'w')
    uv_cmds_file = open(uv_cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')
    category_file = open(category_txt, 'w')
    instance_file = open(instance_txt, 'w')

    uv_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "texture/uv.py")
    bake_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "texture/bake.py")

    instance_name_folder_map = {}
    command_list = []

    for index in range(len(source_mesh_paths)):
        source_mesh_path = source_mesh_paths[index]
        if not os.path.exists(source_mesh_path):
            print('Cannot find source mesh file ', source_mesh_path)
            continue

        destination_mesh_path = destination_mesh_paths[index]
        if not os.path.exists(destination_mesh_path):
            print('Cannot find destination mesh file ', destination_mesh_path)
            continue

        instance_name = instance_name_list[index]
        category = mesh_categories[index]
        if category not in instance_name_folder_map.keys():
            instance_name_folder_map[category] = {}
        instance_output_folder = os.path.join(output_folder, instance_name)
        if not os.path.exists(instance_output_folder):
            os.mkdir(instance_output_folder)
        final_output_folder = os.path.join(instance_output_folder, "bake")
        if not os.path.exists(final_output_folder):
            os.mkdir(final_output_folder)
        final_baked_mesh_fullpath = os.path.join(final_output_folder, "bake.obj")
        instance_name_folder_map[category][instance_name] = {}
        instance_name_folder_map[category][instance_name]["Bake"] = final_baked_mesh_fullpath

        if internal_uv:
            uv_folder = os.path.join(final_output_folder, "uv")
            if not os.path.exists(uv_folder):
                os.mkdir(uv_folder)
            uv_mesh = os.path.join(uv_folder, "mesh.obj")

            uv_cmd = "python {} --source_mesh_path '{}' --output_mesh_path '{}'".format(
                uv_op_fullpath, destination_mesh_path, uv_mesh)

            uv_command_struct = {}
            uv_command_struct["name"] = "uv"
            uv_command_struct["cmd"] = uv_cmd
            command_list.append(uv_command_struct)

        bake_cmd = "{} -b -P {} --".format(blender_root, bake_op_fullpath)
        bake_cmd = bake_cmd + \
                   " --source_mesh_path '{}' ".format(source_mesh_path)

        if internal_uv:
            bake_cmd = bake_cmd + \
                       " --destination_mesh_path '{}' ".format(uv_mesh)
        else:
            bake_cmd = bake_cmd + \
                       " --destination_mesh_path '{}' ".format(destination_mesh_path)

        bake_cmd = bake_cmd + \
                   " --output_mesh_folder '{}' ".format(final_output_folder)

        if config_data_struct["scale_mesh"]:
            if proc_data_folder_list[index] is not None:
                transformation_txt = os.path.join(
                    proc_data_folder_list[index], "transformation.txt")
                if not os.path.exists(transformation_txt):
                    print('Cannot find transformation txt file ',
                          transformation_txt)
                    continue
                bake_cmd = bake_cmd + \
                           " --source_trans_txt '{}' ".format(transformation_txt)

        if config_data_struct["source_z_up_axis"]:
            bake_cmd = bake_cmd + " --source_z_up_axis "
        if config_data_struct["destination_z_up_axis"]:
            bake_cmd = bake_cmd + " --destination_z_up_axis "
        if not config_data_struct["emission_baking"]:
            bake_cmd = bake_cmd + " --no_emission_baking "

        bake_cmd = bake_cmd + \
                   " --cage_extrusion {} ".format(
                       config_data_struct["cage_extrusion"])
        bake_cmd = bake_cmd + \
                   " --max_ray_distance {} ".format(
                       config_data_struct["max_ray_distance"])
        if "texture_image_width" in config_data_struct.keys():
            bake_cmd = bake_cmd + \
                       " --texture_image_width {} ".format(
                           config_data_struct["texture_image_width"])
        if "texture_image_height" in config_data_struct.keys():
            bake_cmd = bake_cmd + \
                       " --texture_image_height {} ".format(
                           config_data_struct["texture_image_height"])

        bake_command_struct = {}
        bake_command_struct["name"] = "baking"
        bake_command_struct["cmd"] = bake_cmd
        command_list.append(bake_command_struct)

        pool.apply_async(func=convert_once, args=(command_list, destination_mesh_path, final_output_folder,
                                                  category, instance_name, stat_txt, time_txt, folder_txt, category_txt,
                                                  instance_txt))

        if internal_uv:
            with open(uv_cmds_txt, 'a') as f:
                f.write(uv_cmd + '\n')
        with open(bake_cmds_txt, 'a') as f:
            f.write(bake_cmd + '\n')

    pool.close()
    pool.join()

    time.sleep(0.1)

    if len(already_baked_data_json_path) < 1:
        already_baked_data_json_path = os.path.join(
            log_folder, "bake_info.json")
    bake_info_struct = {}
    bake_info_struct["data"] = {}
    success_list = read_list(stat_txt)
    bake_folder_list = read_list(folder_txt)
    category_list = read_list(category_txt)
    instance_list = read_list(instance_txt)

    for index in range(len(success_list)):
        success_mesh_path = success_list[index]
        bake_folder = bake_folder_list[index]
        category = category_list[index]
        instance_name = instance_list[index]
        if category not in bake_info_struct["data"].keys():
            bake_info_struct["data"][category] = {}
        if instance_name not in bake_info_struct["data"][category].keys():
            bake_info_struct["data"][category][instance_name] = {}
        bake_info_struct["data"][category][instance_name]["Bake"] = success_mesh_path

    # bake_info_struct = {}
    # bake_info_struct["data"] = instance_name_folder_map
    write_json(already_baked_data_json_path, bake_info_struct)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All bake processes done. Local time is %s" % (local_time_str))
