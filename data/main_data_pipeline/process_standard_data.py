import argparse
import json
import multiprocessing
import os
import time


def convert_once(cmd_list, mesh_path, output_folder, mesh_category, stat_txt, time_txt, folder_txt, category_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for manifold converting command is %s....' %
          (str(start_time_str)))

    for cmd_struct in cmd_list:
        time.sleep(0.1)
        print("Start command %s -----> %s" %
              (str(cmd_struct["name"]), str(cmd_struct["cmd"])))

        convert_cmd = str(cmd_struct["cmd"])
        stat = os.system(convert_cmd)
        if str(cmd_struct["name"]) == 'resize':
            if stat != 0:
                time.sleep(0.1)
                stat = os.system(convert_cmd)

        time.sleep(0.1)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After manifold command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    if stat == 0:
        with open(stat_txt, 'a') as f:
            f.write('{}\n'.format(mesh_path))

        with open(folder_txt, 'a') as f:
            f.write('{}\n'.format(output_folder))

        with open(time_txt, 'a') as f:
            f.write('%s starts at %s, finish at %s....\n' %
                    (mesh_path, start_time_str, end_time_str))

        with open(category_txt, 'a') as f:
            f.write('{}\n'.format(mesh_category))


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


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


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


def generate_unrepeated_render_folder_name(path: str, connection_starts='', hash_value: bool = False):
    folder = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    file_basename = os.path.splitext(filename)[0]
    folder_name = os.path.split(folder)[1]
    if hash_value:
        return folder_name
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


def check_data_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    total_data_number = 0
    for category_name in category:
        total_data_number = total_data_number + \
                            len(json_struct["data"][category_name])
    return total_data_number


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        number_info[category_name] = len(json_struct["data"][category_name])
    return number_info


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Standard data processing start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--config_json_path', type=str, default="",
                        help='standard data processing config json file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.2-linux-x64/blender', help='path for blender executable file')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')
    parser.add_argument('--hash_value', action='store_true',
                        help='only use this when the mesh file name is a hash value. will override --connection_starts')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='total num of pods in the cluster')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be processed (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be processed (left close right open)')
    parser.add_argument('--texture_bake', action='store_true',
                        help='Use bake process in texture op')
    parser.add_argument('--no_manifold', action='store_true',
                        help='Close manifold process')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    data_json_path = args.data_json_path
    config_json_path = args.config_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder
    texture_bake = args.texture_bake
    no_manifold = args.no_manifold
    data_start = args.data_start
    data_end = args.data_end

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if not os.path.exists(config_json_path):
        config_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/standard_data.json")

    config_data_struct = read_json(config_json_path)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    mesh_categories = []
    mesh_instance_names = []
    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    if len(in_mesh_list_txt) > 1:
        mesh_paths = read_list(in_mesh_list_txt)
    if len(data_json_path) > 1:
        mesh_paths, _, _, _, mesh_categories, mesh_instance_names = read_mesh_list_from_data_json(
            data_json_path)

    standard_height = config_data_struct["standard_height"]

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

        output_folder = os.path.join(
            output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        log_folder = os.path.join(log_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        print("Number of mesh on this pod: %i, (start: %i, end: %i)" %
              (len(mesh_paths), idx_start, idx_end))

    resize_cmds_txt = os.path.join(log_folder, 'resize_cmds.txt')
    geometry_cmds_txt = os.path.join(log_folder, 'geometry_cmds.txt')
    uv_cmds_txt = os.path.join(log_folder, 'uv_cmds.txt')
    bake_cmds_txt = os.path.join(log_folder, 'bake_cmds.txt')
    texture_cmds_txt = os.path.join(log_folder, 'texture_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    category_txt = os.path.join(log_folder, 'category.txt')

    resize_cmds_file = open(resize_cmds_txt, 'w')
    geometry_cmds_file = open(geometry_cmds_txt, 'w')
    if texture_bake:
        uv_cmds_file = open(uv_cmds_txt, 'w')
        bake_cmds_file = open(bake_cmds_txt, 'w')
    else:
        texture_cmds_file = open(texture_cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')
    category_file = open(category_txt, 'w')

    resize_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "manifold/resize_object.py")
    manifold_geometry_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "manifold/manifold_geometry.py")
    uv_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "texture/uv.py")
    bake_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "texture/bake.py")
    manifold_texture_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "manifold/manifold_texture.py")

    for index in range(len(mesh_paths)):
        mesh_path = mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_filename = os.path.split(mesh_path)[1]
        if index < len(mesh_categories):
            mesh_category = mesh_categories[index]
        else:
            mesh_parent_folder = os.path.split(mesh_folder)[0]
            mesh_category = os.path.split(mesh_parent_folder)[1]
            mesh_categories.append(mesh_category)
        mesh_basename = os.path.splitext(mesh_filename)[0]

        if index >= len(mesh_instance_names):
            unique_mesh_folder_name = generate_unrepeated_render_folder_name(
                mesh_path, connection_starts=args.connection_starts, hash_value=args.hash_value)
            mesh_instance_names.append(unique_mesh_folder_name)
        else:
            unique_mesh_folder_name = mesh_instance_names[index]
        mesh_category_folder = os.path.join(output_folder, mesh_category)
        if not os.path.exists(mesh_category_folder):
            os.mkdir(mesh_category_folder)
        new_mesh_folder = os.path.join(
            mesh_category_folder, unique_mesh_folder_name)
        if not os.path.exists(new_mesh_folder):
            os.mkdir(new_mesh_folder)

        command_list = []
        transformation_txt = os.path.join(new_mesh_folder, "transformation.txt")
        resize_mesh_folder = os.path.join(new_mesh_folder, "resize")
        if not os.path.exists(resize_mesh_folder):
            os.mkdir(resize_mesh_folder)
        resize_mesh_path = os.path.join(resize_mesh_folder, "resize.obj")

        resize_cmd = "{} -b -P {} -- --mesh_path '{}' --resize_mesh_path '{}' --output_transformation_txt_path '{}' --copy_texture ".format(
            blender_root, resize_op_fullpath, mesh_path, resize_mesh_path, transformation_txt)
        resize_cmd = resize_cmd + " --standard_height {} ".format(standard_height)

        resize_command_struct = {}
        resize_command_struct["name"] = "resize"
        resize_command_struct["cmd"] = resize_cmd
        command_list.append(resize_command_struct)

        if not no_manifold:
            geometry_mesh_folder = os.path.join(new_mesh_folder, "manifold")
            if not os.path.exists(geometry_mesh_folder):
                os.mkdir(geometry_mesh_folder)
            geometry_mesh_path = os.path.join(
                geometry_mesh_folder, "manifold.obj")

            geometry_cmd = "{} -b -P {} -- --mesh_path '{}' --output_mesh_path '{}'  ".format(
                blender_root, manifold_geometry_fullpath, resize_mesh_path, geometry_mesh_path)
            geometry_cmd = geometry_cmd + \
                           " --remesh_voxel_size {}  ".format(
                               config_data_struct["remesh_voxel_size"])
            geometry_cmd = geometry_cmd + \
                           " --solidify_thickness {} ".format(
                               config_data_struct["solidify_thickness"])
            geometry_cmd = geometry_cmd + \
                           " --decimate_faces_num {} ".format(
                               config_data_struct["decimate_faces_num"])

            geometry_command_struct = {}
            geometry_command_struct["name"] = "manifold"
            geometry_command_struct["cmd"] = geometry_cmd
            command_list.append(geometry_command_struct)

        texture_mesh_folder = os.path.join(new_mesh_folder, "texture")
        if not os.path.exists(texture_mesh_folder):
            os.mkdir(texture_mesh_folder)

        texture_mesh_path = os.path.join(texture_mesh_folder, "texture.obj")

        texture_cmd = "{} -b -P {} -- ".format(
            blender_root, manifold_texture_fullpath)
        texture_cmd = texture_cmd + \
                      " --mesh_path '{}' ".format(resize_mesh_path)
        texture_cmd = texture_cmd + \
                      " --output_mesh_path '{}' ".format(texture_mesh_path)
        texture_cmd = texture_cmd + " --copy_texture --skip_occlusion "

        if "fix_transparency" in config_data_struct.keys() and "fix_mode" in config_data_struct.keys():
            if config_data_struct["fix_transparency"]:
                fix_transparecy_steps = config_data_struct["fix_mode"]
                if len(fix_transparecy_steps) > 0:
                    texture_cmd = texture_cmd + " --fix_transparency "
                    for fix_step in fix_transparecy_steps:
                        texture_cmd = texture_cmd + " --{} ".format(fix_step)

        texture_command_struct = {}
        texture_command_struct["name"] = "texture"
        texture_command_struct["cmd"] = texture_cmd
        command_list.append(texture_command_struct)

        if texture_bake:
            texture_uv_folder = os.path.join(texture_mesh_folder, "uv")
            if not os.path.exists(texture_uv_folder):
                os.mkdir(texture_uv_folder)
            texture_bake_folder = os.path.join(texture_mesh_folder, "bake")
            if not os.path.exists(texture_bake_folder):
                os.mkdir(texture_bake_folder)
            uv_mesh = os.path.join(texture_uv_folder, "mesh.obj")

            uv_cmd = "python {} --source_mesh_path '{}' --output_mesh_path '{}'".format(
                uv_op_fullpath, geometry_mesh_path, uv_mesh)

            bake_cmd = "{} -b -P {} -- --source_mesh_path '{}' --destination_mesh_path '{}'  ".format(
                blender_root, bake_op_fullpath, resize_mesh_path, uv_mesh)
            bake_cmd = bake_cmd + \
                       " --output_mesh_folder '{}' ".format(texture_bake_folder)
            bake_cmd = bake_cmd + " --output_mesh_filename \"baked_texture\" "
            bake_cmd = bake_cmd + \
                       " --cage_extrusion {} ".format(
                           config_data_struct["cage_extrusion"])
            bake_cmd = bake_cmd + \
                       " --max_ray_distance {} ".format(
                           config_data_struct["max_ray_distance"])
            bake_cmd = bake_cmd + \
                       " --texture_image_width {} ".format(
                           config_data_struct["bake_image_width"])
            bake_cmd = bake_cmd + \
                       " --texture_image_height {} ".format(
                           config_data_struct["bake_image_height"])

            if not config_data_struct["emission_baking"]:
                bake_cmd = bake_cmd + " --no_emission_baking "

            uv_command_struct = {}
            uv_command_struct["name"] = "uv"
            uv_command_struct["cmd"] = uv_cmd
            command_list.append(uv_command_struct)

            bake_command_struct = {}
            bake_command_struct["name"] = "baking"
            bake_command_struct["cmd"] = bake_cmd
            command_list.append(bake_command_struct)

        pool.apply_async(func=convert_once, args=(
            command_list, mesh_path, new_mesh_folder, mesh_category, stat_txt, time_txt, folder_txt, category_txt))

        with open(resize_cmds_txt, 'a') as f:
            f.write(resize_command_struct["cmd"] + '\n')
        if not no_manifold:
            with open(geometry_cmds_txt, 'a') as f:
                f.write(geometry_command_struct["cmd"] + '\n')
        with open(texture_cmds_txt, 'a') as f:
            f.write(texture_command_struct["cmd"] + '\n')
        if texture_bake:
            with open(uv_cmds_txt, 'a') as f:
                f.write(uv_command_struct["cmd"] + '\n')
            with open(bake_cmds_txt, 'a') as f:
                f.write(bake_command_struct["cmd"] + '\n')

    pool.close()
    pool.join()

    time.sleep(1)

    success_mesh_list = read_list(stat_txt)
    result_folder_list = read_list(folder_txt)
    category_list = read_list(category_txt)
    output_json_path = os.path.join(log_folder, "standard.json")

    print("Read from %s mesh list length %i and folder list length %i ....." %
          (log_folder, len(success_mesh_list), len(result_folder_list)))
    data_struct = {}
    data_struct["data"] = {}

    for index in range(len(result_folder_list)):
        mesh_folder = result_folder_list[index]
        mesh_parent_folder = os.path.split(mesh_folder)[0]
        mesh_instance_name = os.path.split(mesh_folder)[1]
        mesh_category = category_list[index]
        if mesh_category not in data_struct["data"].keys():
            data_struct["data"][mesh_category] = {}

        original_mesh_name = os.path.join(
            mesh_parent_folder, "resize/resize.obj")
        manifold_mesh_name = os.path.join(
            mesh_parent_folder, "manifold/manifold.obj")
        texture_mesh_name = os.path.join(
            mesh_parent_folder, "texture/texture.obj")
        unify_mesh_name = os.path.join(
            mesh_parent_folder, "texture/bake/baked_texture.obj")
        transformation_txt = os.path.join(
            mesh_parent_folder, "transformation.txt")

        instance_struct = {}
        if not os.path.exists(original_mesh_name):
            continue
        if not os.path.exists(texture_mesh_name):
            continue
        if not os.path.exists(transformation_txt):
            continue

        instance_struct["Mesh"] = texture_mesh_name
        instance_struct["Obj_Mesh"] = texture_mesh_name
        instance_struct["Original"] = original_mesh_name
        instance_struct["Original_Transformation"] = transformation_txt

        if not no_manifold:
            if not os.path.exists(manifold_mesh_name):
                continue
            instance_struct["Manifold"] = manifold_mesh_name
        if texture_bake:
            if not os.path.exists(unify_mesh_name):
                continue
            instance_struct["Unify"] = unify_mesh_name

        data_struct["data"][mesh_category][mesh_instance_name] = instance_struct

    write_json(output_json_path, data_struct)
    print(check_data_number(data_struct))
    print(check_individual_number(data_struct))

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Standard data processing end. Start time is %s; end time is %s" %
          (local_time_str, end_time_str))
