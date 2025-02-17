import argparse
import json
import multiprocessing
import os
import time


def convert_once(convert_cmd, mesh_path, output_folder, stat_txt, time_txt, folder_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for convert file (at %s) format cmd is %s....' %
          (mesh_path, str(start_time_str)))

    print(convert_cmd)
    stat = os.system(convert_cmd)
    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After convert format command status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    if stat == 0:
        with open(stat_txt, 'a') as f:
            f.write('{}\n'.format(mesh_path))

        with open(folder_txt, 'a') as f:
            f.write('{}\n'.format(output_folder))

        with open(time_txt, 'a') as f:
            f.write('%s starts at %s, finish at %s....\n' % (mesh_path, start_time_str, end_time_str))


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


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


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
                last_model_index = len(elements) - elements[::-1].index('3dAsset_artcenter')
                return "_".join(elements[last_model_index:])
            elif 'share_2909871' in elements:
                last_model_index = len(elements) - elements[::-1].index('share_2909871')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_1' in elements:
                last_model_index = len(elements) - elements[::-1].index('aigc_bucket_1')
                return "_".join(elements[last_model_index:])
            elif 'aigc_bucket_2' in elements:
                last_model_index = len(elements) - elements[::-1].index('aigc_bucket_2')
                return "_".join(elements[last_model_index:])
            elif 'model_denoise' in elements:
                last_model_index = len(elements) - elements[::-1].index('model_denoise')
                return "_".join(elements[last_model_index:])
            elif 'game_character_obj' in elements:
                last_model_index = len(elements) - elements[::-1].index('game_character_obj')
                return "_".join(elements[last_model_index:])
            else:
                return "__".join(elements)


def decorate_folder_name(folder_name: str):
    folder_elements = folder_name.split("_")
    to_remove_element = "_" + folder_elements[-1]
    new_folder_name = folder_name.replace(to_remove_element, "")
    no_splace_folder_name = new_folder_name.replace(" ", "_")
    return no_splace_folder_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert file format.')
    parser.add_argument('--in_mesh_list_txt', type=str, default="",
                        help='input mesh (format: fbx/glb) list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='output mesh folder')
    parser.add_argument('--config_json_path', type=str, default="",
                        help='format conversion processing config json file path')
    parser.add_argument('--success_list_txt', type=str, default="",
                        help='list of meshes successfully converted')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender', help='path for blender binary exe')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--to_glb', action='store_true',
                        help='force convert all mesh to glb files')
    parser.add_argument('--clean_dds', action='store_true',
                        help='clean all dds texture image files to png format')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better_fbx to import fbx file')
    parser.add_argument('--force_old_obj_format', action='store_true',
                        help='force to use old obj importer im blender for file conversion')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be converted (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be converted (left close right open)')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='total number of pods of cluster')
    args = parser.parse_args()

    in_mesh_list_txt = args.in_mesh_list_txt
    data_json_path = args.data_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder
    output_folder = args.output_folder
    config_json_path = args.config_json_path
    data_start = args.data_start
    data_end = args.data_end
    to_glb = args.to_glb
    clean_dds = args.clean_dds

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if len(config_json_path) > 0:
        config_struct = read_json(config_json_path)
    mesh_categories = []
    mesh_instance_names = []
    mesh_info_struct = {}
    mesh_info_struct["data"] = {}
    if len(in_mesh_list_txt) > 1:
        mesh_paths = read_list(in_mesh_list_txt)
    if len(data_json_path) > 1:
        mesh_paths, _, _, _, mesh_categories, mesh_instance_names = read_mesh_list_from_data_json(data_json_path)

    successful_list = []
    if len(args.success_list_txt) > 1:
        successful_list = read_list(args.success_list_txt)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(mesh_paths):
                data_end = len(mesh_paths)
            mesh_path_list = mesh_paths[data_start:data_end]
            mesh_categories = mesh_categories[data_start:data_end]
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

        print("Rendering started using cluster with %i pods" % (args.pod_num))
        print("Number of mesh on this pod: %i, (start: %i, end: %i)" %
              (len(mesh_paths), idx_start, idx_end))

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')
    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    for index in range(len(mesh_paths)):
        mesh_path = mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_parent_folder = os.path.split(mesh_folder)[0]
        mesh_category_folder = os.path.split(mesh_parent_folder)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]
        mesh_extension = os.path.splitext(mesh_filename)[1]
        mesh_extension = mesh_extension.lower()

        if index > len(mesh_categories):
            mesh_category = os.path.split(mesh_category_folder)[1]
        else:
            mesh_category = mesh_categories[index]
        mesh_category_folder = os.path.join(output_folder, mesh_category)
        if not os.path.exists(mesh_category_folder):
            os.mkdir(mesh_category_folder)

        if index > len(mesh_instance_names):
            unique_mesh_folder_name = generate_unrepeated_render_folder_name(mesh_path,
                                                                             connection_starts=args.connection_starts)
        else:
            unique_mesh_folder_name = mesh_instance_names[index]
        new_mesh_folder = os.path.join(mesh_category_folder, unique_mesh_folder_name)
        if not os.path.exists(new_mesh_folder):
            os.mkdir(new_mesh_folder)

        if mesh_path in successful_list:
            continue

        if clean_dds:
            converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texture/dds.py")
            new_obj_name = unique_mesh_folder_name + ".obj"
            new_obj_fullpath = os.path.join(new_mesh_folder, new_obj_name)

            print("[%i] Convert obj file from %s to %s" % (index, mesh_path, new_mesh_folder))

            convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_mesh_path \'{}\' ".format(
                blender_root, converter_fullpath, mesh_path, new_obj_fullpath)
            if args.copy_texture:
                convert_cmd = convert_cmd + " --copy_texture  "

            pool.apply_async(func=convert_once,
                             args=(convert_cmd, mesh_path, new_obj_fullpath, stat_txt, time_txt, folder_txt))

            with open(cmds_txt, 'a') as f:
                f.write(convert_cmd + '\n')
        else:
            if to_glb:
                converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "conversion/mesh_glb_converter.py")

                new_glb_name = unique_mesh_folder_name + ".glb"
                new_glb_fullpath = os.path.join(new_mesh_folder, new_glb_name)
                convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_fullpath \'{}\' ".format(
                    blender_root, converter_fullpath, mesh_path, new_glb_fullpath)
                print(convert_cmd)
                pool.apply_async(func=convert_once, args=(
                    convert_cmd, mesh_path, new_glb_fullpath, stat_txt, time_txt, folder_txt))

                with open(cmds_txt, 'a') as f:
                    f.write(convert_cmd + '\n')
            else:
                if mesh_extension == ".fbx":
                    converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "conversion/fbx_obj_converter.py")

                    new_obj_name = unique_mesh_folder_name + ".obj"
                    new_obj_fullpath = os.path.join(new_mesh_folder, new_obj_name)

                    print("[%i] Convert fbx file from %s to %s" % (index, mesh_path, new_mesh_folder))

                    convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_fullpath \'{}\' ".format(
                        blender_root, converter_fullpath, mesh_path, new_obj_fullpath)
                    if args.copy_texture:
                        convert_cmd = convert_cmd + " --copy_texture  "
                    if args.force_better_fbx:
                        convert_cmd = convert_cmd + " --force_better_fbx  "

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, new_obj_fullpath, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')
                elif mesh_extension == ".obj":
                    converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "conversion/anything_obj_converter.py")

                    new_obj_name = unique_mesh_folder_name + ".obj"
                    new_obj_fullpath = os.path.join(new_mesh_folder, new_obj_name)

                    print("[%i] Convert obj file from %s to %s" % (index, mesh_path, new_mesh_folder))

                    convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_mesh_path \'{}\' ".format(
                        blender_root, converter_fullpath, mesh_path, new_obj_fullpath)
                    convert_cmd = convert_cmd + " --no_additional_texture  "
                    if args.copy_texture:
                        convert_cmd = convert_cmd + " --copy_texture  "
                    if args.force_old_obj_format:
                        convert_cmd = convert_cmd + " --force_old_obj_importer  "

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, new_obj_fullpath, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')
                elif mesh_extension == ".glb" or mesh_extension == ".gltf":
                    converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "conversion/glb_to_obj_final.py")

                    glb_folder = os.path.join(mesh_category_folder, unique_mesh_folder_name)
                    convert_cmd = "{} -b -P {} -- --mesh_path '{}' --output_mesh_folder '{}' ".format(
                        blender_root, converter_fullpath, mesh_path, glb_folder)

                    mesh_filename = os.path.split(mesh_path)[1]
                    mesh_basename = os.path.splitext(mesh_filename)[0]
                    new_mesh_fullname = os.path.join(glb_folder, mesh_basename + ".obj")

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, new_mesh_fullname, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')
                elif mesh_extension == ".dae" or mesh_extension == ".stl":
                    if mesh_extension == ".dae":
                        converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "conversion/dae_to_obj.py")
                    elif mesh_extension == ".stl":
                        converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "conversion/stl_obj_converter.py")

                    final_obj_fullpath = os.path.join(new_mesh_folder, unique_mesh_folder_name + ".obj")

                    convert_cmd = "{} -b -P {} -- --mesh_path \'{}\' --output_fullpath \'{}\' ".format(
                        blender_root, converter_fullpath, mesh_path, final_obj_fullpath)

                    if args.copy_texture:
                        convert_cmd = convert_cmd + " --copy_texture  "

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, final_obj_fullpath, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')
                elif mesh_extension == ".pmx" or mesh_extension == ".vrm":
                    pmx_converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "pmx/pmx_multi_converter.py")
                    vrm_converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "pmx/vrm_multi_converter.py")

                    print("Convert pmx file from %s to %s" % (mesh_path, new_mesh_folder))

                    if mesh_extension == ".pmx":
                        convert_cmd = "{} -b -P {} -- --pmx_path \'{}\' --output_folder \'{}\'  > /dev/null".format(
                            blender_root, pmx_converter_fullpath, mesh_path, new_mesh_folder)
                    else:
                        convert_cmd = "{} -b -P {} -- --vrm_path \'{}\' --output_folder \'{}\'  > /dev/null".format(
                            blender_root, vrm_converter_fullpath, mesh_path, new_mesh_folder)

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, new_mesh_folder, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')
                elif mesh_extension == ".off":
                    converter_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "conversion/off_obj_converter.py")
                    final_obj_fullpath = os.path.join(new_mesh_folder, unique_mesh_folder_name + ".obj")

                    convert_cmd = "python {} --input_mesh_path \'{}\' --output_mesh_path \'{}\' ".format(
                        converter_fullpath,
                        mesh_path,
                        final_obj_fullpath)

                    pool.apply_async(func=convert_once,
                                     args=(convert_cmd, mesh_path, final_obj_fullpath, stat_txt, time_txt, folder_txt))

                    with open(cmds_txt, 'a') as f:
                        f.write(convert_cmd + '\n')

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Format converting done. Local time is %s" % (local_time_str))
