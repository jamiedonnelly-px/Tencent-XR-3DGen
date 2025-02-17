import argparse
import json
import multiprocessing
import os
import time


def convert_once(bake_cmd, mesh_path, output_folder, stat_txt, time_txt, folder_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for manifold converting command is %s....' %
          (str(start_time_str)))

    time.sleep(0.1)
    print("Start command %s: %s" % ("combine", bake_cmd))
    exec_result = os.system(bake_cmd)
    time.sleep(0.1)

    stat = exec_result

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After bake command status is %s; time for this status is %s....' %
          (str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (output_folder, start_time_str, end_time_str))


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
            dest_instance_name = None
            for current_dest_name in dest_instance_path_struct.keys():
                if current_dest_name in instance_name:
                    dest_instance_name = current_dest_name
                    break
            if dest_instance_name is None:
                continue
            dest_instance_paths = dest_instance_path_struct[dest_instance_name]

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
            instance_name_list.append(dest_instance_name)
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
    parser.add_argument('--nake_model_path', type=str, default="",
                        help='nake model file path')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.0.1-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    nake_model_path = args.nake_model_path
    blender_root = args.blender_root
    log_folder = args.log_folder

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    source_mesh_paths = read_list(in_mesh_list_txt)

    print("Number of mesh on this pod: %i" % (len(source_mesh_paths)))

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    combine_cmds_txt = os.path.join(log_folder, 'cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')

    combine_cmds_file = open(combine_cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    combiner_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../geometry/combine_models.py")

    instance_name_folder_map = {}

    for index in range(len(source_mesh_paths)):
        mesh_path = source_mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find source mesh file ', mesh_path)
            continue

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        mesh_parent_folder = os.path.split(mesh_folder)[0]
        mesh_parent_folder_name = os.path.split(mesh_parent_folder)[1]
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]

        input_mesh_list = [mesh_path, nake_model_path]

        joint_folder = os.path.join(output_folder, mesh_folder_name)

        combine_cmd = "{} -b -P {} -- --joint_mesh_folder \"{}\" --joint_mesh_name \"{}\" --mesh_path_list ".format(
            blender_root, combiner_fullpath, joint_folder, mesh_basename)
        for to_combine in input_mesh_list:
            combine_cmd = combine_cmd + " \"{}\" ".format(to_combine)

        pool.apply_async(func=convert_once, args=(
            combine_cmd, mesh_path, joint_folder, stat_txt, time_txt, folder_txt))

        with open(combine_cmds_txt, 'a') as f:
            f.write(combine_cmd + '\n')

    pool.close()
    pool.join()

    time.sleep(0.1)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All bake processes done. Local time is %s" % (local_time_str))
