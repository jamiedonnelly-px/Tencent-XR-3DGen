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

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        f.write('%s starts at %s, finish at %s....\n' %
                (mesh_path, start_time_str, end_time_str))


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


def decorate_folder_name(folder_name: str):
    folder_elements = folder_name.split("_")
    to_remove_element = "_" + folder_elements[-1]
    new_folder_name = folder_name.replace(to_remove_element, "")
    no_splace_folder_name = new_folder_name.replace(" ", "_")
    return no_splace_folder_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert file format.')
    parser.add_argument('--merged_json_path', type=str, default="",
                        help='json file containing uv merged meshes')
    parser.add_argument('--output_folder', type=str, default="",
                        help='output mesh folder')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json file path containing raw meshes')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender', help='path for blender binary exe')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    args = parser.parse_args()

    merged_json_path = args.merged_json_path
    data_json_path = args.data_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    mesh_categories = []
    mesh_instance_names = []
    mesh_info_struct = {}
    mesh_info_struct["data"] = {}

    mesh_paths, _, _, _, mesh_categories, mesh_instance_names = read_mesh_list_from_data_json(
        data_json_path)
    uv_merged_data_struct = read_json(merged_json_path)

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

    converter_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "texture/transfer_uv.py")

    for index in range(len(mesh_paths)):
        mesh_path = mesh_paths[index]
        if not os.path.exists(mesh_path):
            print('Cannot find input mesh file ', mesh_path)
            continue

        mesh_category = mesh_categories[index]
        mesh_instance_name = mesh_instance_names[index]

        if "MCWY_2_" not in mesh_category:
            continue
        if mesh_category not in uv_merged_data_struct["data"].keys():
            continue
        if mesh_instance_name not in uv_merged_data_struct["data"][mesh_category].keys():
            continue

        uv_merged_mesh_path = uv_merged_data_struct["data"][mesh_category][mesh_instance_name]["Obj_Mesh"]

        mesh_category_folder = os.path.join(output_folder, mesh_category)
        if not os.path.exists(mesh_category_folder):
            os.mkdir(mesh_category_folder)
        mesh_instance_folder = os.path.join(mesh_category_folder, mesh_instance_name)
        if not os.path.exists(mesh_instance_folder):
            os.mkdir(mesh_instance_folder)

        mesh_folder = os.path.split(mesh_path)[0]
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]
        mesh_extension = os.path.splitext(mesh_filename)[1]

        copied_mesh_name = mesh_instance_name + ".obj"
        copied_mesh_path = os.path.join(mesh_instance_folder, copied_mesh_name)

        convert_cmd = "{} -b -P {} -- ".format(
            blender_root, converter_fullpath)
        convert_cmd = convert_cmd + " --mesh_path \'{}\' ".format(mesh_path)
        convert_cmd = convert_cmd + \
                      " --destination_mesh_path \'{}\' ".format(uv_merged_mesh_path)
        convert_cmd = convert_cmd + \
                      " --output_mesh_path \'{}\' ".format(copied_mesh_path)
        print(convert_cmd)
        # pool.apply_async(func=convert_once, args=(
        #     convert_cmd, uv_merged_mesh_path, copied_mesh_path, stat_txt, time_txt, folder_txt))

        with open(cmds_txt, 'a') as f:
            f.write(convert_cmd + '\n')

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Format converting done. Local time is %s" % (local_time_str))
