import argparse
import json
import multiprocessing
import os
import sys
import time

from easydict import EasyDict as edict


def parse_config_json(json_path: str):
    print("Parse config json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        config = json.load(f)

    config = edict(config)
    return config


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


def render_once(mesh_cmd_struct,
                mesh_path, proc_data_folder,
                stat_txt, time_txt, proc_txt,
                apply_preprocess_mesh: bool = False):
    stat = 1
    preprocess_stat = 1
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for render and sample cmd is %s....' % (str(start_time_str)))
    print(mesh_cmd_struct)

    if apply_preprocess_mesh:
        for mesh_cmd_name in mesh_cmd_struct.keys():
            print("Name is %s; cmd is %s......" %
                  (mesh_cmd_name, mesh_cmd_struct[mesh_cmd_name]))
            preprocess_stat = os.system(mesh_cmd_struct[mesh_cmd_name])
            time.sleep(0.1)
            if preprocess_stat != 0:
                if mesh_cmd_name != "texture":
                    return

    t_middle = time.time()
    middle_time = time.localtime(t_middle)
    middle_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', middle_time)

    print('After point sampling command status is %s; time for this status is %s....' % (
        str(preprocess_stat), str(middle_time_str)))

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    if preprocess_stat == 0:
        stat = 0

    print('After rendering command status is %s; time for this status is %s....' % (str(stat), str(end_time_str)))

    if stat == 0:
        with open(stat_txt, 'a', encoding='UTF-8') as f:
            f.write('{}\n'.format(mesh_path))

        with open(time_txt, 'a', encoding='UTF-8') as f:
            f.write('%s starts at %s, finish at %s....\n' % (mesh_path, start_time_str, end_time_str))

        with open(proc_txt, 'a', encoding='UTF-8') as f:
            f.write('{}\n'.format(proc_data_folder))


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
            elif "Manifold_Mesh" in instance_paths.keys():
                if instance_paths["Manifold_Mesh"] is not None:
                    manifold_path = instance_paths["Manifold_Mesh"]
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


def read_mesh_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r', encoding='UTF-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = os.path.abspath(line.strip())
            if len(mesh_path) > 2:
                mesh_path_list.append(mesh_path)
    print('load mesh_path_list ', len(mesh_path_list))
    return mesh_path_list


def check_str_in_list(check_str: str, str_list: list):
    for str_element in str_list:
        if check_str in str_element:
            return True
    return False


def generate_unrepeated_render_folder_name(path: str, connection_starts='', hash_value: bool = False):
    folder = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    file_basename = os.path.splitext(filename)[0]
    if hash_value:
        return file_basename
    else:
        if sys.platform.startswith('win'):
            original_elements = folder.split("\\")
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
            if elements[-1].lower() == "mesh" and elements[-2] == "OBJ":
                return "_".join(elements[last_model_index:-2])
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
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Rendering start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Renders multi-gpu with pool.')
    parser.add_argument('--in_mesh_list_txt', type=str, default="",
                        help='input mesh list txt file')
    parser.add_argument('--data_json_path', type=str, default="",
                        help='data json path (database json), choose between this or mesh list')
    parser.add_argument('--data_json_tag_str', type=str, default="",
                        help='only render tag in data json')
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
    parser.add_argument('--donot_apply_any_process', action='store_true',
                        help='only generate cmd lists; do not start render or mesh process')
    parser.add_argument('--force_outside_trans', action='store_true',
                        help='use outside transformation txt in all processes')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')
    parser.add_argument('--hash_value', action='store_true',
                        help='only use this when the mesh file name is a hash value. will override --connection_starts')
    parser.add_argument('--preprocess_scale_mesh', action='store_true',
                        help='scale mesh in preprocessing process')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    parser.add_argument('--pod_id', type=int, default=-1,
                        help='index of pods used in cluster')
    parser.add_argument('--pod_num', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')
    args = parser.parse_args()

    data_start = args.data_start
    data_end = args.data_end
    mesh_list_path = args.in_mesh_list_txt
    data_json_path = args.data_json_path
    data_json_tag_str = args.data_json_tag_str
    log_folder = args.log_folder
    blender_root = args.blender_root
    output_folder = args.output_folder
    proc_data_output_folder = args.proc_data_output_folder
    success_list_path = args.success_list

    apply_preprocess_mesh = args.apply_preprocess_mesh
    force_outside_trans = args.force_outside_trans
    hash_value = args.hash_value
    preprocess_scale_mesh = args.preprocess_scale_mesh

    success_mesh_list = []
    if os.path.exists(success_list_path) and len(success_list_path) > 1:
        success_mesh_list = read_list(success_list_path)

    config_json_path = args.config_json_path
    if len(config_json_path) < 1 or not os.path.exists(config_json_path):
        config_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/default.json")
    config_struct = parse_config_json(config_json_path)
    print("Render config is: \n\n", config_struct)

    mesh_path_list = []
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
        mesh_path_list, _, input_proc_data_folder_list, _, category_list, instance_name_list = read_mesh_list_from_data_json(
            data_json_path)
    elif len(mesh_list_path) > 1:
        mesh_path_list = read_mesh_list(mesh_list_path)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(mesh_path_list):
                data_end = len(mesh_path_list)
            mesh_path_list = mesh_path_list[data_start:data_end]
            input_proc_data_folder_list = input_proc_data_folder_list[data_start:data_end]
            category_list = category_list[data_start:data_end]
            instance_name_list = instance_name_list[data_start:data_end]

    if args.pod_num >= 0 and args.pod_id >= 0:
        mesh_path_len = len(mesh_path_list)
        per_pod_len = mesh_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = mesh_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        mesh_path_list = mesh_path_list[idx_start:idx_end]
        input_proc_data_folder_list = input_proc_data_folder_list[idx_start:idx_end]
        category_list = category_list[idx_start:idx_end]
        instance_name_list = instance_name_list[idx_start:idx_end]

        output_folder = os.path.join(output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if len(proc_data_output_folder) > 1:
            proc_data_output_folder = os.path.join(proc_data_output_folder, "pod_{}".format(args.pod_id))
            if not os.path.exists(proc_data_output_folder):
                os.mkdir(proc_data_output_folder)

        log_folder = os.path.join(log_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        print("Rendering started using cluster with %i pods" % (args.pod_num))
        print("Number of mesh on this pod: %i, (start: %i, end: %i)" % (len(mesh_path_list), idx_start, idx_end))

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

            if len(proc_data_output_folder) > 1:
                proc_category_folder = os.path.join(proc_data_output_folder, mesh_category)
                if apply_preprocess_mesh:
                    if not os.path.exists(proc_category_folder):
                        os.mkdir(proc_category_folder)
                mesh_proc_folder = os.path.join(proc_category_folder, mesh_folder_name)
                if apply_preprocess_mesh:
                    if not os.path.exists(mesh_proc_folder):
                        print("Creating proc data folder on %s" % (mesh_proc_folder))
                        os.mkdir(mesh_proc_folder)

        else:
            mesh_folder_name = generate_unrepeated_render_folder_name(mesh_path,
                                                                      connection_starts=args.connection_starts,
                                                                      hash_value=hash_value)

            if len(proc_data_output_folder) > 1:
                mesh_proc_folder = os.path.join(proc_data_output_folder, mesh_folder_name)
                if apply_preprocess_mesh:
                    if not os.path.exists(mesh_proc_folder):
                        print("Creating proc data folder on %s" % (mesh_proc_folder))
                        os.mkdir(mesh_proc_folder)

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

        trans_txt = os.path.join(final_proc_folder, "transformation.txt")
        z_txt = os.path.join(final_proc_folder, "z.txt")

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
    print('Find {} cpus, use {} threads in sampling......'.format(cpu_cnt, args.pool_cnt))

    sdf_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdf_sample.py")

    cnt = 0
    for index in range(len(mesh_path_list)):
        mesh_path = mesh_path_list[index]
        manifold_mesh_path = mesh_path
        proc_data_folder = proc_data_folder_list[index]

        final_trans_txt = trans_txt_list[index]
        final_z_txt = z_txt_list[index]

        config_struct["transformation_txt"] = final_trans_txt
        config_struct["z_txt"] = final_z_txt
        mesh_cmd_struct = {}

        sdf_sample_cmd = "python \"{}\" --mesh_path \"{}\" ".format(sdf_script_fullpath, manifold_mesh_path)
        sdf_sample_cmd = sdf_sample_cmd + " --output_folder \"{}\" ".format(proc_data_folder)
        sdf_sample_cmd = sdf_sample_cmd + " --transform_path \"{}\" ".format(final_trans_txt)
        sdf_sample_cmd = sdf_sample_cmd + " --z_transform_path \"{}\" ".format(final_z_txt)
        sdf_sample_cmd = sdf_sample_cmd + " --standard_height {} ".format(config_struct["standard_height"])
        sdf_sample_cmd = sdf_sample_cmd + " --space_sample_number {} ".format(
            config_struct["geometry_space_sample_number"])
        sdf_sample_cmd = sdf_sample_cmd + " --near_surface_sample_number {} ".format(
            config_struct["geometry_near_surface_sample_number"])
        sdf_sample_cmd = sdf_sample_cmd + " --surface_sample_number {} ".format(
            config_struct["geometry_surface_sample_number"])
        sdf_sample_cmd = sdf_sample_cmd + " --sample_format \'{}\' ".format(config_struct["sample_format"])
        sdf_sample_cmd = sdf_sample_cmd + " --chunk_size {} ".format(config_struct["h5_chunk_size"])
        if config_struct["shuffle_sample"]:
            sdf_sample_cmd = sdf_sample_cmd + " --shuffle "

        mesh_cmd_struct["sdf"] = sdf_sample_cmd

        pool.apply_async(func=render_once, args=(mesh_cmd_struct,
                                                 mesh_path, proc_data_folder,
                                                 stat_txt, time_txt, proc_txt,
                                                 apply_preprocess_mesh))
        cnt += 1
        with open(proc_cmds_txt, 'a', encoding='UTF-8') as f:
            for mesh_stage in mesh_cmd_struct.keys():
                f.write(mesh_cmd_struct[mesh_stage] + '\n')

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All rendering tasks DONE. These tasks end at time %s" %
          (local_time_str))
