import argparse
import json
import multiprocessing
import os
import random
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


def correct_disk_name(path_list: list):
    new_path_list = []
    for path_name in path_list:
        if 'apdcephfs_data_cq3' in path_name:
            new_path_name = path_name.replace(
                'apdcephfs_data_cq3', 'apdcephfs_cq3')
            new_path_list.append(new_path_name)
        else:
            new_path_list.append(path_name)
    return new_path_list


def render_once(render_cmd_struct,
                mesh_path, output_folder,
                stat_txt, time_txt, folder_txt,
                apply_render_cmd: bool = False,
                render_stages: list = ["common", "emission", "no_smooth"]):
    stat = 1
    render_stat = 1
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    if apply_render_cmd:
        for render_cmd_name in render_stages:
            if render_cmd_name in render_cmd_struct.keys():
                print("Name is %s; cmd is %s......" %
                      (render_cmd_name, render_cmd_struct[render_cmd_name]))
                render_stat = os.system(render_cmd_struct[render_cmd_name])
                time.sleep(0.1)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print(
        'After rendering command status is %s; time for this status is %s....' % (str(render_stat), str(end_time_str)))

    if render_stat != 0:
        return

    with open(stat_txt, 'a', encoding='UTF-8') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a', encoding='UTF-8') as f:
        f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a', encoding='UTF-8') as f:
        f.write('%s starts at %s, finish at %s....\n' % (mesh_path, start_time_str, end_time_str))


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
            last_model_index = len(elements) - elements[::-1].index(connection_starts)
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
    parser.add_argument('--pose_json_path', type=str, default="",
                        help='cam_parameters.json including poses json path')
    parser.add_argument('--config_json_path', type=str, default="",
                        help='render config json path')
    parser.add_argument('--output_folder', type=str,
                        help='output folder for render results')
    parser.add_argument('--success_list', type=str, default="",
                        help='mesh already rendered in mesh list')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str,
                        help='log folder to store information')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-4.2.0-linux-x64/blender', help='path for blender binary exe')
    parser.add_argument('--silent', action='store_true',
                        help='no log output of rendering scripts')
    parser.add_argument('--thumbnail', action='store_true',
                        help='start rendering thumbnails')
    parser.add_argument('--vae_latent', action='store_true',
                        help='start rendering only 21 images for VAE training')
    parser.add_argument('--apply_render', action='store_true',
                        help='start the rendering process')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better fbx as import plugin')
    parser.add_argument('--parse_exr', action='store_true',
                        help='set this will parse exr to color/depth/...')
    parser.add_argument('--only_render_png', action='store_true',
                        help='set this will only render color files; effective when not set parse_exr')
    parser.add_argument('--connection_starts', type=str, default="",
                        help='we connect words on path after this to generate output folder')
    parser.add_argument('--hash_value', action='store_true',
                        help='only use this when the mesh file name is a hash value. will override --connection_starts')
    parser.add_argument('--render_stage_string', type=str, default="common+emission+no_smooth",
                        help='stages of render to be used')
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
    pose_json_path = args.pose_json_path
    log_folder = args.log_folder
    blender_root = args.blender_root
    output_folder = args.output_folder
    success_list_path = args.success_list

    apply_render = args.apply_render
    hash_value = args.hash_value

    render_stages = args.render_stage_string.split("+")

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
    mesh_category_map = {}
    category_list = []
    instance_name_list = []

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if len(data_json_path) > 1:
        mesh_path_list, _, _, _, category_list, instance_name_list = read_mesh_list_from_data_json(
            data_json_path)
    elif len(mesh_list_path) > 1:
        mesh_path_list = read_mesh_list(mesh_list_path)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(mesh_path_list):
                data_end = len(mesh_path_list)
            mesh_path_list = mesh_path_list[data_start:data_end]
            if len(category_list) > 0:
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
        if len(category_list) > 0:
            category_list = category_list[idx_start:idx_end]
            instance_name_list = instance_name_list[idx_start:idx_end]

        output_folder = os.path.join(output_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        log_folder = os.path.join(log_folder, "pod_{}".format(args.pod_id))
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        print("Rendering started using cluster with %i pods" % (args.pod_num))
        print("Number of mesh on this pod: %i, (start: %i, end: %i)" % (len(mesh_path_list), idx_start, idx_end))

    mesh_number = len(mesh_path_list)
    real_mesh_path_list = []
    for mesh_index in range(mesh_number):
        current_output_folder = output_folder
        mesh_path = mesh_path_list[mesh_index]

        if not os.path.exists(mesh_path):
            continue
        if check_str_in_list(mesh_path, success_mesh_list):
            continue

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
                    print("Creating render data folder on %s" % (mesh_render_folder))
                    os.mkdir(mesh_render_folder)
        else:
            mesh_folder_name = generate_unrepeated_render_folder_name(
                mesh_path, connection_starts=args.connection_starts, hash_value=hash_value)
            mesh_render_folder = os.path.join(current_output_folder, mesh_folder_name)
            if apply_render:
                if not os.path.exists(mesh_render_folder):
                    print("Creating render data folder on %s" % (mesh_render_folder))
                    os.mkdir(mesh_render_folder)

        final_output_folder = os.path.join(mesh_render_folder,
                                           "render_{}_Valour".format(config_struct["render_height"]))
        output_folder_list.append(final_output_folder)
        real_mesh_path_list.append(mesh_path)

        trans_txt = os.path.join(final_output_folder, "transformation.txt")
        z_txt = os.path.join(final_output_folder, "z.txt")
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
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in rendering......'.format(cpu_cnt, args.pool_cnt))

    render_script_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_one_layer_mesh.py")

    cnt = 0
    for index in range(len(real_mesh_path_list)):
        mesh_path = real_mesh_path_list[index]

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

        config_struct["transformation_txt"] = final_trans_txt
        config_struct["z_txt"] = final_z_txt
        config_struct["pose_json"] = pose_json_path
        config_struct["mesh_json_path"] = data_json_path
        config_struct["mesh_list_path"] = mesh_list_path

        render_cmd = "{} -b -P {} -- --mesh_path \"{}\" ".format(blender_root, render_script_fullpath, mesh_path)
        render_stage_map = {}
        render_stage_map["cmd"] = {}

        if apply_render:
            camera_pose_json_data = read_json(pose_json_path)
            camera_pose_data = camera_pose_json_data["poses"]

            render_cmd = render_cmd + " --transform_path \"{}\" --pose_json_path \"{}\" ".format(final_z_txt,
                                                                                                 pose_json_path)

            if args.force_better_fbx:
                render_cmd = render_cmd + " --use_better_fbx "

            if "rotate_object" in config_struct.keys():
                if config_struct["rotate_object"]:
                    render_cmd = render_cmd + " --rotate_object "

            if "color_background" in config_struct.keys():
                if config_struct["color_background"]:
                    render_cmd = render_cmd + " --color_background "

            if "unit_trans" in config_struct.keys():
                if config_struct["unit_trans"]:
                    render_cmd = render_cmd + " --use_unit_transform "

            if config_struct["daz"]:
                render_cmd = render_cmd + " --render_daz "

            if config_struct["solidify"]:
                render_cmd = render_cmd + " --solidify "

            if "color_attribute" in config_struct.keys():
                if config_struct["color_attribute"]:
                    render_cmd = render_cmd + " --use_color_attribute "

            if "aux_format" in config_struct.keys():
                render_cmd = render_cmd + " --aux_image_type \"{}\" ".format(config_struct["aux_format"])

            render_basic_str = " --engine \"{}\" --render_height {} --render_width {} ".format(
                config_struct["engine"], config_struct["render_height"], config_struct["render_width"])
            render_cmd = render_cmd + render_basic_str

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

                    random_index = random.randint(0, hdr_map_number - 1)
                    random_map_path = hdr_map_path_list[random_index]
                    hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), random_map_path)
                    for camera_name in camera_pose_data.keys():
                        config_struct["hdr"]["paths"][camera_name] = hdr_map_path

                elif hdr_usage == 'static' or hdr_usage is None:
                    config_struct["hdr"] = {}
                    config_struct["hdr"]["paths"] = {}
                    if "hdr_path" in config_struct:
                        relative_hdr_path = os.path.join("../", config_struct["hdr_path"])
                        hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_hdr_path)
                        if not os.path.exists(hdr_map_path):
                            hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "../irrmaps/mud_road_puresky_4k.hdr")
                    else:
                        hdr_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "../irrmaps/mud_road_puresky_4k.hdr")
                    for camera_name in camera_pose_data.keys():
                        config_struct["hdr"]["paths"][camera_name] = hdr_map_path
            elif config_struct["light_source"] == "point":
                render_cmd = render_cmd + " --use_point_light  "

            if "equilibrium_path" in config_struct.keys():
                relative_equilibrium_path = config_struct["equilibrium_path"]
                abs_equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    relative_equilibrium_path)
                config_struct["equilibrium_path"] = abs_equilibrium_path

            if args.thumbnail:
                thumbnail_str = " --render_thumbnail "
                render_cmd = render_cmd + thumbnail_str
            else:
                if args.vae_latent:
                    vae_latent_str = " --render_vae_latent "
                    render_cmd = render_cmd + vae_latent_str

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
                render_cmd = render_cmd + " --camera_config_path \"{}\" ".format(config_json_output_path)

            for render_stage in render_stages:
                if render_stage == "emission":
                    render_stage_map["cmd"][render_stage] = render_cmd + " --only_emission "
                if render_stage == "equilibrium":
                    render_stage_map["cmd"][render_stage] = render_cmd + " --render_equilibrium "
                if render_stage == "PBR":
                    render_stage_map["cmd"][render_stage] = render_cmd + " --render_material --material_type \"PBR\" "
                if render_stage == "bump":
                    render_stage_map["cmd"][render_stage] = render_cmd + " --render_material --material_type \"bump\"  "
                if render_stage == "common":
                    render_stage_map["cmd"][render_stage] = render_cmd
                if render_stage == "no_smooth":
                    render_stage_map["cmd"][render_stage] = render_cmd

                render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][
                                                            render_stage] + " --output_folder \"{}\" ".format(
                    render_stage_map["output_folder"][render_stage])

                if args.parse_exr:
                    if render_stage == "common" or render_stage == "no_smooth":
                        render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][render_stage] + " --parse_exr "
                else:
                    if args.only_render_png:
                        if render_stage == "common" or render_stage == "no_smooth":
                            render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][render_stage] + \
                                                                    " --only_render_png "

                if render_stage in ["emission", "equilibrium", "PBR", "bump"]:
                    render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][
                                                                render_stage] + " --only_render_png --smooth --no_camera_export "

                if render_stage == "no_smooth":
                    render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][render_stage] + \
                                                            " --no_camera_export "

                if render_stage == "common":
                    render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][render_stage] + " --smooth "

                # must at the end of connecting cmd
                if args.silent:
                    render_stage_map["cmd"][render_stage] = render_stage_map["cmd"][render_stage] + " > /dev/null"

            pool.apply_async(func=render_once, args=(render_stage_map["cmd"],
                                                     mesh_path,
                                                     final_output_folder,
                                                     stat_txt,
                                                     time_txt,
                                                     folder_txt,
                                                     apply_render,
                                                     render_stages))
        cnt += 1
        if apply_render:
            with open(cmds_txt, 'a', encoding='UTF-8') as f:
                for render_stage in render_stages:
                    if render_stage in render_stage_map["cmd"].keys():
                        f.write(render_stage_map["cmd"][render_stage] + '\n')

    pool.close()
    pool.join()

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All rendering tasks DONE. These tasks end at time %s" %
          (local_time_str))
