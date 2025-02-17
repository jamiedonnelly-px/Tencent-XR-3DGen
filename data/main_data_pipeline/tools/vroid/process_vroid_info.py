import argparse
import json
import multiprocessing
import os
import time

import clip
import torch
from PIL import Image
from tqdm import tqdm


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


def convert_once(command_struct, destination_mesh_path, output_folder,
                 stat_txt, time_txt, folder_txt):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for manifold converting command is %s....' %
          (str(start_time_str)))

    for cmd_name in command_struct.keys():
        print("Start command %s: %s" % (cmd_name, command_struct[cmd_name]))
        exec_result = os.system(command_struct[cmd_name])
        time.sleep(0.1)

    stat = exec_result

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After mesh remove command status is %s; time for this status is %s....' %
          (str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(destination_mesh_path))

    with open(folder_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('{}\n'.format(output_folder))

    with open(time_txt, 'a') as f:
        is_suc = 1 if stat == 0 else 0
        if stat == 0:
            f.write('%s starts at %s, finish at %s....\n' %
                    (output_folder, start_time_str, end_time_str))


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")


def clip_obj_texture_images(obj_path_list: list):
    clip_model.eval()
    obj_name_list = []
    obj_picture_tensor_list = []

    for index in tqdm(range(len(obj_path_list))):
        obj_path = obj_path_list[index]
        obj_folder = os.path.split(obj_path)[0]
        mtl_path = obj_path.replace(".obj", ".mtl")
        mtl_info = read_list(mtl_path)
        picture_tensor = None
        for mtl_line in mtl_info:
            if "map_Kd" in mtl_line:
                picture_name = mtl_line.split(" ")[1]
                picture_fullpath = os.path.join(obj_folder, picture_name)

                if not os.path.exists(picture_fullpath):
                    print("Cannot find texture image located at %s" %
                          (picture_fullpath))
                    break

                picture_tensor = clip_preprocess(
                    Image.open(picture_fullpath)).unsqueeze(0)
                # picture_tensor = clip_model.encode_image(picture_data)
                # picture_tensor /= picture_tensor.norm(dim=-1, keepdim=True)
                break
        if picture_tensor is None:
            continue
        obj_name_list.append(obj_path)
        obj_picture_tensor_list.append(picture_tensor)

    obj_picture_tensor = torch.stack((obj_picture_tensor_list)).squeeze(dim=1)
    clip_picture_feature = clip_model.encode_image(obj_picture_tensor)

    del obj_picture_tensor
    torch.cuda.empty_cache()

    return obj_name_list, clip_picture_feature


def compare_obj_texture_images(obj_path_list: list):
    clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    # loss_fn = lpips.LPIPS(net='alex')
    # loss_fn.cuda()
    loss = torch.nn.MSELoss()
    image_tensor_list = []
    no_duplicate_obj_list = []
    for obj_path in obj_path_list:
        obj_folder = os.path.split(obj_path)[0]
        mtl_path = obj_path.replace(".obj", ".mtl")
        mtl_info = read_list(mtl_path)
        picture_tensor = None
        for mtl_line in mtl_info:
            if "map_Kd" in mtl_line:
                picture_name = mtl_line.split(" ")[1]
                picture_fullpath = os.path.join(obj_folder, picture_name)
                picture_data = clip_preprocess(
                    Image.open(picture_fullpath)).unsqueeze(0)
                picture_tensor = clip_model.encode_image(picture_data)
                # picture_tensor /= picture_tensor.norm(dim=-1, keepdim=True)
                break
        if picture_tensor is None:
            continue
        obj_is_duplicate = False
        for index in range(len(image_tensor_list)):
            image_tensor = image_tensor_list[index]
            dist_old_new = loss(image_tensor, picture_tensor)
            if dist_old_new < 0.1:
                print(dist_old_new)
                print(no_duplicate_obj_list[index], obj_path)
                obj_is_duplicate = True
                break
        if obj_is_duplicate:
            continue
        else:
            image_tensor_list.append(picture_tensor)
            no_duplicate_obj_list.append(obj_path)
    return no_duplicate_obj_list


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


def read_mesh_list_from_data_json(json_path: str, data_tag=""):
    print("Parse data json at path %s" % (json_path))
    with open(json_path, encoding='utf-8') as f:
        data_config = json.load(f)

    obj_path_struct = {}
    object_category_list = []
    src_obj_path_list = []
    instance_name_list = []
    manifold_path_list = []
    proc_data_folder_list = []
    render_data_list = []

    data_path_struct = data_config["data"]
    data_path_name_list = list(data_path_struct.keys())

    for data_name in data_path_name_list:
        if len(data_tag) > 1:
            if data_name != data_tag:
                continue

        all_instance_path_struct = data_path_struct[data_name]
        for instance_name in all_instance_path_struct.keys():
            instance_paths = all_instance_path_struct[instance_name]
            if "Mesh" not in instance_paths.keys():
                continue

            src_mesh_path = instance_paths["Mesh"]
            if src_mesh_path is None:
                continue

            if "TexPcd" in instance_paths.keys():
                if instance_paths["TexPcd"] is not None:
                    tex_pcd_path = instance_paths["TexPcd"]
                    proc_data_folder = os.path.split(tex_pcd_path)[0]
                else:
                    proc_data_folder = None
            else:
                proc_data_folder = None

            category_name = data_name.split("_")[-1]
            src_obj_path_list.append(correct_disk_name(src_mesh_path))
            object_category_list.append(category_name)
            instance_name_list.append(instance_name)
            proc_data_folder_list.append(correct_disk_name(proc_data_folder))

    return src_obj_path_list, object_category_list, instance_name_list, proc_data_folder_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove mesh with certain material name.')
    parser.add_argument('--vroid_info_json', type=str, default="",
                        help='input vroid mesh info json file')
    parser.add_argument('--material_mesh_json_path', type=str, default="",
                        help='json file containing information between material name and mesh')
    parser.add_argument('--output_json_path', type=str, default="",
                        help='output json information file')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')

    args = parser.parse_args()
    vroid_info_json = args.vroid_info_json
    material_mesh_json_path = args.material_mesh_json_path
    output_json_path = args.output_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    source_mesh_path_list, category_list, instance_name_list, _ = read_mesh_list_from_data_json(
        vroid_info_json)
    source_mesh_folder_set = set()
    source_mesh_path_category_map = {}
    source_mesh_path_instance_map = {}
    for index in range(len(source_mesh_path_list)):
        mesh_path = source_mesh_path_list[index]
        mesh_category = category_list[index]
        instance_name = instance_name_list[index]

        mesh_part_folder = os.path.split(mesh_path)[0]
        mesh_split_folder = os.path.split(mesh_part_folder)[0]
        source_mesh_folder_set.add(mesh_split_folder)
        source_mesh_path_category_map[mesh_split_folder] = mesh_category
        source_mesh_path_instance_map[mesh_split_folder] = instance_name

    source_mesh_folder_paths = list(source_mesh_folder_set)

    print("Number of mesh on this pod: %i" % (len(source_mesh_folder_paths)))
    cmds_txt = os.path.join(log_folder, 'split_info_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    category_txt = os.path.join(log_folder, 'category.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')

    cmds_file = open(cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    category_file = open(category_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    vroid_info_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "vroid_mesh_info.py")

    # for index in range(len(source_mesh_folder_paths)):
    #     source_mesh_folder = source_mesh_folder_paths[index]
    #     if not os.path.exists(source_mesh_folder):
    #         print('Cannot find source mesh file ', source_mesh_folder)
    #         continue

    #     command_struct = {}

    #     mesh_folder = os.path.split(source_mesh_folder)[0]
    #     mesh_name = os.path.split(mesh_folder)[1]
    #     vroid_instance_info_json = os.path.join(mesh_folder, "info.json")
    #     if os.path.exists(vroid_instance_info_json):
    #         print("Vroid info at %s already exist...." %
    #               (vroid_instance_info_json))
    #         continue

    #     vroid_info_cmd = "{} -b -P {} -- ".format(
    #         blender_root, vroid_info_op_fullpath)
    #     vroid_info_cmd = vroid_info_cmd + \
    #         " --source_mesh_folder \"{}\" ".format(source_mesh_folder)
    #     vroid_info_cmd = vroid_info_cmd + \
    #         " --output_json_path \"{}\" ".format(vroid_instance_info_json)

    #     command_struct["info"] = vroid_info_cmd

    #     print("Generate mesh info in splited mesh folder %s...." %
    #           (source_mesh_folder))

    #     pool.apply_async(func=convert_once, args=(
    #         command_struct, source_mesh_folder, vroid_instance_info_json, stat_txt, time_txt, folder_txt))

    #     with open(cmds_txt, 'a') as f:
    #         f.write(vroid_info_cmd + '\n')

    pool.close()
    pool.join()

    time.sleep(0.1)

    input_data_struct = read_json(vroid_info_json)
    mesh_data_struct = input_data_struct["data"]
    mesh_material_map = {}

    if not os.path.exists(material_mesh_json_path):
        source_folder_vroid_info_map = {}
        source_folder_number = len(source_mesh_folder_paths)
        for index in range(source_folder_number):
            source_mesh_folder = source_mesh_folder_paths[index]
            mesh_folder = os.path.split(source_mesh_folder)[0]
            vroid_info_json_path = os.path.join(mesh_folder, "info.json")

            if os.path.exists(vroid_info_json_path):
                vroid_info_all_parts = read_json(vroid_info_json_path)
                print("[%i / %i], %s, %s" % (index, source_folder_number,
                                             vroid_info_json_path, vroid_info_all_parts))
                source_folder_vroid_info_map[mesh_folder] = vroid_info_all_parts
        for data_name in mesh_data_struct.keys():
            vroid_part = data_name.split("_")[-1].lower()
            if data_name not in mesh_material_map.keys():
                mesh_material_map[data_name] = {}

            for mesh_name in mesh_data_struct[data_name].keys():
                mesh_path = mesh_data_struct[data_name][mesh_name]["Mesh"]
                mesh_part_folder = os.path.split(mesh_path)[0]
                mesh_split_folder = os.path.split(mesh_part_folder)[0]
                mesh_folder = os.path.split(mesh_split_folder)[0]
                # vroid_info_json_path = os.path.join(mesh_folder, "info.json")

                if mesh_folder in source_folder_vroid_info_map.keys():
                    vroid_info_all_parts = source_folder_vroid_info_map[mesh_folder]
                    vroid_info_current_part = vroid_info_all_parts[vroid_part]
                    material_name = vroid_info_current_part["material"]
                    if material_name not in mesh_material_map[data_name].keys():
                        mesh_material_map[data_name][material_name] = {}
                        mesh_material_map[data_name][material_name]["path"] = [
                        ]
                        mesh_material_map[data_name][material_name]["name"] = [
                        ]
                    mesh_material_map[data_name][material_name]["path"].append(
                        mesh_path)
                    mesh_material_map[data_name][material_name]["name"].append(
                        mesh_name)
                    print(data_name, vroid_part, material_name, mesh_path)

        write_json(material_mesh_json_path, mesh_material_map)
    else:
        mesh_material_map = read_json(material_mesh_json_path)

    output_no_duplicate_struct = {}
    output_no_duplicate_struct["data"] = {}
    for data_name in mesh_material_map.keys():
        if data_name not in output_no_duplicate_struct["data"].keys():
            output_no_duplicate_struct["data"][data_name] = {}
        for material_name in mesh_material_map[data_name].keys():
            original_obj_list = mesh_material_map[data_name][material_name]["path"]
            obj_path_list, obj_tensor = clip_obj_texture_images(
                original_obj_list)
            object_number = len(obj_path_list)

            # obj_tensor = torch.stack((obj_tensor_list))
            obj_tensor_x_axis = obj_tensor.unsqueeze(dim=0)
            obj_tensor_y_axis = obj_tensor.unsqueeze(dim=1)
            obj_similarity = torch.nn.functional.cosine_similarity(
                obj_tensor_x_axis, obj_tensor_y_axis, dim=2)

            similary_tuple = torch.where(obj_similarity > 0.92)
            similarity_list_x = similary_tuple[0].cpu().numpy().tolist()
            similarity_list_y = similary_tuple[1].cpu().numpy().tolist()
            similarity_map = {}
            for index in range(object_number):
                similarity_map[index] = True
            for pair_index in range(len(similarity_list_x)):
                index_x = similarity_list_x[pair_index]
                index_y = similarity_list_y[pair_index]
                if index_x == index_y:
                    continue
                if not similarity_map[index_x]:
                    continue
                similarity_map[index_y] = False

            active_index_list = []
            for object_index in similarity_map.keys():
                if similarity_map[object_index]:
                    active_index_list.append(object_index)
                    active_mesh_name = mesh_material_map[data_name][material_name]["name"][object_index]
                    active_mesh_path = mesh_material_map[data_name][material_name]["path"][object_index]
                    if active_mesh_name not in output_no_duplicate_struct["data"][data_name].keys():
                        output_no_duplicate_struct["data"][data_name][active_mesh_name] = {
                        }
                    output_no_duplicate_struct["data"][data_name][active_mesh_name]["Mesh"] = active_mesh_path
            print(len(active_index_list))
            print(check_individual_number(output_no_duplicate_struct))

    write_json(output_json_path, output_no_duplicate_struct)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All daz split processes done. Local time is %s" % (local_time_str))
