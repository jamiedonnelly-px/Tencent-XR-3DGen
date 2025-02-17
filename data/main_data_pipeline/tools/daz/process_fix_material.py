import argparse
import json
import multiprocessing
import os
import time


def convert_once(command_struct, destination_mesh_path, output_folder, stat_txt, time_txt, folder_txt):
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
    parser.add_argument('--in_mesh_list', type=str, default="",
                        help='mesh list txt file')
    parser.add_argument('--output_folder', type=str, default="",
                        help='folder containing all output mesh files')
    parser.add_argument('--daz_file_label', type=str, default="",
                        help='name of label of this daz file')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--data_start', type=int, default=0,
                        help='start (left) of interval of data to be rendered (left close right open)')
    parser.add_argument('--data_end', type=int, default=-1,
                        help='end (right) of interval of data to be rendered (left close right open)')

    args = parser.parse_args()
    in_mesh_list_txt = args.in_mesh_list
    output_folder = args.output_folder
    daz_file_label = args.daz_file_label
    blender_root = args.blender_root
    log_folder = args.log_folder

    data_start = args.data_start
    data_end = args.data_end

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    mesh_categories = []
    instance_name_list = []
    cloth_name_path_map = {}
    if len(in_mesh_list_txt) > 1:
        source_mesh_paths = read_list(in_mesh_list_txt)
        for mesh_path in source_mesh_paths:
            mesh_filename = os.path.split(mesh_path)[1]
            mesh_file_basename = os.path.splitext(mesh_filename)[0]
            mesh_file_extension = os.path.splitext(mesh_filename)[1]
            cloth_name = mesh_file_basename.replace(" ", "_")
            cloth_name = cloth_name.replace(".duf_", "_")
            mesh_categories.append(daz_file_label)
            instance_name_list.append(cloth_name)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(source_mesh_paths):
                data_end = len(source_mesh_paths)
            source_mesh_paths = source_mesh_paths[data_start:data_end]

    print("Number of mesh on this pod: %i" % (len(source_mesh_paths)))

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    material_cmds_txt = os.path.join(log_folder, 'material_cmds.txt')
    split_cmds_txt = os.path.join(log_folder, 'split_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')

    material_cmds_file = open(material_cmds_txt, 'w')
    split_cmds_file = open(split_cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    fix_material_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "fix_material.py")
    daz_split_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "split_daz.py")

    instance_name_cloth_info_map = {}
    cloth_info_instance_map = {}
    cloth_info_instance_map["clothes"] = {}
    cloth_info_instance_map["hair"] = {}
    cloth_info_instance_map["body"] = {}

    for index in range(len(source_mesh_paths)):
        source_mesh_path = source_mesh_paths[index]
        if not os.path.exists(source_mesh_path):
            print('Cannot find source mesh file ', source_mesh_path)
            continue

        mesh_folder = os.path.split(source_mesh_path)[0]
        mesh_name = os.path.split(source_mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]

        mesh_info_file_name = mesh_basename + ".txt"
        mesh_info_fullpath = os.path.join(mesh_folder, mesh_info_file_name)
        if not os.path.exists(mesh_info_fullpath):
            print(mesh_info_fullpath)
            continue
        mesh_info = read_list(mesh_info_fullpath)

        if len(mesh_info) < 3:
            print(mesh_info, mesh_info_file_name)
            continue

        mesh_body_name = mesh_info[0].split(":")[1]
        mesh_clothes_name = mesh_info[1].split(":")[1]
        mesh_hair_name = mesh_info[2].split(":")[1]

        print("body: %s | clothes: %s | hair: %s " %
              (mesh_body_name, mesh_clothes_name, mesh_hair_name))

        output_mesh_name = mesh_basename.replace(" ", "_")
        output_mesh_name = output_mesh_name.replace(".duf_", "_")
        output_mesh_name = output_mesh_name + ".obj"

        instance_name = instance_name_list[index]

        instance_name_cloth_info_map[instance_name] = [
            mesh_body_name, mesh_clothes_name, mesh_hair_name]

        category = mesh_categories[index]
        category_output_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_output_folder):
            os.mkdir(category_output_folder)
        instance_output_folder = os.path.join(
            category_output_folder, instance_name)
        if not os.path.exists(instance_output_folder):
            os.mkdir(instance_output_folder)

        complete_mesh_folder = os.path.join(instance_output_folder, "complete")
        splited_mesh_folder = os.path.join(instance_output_folder, "split")
        if not os.path.exists(complete_mesh_folder):
            os.mkdir(complete_mesh_folder)
        if not os.path.exists(splited_mesh_folder):
            os.mkdir(splited_mesh_folder)

        final_output_path = os.path.join(
            complete_mesh_folder, output_mesh_name)

        fix_material_cmd = "{} -b -P {} -- ".format(
            blender_root, fix_material_fullpath)
        fix_material_cmd = fix_material_cmd + \
                           " --source_mesh_path \"{}\" ".format(source_mesh_path)
        fix_material_cmd = fix_material_cmd + \
                           " --output_mesh_path \"{}\" --clear_pose ".format(
                               final_output_path)

        command_struct = {}
        command_struct["fix"] = fix_material_cmd

        daz_split_cmd = "{} -b -P {} -- ".format(
            blender_root, daz_split_fullpath)
        daz_split_cmd = daz_split_cmd + \
                        " --source_mesh_path \"{}\" ".format(final_output_path)
        daz_split_cmd = daz_split_cmd + \
                        " --output_mesh_folder \"{}\" ".format(splited_mesh_folder)

        cloth_info_instance_map["body"][mesh_body_name] = []
        cloth_info_instance_map["body"][mesh_body_name].append(
            instance_name)
        body_file_name = mesh_body_name.replace(" ", "_")
        body_file_name = body_file_name.replace("!", "_")
        body_file_name = body_file_name.replace(".duf", "_")
        daz_split_cmd = daz_split_cmd + \
                        " --body_name \"{}\" ".format(body_file_name)

        if (mesh_hair_name not in cloth_info_instance_map["hair"]) or (
                mesh_clothes_name not in cloth_info_instance_map["clothes"]):
            if mesh_clothes_name not in cloth_info_instance_map["clothes"]:
                cloth_info_instance_map["clothes"][mesh_clothes_name] = []
                cloth_info_instance_map["clothes"][mesh_clothes_name].append(
                    instance_name)
                cloth_file_name = mesh_clothes_name.replace(" ", "_")
                cloth_file_name = cloth_file_name.replace("!", "_")
                cloth_file_name = cloth_file_name.replace(".duf", "_")
                daz_split_cmd = daz_split_cmd + \
                                " --split_clothes --clothes_name \"{}\" ".format(
                                    cloth_file_name)

            if mesh_hair_name not in cloth_info_instance_map["hair"]:
                cloth_info_instance_map["hair"][mesh_hair_name] = []
                cloth_info_instance_map["hair"][mesh_hair_name].append(
                    instance_name)
                hair_file_name = mesh_hair_name.replace(" ", "_")
                hair_file_name = hair_file_name.replace("!", "_")
                hair_file_name = hair_file_name.replace(".duf", "_")
                daz_split_cmd = daz_split_cmd + \
                                " --split_hair --hair_name \"{}\" ".format(hair_file_name)
            command_struct["split"] = daz_split_cmd
        else:
            cloth_info_instance_map["body"][mesh_body_name].append(
                instance_name)
            cloth_info_instance_map["clothes"][mesh_clothes_name].append(
                instance_name)
            cloth_info_instance_map["hair"][mesh_hair_name].append(
                instance_name)

        print("Fix material in mesh %s and split to categories...." %
              (source_mesh_path))

        pool.apply_async(func=convert_once, args=(
            command_struct, source_mesh_path, final_output_path, stat_txt, time_txt, folder_txt))

        with open(material_cmds_txt, 'a') as f:
            f.write(fix_material_cmd + '\n')
        with open(split_cmds_txt, 'a') as f:
            f.write(daz_split_cmd + '\n')

    pool.close()
    pool.join()

    time.sleep(0.1)


    def find_part_obj_in_folder(folder_path: str):
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            for filename in files:
                file_extension = os.path.splitext(filename)[1]
                file_basename = os.path.splitext(filename)[0]
                if file_extension == ".obj":
                    file_fullpath = os.path.join(folder_path, filename)
                    return file_fullpath, file_basename
        return None, None


    mesh_list = read_list(folder_txt)
    output_struct = {}
    output_struct["data"] = {}
    output_struct["data"]["DAZ_Top"] = {}
    output_struct["data"]["DAZ_Bottom"] = {}
    output_struct["data"]["DAZ_Hair"] = {}
    output_struct["data"]["DAZ_Outfit"] = {}
    output_struct["data"]["DAZ_Shoe"] = {}
    output_struct["data"]["DAZ_Body"] = {}

    body_cloth_names_map = {}

    for mesh_path in mesh_list:
        mesh_complete_folder = os.path.split(mesh_path)[0]
        mesh_folder = os.path.split(mesh_complete_folder)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        split_folder = os.path.join(mesh_folder, "split")

        body_folder = os.path.join(split_folder, "body")
        if not os.path.exists(body_folder):
            print("Cannot find body obj folder %s" % (body_folder))
            continue
        body_obj_name, body_instance_name = find_part_obj_in_folder(
            body_folder)
        if body_obj_name is not None and body_instance_name is not None:
            data_name = "DAZ_" + body_instance_name + "_Body"
            if body_obj_name not in body_cloth_names_map.keys():
                body_cloth_names_map[body_obj_name] = {}
            # if data_name not in output_struct["data"]["DAZ_Body"].keys():
            #     output_struct["data"]["DAZ_Body"][data_name] = {}
            #     output_struct["data"]["DAZ_Body"][data_name]["Mesh"] = body_obj_name
            #     print(body_obj_name)

        top_folder = os.path.join(split_folder, "top")
        if os.path.exists(top_folder):
            top_obj_name, top_instance_name = find_part_obj_in_folder(
                top_folder)
            if top_obj_name is not None and top_instance_name is not None:
                data_name = "DAZ_" + top_instance_name + "_Top"
                output_struct["data"]["DAZ_Top"][data_name] = {}
                output_struct["data"]["DAZ_Top"][data_name]["Mesh"] = top_obj_name
                print(top_obj_name)
                if body_obj_name is not None and body_instance_name is not None:
                    if data_name not in body_cloth_names_map[body_obj_name].keys():
                        body_cloth_names_map[body_obj_name][data_name] = {}
                    body_cloth_names_map[body_obj_name][data_name]["top"] = top_obj_name

        bottom_folder = os.path.join(split_folder, "bottom")
        if os.path.exists(bottom_folder):
            bottom_obj_name, bottom_instance_name = find_part_obj_in_folder(
                bottom_folder)
            if bottom_obj_name is not None and bottom_instance_name is not None:
                data_name = "DAZ_" + bottom_instance_name + "_Bottom"
                output_struct["data"]["DAZ_Bottom"][data_name] = {}
                output_struct["data"]["DAZ_Bottom"][data_name]["Mesh"] = bottom_obj_name
                print(bottom_obj_name)
                if body_obj_name is not None and body_instance_name is not None:
                    if data_name not in body_cloth_names_map[body_obj_name].keys():
                        body_cloth_names_map[body_obj_name][data_name] = {}
                    body_cloth_names_map[body_obj_name][data_name]["bottom"] = bottom_obj_name

        hair_folder = os.path.join(split_folder, "hair")
        if os.path.exists(hair_folder):
            hair_obj_name, hair_instance_name = find_part_obj_in_folder(
                hair_folder)
            if hair_obj_name is not None and hair_instance_name is not None:
                data_name = "DAZ_" + hair_instance_name + "_Hair"
                output_struct["data"]["DAZ_Hair"][data_name] = {}
                output_struct["data"]["DAZ_Hair"][data_name]["Mesh"] = hair_obj_name
                print(hair_obj_name)
                if body_obj_name is not None and body_instance_name is not None:
                    if data_name not in body_cloth_names_map[body_obj_name].keys():
                        body_cloth_names_map[body_obj_name][data_name] = {}
                    body_cloth_names_map[body_obj_name][data_name]["hair"] = hair_obj_name

        outfit_folder = os.path.join(split_folder, "outfit")
        if os.path.exists(outfit_folder):
            outfit_obj_name, outfit_instance_name = find_part_obj_in_folder(
                outfit_folder)
            if outfit_obj_name is not None and outfit_instance_name is not None:
                data_name = "DAZ_" + outfit_instance_name + "_Outfit"
                output_struct["data"]["DAZ_Outfit"][data_name] = {}
                output_struct["data"]["DAZ_Outfit"][data_name]["Mesh"] = outfit_obj_name
                print(outfit_obj_name)
                if body_obj_name is not None and body_instance_name is not None:
                    if data_name not in body_cloth_names_map[body_obj_name].keys():
                        body_cloth_names_map[body_obj_name][data_name] = {}
                    body_cloth_names_map[body_obj_name][data_name]["outfit"] = outfit_obj_name

        shoe_folder = os.path.join(split_folder, "shoe")
        if os.path.exists(shoe_folder):
            shoe_obj_name, shoe_instance_name = find_part_obj_in_folder(
                shoe_folder)
            if shoe_obj_name is not None and shoe_instance_name is not None:
                data_name = "DAZ_" + shoe_instance_name + "_Shoe"
                output_struct["data"]["DAZ_Shoe"][data_name] = {}
                output_struct["data"]["DAZ_Shoe"][data_name]["Mesh"] = shoe_obj_name
                print(shoe_obj_name)
                if body_obj_name is not None and body_instance_name is not None:
                    if data_name not in body_cloth_names_map[body_obj_name].keys():
                        body_cloth_names_map[body_obj_name][data_name] = {}
                    body_cloth_names_map[body_obj_name][data_name]["shoe"] = shoe_obj_name

    output_json_path = os.path.join(log_folder, "daz_info.json")
    body_cloth_json_path = os.path.join(log_folder, "body_clothes.json")
    write_json(output_json_path, output_struct)
    write_json(body_cloth_json_path, body_cloth_names_map)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All daz split processes done. Local time is %s" % (local_time_str))
