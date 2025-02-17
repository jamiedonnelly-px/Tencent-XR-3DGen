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
    parser.add_argument('--file_label', type=str, default="",
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
    file_label = args.file_label
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
            mesh_path_elements = mesh_path.split("/")
            cloth_name = mesh_path_elements[-3] + "_" + mesh_path_elements[-2]
            cloth_name = cloth_name.replace(" ", "_")
            cloth_name = cloth_name.replace(".duf_", "_")
            mesh_categories.append(file_label)
            instance_name_list.append(cloth_name)

    if data_start >= 0 and data_end > 0:
        if data_end > data_start:
            if data_end > len(source_mesh_paths):
                data_end = len(source_mesh_paths)
            source_mesh_paths = source_mesh_paths[data_start:data_end]

    print("Number of mesh on this pod: %i" % (len(source_mesh_paths)))

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    hair_cmds_txt = os.path.join(log_folder, 'hair_cmds.txt')
    body_cmds_txt = os.path.join(log_folder, 'body_cmds.txt')
    split_cmds_txt = os.path.join(log_folder, 'split_cmds.txt')
    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    time_txt = os.path.join(log_folder, 'time.txt')

    hair_cmds_file = open(hair_cmds_txt, 'w')
    body_cmds_file = open(body_cmds_txt, 'w')
    split_cmds_file = open(split_cmds_txt, 'w')
    stat_file = open(stat_txt, 'w')
    time_file = open(time_txt, 'w')
    folder_file = open(folder_txt, 'w')

    object_direction_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../geometry/object_direction.py")
    vroid_split_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "split_vroid.py")

    for index in range(len(source_mesh_paths)):
        source_mesh_path = source_mesh_paths[index]
        if not os.path.exists(source_mesh_path):
            print('Cannot find source mesh file ', source_mesh_path)
            continue

        mesh_folder = os.path.split(source_mesh_path)[0]
        mesh_name = os.path.split(source_mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_name)[0]

        hair_obj_name = os.path.join(mesh_folder, "raw_hair.obj")
        body_obj_name = os.path.join(mesh_folder, "raw_body.obj")

        print("body: %s | clothes: %s | hair: %s " %
              (body_obj_name, source_mesh_path, hair_obj_name))

        instance_name = instance_name_list[index]

        category = mesh_categories[index]
        category_output_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_output_folder):
            os.mkdir(category_output_folder)
        instance_output_folder = os.path.join(
            category_output_folder, instance_name)
        if not os.path.exists(instance_output_folder):
            os.mkdir(instance_output_folder)

        splited_mesh_folder = os.path.join(instance_output_folder, "split")
        if not os.path.exists(splited_mesh_folder):
            os.mkdir(splited_mesh_folder)
        hair_folder = os.path.join(splited_mesh_folder, "hair")
        if not os.path.exists(hair_folder):
            os.mkdir(hair_folder)
        body_folder = os.path.join(splited_mesh_folder, "body")
        if not os.path.exists(body_folder):
            os.mkdir(body_folder)

        command_struct = {}

        vroid_split_cmd = "{} -b -P {} -- ".format(
            blender_root, vroid_split_fullpath)
        vroid_split_cmd = vroid_split_cmd + \
                          " --source_mesh_path \"{}\" ".format(source_mesh_path)
        vroid_split_cmd = vroid_split_cmd + \
                          " --output_mesh_folder \"{}\" ".format(splited_mesh_folder)

        command_struct["clothes"] = vroid_split_cmd

        hair_output_mesh_path = os.path.join(hair_folder, "hair.obj")
        direction_hair_cmd = "{} -b -P {} -- ".format(
            blender_root, object_direction_fullpath)
        direction_hair_cmd = direction_hair_cmd + " --mesh_path '{}' --direction_corrected_mesh_path '{}' --input_direction '{}' '{}'".format(
            hair_obj_name, hair_output_mesh_path, 'Y', 'Z'
        )

        command_struct["hair"] = direction_hair_cmd

        body_output_mesh_path = os.path.join(body_folder, "body.obj")
        direction_body_cmd = "{} -b -P {} -- ".format(
            blender_root, object_direction_fullpath)
        direction_body_cmd = direction_body_cmd + " --mesh_path '{}' --direction_corrected_mesh_path '{}' --input_direction '{}' '{}'".format(
            body_obj_name, body_output_mesh_path, 'Y', 'Z'
        )

        command_struct["body"] = direction_body_cmd

        print("Fix material in mesh %s and split to categories...." %
              (source_mesh_path))

        pool.apply_async(func=convert_once, args=(
            command_struct, source_mesh_path, splited_mesh_folder, stat_txt, time_txt, folder_txt))

        with open(hair_cmds_txt, 'a') as f:
            f.write(direction_hair_cmd + '\n')
        with open(split_cmds_txt, 'a') as f:
            f.write(vroid_split_cmd + '\n')
        with open(body_cmds_txt, 'a') as f:
            f.write(direction_body_cmd + '\n')

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
    output_struct["data"]["VRoid_Top"] = {}
    output_struct["data"]["VRoid_Bottom"] = {}
    output_struct["data"]["VRoid_Hair"] = {}
    output_struct["data"]["VRoid_Outfit"] = {}
    output_struct["data"]["VRoid_Shoe"] = {}
    output_struct["data"]["VRoid_Body"] = {}

    body_cloth_names_map = {}

    for mesh_path in mesh_list:
        mesh_folder = os.path.split(mesh_path)[0]
        mesh_folder_name = os.path.split(mesh_folder)[1]
        split_folder = os.path.join(mesh_folder, "split")

        body_folder = os.path.join(split_folder, "body")
        body_obj_name = os.path.join(body_folder, "body.obj")
        if not os.path.exists(body_obj_name):
            print("Cannot find body obj file %s" % (body_obj_name))
            continue

        data_name = "VRoid_" + mesh_folder_name + "_Body"
        if data_name not in output_struct["data"]["VRoid_Body"].keys():
            output_struct["data"]["VRoid_Body"][data_name] = {}
            output_struct["data"]["VRoid_Body"][data_name]["Mesh"] = body_obj_name
            print(body_obj_name)

        top_folder = os.path.join(split_folder, "top")
        top_obj_name = os.path.join(top_folder, "top.obj")
        if os.path.exists(top_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Top"
            output_struct["data"]["VRoid_Top"][data_name] = {}
            output_struct["data"]["VRoid_Top"][data_name]["Mesh"] = top_obj_name
            print(top_obj_name)

        bottom_folder = os.path.join(split_folder, "bottom")
        bottom_obj_name = os.path.join(bottom_folder, "bottom.obj")
        if os.path.exists(bottom_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Bottom"
            output_struct["data"]["VRoid_Bottom"][data_name] = {}
            output_struct["data"]["VRoid_Bottom"][data_name]["Mesh"] = bottom_obj_name
            print(bottom_obj_name)

        hair_folder = os.path.join(split_folder, "hair")
        hair_obj_name = os.path.join(hair_folder, "hair.obj")
        if os.path.exists(hair_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Hair"
            output_struct["data"]["VRoid_Hair"][data_name] = {}
            output_struct["data"]["VRoid_Hair"][data_name]["Mesh"] = hair_obj_name
            print(hair_obj_name)

        outfit_folder = os.path.join(split_folder, "outfit")
        outfit_obj_name = os.path.join(outfit_folder, "outfit.obj")
        if os.path.exists(outfit_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Outfit"
            output_struct["data"]["VRoid_Outfit"][data_name] = {}
            output_struct["data"]["VRoid_Outfit"][data_name]["Mesh"] = outfit_obj_name
            print(outfit_obj_name)

        shoe_folder = os.path.join(split_folder, "shoe")
        shoe_obj_name = os.path.join(shoe_folder, "shoe.obj")
        if os.path.exists(shoe_obj_name):
            data_name = "VRoid_" + mesh_folder_name + "_Shoe"
            output_struct["data"]["VRoid_Shoe"][data_name] = {}
            output_struct["data"]["VRoid_Shoe"][data_name]["Mesh"] = shoe_obj_name
            print(shoe_obj_name)

    output_json_path = os.path.join(log_folder, "daz_info.json")
    write_json(output_json_path, output_struct)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("All daz split processes done. Local time is %s" % (local_time_str))
