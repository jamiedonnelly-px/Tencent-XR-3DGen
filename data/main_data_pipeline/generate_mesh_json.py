import argparse
import json
import multiprocessing
import os
import time


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


def json_once(log_foldername: str, output_json_path: str):
    stat_txt = os.path.join(log_foldername, "success.txt")
    folder_txt = os.path.join(log_foldername, "folder.txt")
    category_txt = os.path.join(log_foldername, "category.txt")

    success_mesh_list = read_list(stat_txt)
    result_folder_list = read_list(folder_txt)
    category_list = read_list(category_txt)

    print("Read from %s mesh list length %i and folder list length %i ....." %
          (log_foldername, len(success_mesh_list), len(result_folder_list)))

    output_data_struct = {}
    output_data_struct["data"] = {}

    for index in range(len(result_folder_list)):
        mesh_folder = result_folder_list[index]
        mesh_instance_name = os.path.split(mesh_folder)[1]
        if index >= len(category_list):
            continue
        mesh_category = category_list[index]
        if mesh_category not in output_data_struct["data"].keys():
            output_data_struct["data"][mesh_category] = {}

        original_mesh_name = os.path.join(mesh_folder, "resize/resize.obj")
        manifold_mesh_name = os.path.join(mesh_folder, "manifold/manifold.obj")
        texture_mesh_name = os.path.join(mesh_folder, "texture/texture.obj")

        instance_struct = {}
        if not os.path.exists(original_mesh_name):
            continue
        if not os.path.exists(texture_mesh_name):
            continue
        if not os.path.exists(manifold_mesh_name):
            continue

        instance_struct["Manifold"] = manifold_mesh_name
        instance_struct["Mesh"] = texture_mesh_name
        instance_struct["Original"] = original_mesh_name

        before_txt = os.path.join(mesh_folder, "manifold/before.txt")
        if os.path.exists(before_txt):
            before_mesh = read_txt(before_txt)
            instance_struct["Fine"] = before_mesh
        output_data_struct["data"][mesh_category][mesh_instance_name] = instance_struct

    write_json(output_json_path, output_data_struct)


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_txt(in_txt: str):
    list_content = read_list(in_txt)
    return list_content[0]


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


def scan_source_log_folder(source_log_folder: str):
    log_folder_list = []
    first_layer_geometry_cmd_txt_file = os.path.join(source_log_folder, "geometry_cmds.txt")
    first_layer_texture_cmd_txt_file = os.path.join(source_log_folder, "texture_cmds.txt")
    if os.path.exists(first_layer_geometry_cmd_txt_file) and os.path.exists(first_layer_texture_cmd_txt_file):
        log_folder_list.append(source_log_folder)

    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        if "verify" in folder_name:
            continue

        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            geometry_cmd_txt_file = os.path.join(folder_fullpath, "geometry_cmds.txt")
            proc_cmd_txt_file = os.path.join(folder_fullpath, "texture_cmds.txt")
            if os.path.exists(geometry_cmd_txt_file) and os.path.exists(proc_cmd_txt_file):
                log_folder_list.append(folder_fullpath)
    log_folder_list.sort()
    return log_folder_list


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--source_log_folder', type=str, default="",
                        help='folder of render logs')
    parser.add_argument('--stat_txt', type=str, default="",
                        help='txt of success meshes')
    parser.add_argument('--folder_txt', type=str, default="",
                        help='txt of success render folders')
    parser.add_argument('--category_txt', type=str, default="",
                        help='txt of mesh categories')
    parser.add_argument('--merge_output_json', type=str, default="",
                        help='merge all data in json output')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')

    args = parser.parse_args()
    source_log_folder = args.source_log_folder
    merge_output_json = args.merge_output_json

    log_folder_list = scan_source_log_folder(source_log_folder)
    log_json_path_list = []

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(cpu_cnt, args.pool_cnt))

    json_merger_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "tools/lists/merge_json_file_list.py")

    for log_foldername in log_folder_list:
        output_json_path = os.path.join(log_foldername, "mesh_info.json")
        log_json_path_list.append(output_json_path)
        pool.apply_async(func=json_once, args=(log_foldername, output_json_path))

    pool.close()
    pool.join()

    time.sleep(1)

    json_merge_cmd = "python {} --output_json_path {} --json_files ".format(json_merger_fullpath, merge_output_json)
    for log_json_path in log_json_path_list:
        json_merge_cmd = json_merge_cmd + " \"{}\" ".format(log_json_path)
    print(json_merge_cmd)
    os.system(json_merge_cmd)
