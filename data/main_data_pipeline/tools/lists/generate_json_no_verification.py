import argparse
import json
import os


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


def scan_source_log_folder(source_log_folder: str):
    log_folder_list = []
    first_layer_cmd_txt_file = os.path.join(source_log_folder, "cmds.txt")
    first_layer_proc_cmd_txt_file = os.path.join(source_log_folder, "proc_cmds.txt")
    if os.path.exists(first_layer_cmd_txt_file) and os.path.exists(first_layer_proc_cmd_txt_file):
        log_folder_list.append(source_log_folder)

    source_log_folder_files = os.listdir(source_log_folder)
    for folder_name in source_log_folder_files:
        if "verify" in folder_name:
            continue

        folder_fullpath = os.path.join(source_log_folder, folder_name)
        if os.path.isdir(folder_fullpath):
            cmd_txt_file = os.path.join(folder_fullpath, "cmds.txt")
            proc_cmd_txt_file = os.path.join(folder_fullpath, "proc_cmds.txt")
            if os.path.exists(cmd_txt_file) and os.path.exists(proc_cmd_txt_file):
                log_folder_list.append(folder_fullpath)
    log_folder_list.sort()
    return log_folder_list


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
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--log_folder', type=str,
                        help='folder of json')
    parser.add_argument('--output_json_path', type=str,
                        help='merged json output path')
    args = parser.parse_args()

    log_folder_list = scan_source_log_folder(args.log_folder)
    data_struct = {}
    data_struct["data"] = {}
    for log_index in range(len(log_folder_list)):
        log_folder_fullpath = log_folder_list[log_index]

        stat_txt = os.path.join(log_folder_fullpath, "success.txt")
        folder_txt = os.path.join(log_folder_fullpath, "folder.txt")
        proc_txt = os.path.join(log_folder_fullpath, "proc.txt")

        success_mesh_list = read_list(stat_txt)
        result_folder_list = read_list(folder_txt)
        proc_folder_list = read_list(proc_txt)

        mesh_number = len(success_mesh_list)
        print("Read from %s mesh list length %i and folder list length %i ....." %
              (log_folder_fullpath, mesh_number, len(result_folder_list)))

        for index in range(mesh_number):
            success_mesh_path = success_mesh_list[index]
            render_folder = result_folder_list[index]

            render_parent_folder = os.path.split(render_folder)[0]
            mesh_instance_name = os.path.split(render_parent_folder)[1]
            render_category_folder = os.path.split(render_parent_folder)[0]
            mesh_category = os.path.split(render_category_folder)[1]
            if mesh_category not in data_struct["data"].keys():
                data_struct["data"][mesh_category] = {}

            texture_mesh_name = success_mesh_path

            transformation_txt = os.path.join(render_folder, "transformation.txt")

            print("[%i / %i / %i] Add render mesh %s (category %s) to final json.." %
                  (index, mesh_number, log_index, texture_mesh_name, mesh_category))
            instance_struct = {}
            instance_struct["Mesh"] = texture_mesh_name
            instance_struct["ImgDir"] = render_folder
            if os.path.exists(transformation_txt):
                instance_struct["Z_Transformation"] = transformation_txt
            data_struct["data"][mesh_category][mesh_instance_name] = instance_struct

    write_json(args.output_json_path, data_struct)
    print(check_data_number(data_struct))
    print(check_individual_number(data_struct))
