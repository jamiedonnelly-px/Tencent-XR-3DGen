import argparse
import json
import os
import time

import h5py
import numpy as np


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        number_info[category_name] = len(json_struct["data"][category_name])
    return number_info


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt: str):
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


def write_done(path: str):
    file_name = "task.done"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("done")


def check_done(path: str):
    file_name = "task.done"
    file_fullpath = os.path.join(path, file_name)
    if os.path.exists(file_fullpath):
        return True
    return False


def write_valid(path: str):
    file_name = "mesh.valid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("valid")


def check_file_number(folder_path: str, extension: str):
    exr_counter = 0
    if os.path.exists(folder_path):
        exr_files = os.listdir(folder_path)
        for exr_file in exr_files:
            if extension in exr_file:
                exr_counter = exr_counter + 1

    return exr_counter


def check_sample_h5(sample_folder: str, sample_number_struct: dict = None):
    geometry_folder = os.path.join(sample_folder, "geometry")
    geometry_sample_h5 = os.path.join(geometry_folder, "sample.h5")
    if not os.path.exists(geometry_sample_h5):
        return
    with h5py.File(geometry_sample_h5, 'r') as f:
        for data_name in f.keys():
            npy_data = np.array(f[data_name])
            if sample_number_struct is None:
                if npy_data.shape[0] < 100000:
                    return
            else:
                data_position = data_name.rsplit('_', 1)[0]
                if data_position not in sample_number_struct.keys():
                    return
                sample_number = sample_number_struct[data_position]
                print("Actual sample number is %i, config sample number is %i" % (npy_data.shape[0], sample_number))
                if npy_data.shape[0] != sample_number:
                    return
    write_valid(sample_folder)


def check_sample_npy(sample_folder: str):
    geometry_folder = os.path.join(sample_folder, "geometry")
    geometry_files = os.listdir(geometry_folder)
    for geometry_filename in geometry_files:
        file_basename = os.path.splitext(geometry_filename)[0]
        file_extension = os.path.splitext(geometry_filename)[1]
        if file_extension == ".npy":
            file_elements = file_basename.split("_")
            point_number = int(file_elements[-1])
            file_fullpath = os.path.join(geometry_folder, geometry_filename)
            npy_data = np.load(file_fullpath)
            if npy_data.shape[0] != point_number:
                return

    write_valid(sample_folder)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("SDF sample verification starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--proc_folder", type=str, default="",
                        help="path to sample points results")
    parser.add_argument("--render_folder", type=str, default="",
                        help="output folder of sampled points")
    parser.add_argument("--check_format", type=str, default="h5",
                        help="check sample data format, choose between h5 or npy")
    args = parser.parse_args()

    proc_folder = args.proc_folder
    render_folder = args.render_folder
    check_format = args.check_format

    done_filename = os.path.join(proc_folder, "task.done")
    # if not os.path.exists(done_filename):
    #     exit(-1)

    valid_filename = os.path.join(proc_folder, "mesh.valid")
    if not os.path.exists(valid_filename):
        config_file = os.path.join(render_folder, "config.json")
        sampled_number_struct = None
        if os.path.exists(config_file) and len(render_folder) > 1:
            config_data = read_json(config_file)
            sampled_number_struct = {}
            sampled_number_struct["space"] = config_data["geometry_space_sample_number"]
            sampled_number_struct["near_surface"] = config_data["geometry_near_surface_sample_number"]
            sampled_number_struct["surface"] = config_data["geometry_surface_sample_number"]

        if check_format == "h5":
            check_sample_h5(sample_folder=proc_folder, sample_number_struct=sampled_number_struct)
        else:
            check_sample_npy(sample_folder=proc_folder)
