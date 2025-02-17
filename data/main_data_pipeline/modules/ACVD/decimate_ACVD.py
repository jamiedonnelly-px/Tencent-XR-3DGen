import argparse
import json
import os
import time


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


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Remesh starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Management json file to data json file')
    parser.add_argument('--input_mesh_path', type=str, default='',
                        help='abspath of input mesh file')
    parser.add_argument('--output_mesh_path', type=str, default='',
                        help='output mesh file path')
    parser.add_argument('--point_number', type=int, default=10000,
                        help='output mesh point number')
    parser.add_argument('--gradation', type=float, default=1.5,
                        help='the influence of local curvature (0=uniform mesh, 1.5=non-uniform mesh)')
    parser.add_argument('--force_manifold', action='store_true',
                        help='force the output mesh file to be manifold')
    parser.add_argument('--minimal_face_number', type=int, default=-1,
                        help='clean up floating artifacts in mesh; -1 for not use this step')
    args = parser.parse_args()

    input_mesh_path = args.input_mesh_path
    output_mesh_path = args.output_mesh_path
    point_number = args.point_number
    gradation = args.gradation
    force_manifold = args.force_manifold
    minimal_face_number = args.minimal_face_number

    output_mesh_folder = os.path.split(output_mesh_path)[0]
    output_mesh_filename = os.path.split(output_mesh_path)[1]

    ACVD_op_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./release/ACVD")
    clean_op_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry/clean_mesh.py")

    ACVD_command = "{} \'{}\' {} {} ".format(ACVD_op_path, input_mesh_path, point_number, gradation)
    if force_manifold:
        ACVD_command = ACVD_command + " -m 1 "

    ACVD_command = ACVD_command + " -o \'{}\' ".format(output_mesh_folder)
    ACVD_command = ACVD_command + " -of \'{}\' ".format(output_mesh_filename)

    print("Remeshing using ACVD from path %s to path %s" % (input_mesh_path, output_mesh_path))
    print(ACVD_command)
    os.system(ACVD_command)
    time.sleep(0.1)

    if minimal_face_number > 1:
        cleanup_command = "python {} ".format(clean_op_path)
        cleanup_command = cleanup_command + " --mesh_path \'{}\'".format(output_mesh_path)
        cleanup_command = cleanup_command + " --output_mesh_path \'{}\'".format(output_mesh_path)
        cleanup_command = cleanup_command + " --minimal_face_number {} ".format(minimal_face_number)
        print(cleanup_command)
        os.system(cleanup_command)
        time.sleep(0.1)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("ACVD Remesh finished. Local time is %s" % (local_time_str))
