import argparse
import json
import multiprocessing
import os
import time


def check_once(check_cmd: str,
               output_json_file: str,
               input_mesh_path: str,
               roughness_txt: str,
               metallic_txt: str,
               specular_txt: str):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for glb stat calculation command is %s....' %
          (str(start_time_str)))

    print("Start command %s: %s" % ("glb stat", check_cmd))
    exec_result = os.system(check_cmd)
    time.sleep(0.1)
    stat = exec_result

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print('After glb stat command status is %s; time for this status is %s....' %
          (str(stat), str(end_time_str)))

    if os.path.exists(output_json_file):
        glb_stat_struct = read_json(output_json_file)

        if glb_stat_struct["roughness"]["image"] or glb_stat_struct["roughness"]["value"]:
            with open(roughness_txt, 'a') as f:
                f.write('{}\n'.format(input_mesh_path))

        if glb_stat_struct["metallic"]["image"] or glb_stat_struct["metallic"]["value"]:
            with open(metallic_txt, 'a') as f:
                f.write('{}\n'.format(input_mesh_path))

        if glb_stat_struct["specular"]["image"] or glb_stat_struct["specular"]["value"]:
            with open(specular_txt, 'a') as f:
                f.write('{}\n'.format(specular_txt))


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


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--mesh_list_txt', type=str, default="",
                        help='mesh list txt full path')
    parser.add_argument('--mesh_info_json_folder', type=str, default="",
                        help='folder for storing mesh info json file')
    parser.add_argument('--output_json_path', type=str, default="",
                        help='json file of meshes with correct material')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender',
                        help='path for blender 3.6.2 version executable file')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')

    args = parser.parse_args()
    mesh_list_txt = args.mesh_list_txt
    mesh_info_json_folder = args.mesh_info_json_folder
    if not os.path.exists(mesh_info_json_folder):
        os.mkdir(mesh_info_json_folder)
    output_json_path = args.output_json_path
    blender_root = args.blender_root
    log_folder = args.log_folder

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    mesh_path_list = read_list(mesh_list_txt)
    stat_json_path_list = []
    glb_stat_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "check_glb_material.py")

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    cmds_txt = os.path.join(log_folder, 'cmds.txt')
    roughness_txt = os.path.join(log_folder, 'roughness.txt')
    specular_txt = os.path.join(log_folder, 'specular.txt')
    metallic_txt = os.path.join(log_folder, 'metallic.txt')

    cmds_file = open(cmds_txt, 'w')
    roughness_file = open(roughness_txt, 'w')
    specular_file = open(specular_txt, 'w')
    metallic_file = open(metallic_txt, 'w')

    for mesh_path in mesh_path_list:
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]
        mesh_extension = os.path.splitext(mesh_filename)[1]
        if mesh_extension != ".glb":
            mesh_path = mesh_path + ".glb"

        stat_json_path = os.path.join(
            mesh_info_json_folder, mesh_basename + ".json")

        glb_stat_cmd = "{} -b -P {} --".format(
            blender_root, glb_stat_fullpath)
        glb_stat_cmd = glb_stat_cmd + \
                       " --source_mesh_path \"{}\" ".format(mesh_path)
        glb_stat_cmd = glb_stat_cmd + \
                       " --output_json_path \"{}\" ".format(stat_json_path)

        print("Fix material in mesh %s " % (mesh_path))
        stat_json_path_list.append(stat_json_path)

        pool.apply_async(func=check_once, args=(
            glb_stat_cmd, stat_json_path, mesh_path, roughness_txt, metallic_txt, specular_txt))

        # check_once(glb_stat_cmd, stat_json_path, mesh_path,
        #            roughness_txt, metallic_txt, specular_txt)

        with open(cmds_txt, 'a') as f:
            f.write(glb_stat_cmd + '\n')

    pool.close()
    pool.join()

    roughness_info = read_list(roughness_txt)
    metallic_info = read_list(metallic_txt)
    specular_info = read_list(specular_txt)
    output_struct = {}
    output_struct["data"] = {}
    output_struct["data"]["objaverse"] = {}

    for mesh_path in roughness_info:
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]

        if mesh_basename not in output_struct["data"]["objaverse"].keys():
            output_struct["data"]["objaverse"][mesh_basename] = {}
            output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        else:
            if "Mesh" not in output_struct["data"]["objaverse"][mesh_basename].keys():
                output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        output_struct["data"]["objaverse"][mesh_basename]["Roughness"] = True

    for mesh_path in metallic_info:
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]

        if mesh_basename not in output_struct["data"]["objaverse"].keys():
            output_struct["data"]["objaverse"][mesh_basename] = {}
            output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        else:
            if "Mesh" not in output_struct["data"]["objaverse"][mesh_basename].keys():
                output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        output_struct["data"]["objaverse"][mesh_basename]["Metallic"] = True

    for mesh_path in specular_info:
        mesh_filename = os.path.split(mesh_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]

        if mesh_basename not in output_struct["data"]["objaverse"].keys():
            output_struct["data"]["objaverse"][mesh_basename] = {}
            output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        else:
            if "Mesh" not in output_struct["data"]["objaverse"][mesh_basename].keys():
                output_struct["data"]["objaverse"][mesh_basename]["Mesh"] = mesh_path
        output_struct["data"]["objaverse"][mesh_basename]["Specular"] = True

    write_json(output_json_path, output_struct)
