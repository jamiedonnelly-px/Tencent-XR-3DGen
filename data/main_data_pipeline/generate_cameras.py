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


def write_list(path: str, write_list: list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Pose generation during render start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str,
                        help="output directory to save cameras to")
    parser.add_argument("--config_json", type=str,
                        help="config json file path")
    parser.add_argument("--image_size_x", type=int, default=512,
                        help="rendered image size on x axis")
    parser.add_argument("--image_size_y", type=int, default=512,
                        help="rendered image size on y axis")
    parser.add_argument("--mode", type=str, default="V",
                        help="mode of camera generation, choose between V/S/RIGID/RIGID_RANDOM")

    all_args = parser.parse_args()

    HEAD_VAR = 0.8

    output_folder = all_args.output_folder
    config_json = all_args.config_json
    image_size_x = all_args.image_size_x
    image_size_y = all_args.image_size_y
    mode = all_args.mode

    config_struct = read_json(config_json)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if mode == "ORTHO":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose/ortho_camera.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)

    elif mode == 'S':
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose/sds_random.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)

    elif mode == "SDS_RANDOM":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose/sds_random.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)
    elif mode == "RSVC":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "pose/random_single_view_condition.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)
    elif mode == "Material_Transfer":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "pose/material_transfer.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)

    elif mode == "RTriVC":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "pose/random_three_view_condition.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)

    elif mode == "RC":
        pose_generator_fullpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "pose/random_condition_perspective.py")
        pose_cmd = "python \'{}\' ".format(pose_generator_fullpath)
        pose_cmd = pose_cmd + " --output_folder \'{}\' ".format(output_folder)
        pose_cmd = pose_cmd + " --config_json \'{}\' ".format(config_json)
        pose_cmd = pose_cmd + " --image_size_x \'{}\' ".format(image_size_x)
        pose_cmd = pose_cmd + " --image_size_y \'{}\' ".format(image_size_y)

        print(pose_cmd)
        os.system(pose_cmd)
        time.sleep(0.1)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Pose generation during render end. Start time is %s; end time is %s" % (local_time_str, end_time_str))
