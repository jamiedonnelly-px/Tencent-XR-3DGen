import argparse
import json
import os
import time

import cv2
import numpy as np


def remove_exr(render_folder):
    exr_folder = os.path.join(render_folder, "exr")
    exr_files = os.listdir(exr_folder)
    for exr_file in exr_files:
        exr_filename = os.path.join(exr_folder, exr_file)
        os.remove(exr_filename)


def write_done(path: str):
    file_name = "task.done"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("done")


def write_undone(path: str):
    file_name = "task.undone"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("undone")


def write_valid(path: str):
    file_name = "mesh.valid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("valid")


def write_fulfill(path: str):
    file_name = "mesh.fulfill"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("fulfill")


def write_invalid(path: str):
    file_name = "mesh.invalid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("invalid")


def check_file_number(folder_path: str, extension: str):
    exr_counter = 0
    if os.path.exists(folder_path):
        exr_files = os.listdir(folder_path)
        for exr_file in exr_files:
            if extension in exr_file:
                exr_counter = exr_counter + 1

    return exr_counter


def random_some_image(camera_info, camera_number: int, random_number: int):
    camera_name_list = list(camera_info.keys())
    random_result_list = []
    camera_no_list = np.random.randint(
        0, camera_number, size=(random_number), dtype='int').tolist()
    for camera_no in camera_no_list:
        camera_str = camera_name_list[camera_no]
        camera_str = camera_str + ".png"
        random_result_list.append(camera_str)
    return random_result_list


def check_view_normal_valid(camera_info,
                            folder_path: str,
                            camera_number: int,
                            random_number: int,
                            read_image: bool = False,
                            aux_data_format: str = 'png'):
    max_empty_ratio = 0.99
    max_negative_z_ratio = 0.01
    correct_counter = 0
    error_counter = 0
    normal_folder = os.path.join(folder_path, "view_normal")

    AUX_DATA_SCALE = 125.0
    if aux_data_format == 'png_16bit':
        AUX_DATA_SCALE = 30000.0

    if not os.path.exists(normal_folder):
        error_counter = 8
        return correct_counter, error_counter

    image_files = random_some_image(camera_info, camera_number, random_number)
    for image_file in image_files:
        image_file_abspath = os.path.join(normal_folder, image_file)
        file_size = os.path.getsize(image_file_abspath)
        if file_size > 3000:
            if read_image:
                print("Read view normal from %s for verification..." % (image_file_abspath))
                temp_image_cv = cv2.imread(image_file_abspath, cv2.IMREAD_UNCHANGED)
                temp_image_float = np.array(temp_image_cv).astype(np.float32)
                normals_view = temp_image_float / AUX_DATA_SCALE - 1
                norms = np.linalg.norm(normals_view, axis=-1, ord=2)  # [h,w]
                z = normals_view[..., 2]  # [h,w]

                empty_normal_ratio = (norms < 0.5).sum() / norms.size
                # we treat values < -0.02 as negative
                negative_z_ratio = ((z < -0.02) & (norms >= 0.5)).sum() / ((norms >= 0.5).sum() + 0.001)
                # or min_z_normas < -0.05:
                if empty_normal_ratio > max_empty_ratio or negative_z_ratio > max_negative_z_ratio:
                    error_counter = error_counter + 1
                else:
                    correct_counter = correct_counter + 1
            else:
                correct_counter = correct_counter + 1
        else:
            error_counter = error_counter + 1
    return correct_counter, error_counter


def check_folder_filesize(camera_info, folder_path: str, camera_number: int,
                          random_number: int, read_image: bool = False):
    correct_counter = 0
    error_counter = 0
    if os.path.exists(folder_path):
        exr_files = random_some_image(camera_info, camera_number, random_number)
        for exr_file in exr_files:
            exr_file_abspath = os.path.join(folder_path, exr_file)
            file_size = os.path.getsize(exr_file_abspath)
            if file_size > 3000:
                try:
                    if read_image:
                        print("Read image from %s for verification..." % (exr_file_abspath))
                        temp_image = cv2.imread(exr_file_abspath, cv2.IMREAD_UNCHANGED)
                        temp_image_shape = temp_image.shape
                        pixel_number = temp_image_shape[0] * temp_image_shape[1]
                        rgb_number = np.count_nonzero(temp_image[:, :, 3])

                        if rgb_number > int(0.01 * pixel_number):
                            correct_counter = correct_counter + 1
                        else:
                            error_counter = error_counter + 1
                    else:
                        correct_counter = correct_counter + 1
                except:
                    error_counter = error_counter + 1
            else:
                error_counter = error_counter + 1
    return correct_counter, error_counter


def get_image_number(camera_info_path: str):
    if os.path.exists(camera_info_path):
        with open(camera_info_path, encoding='utf-8') as f:
            camera_info = json.load(f)
        camera_number = len(camera_info)
        return camera_number
    return 0


def test_dataset_integrity(real_render_folder: str,
                           camera_parameters_json: str = "",
                           check_color_only: bool = False):
    if len(camera_parameters_json) < 1 or not os.path.exists(camera_parameters_json):
        camera_info_path = os.path.join(real_render_folder, "cam_parameters.json")
    else:
        camera_info_path = camera_parameters_json

    if os.path.exists(camera_info_path):
        with open(camera_info_path, encoding='utf-8') as f:
            camera_info = json.load(f)
        camera_number = len(camera_info)
        if check_color_only:
            color_folder = os.path.join(real_render_folder, "color")
            color_number = check_file_number(color_folder, ".png")
            if camera_number == color_number:
                return True
        else:
            color_folder = os.path.join(real_render_folder, "color")
            color_number = check_file_number(color_folder, ".png")

            depth_folder = os.path.join(real_render_folder, "depth")
            normal_folder = os.path.join(real_render_folder, "normal")
            xyz_folder = os.path.join(real_render_folder, "xyz")
            depth_number = check_file_number(depth_folder, ".png")
            normal_number = check_file_number(normal_folder, ".png")
            xyz_number = check_file_number(xyz_folder, ".png")

            file_number_set = {color_number, depth_number, normal_number, xyz_number}
            if len(file_number_set) == 1 and color_number == camera_number:
                return True

    return False


def test_dataset_filesize(real_render_folder: str,
                          camera_parameters_json: str = "",
                          verify_color_only: bool = False,
                          read_verify: bool = False,
                          random_number: int = -1,
                          aux_data_format: str = 'png'):
    if len(camera_parameters_json) < 1 or not os.path.exists(camera_parameters_json):
        camera_info_path = os.path.join(real_render_folder, "cam_parameters.json")
    else:
        camera_info_path = camera_parameters_json

    if os.path.exists(camera_info_path):
        with open(camera_info_path, encoding='utf-8') as f:
            camera_info = json.load(f)
        camera_number = len(camera_info)
        if random_number <= 0:
            random_number = camera_number
        color_folder = os.path.join(real_render_folder, "color")
        color_number, color_error = check_folder_filesize(camera_info,
                                                          color_folder,
                                                          camera_number,
                                                          random_number,
                                                          read_image=read_verify)
        print("color number is %i, error number is %i" %
              (color_number, color_error))

        if not verify_color_only:
            normal_number, normal_error = check_view_normal_valid(camera_info,
                                                                  real_render_folder,
                                                                  camera_number,
                                                                  random_number,
                                                                  read_image=read_verify,
                                                                  aux_data_format=aux_data_format)
            print("normal number is %i, error number is %i" %
                  (normal_number, normal_error))

        if color_error <= 1 and color_number >= (random_number - 1):
            if not verify_color_only:
                if normal_error <= 1 and normal_number >= (random_number - 1):
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            return 1

    return 1


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Render folder verification starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--render_folder', type=str,
                        help='folder that stores these meshes')
    parser.add_argument('--camera_parameters_json', type=str, default="",
                        help='json file containing camera parameters')
    parser.add_argument('--render_config_json', type=str, default="",
                        help='json file containing camera parameters')
    parser.add_argument('--read_verify', action='store_true',
                        help='verify image deeply by imread function')
    parser.add_argument('--size_verify', action='store_true',
                        help='verify image by checking image size')
    parser.add_argument('--check_color_only', action='store_true',
                        help='verify only rendered RGB images, do not verify depth images and normal images')
    parser.add_argument('--random_number', type=int, default=-1,
                        help='number of random')
    args = parser.parse_args()
    render_folder = args.render_folder
    camera_parameters_json = args.camera_parameters_json
    render_config_json = args.render_config_json
    random_number = args.random_number

    with open(render_config_json, encoding='utf-8') as f:
        render_config_struct = json.load(f)

    aux_data_format = render_config_struct["aux_format"]

    valid_filename = os.path.join(render_folder, "mesh.valid")
    if not os.path.exists(valid_filename):
        integrity_result = test_dataset_integrity(render_folder,
                                                  camera_parameters_json=camera_parameters_json,
                                                  check_color_only=args.check_color_only)
        if integrity_result:
            filesize_test_result = test_dataset_filesize(render_folder,
                                                         camera_parameters_json=camera_parameters_json,
                                                         verify_color_only=args.check_color_only,
                                                         read_verify=args.read_verify,
                                                         random_number=random_number,
                                                         aux_data_format=aux_data_format)
            if filesize_test_result == 0:
                write_valid(render_folder)
