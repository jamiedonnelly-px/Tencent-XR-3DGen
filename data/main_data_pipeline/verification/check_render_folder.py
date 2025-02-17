import argparse
import json
import os
import time

import cv2


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
    file_name = "task.valid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("valid")


def write_invalid(path: str):
    file_name = "task.invalid"
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


def check_folder_filesize(folder_path: str, extension: str, read_image: bool = False):
    exr_counter = 0
    empty_counter = 0
    error_counter = 0
    if os.path.exists(folder_path):
        exr_files = os.listdir(folder_path)
        for exr_file in exr_files:
            if extension in exr_file:
                exr_file_abspath = os.path.join(folder_path, exr_file)
                file_size = os.path.getsize(exr_file_abspath)
                if file_size > 5000:
                    try:
                        if read_image:
                            print("Read image from %s for verification..." %
                                  (exr_file_abspath))
                            temp_image = cv2.imread(exr_file_abspath)
                            exr_counter = exr_counter + 1
                        else:
                            exr_counter = exr_counter + 1
                    except:
                        error_counter = error_counter + 1
                else:
                    empty_counter = empty_counter + 1
    return exr_counter, empty_counter, error_counter


def test_exr_file_number(real_render_folder: str, outside_camera_param: str = ""):
    time.sleep(0.1)
    if len(outside_camera_param) < 1 or not os.path.exists(outside_camera_param):
        camera_info_path = os.path.join(
            real_render_folder, "cam_parameters.json")
    else:
        camera_info_path = outside_camera_param
    if os.path.exists(camera_info_path):
        with open(camera_info_path, encoding='utf-8') as f:
            camera_info = json.load(f)
        camera_number = len(camera_info)
        exr_folder = os.path.join(real_render_folder, "exr")
        exr_number = check_file_number(exr_folder, ".exr")

        if camera_number == exr_number:
            return True
    return False


def test_dataset_integrity(real_render_folder: str,
                           outside_camera_param: str = "",
                           check_color_only: bool = False):
    if len(outside_camera_param) < 1 or not os.path.exists(outside_camera_param):
        camera_info_path = os.path.join(
            real_render_folder, "cam_parameters.json")
    else:
        camera_info_path = outside_camera_param
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
            depth_folder = os.path.join(real_render_folder, "depth")
            normal_folder = os.path.join(real_render_folder, "normal")
            color_number = check_file_number(color_folder, ".png")
            depth_number = check_file_number(depth_folder, ".png")
            normal_number = check_file_number(normal_folder, ".png")

            if camera_number == color_number and camera_number == depth_number and normal_number == camera_number:
                return True
    return False


def test_dataset_filesize(real_render_folder: str,
                          outside_camera_param: str = "",
                          read_verify: bool = False,
                          size_verify: bool = False,
                          check_color_only: bool = False):
    if len(outside_camera_param) < 1 or not os.path.exists(outside_camera_param):
        camera_info_path = os.path.join(
            real_render_folder, "cam_parameters.json")
    else:
        camera_info_path = outside_camera_param
    if os.path.exists(camera_info_path):
        with open(camera_info_path, encoding='utf-8') as f:
            camera_info = json.load(f)
        camera_number = len(camera_info)
        color_folder = os.path.join(real_render_folder, "color")
        depth_folder = os.path.join(real_render_folder, "depth")
        normal_folder = os.path.join(real_render_folder, "normal")
        if check_color_only:
            if size_verify:
                color_number, _, color_error = check_folder_filesize(
                    color_folder, ".png", read_image=read_verify)
                if color_error > 0:
                    return 2
                if camera_number == color_number:
                    return 0
            else:
                return 0
        else:
            if size_verify:
                color_number, _, color_error = check_folder_filesize(
                    color_folder, ".png", read_image=read_verify)
                depth_number, _, depth_error = check_folder_filesize(
                    depth_folder, ".png", read_image=read_verify)
                normal_number, _, normal_error = check_folder_filesize(
                    normal_folder, ".png", read_image=read_verify)
                if color_error > 0 or depth_error > 0 or normal_error > 0:
                    return 2

                if camera_number == color_number and camera_number == depth_number and normal_number == camera_number:
                    return 0
            else:
                return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--render_folder', type=str,
                        help='folder that stores these meshes')
    parser.add_argument('--outside_camera_param', type=str, default="",
                        help='file of camera parameters json')
    parser.add_argument('--read_verify', action='store_true',
                        help='verify image deeply by imread function')
    parser.add_argument('--size_verify', action='store_true',
                        help='verify image by checking image size')
    parser.add_argument('--check_color_only', action='store_true',
                        help='verify only rendered RGB images, do not verify depth images and normal images')
    args = parser.parse_args()

    render_folder = args.render_folder
    outside_camera_param = args.outside_camera_param

    filesize_test_result = test_dataset_filesize(
        render_folder,
        outside_camera_param=outside_camera_param,
        read_verify=args.read_verify,
        size_verify=args.size_verify,
        check_color_only=args.check_color_only)
    integrity_test_result = test_dataset_integrity(
        render_folder,
        outside_camera_param=outside_camera_param,
        check_color_only=args.check_color_only)

    if integrity_test_result and (filesize_test_result == 0):
        done_path = os.path.join(render_folder, "task.done")
        if not os.path.exists(done_path):
            write_done(render_folder)
        write_valid(render_folder)
