#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import shlex
import subprocess
import time
import unittest

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')


def current_time():
    """
    Get current time string.
    :return: current time string
    """
    t_current = time.time()
    local_time = time.localtime(t_current)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    return local_time_str


def read_list(in_list_txt: str):
    """
    Read a list of contents from a txt file.
    :param in_list_txt: path to the list txt file
    :return: contents in the list
    """
    str_list = []
    if not os.path.exists(in_list_txt):
        logging.error('Cannot find input list txt file ', in_list_txt)
        return str_list
    try:
        with open(in_list_txt, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                one_line_content = line.strip()
                if len(one_line_content) > 1:
                    str_list.append(one_line_content)
    except (IOError, FileNotFoundError) as e:
        logging.error("Cannot read list file %s" % in_list_txt)
    return str_list


def read_json(json_path: str):
    """
    Read a json file to a json struct.
    :param json_path: path of the json file
    :return: result json struct
    """
    try:
        with open(json_path, encoding='utf-8') as f:
            json_struct = json.load(f)
            return json_struct
    except (IOError, FileNotFoundError) as e:
        logging.error("Cannot read json file from %s" % json_path)


class TestDatasetGeneration(unittest.TestCase):
    def __init__(self, test_name='test_render', input_data_folder="datas/render_op_data"):
        super(TestDatasetGeneration, self).__init__(test_name)

        self.render_mesh_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "../../render_color_depth_normal_helper.py")
        self.data_folder = input_data_folder
        self.path_config_json = os.path.join(os.path.abspath(self.data_folder), "path_config.json")

    def __check_file_number(self, folder_path: str, extension: str):
        exr_counter = 0
        if os.path.exists(folder_path):
            exr_files = os.listdir(folder_path)
            for exr_file in exr_files:
                if extension in exr_file:
                    exr_counter = exr_counter + 1

        return exr_counter

    def test_render(self):
        # step 1: read path configuration file and prepare folders
        path_config_struct = read_json(self.path_config_json)
        output_folder = path_config_struct["output_folder"]
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        transform_path = os.path.join(output_folder, "z.txt")

        # step 2: run the render script
        render_cmd = "{} -b -P {} ".format(path_config_struct["blender_root"], self.render_mesh_script_path)
        render_cmd = "{} -- --mesh_path \'{}\' ".format(render_cmd, path_config_struct["mesh_path"])
        render_cmd = "{} --transform_path \'{}\' ".format(render_cmd, transform_path)
        render_cmd = "{} --pose_json_path \'{}\' ".format(render_cmd, path_config_struct["pose_config"])
        render_cmd = "{} --camera_config_path \'{}\' ".format(render_cmd, path_config_struct["render_config"])
        render_cmd = "{} --output_folder \'{}\' ".format(render_cmd, output_folder)
        render_cmd = "{} --aux_image_type \"png_16bit\"  --engine \"eevee_next\" ".format(render_cmd)
        render_cmd = "{} --render_height 768 --render_width 768 ".format(render_cmd)
        render_cmd = "{} --parse_exr  --smooth".format(render_cmd)
        cmd_elements = shlex.split(render_cmd)
        subprocess.run(cmd_elements, check=True, text=True)

        camera_info_path = os.path.join(output_folder, "cam_parameters.json")
        self.assertTrue(os.path.exists(camera_info_path), f"Camera parameters at {camera_info_path}")
        camera_info = read_json(camera_info_path)
        camera_number = len(camera_info)

        color_folder = os.path.join(output_folder, "color")
        color_number = self.__check_file_number(color_folder, ".png")
        self.assertEqual(camera_number, color_number, "Color number")

        depth_folder = os.path.join(output_folder, "depth")
        depth_number = self.__check_file_number(depth_folder, ".png")
        self.assertEqual(camera_number, depth_number, "Depth number")

        normal_folder = os.path.join(output_folder, "normal")
        normal_number = self.__check_file_number(normal_folder, ".png")
        self.assertEqual(camera_number, normal_number, "Normal number")

        xyz_folder = os.path.join(output_folder, "xyz")
        xyz_number = self.__check_file_number(xyz_folder, ".png")
        self.assertEqual(camera_number, xyz_number, "XYZ number")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dataset generation unittest.')
    parser.add_argument('--input_data_folder',
                        type=str,
                        default="datas/render_op_data",
                        help='Test data folder')
    args = parser.parse_args()

    suite = unittest.TestSuite()
    test_case = TestDatasetGeneration(input_data_folder=args.input_data_folder)
    suite.addTest(test_case)

    # Run the test here
    runner = unittest.TextTestRunner()
    runner.run(suite)
