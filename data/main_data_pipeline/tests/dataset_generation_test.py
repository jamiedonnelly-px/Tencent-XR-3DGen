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

import h5py
import numpy as np

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
    def __init__(self, test_name='test_dataset_generation', input_data_folder="datas/dataset_generation"):
        super(TestDatasetGeneration, self).__init__(test_name)

        self.dataset_generation_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "../render_mesh_batch.py")
        self.data_folder = input_data_folder
        self.path_config_json = os.path.join(os.path.abspath(self.data_folder), "path_config.json")

    def __check_sample_h5(self, sample_folder: str, sample_number_struct: dict = None):
        geometry_folder = os.path.join(sample_folder, "geometry")
        geometry_sample_h5 = os.path.join(geometry_folder, "sample.h5")
        if not os.path.exists(geometry_sample_h5):
            return False
        with h5py.File(geometry_sample_h5, 'r') as f:
            for data_name in f.keys():
                npy_data = np.array(f[data_name])
                if sample_number_struct is None:
                    if npy_data.shape[0] < 100000:
                        return False
                else:
                    data_position = data_name.rsplit('_', 1)[0]
                    if data_position not in sample_number_struct.keys():
                        return False
                    sample_number = sample_number_struct[data_position]
                    if npy_data.shape[0] != sample_number:
                        return False
        return True

    def __check_file_number(self, folder_path: str, extension: str):
        exr_counter = 0
        if os.path.exists(folder_path):
            exr_files = os.listdir(folder_path)
            for exr_file in exr_files:
                if extension in exr_file:
                    exr_counter = exr_counter + 1

        return exr_counter

    def __check_dataset_integrity(self,
                                  camera_info_path: str,
                                  real_render_folder: str,
                                  check_color_only: bool = True):
        self.assertTrue(os.path.exists(camera_info_path), f"Camera parameters at {camera_info_path}")
        camera_info = read_json(camera_info_path)
        camera_number = len(camera_info)

        color_folder = os.path.join(real_render_folder, "color")
        color_number = self.__check_file_number(color_folder, ".png")
        self.assertEqual(camera_number, color_number, "Color number")

        if not check_color_only:
            depth_folder = os.path.join(real_render_folder, "depth")
            depth_number = self.__check_file_number(depth_folder, ".png")
            self.assertEqual(camera_number, depth_number, "Depth number")

            normal_folder = os.path.join(real_render_folder, "normal")
            normal_number = self.__check_file_number(normal_folder, ".png")
            self.assertEqual(camera_number, normal_number, "Normal number")

            xyz_folder = os.path.join(real_render_folder, "xyz")
            xyz_number = self.__check_file_number(xyz_folder, ".png")
            self.assertEqual(camera_number, xyz_number, "XYZ number")

    def test_dataset_generation(self):
        # step 1: read path configuration file and prepare folders
        current_time_str = current_time()
        path_config_struct = read_json(self.path_config_json)
        output_folder = path_config_struct["output_folder"]
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        current_run_folder = os.path.join(output_folder, "test_dataset_{}".format(current_time_str))
        if not os.path.exists(current_run_folder):
            os.mkdir(current_run_folder)

        render_data_folder = os.path.join(current_run_folder, "render_data")
        if not os.path.exists(render_data_folder):
            os.mkdir(render_data_folder)

        proc_data_folder = os.path.join(current_run_folder, "proc_data")
        if not os.path.exists(proc_data_folder):
            os.mkdir(proc_data_folder)

        log_folder = os.path.join(current_run_folder, "log")
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        # step 2: run the dataset generation script
        dataset_cmd = "python {} ".format(self.dataset_generation_script_path)
        dataset_cmd = dataset_cmd + " --data_json_path \"{}\" ".format(path_config_struct["test_data_path"])
        dataset_cmd = dataset_cmd + " --output_folder \"{}\" ".format(render_data_folder)
        dataset_cmd = dataset_cmd + " --log_folder \"{}\" ".format(log_folder)
        dataset_cmd = dataset_cmd + " --proc_data_output_folder \"{}\" ".format(proc_data_folder)
        dataset_cmd = dataset_cmd + " --blender_root \"{}\" ".format(path_config_struct["blender_root"])
        dataset_cmd = dataset_cmd + " --config_json_path  \"{}\" ".format(path_config_struct["render_config"])
        dataset_cmd = dataset_cmd + " --generate_pose_config_json_path  \"{}\" ".format(
            path_config_struct["pose_generation_config"])
        dataset_cmd = dataset_cmd + " --pose_generation_mode \"{}\" ".format(path_config_struct["pose_generation_mode"])
        dataset_cmd = dataset_cmd + " --pool_cnt 12 --silent --parse_exr --apply_render "
        dataset_cmd = dataset_cmd + " --apply_preprocess_mesh --preprocess_scale_mesh --force_sample_on_before "
        dataset_cmd = dataset_cmd + " --render_stage_string {}".format(path_config_struct["stages"])
        cmd_elements = shlex.split(dataset_cmd)
        subprocess.run(cmd_elements, check=True, text=True)

        # step 3: verify render results
        folder_txt = os.path.join(log_folder, "folder.txt")
        proc_txt = os.path.join(log_folder, "proc.txt")
        self.assertTrue(os.path.exists(folder_txt))
        self.assertTrue(os.path.exists(proc_txt))

        folder_contents = read_list(folder_txt)
        proc_contents = read_list(proc_txt)
        self.assertEqual(len(folder_contents), len(proc_contents))
        for index in range(len(folder_contents)):
            render_folder_path = folder_contents[index]
            sample_folder_path = proc_contents[index]
            logging.info(f"Check render folder {render_folder_path} and sample folder {sample_folder_path}")
            render_config_json = os.path.join(render_folder_path, "config.json")
            config_json_struct = read_json(render_config_json)
            sampled_number_struct = {}
            sampled_number_struct["space"] = config_json_struct["geometry_space_sample_number"]
            sampled_number_struct["near_surface"] = config_json_struct["geometry_near_surface_sample_number"]
            sampled_number_struct["surface"] = config_json_struct["geometry_surface_sample_number"]

            check_sample_flag = self.__check_sample_h5(sample_folder_path, sampled_number_struct)
            self.assertTrue(check_sample_flag)

            stage_list = path_config_struct["stages"].split("+")
            camera_info_path = os.path.join(config_json_struct["stages"]["common"], "cam_parameters.json")
            for stage_name in stage_list:
                if stage_name == "common":
                    self.__check_dataset_integrity(camera_info_path,
                                                   config_json_struct["stages"][stage_name],
                                                   False)
                else:
                    self.__check_dataset_integrity(camera_info_path,
                                                   config_json_struct["stages"][stage_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dataset generation unittest.')
    parser.add_argument('--input_data_folder', type=str, help='Test data folder')
    args = parser.parse_args()

    suite = unittest.TestSuite()
    test_case = TestDatasetGeneration(input_data_folder=args.input_data_folder)
    suite.addTest(test_case)

    # Run the test here
    runner = unittest.TextTestRunner()
    runner.run(suite)
