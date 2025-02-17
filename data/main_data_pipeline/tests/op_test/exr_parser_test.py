#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shlex
import subprocess
import unittest

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')


class TestDatasetGeneration(unittest.TestCase):
    def __init__(self, test_name='test_exr_parser'):
        super(TestDatasetGeneration, self).__init__(test_name)

        self.exr_parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../exr_parser.py")
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/datas/exr_op_data")
        self.test_exr_folder = os.path.join(self.data_folder, "test")
        self.exr_file = os.path.join(self.test_exr_folder, "exr/cam-0000.exr")
        self.gt_folder = os.path.join(self.data_folder, "gt")
        self.camera_parameter_file = os.path.join(self.test_exr_folder, "cam_parameters.json")

    def __compare_two_images(self, image_path1: str, image_path2: str):
        SMALL_DISTANCE = 0.00001
        temp_image_1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
        temp_image_1_float = np.array(temp_image_1).astype(np.float32)
        temp_image_2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)
        temp_image_2_float = np.array(temp_image_2).astype(np.float32)
        image_diff = temp_image_1_float - temp_image_2_float
        image_diff_abs = np.abs(image_diff)
        image_diff_sum = np.sum(image_diff_abs)
        return image_diff_sum < SMALL_DISTANCE

    def __test_one_type_data(self, data_name: str, exr_basename: str):
        color_result_folder = os.path.join(self.test_exr_folder, data_name)
        color_gt_folder = os.path.join(self.gt_folder, data_name)
        self.assertTrue(os.path.exists(color_result_folder))

        color_result_file = os.path.join(color_result_folder, exr_basename + ".png")
        color_gt_file = os.path.join(color_gt_folder, exr_basename + ".png")
        self.assertTrue(os.path.exists(color_result_file))

        compare_result = self.__compare_two_images(color_result_file, color_gt_file)
        self.assertTrue(compare_result)

    def test_exr_parser(self):
        # step 1: build the cmd
        exr_cmd = "python {} --exr_file \"{}\" --camera_info_path \"{}\"".format(self.exr_parser_path,
                                                                                 self.exr_file,
                                                                                 self.camera_parameter_file)
        exr_filename = os.path.split(self.exr_file)[1]
        exr_basename = os.path.splitext(exr_filename)[0]
        cmd_elements = shlex.split(exr_cmd)
        subprocess.run(cmd_elements, check=True, text=True)

        self.__test_one_type_data("color", exr_basename)
        self.__test_one_type_data("depth", exr_basename)
        self.__test_one_type_data("normal", exr_basename)
        self.__test_one_type_data("xyz", exr_basename)
        self.__test_one_type_data("view_normal", exr_basename)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    test_case = TestDatasetGeneration()
    suite.addTest(test_case)

    # Run the test here
    runner = unittest.TextTestRunner()
    runner.run(suite)
