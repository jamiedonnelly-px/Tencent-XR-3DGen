#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shlex
import subprocess
import unittest

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')


class TestDatasetGeneration(unittest.TestCase):
    def __init__(self, test_name='test_sdf_sampler'):
        super(TestDatasetGeneration, self).__init__(test_name)

        self.sdf_sampler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "../../geometry/sdf_sample.py")
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/datas/sdf_op_data")
        self.result_folder = os.path.join(self.data_folder, "result")
        self.mesh_file = os.path.join(self.data_folder, "manifold.obj")
        self.config_file = os.path.join(self.data_folder, "config.json")

    def test_sdf_sampler(self):
        sample_point_number = 500000
        output_trans_txt = os.path.join(self.result_folder, "transformation.txt")
        output_z_txt = os.path.join(self.result_folder, "z.txt")

        # step 1: build sdf sample scripts
        sdf_cmd = "python {}".format(self.sdf_sampler_path)
        sdf_cmd = "{} --mesh_path \"{}\" ".format(sdf_cmd, self.mesh_file)
        sdf_cmd = "{} --output_folder \"{}\" ".format(sdf_cmd, self.result_folder)
        sdf_cmd = "{} --transform_path \"{}\" ".format(sdf_cmd, output_trans_txt)
        sdf_cmd = "{} --z_transform_path \"{}\" ".format(sdf_cmd, output_z_txt)
        sdf_cmd = "{} --standard_height 1.92 ".format(sdf_cmd)
        sdf_cmd = "{} --sample_format 'h5_chunk' --chunk_size 4096 --shuffle ".format(sdf_cmd)
        sdf_cmd = "{} --space_sample_number {} ".format(sdf_cmd, sample_point_number)
        sdf_cmd = "{} --near_surface_sample_number {} ".format(sdf_cmd, sample_point_number)
        sdf_cmd = "{} --surface_sample_number {} ".format(sdf_cmd, sample_point_number)
        cmd_elements = shlex.split(sdf_cmd)
        subprocess.run(cmd_elements, check=True, text=True)

        geometry_folder = os.path.join(self.result_folder, "geometry")
        geometry_sample_h5 = os.path.join(geometry_folder, "sample.h5")
        self.assertTrue(os.path.exists(geometry_sample_h5))
        with h5py.File(geometry_sample_h5, 'r') as f:
            for data_name in f.keys():
                npy_data = np.array(f[data_name])
                self.assertEqual(npy_data.shape[0], sample_point_number)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    test_case = TestDatasetGeneration()
    suite.addTest(test_case)

    # Run the test here
    runner = unittest.TextTestRunner()
    runner.run(suite)
