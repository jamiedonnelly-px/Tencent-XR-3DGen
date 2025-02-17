#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pose_common


def pose_generation(elevation_start: float, elevation_end: float,
                    fov_start: float, fov_end: float,
                    partition_camera_angle_list: list,
                    partition_camera_num_list: list,
                    image_percentage: float,
                    object_rotation: np.ndarray):
    """
    Generate camera parameters for RC mode.
    See https://doc.weixin.qq.com/doc/w3_AeAAvAbaAPMFXJULJsrSKeW22nDnZ for detailed information
    """
    azimuth_list = []
    elevation_list = []
    ortho_scale_list = []
    fov_list = []
    camera_config_list = []
    cam2worlds = []
    radius_list = []

    partition_number = len(partition_camera_angle_list)
    partition_elevation_angle = (elevation_end - elevation_start) / partition_number

    for index in range(partition_number):
        partition_length = partition_camera_num_list[index]
        p_azimuth_start = partition_camera_angle_list[index]["start"]
        p_azimuth_end = partition_camera_angle_list[index]["end"]
        p_elevation_start = elevation_start + index * partition_elevation_angle
        p_elevation_end = elevation_start + (index + 1) * partition_elevation_angle

        for _ in range(partition_length):
            perturb_azimuth = (p_azimuth_start - p_azimuth_end) * np.random.rand() + p_azimuth_end
            perturb_elevation = (elevation_start - elevation_end) * np.random.rand() + elevation_end
            if perturb_azimuth < 0:
                perturb_azimuth = perturb_azimuth + 360.0
            random_fov = (fov_start - fov_end) * np.random.rand() + fov_end
            fov_1d_array = np.array([random_fov])
            radius = pose_common.get_radius(fov_1d_array, image_percentage)
            radius_list.append(radius[0])

            azimuth_list.append(perturb_azimuth)
            elevation_list.append(perturb_elevation)
            ortho_scale_list.append(-1.0)
            camera_config_list.append("persp")
            fov_list.append(random_fov)

    azimuth_array = np.array(azimuth_list)
    elevation_array = np.array(elevation_list)
    radius_array = np.array(radius_list)

    cam2world = pose_common.generate_cam2world(radius_array, azimuth_array, elevation_array)

    cam2worlds.extend(object_rotation @ cam2world)

    return cam2worlds, azimuth_list, elevation_list, radius_list, fov_list, ortho_scale_list, camera_config_list


def final_data_generation(index: int,
                          cam_name: str,
                          result_c2w: list,
                          res_fov_list: list,
                          res_ortho_scale_list: list,
                          image_size: list,
                          pose_data: dict):
    """
    Write camera intrinsic and extrisic parameters to pose_data struct
    :param index: index of camera
    :param cam_name: camera name
    :param result_c2w: a list of camera pose in world coordinate
    :param res_fov_list: a list of camera field of view
    :param res_ortho_scale_list: a list of camera orthographic scale
    :param image_size: image size, one side length of a square image
    :param pose_data: output dictionary to store camera poses
    """
    scale = res_ortho_scale_list[index]
    fov = res_fov_list[index]
    pose = result_c2w[index]
    pose_blender = pose_common.opencv_to_blender(pose)
    pose_data[cam_name] = {
        "k": None,
        "scale": None,
        "pose": pose_blender.tolist(),
        "model": None
    }

    if scale < 0:
        pose_data[cam_name]["scale"] = None
        pose_data[cam_name]["model"] = "perspective"
    else:
        pose_data[cam_name]["scale"] = scale
        pose_data[cam_name]["model"] = "orthographic"

    if fov < 0:
        pose_data[cam_name]["k"] = None
        pose_data[cam_name]["model"] = "orthographic"
    else:
        K_array = pose_common.get_intrinsic_matrix(fov, image_size).numpy()
        pose_data[cam_name]["k"] = K_array.tolist()
        pose_data[cam_name]["model"] = "perspective"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RC camera generation script.')
    parser.add_argument("--output_folder", type=str,
                        help="output directory to save cameras to")
    parser.add_argument("--config_json", type=str,
                        help="config json file path")
    parser.add_argument("--image_size_x", type=int, default=512,
                        help="rendered image size on x axis")
    parser.add_argument("--image_size_y", type=int, default=512,
                        help="rendered image size on y axis")
    all_args = parser.parse_args()

    output_folder = all_args.output_folder
    config_json = all_args.config_json
    image_size_x = all_args.image_size_x
    image_size_y = all_args.image_size_y
    image_size = [image_size_x, image_size_y]

    all_camera_generation_config = pose_common.read_json(config_json)
    random_config = all_camera_generation_config["RC"]

    camera_parameters_data = {}
    camera_parameters_data["config"] = {}
    camera_parameters_data["config"]["RC"] = random_config
    pose_data = {}
    camera_parameters_data["poses"] = pose_data
    final_aux_camera_pose = []

    total_index = 0

    ortho_scale = random_config["ortho_scale"]
    ortho_radius = random_config["ortho_radius"]
    image_percentage = random_config["image_percentage"]
    partition_angle_list = random_config["partition_angle_list"]
    partition_camera_number_list = random_config["partition_camera_number_list"]

    condition_partition_number = random_config["condition_partition"]
    condition_fov_start = random_config["condition_fov_range"]["start"]
    condition_fov_end = random_config["condition_fov_range"]["end"]
    condition_elevation_start = random_config["condition_elevation_range"]["start"]
    condition_elevation_end = random_config["condition_elevation_range"]["end"]
    condition_azimuth_start = random_config["condition_azimuth_range"]["start"]
    condition_azimuth_end = random_config["condition_azimuth_range"]["end"]

    azimuth_list = random_config["azimuth_list"]
    elevation_list = random_config["elevation_list"]

    object_rotation = pose_common.gen_rotation_blender()

    total_azimuth_list = copy.deepcopy(azimuth_list)
    total_elevation_list = copy.deepcopy(elevation_list)
    total_radius_list = [ortho_radius] * len(total_azimuth_list)
    total_fov_list = [-1.0] * len(total_azimuth_list)
    total_ortho_scale_list = [ortho_scale] * len(total_azimuth_list)
    total_camera_config_list = ["ortho"] * len(total_azimuth_list)

    result_tuple_condition = pose_generation(elevation_start=condition_elevation_start,
                                             elevation_end=condition_elevation_end,
                                             fov_start=condition_fov_start,
                                             fov_end=condition_fov_end,
                                             partition_camera_angle_list=partition_angle_list,
                                             partition_camera_num_list=partition_camera_number_list,
                                             image_percentage=image_percentage,
                                             object_rotation=object_rotation)

    condition_camera_pose = result_tuple_condition[0]
    condition_azimuth_list = result_tuple_condition[1]
    condition_elevation_list = result_tuple_condition[2]
    condition_radius_list = result_tuple_condition[3]
    condition_fov_list = result_tuple_condition[4]
    condition_ortho_scale_list = result_tuple_condition[5]
    condition_camera_config_list = result_tuple_condition[6]

    total_azimuth_list.extend(condition_azimuth_list)
    total_elevation_list.extend(condition_elevation_list)
    total_radius_list.extend(condition_radius_list)
    total_fov_list.extend(condition_fov_list)
    total_ortho_scale_list.extend(condition_ortho_scale_list)

    aux_azimuth_array = np.array(azimuth_list)
    aux_elevation_array = np.array(elevation_list)
    ortho_radius_list = [ortho_radius] * len(azimuth_list)
    aux_radius_array = np.array(ortho_radius_list)
    aux_camera_pose = pose_common.generate_cam2world(aux_radius_array, aux_azimuth_array, aux_elevation_array)
    final_aux_camera_pose.extend(object_rotation @ aux_camera_pose)

    all_camera_pose = final_aux_camera_pose + condition_camera_pose
    total_camera_number = len(all_camera_pose)

    object_scale_list = [ortho_scale] * total_camera_number
    cam_names = ["cam-%04d" % i for i in range(total_camera_number)]

    for index in range(total_camera_number):
        final_data_generation(index=index,
                              cam_name=cam_names[total_index],
                              result_c2w=all_camera_pose,
                              res_fov_list=total_fov_list,
                              res_ortho_scale_list=total_ortho_scale_list,
                              image_size=image_size,
                              pose_data=pose_data)
        total_index = total_index + 1

    random_config["real_azimuth"] = total_azimuth_list
    random_config["real_elevation"] = total_elevation_list
    random_config["real_fov"] = total_fov_list

    logging.info("Camera number is %i........" % (total_camera_number))

    camera_parameter_filename = "internal_cam_parameters.json"
    json_output_path = os.path.join(output_folder, camera_parameter_filename)
    pose_common.write_json(json_output_path, camera_parameters_data)
