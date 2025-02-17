import argparse
import copy
import json
import os
import time

import numpy as np
import scipy.spatial
import torch


def euler_to_rotation_matrix(euler_angles):
    r = scipy.spatial.transform.Rotation.from_euler(
        'xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


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


def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0),
                       (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)  # T * origin


def get_intrinsic_matrix(fov: float, image_size: list):
    """
    Calculate the intrinsic camera matrix for a square image.

    :param fov: Field of view in degrees.
    :param image_size: Width or height of the square image in pixels.
    :return: 3x3 numpy array representing the intrinsic camera matrix.
    """
    # Calculate the focal length in pixels
    focal_length_px = image_size[0] / (2 * np.tan(np.radians(fov) / 2))

    # The principal point is typically at the center of the image
    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0

    # Constructing the intrinsic matrix
    intrinsic_matrix = np.array([[focal_length_px, 0, cx],
                                 [0, focal_length_px, cy],
                                 [0, 0, 1]])
    intrinsic_tensor = torch.from_numpy(intrinsic_matrix)
    return intrinsic_tensor


def angles_to_unit_vectors(azimuth, elevation):
    """
    Converts azimuth and elevation angles to unit vector coordinates.

    Parameters:
    azimuth (numpy.ndarray): Array of azimuth angles in degrees. Shape: (n,)
    elevation (numpy.ndarray): Array of elevation angles in degrees. Shape: (n,)

    Returns:
    numpy.ndarray: Array of unit vector coordinates corresponding to the angles. Shape: (n, 3)
    """
    # Convert angles from degrees to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    # Calculate Cartesian coordinates
    x = np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = np.sin(elevation_rad)

    # Combine into nx3 array
    vectors = np.stack((x, y, z), axis=-1)

    return vectors


def look_at_opencv(camera_positions, target=np.array([0, 0, 0]), world_up=np.array([0, 0, 1])):
    """
    Generates camera-to-world transformation matrices for cameras following the OpenCV convention.

    Parameters:
    camera_positions (numpy.ndarray): An array of camera positions.
                                      Shape: (n_views, 3), where n_views is the number of camera views.
                                      Each row should be [x, y, z] coordinates of a camera position.
    target (numpy.ndarray): A 3D point that each camera is looking at. Defaults to the world origin [0, 0, 0].
                            Shape: (3,), representing the [x, y, z] coordinates of the target point.
    world_up (numpy.ndarray): The up direction in world coordinates. Defaults to [0, 0, 1].
                              Shape: (3,), representing the [x, y, z] coordinates of the world's up vector.

    Returns:
    numpy.ndarray: An array of 4x4 transformation matrices corresponding to each camera position.
                   Shape: (n_views, 4, 4), where each 4x4 matrix is a transformation matrix for a camera.
    """
    matrices = []

    for pos in camera_positions:
        # Forward vector (Z-axis)
        forward = target - pos
        forward /= np.linalg.norm(forward)

        # Right vector (X-axis)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        # Camera down vector (Y-axis, pointing downwards in OpenCV convention)
        down = np.cross(forward, right)

        # Construct transformation matrix
        mat = np.eye(4)
        mat[0:3, 0] = right
        mat[0:3, 1] = down
        mat[0:3, 2] = forward
        mat[0:3, 3] = pos

        matrices.append(mat)

    return np.array(matrices)


def get_radius(fov, bbox_size):
    '''
    fov: 1d array
    bbox_size: scalar or 1d array, size of the tight bounding box that covers unit sphere
            projected onto image plane, relative to image dimensions

    returns: radius, 1d array
    '''

    half_fov_rads = np.radians(fov / 2)
    radius = 1 / np.sin(np.arctan(np.tan(half_fov_rads) * bbox_size))
    return radius


def gen_rotation_blender():
    return np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def generate_cam2world(radius, cams_azimuth, cams_elevation):
    """
    Roates a multi-camera system, returns the camera poses of each camera in each rotation

    Parameters:
    radius: array of radius Shape: (n_cams,)
    cams_azimuth (numpy.ndarray): Array of azimuth angles in degrees in a multi-camera system, reletaive to reference camera. Shape: (n_cams,)
    cams_elevation (numpy.ndarray): Array of elevation angles in degrees in a multi-camera system, reletaive to reference camera. Shape: (n_cams,)

    Returns:
    numpy.ndarray: Array of unit vector coordinates corresponding to the angles. Shape: (n_cam, 4, 4)
    """
    cam_locations = angles_to_unit_vectors(cams_azimuth, cams_elevation) * radius.reshape(-1, 1)

    cam2worlds = look_at_opencv(cam_locations)  # [n_cams, 4, 4]

    return cam2worlds


def pose_generation_3view(target_3view_number: int,
                          elevation_start: float, elevation_end: float,
                          azimuth_start: float, azimuth_end: float,
                          elevation_perturb_start: float, elevation_perturb_end: float,
                          ortho_radius: float,
                          object_rotation: np.ndarray):
    azimuth_list = []
    elevation_list = []
    cam2worlds = []

    for _ in range(target_3view_number):
        random_elevation = (elevation_start - elevation_end) * np.random.rand() + elevation_end
        for _ in range(3):
            random_azimuth = (azimuth_start - azimuth_end) * np.random.rand() + azimuth_end
            aux_random_elevation = random_elevation + (
                        elevation_perturb_start - elevation_perturb_end) * np.random.rand() + elevation_perturb_end
            azimuth_list.append(random_azimuth)
            elevation_list.append(aux_random_elevation)

    radius_list = [ortho_radius] * len(azimuth_list)

    azimuth_array = np.array(azimuth_list)
    elevation_array = np.array(elevation_list)
    radius_array = np.array(radius_list)

    cam2world = generate_cam2world(radius_array, azimuth_array, elevation_array)

    cam2worlds.extend(object_rotation @ cam2world)

    return cam2worlds, azimuth_list, elevation_list, radius_list


def pose_generation(azimuth_start: float, azimuth_end: float,
                    elevation_start: float, elevation_end: float,
                    camera_number: int, partition_number: int,
                    ortho_radius: float,
                    object_rotation: np.ndarray):
    azimuth_list = []
    elevation_list = []
    cam2worlds = []

    partition_azimuth_angle = (azimuth_end - azimuth_start) / partition_number
    partition_elevation_angle = (elevation_end - elevation_start) / partition_number
    for index in range(partition_number):
        p_azimuth_start = azimuth_start + index * partition_azimuth_angle
        p_azimuth_end = azimuth_start + (index + 1) * partition_azimuth_angle
        p_elevation_start = elevation_start + index * partition_elevation_angle
        p_elevation_end = elevation_start + (index + 1) * partition_elevation_angle

        partition_length = int(camera_number / partition_number)
        if index == partition_number - 1:
            partition_length = camera_number - (partition_number - 1) * partition_length
        for camera_index in range(partition_length):
            perturb_azimuth = (p_azimuth_start - p_azimuth_end) * np.random.rand() + p_azimuth_end
            perturb_elevation = (elevation_start - elevation_end) * np.random.rand() + elevation_end
            if perturb_azimuth < 0:
                perturb_azimuth = perturb_azimuth + 360.0

            azimuth_list.append(perturb_azimuth)
            elevation_list.append(perturb_elevation)

    radius_list = [ortho_radius] * len(azimuth_list)

    azimuth_array = np.array(azimuth_list)
    elevation_array = np.array(elevation_list)
    radius_array = np.array(radius_list)

    cam2world = generate_cam2world(radius_array, azimuth_array, elevation_array)

    cam2worlds.extend(object_rotation @ cam2world)

    return cam2worlds, azimuth_list, elevation_list, radius_list


def final_data_generation(index: int,
                          cam_name: str,
                          result_c2w: list,
                          scale_list: list,
                          pose_data: dict):
    pose = result_c2w[index]
    pose_blender = opencv_to_blender(pose)
    pose_data[cam_name] = {
        "k": None,
        "scale": scale_list[index],
        "pose": pose_blender.tolist(),
        "model": "orthographic"
    }


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)

    parser = argparse.ArgumentParser(
        description='Augmentation camera generation script.')
    parser.add_argument("--output_folder", type=str,
                        help="output directory to save cameras to")
    parser.add_argument("--config_json", type=str,
                        help="config json file path")
    parser.add_argument("--image_size_x", type=int, default=512,
                        help="rendered image size on x axis")
    parser.add_argument("--image_size_y", type=int, default=512,
                        help="rendered image size on y axis")
    all_args = parser.parse_args()

    print("Pose generation for mode ORTHO start. Local time is %s" %
          (local_time_str))

    output_folder = all_args.output_folder
    config_json = all_args.config_json
    image_size_x = all_args.image_size_x
    image_size_y = all_args.image_size_y
    image_size = [image_size_x, image_size_y]

    all_camera_generation_config = read_json(config_json)
    random_config = all_camera_generation_config["RSVC"]

    camera_parameters_data = {}
    camera_parameters_data["config"] = {}
    camera_parameters_data["config"]["RSVC"] = random_config
    pose_data = {}
    camera_parameters_data["poses"] = pose_data
    final_aux_camera_pose = []

    total_index = 0

    ortho_scale = random_config["ortho_scale"]
    ortho_radius = random_config["ortho_radius"]

    target_3view_number = random_config["3view_target_number"]
    target_3view_azimuth_start = random_config["3view_target_azimuth_range"]["start"]
    target_3view_azimuth_end = random_config["3view_target_azimuth_range"]["end"]
    target_3view_elevation_start = random_config["3view_target_elevation_range"]["start"]
    target_3view_elevation_end = random_config["3view_target_elevation_range"]["end"]
    target_3view_elevation_perturb_start = random_config["3view_target_elevation_perturb_range"]["start"]
    target_3view_elevation_perturb_end = random_config["3view_target_elevation_perturb_range"]["end"]

    condition_camera_number = random_config["condition_camera_number"]
    condition_partition_number = random_config["condition_partition"]
    condition_elevation_start = random_config["condition_elevation_range"]["start"]
    condition_elevation_end = random_config["condition_elevation_range"]["end"]
    condition_azimuth_start = random_config["condition_azimuth_range"]["start"]
    condition_azimuth_end = random_config["condition_azimuth_range"]["end"]

    azimuth_list = random_config["azimuth_list"]
    elevation_list = random_config["elevation_list"]

    object_rotation = gen_rotation_blender()

    total_azimuth_list = copy.deepcopy(azimuth_list)
    total_elevation_list = copy.deepcopy(elevation_list)

    result_tuple_3view = pose_generation_3view(target_3view_number=target_3view_number,
                                               azimuth_start=target_3view_azimuth_start,
                                               azimuth_end=target_3view_azimuth_end,
                                               elevation_start=target_3view_elevation_start,
                                               elevation_end=target_3view_elevation_end,
                                               elevation_perturb_start=target_3view_elevation_perturb_start,
                                               elevation_perturb_end=target_3view_elevation_perturb_end,
                                               ortho_radius=ortho_radius, object_rotation=object_rotation)

    cam2worlds_3view = result_tuple_3view[0]
    azimuth_list_3view = result_tuple_3view[1]
    elevation_list_3view = result_tuple_3view[2]
    radius_list = result_tuple_3view[3]

    total_azimuth_list.extend(azimuth_list_3view)
    total_elevation_list.extend(elevation_list_3view)

    result_tuple_condition = pose_generation(azimuth_start=condition_azimuth_start,
                                             azimuth_end=condition_azimuth_end,
                                             elevation_start=condition_elevation_start,
                                             elevation_end=condition_elevation_end,
                                             camera_number=condition_camera_number,
                                             partition_number=condition_partition_number,
                                             ortho_radius=ortho_radius, object_rotation=object_rotation)

    condition_camera_pose = result_tuple_condition[0]
    condition_azimuth_list = result_tuple_condition[1]
    condition_elevation_list = result_tuple_condition[2]
    condition_radius_list = result_tuple_condition[3]

    total_azimuth_list.extend(condition_azimuth_list)
    total_elevation_list.extend(condition_elevation_list)

    aux_azimuth_array = np.array(azimuth_list)
    aux_elevation_array = np.array(elevation_list)
    ortho_radius_list = [ortho_radius] * len(azimuth_list)
    aux_radius_array = np.array(ortho_radius_list)
    aux_camera_pose = generate_cam2world(aux_radius_array, aux_azimuth_array, aux_elevation_array)
    final_aux_camera_pose.extend(object_rotation @ aux_camera_pose)

    all_camera_pose = final_aux_camera_pose + cam2worlds_3view + condition_camera_pose
    total_camera_number = len(all_camera_pose)

    object_scale_list = [ortho_scale] * total_camera_number
    cam_names = ["cam-%04d" % i for i in range(total_camera_number)]

    for index in range(total_camera_number):
        final_data_generation(index=index,
                              cam_name=cam_names[total_index],
                              result_c2w=all_camera_pose,
                              scale_list=object_scale_list,
                              pose_data=pose_data)
        total_index = total_index + 1

    random_config["real_azimuth"] = total_azimuth_list
    random_config["real_elevation"] = total_elevation_list

    t_date = time.time()
    start_date = time.localtime(t_date)
    start_date_str = time.strftime('%Y%m%d', start_date)

    print("Camera number is %i........" % (total_camera_number))

    camera_parameter_filename = "internal_cam_parameters.json"
    json_output_path = os.path.join(output_folder, camera_parameter_filename)
    write_json(json_output_path, camera_parameters_data)
