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


def gen_random_rotations(n_rotations: int = 1,
                         min_diff_in_degrees: float = 0,
                         exclude_I: bool = False,
                         max_trial: int = 1024):
    """
    Generates random 3D rotation matrices, with a minimum rotational distance in between.

    :param n_rotations: number of random rotations
    :param min_diff_in_degrees: minimum distances between two random rotations in degrees
    :return: np array of shape [n_rotations, 4, 4] representing transformation matrices
    """

    def rotation_distance(R1, R2):
        R = np.dot(R1.T, R2)
        angle_rad = np.arccos((max(min(np.trace(R), 3), -1) - 1) / 2)
        return np.degrees(angle_rad)

    if exclude_I:
        rotations = [np.eye(4)]
    else:
        rotations = []

    n_trial = 0

    while len(rotations) < n_rotations + int(exclude_I):
        random_x_angle = (0.0 - 360.0) * np.random.rand() + 360.0
        random_y_angle = (0.0 - 360.0) * np.random.rand() + 360.0
        random_z_angle = (0.0 - 360.0) * np.random.rand() + 360.0
        random_angles = np.array(
            [random_x_angle, random_y_angle, random_z_angle])
        print(random_angles)
        R_new = euler_to_rotation_matrix(random_angles)
        R_result = np.eye(4)
        R_result[0:3, 0:3] = R_new
        n_trial += 1

        if n_trial > max_trial or all(
                rotation_distance(R_result, R_old) >= min_diff_in_degrees for R_old in rotations):
            rotations.append(R_result)

    return np.array(rotations)[-n_rotations:]


def gen_rotation_blender():
    return np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def gen_random_rotation_azimuth_only(start_angle: float = 0,
                                     end_angle: float = 360.0):
    rot_z_deg = (start_angle - end_angle) * np.random.rand() + end_angle
    rot_z = np.radians(rot_z_deg)
    z_matrix = np.array([
        [np.cos(rot_z), np.sin(rot_z), 0, 0],
        [-np.sin(rot_z), np.cos(rot_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return z_matrix, rot_z_deg


def rotation_matrix_xy_plane(xy_deg: float):
    xy_rad = np.radians(xy_deg)

    # Generate a random angle for the direction of the axis in the x-y plane
    random_angle = np.random.rand() * 2 * np.pi  # Random angle between 0 and 2*pi

    # Compute the components of the random axis in the x-y plane
    u_x = np.cos(random_angle)
    u_y = np.sin(random_angle)
    u_z = 0  # No z component since the axis is in the x-y plane

    # Rotation matrix components
    cos_xy = np.cos(xy_rad)
    sin_xy = np.sin(xy_rad)
    one_minus_cos_x = 1 - cos_xy

    # Compute the rotation matrix using the axis-angle formula
    R = np.array([
        [cos_xy + u_x ** 2 * one_minus_cos_x, u_x * u_y * one_minus_cos_x -
         u_z * sin_xy, u_x * u_z * one_minus_cos_x + u_y * sin_xy],
        [u_y * u_x * one_minus_cos_x + u_z * sin_xy, cos_xy + u_y ** 2 *
         one_minus_cos_x, u_y * u_z * one_minus_cos_x - u_x * sin_xy],
        [u_z * u_x * one_minus_cos_x - u_y * sin_xy, u_z * u_y *
         one_minus_cos_x + u_x * sin_xy, cos_xy + u_z ** 2 * one_minus_cos_x]
    ])

    return R


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


def gen_random_rotation_mostly_azimuth(x_start_angle: float,
                                       x_end_angle: float,
                                       z_start_angle: float,
                                       z_end_angle: float):
    '''first rotates around z then rotate around random axis in x-y plane
    '''
    world2blender = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    rot_z_deg = (z_start_angle - z_end_angle) * np.random.rand() + z_end_angle
    rot_xy_deg = (x_start_angle - x_end_angle) * np.random.rand() + x_end_angle

    rot_z = np.radians(rot_z_deg)

    # Rotation matrix around z-axis
    R_z = np.array([
        [np.cos(rot_z), np.sin(rot_z), 0],
        [-np.sin(rot_z), np.cos(rot_z), 0],
        [0, 0, 1]
    ])

    R_xy = rotation_matrix_xy_plane(rot_xy_deg)

    ret = np.eye(4)
    ret[:3, :3] = R_xy @ R_z @ world2blender
    return ret, rot_z_deg, rot_xy_deg


def pose_generation(azimuth_list: list,
                    elevation_list: list,
                    ortho_scale_list: list,
                    # perturb_azimuth_start: float,
                    # perturb_azimuth_end: float,
                    # perturb_elevation_start: float,
                    # perturb_elevation_end: float,
                    aux_azimuth_start: float,
                    aux_azimuth_end: float,
                    aux_elevation_start: float,
                    aux_elevation_end: float,
                    aux_perturb_elevation_start: float,
                    aux_perturb_elevation_end: float,
                    fov_start: float, fov_end: float,
                    ortho_radius: float,
                    object_rotation: np.ndarray,
                    persp_y_only_number: int = 0,
                    ortho_y_only_number: int = 0,
                    aux_camera_number: int = 0,
                    size_image_percentage: float = 0.96):
    cam2worlds = []

    current_azimuth_list = copy.deepcopy(azimuth_list)
    current_elevation_list = copy.deepcopy(elevation_list)
    current_scale_list = copy.deepcopy(ortho_scale_list)
    current_fov_list = [-1.0] * len(current_azimuth_list)
    current_radius_list = [ortho_radius] * len(current_azimuth_list)
    current_camera_config_list = ["ortho"] * len(current_azimuth_list)

    # for index in range(len(current_azimuth_list)):
    #     azimuth_perturb_coeff = 0.5 if index == 0 else np.random.rand()
    #     elevation_perturb_coeff = 0.5 if index == 0 else np.random.rand()
    #     perturb_azimuth = current_azimuth_list[index] + (perturb_azimuth_start - perturb_azimuth_end) * azimuth_perturb_coeff + perturb_azimuth_end
    #     perturb_elevation = current_elevation_list[index] + (perturb_elevation_start - perturb_elevation_end) * elevation_perturb_coeff + perturb_elevation_end
    #     current_azimuth_list.append(perturb_azimuth)
    #     current_elevation_list.append(perturb_elevation)
    #     current_scale_list.append(current_scale_list[index])
    #     current_fov_list.append(current_fov_list[index])
    #     current_radius_list.append(current_radius_list[index])
    #     current_camera_config_list.append(current_camera_config_list[index])

    persp_fov_list = []
    for _ in range(persp_y_only_number):
        random_azimuth = 0
        random_elevation = (aux_elevation_start - aux_elevation_end) * np.random.rand() + aux_elevation_end

        current_azimuth_list.append(random_azimuth)
        current_elevation_list.append(random_elevation)

        random_fov = (fov_start - fov_end) * np.random.rand() + fov_end
        current_fov_list.append(random_fov)
        current_scale_list.append(-1.0)
        current_camera_config_list.append("persp")
        persp_fov_list.append(random_fov)

        for _ in range(aux_camera_number):
            random_azimuth = (aux_azimuth_start - aux_azimuth_end) * np.random.rand() + aux_azimuth_end
            aux_random_elevation = random_elevation + (
                    aux_perturb_elevation_start - aux_perturb_elevation_end) * np.random.rand() + aux_perturb_elevation_end

            current_azimuth_list.append(random_azimuth)
            current_elevation_list.append(aux_random_elevation)
            random_fov = (fov_start - fov_end) * np.random.rand() + fov_end
            current_fov_list.append(random_fov)
            current_scale_list.append(-1.0)
            current_camera_config_list.append("persp")
            persp_fov_list.append(random_fov)

    persp_fov_array = np.array(persp_fov_list)
    persp_radius_list = get_radius(persp_fov_array, size_image_percentage).tolist()
    # for index in range(len(persp_radius_list)):
    #     if (index % (aux_camera_number + 1)) == 0:
    #         continue
    #     radius = persp_radius_list[index]
    #     random_percentage = (persp_radius_percentage - 1.0) * np.random.rand() + 1.0
    #     persp_radius_list[index] = radius * random_percentage

    current_radius_list = current_radius_list + persp_radius_list

    for _ in range(ortho_y_only_number):
        random_azimuth = 0
        random_elevation = (aux_elevation_start - aux_elevation_end) * np.random.rand() + aux_elevation_end

        current_azimuth_list.append(random_azimuth)
        current_elevation_list.append(random_elevation)
        current_fov_list.append(-1.0)
        current_scale_list.append(ortho_scale_list[0])
        current_camera_config_list.append("ortho")
        current_radius_list.append(ortho_radius)

        for _ in range(aux_camera_number):
            random_azimuth = (aux_azimuth_start - aux_azimuth_end) * np.random.rand() + aux_azimuth_end
            aux_random_elevation = random_elevation + (
                    aux_perturb_elevation_start - aux_perturb_elevation_end) * np.random.rand() + aux_perturb_elevation_end

            current_azimuth_list.append(random_azimuth)
            current_elevation_list.append(aux_random_elevation)
            current_fov_list.append(-1.0)
            current_scale_list.append(ortho_scale_list[0])
            current_camera_config_list.append("ortho")
            current_radius_list.append(ortho_radius)

    azimuth_array = np.array(current_azimuth_list)
    elevation_array = np.array(current_elevation_list)
    radius_array = np.array(current_radius_list)

    cam2world = generate_cam2world(radius_array, azimuth_array, elevation_array)

    cam2worlds.extend(object_rotation @ cam2world)

    pose_result = (cam2worlds, current_azimuth_list, current_elevation_list, current_fov_list, current_scale_list,
                   current_camera_config_list)

    return pose_result


def final_data_generation(index: int,
                          cam_name: str,
                          result_c2w: list,
                          res_fov_list: list,
                          res_scale_list: list,
                          res_camera_name_list,
                          image_size: list,
                          pose_data: dict):
    fov = res_fov_list[index]
    pose = result_c2w[index]
    pose_blender = opencv_to_blender(pose)
    pose_data[cam_name] = {
        "k": None,
        "scale": None,
        "pose": pose_blender.tolist(),
        "model": None
    }

    if res_camera_name_list[index] == 'ortho':
        pose_data[cam_name]["scale"] = res_scale_list[index]
        pose_data[cam_name]["k"] = None
        pose_data[cam_name]["model"] = "orthographic"
    else:
        pose_data[cam_name]["scale"] = None
        K_array = get_intrinsic_matrix(fov, image_size).numpy()
        pose_data[cam_name]["k"] = K_array.tolist()
        pose_data[cam_name]["model"] = "perspective"


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
    online_config = all_camera_generation_config["RTriVC"]
    # object_euler_range = augmentation_config["object_euler_range"]

    ortho_scale_list = online_config["ortho_scale_list"]
    azimuth_list = online_config["azimuth_list"]
    elevation_list = online_config["elevation_list"]

    online_config["real_fov_list"] = []
    online_config["real_azimuth_list"] = []
    online_config["real_elevation_list"] = []

    ortho_radius = online_config["ortho_radius"]
    persp_radius_percentage = online_config["perspective_radius_percentage"]
    size_ortho = online_config["image_percentage"]

    persp_y_only_number = online_config["persp_y_only_number"]
    ortho_y_only_number = online_config["ortho_y_only_number"]
    y_only_number = ortho_y_only_number + persp_y_only_number
    y_aux_camera_number = online_config["y_aux_camera_number"]

    perturb_azimuth_start = online_config["perturb_azimuth_range"]["start"]
    perturb_azimuth_end = online_config["perturb_azimuth_range"]["end"]
    perturb_elevation_start = online_config["perturb_elevation_range"]["start"]
    perturb_elevation_end = online_config["perturb_elevation_range"]["end"]

    aux_azimuth_start = online_config["aux_azimuth_range"]["start"]
    aux_azimuth_end = online_config["aux_azimuth_range"]["end"]
    aux_elevation_start = online_config["aux_elevation_range"]["start"]
    aux_elevation_end = online_config["aux_elevation_range"]["end"]
    aux_perturb_elevation_start = online_config["aux_perturb_elevation_range"]["start"]
    aux_perturb_elevation_end = online_config["aux_perturb_elevation_range"]["end"]

    fov_start = online_config["perspective_fov_start"]
    fov_end = online_config["perspective_fov_end"]
    first_identity = online_config["first_identity"]

    # parameters for 4 input reference views
    azimuth_ref = np.array([0, 0, 0, 0])
    elevation_ref = np.array([0, 0, 0, 0])

    intrinsics = []
    cam2worlds = []

    equator_number = online_config["equator_number"]
    equator_rotation_angles = online_config["equator_range"]
    all_direction_number = 0
    all_direction_range = []
    if "all_direction_number" in online_config.keys():
        all_direction_number = online_config["all_direction_number"]
        all_direction_range = online_config["all_direction_range"]

    c2w = []
    intrinsic = []
    ortho_scale = []

    camera_parameters_data = {}
    camera_parameters_data["config"] = {}
    camera_parameters_data["config"]["RTriVC"] = online_config
    pose_data = {}
    camera_parameters_data["poses"] = pose_data

    single_camera_number = len(azimuth_list) * 2 + y_only_number * (y_aux_camera_number + 1)
    total_camera_number = single_camera_number * (equator_number + all_direction_number)
    if first_identity:
        total_camera_number = total_camera_number + single_camera_number
    cam_names = ["cam-%04d" % i for i in range(total_camera_number)]

    total_index = 0
    total_azimuth_list = []
    total_elevation_list = []
    total_object_azimuth_list = []
    total_object_elevation_list = []

    # 0-th
    if first_identity:
        identity_rotation = gen_rotation_blender()
        pose_result = pose_generation(azimuth_list, elevation_list, ortho_scale_list,
                                      aux_azimuth_start=aux_azimuth_start,
                                      aux_azimuth_end=aux_azimuth_end,
                                      aux_elevation_start=aux_elevation_start,
                                      aux_elevation_end=aux_elevation_end,
                                      aux_perturb_elevation_start=aux_perturb_elevation_start,
                                      aux_perturb_elevation_end=aux_perturb_elevation_end,
                                      fov_start=fov_start,
                                      fov_end=fov_end,
                                      ortho_radius=ortho_radius,
                                      object_rotation=identity_rotation,
                                      persp_y_only_number=persp_y_only_number,
                                      ortho_y_only_number=ortho_y_only_number,
                                      aux_camera_number=y_aux_camera_number,
                                      size_image_percentage=size_ortho)

        result_c2w = pose_result[0]
        res_azimuth_list = pose_result[1]
        res_elevation_list = pose_result[2]
        res_fov_list = pose_result[3]
        res_scale_list = pose_result[4]
        res_camera_name_list = pose_result[5]

        for index in range(len(result_c2w)):
            final_data_generation(index, cam_names[total_index],
                                  result_c2w,
                                  res_fov_list,
                                  res_scale_list,
                                  res_camera_name_list,
                                  image_size,
                                  pose_data)
            total_index = total_index + 1
        total_azimuth_list.extend(res_azimuth_list)
        total_elevation_list.extend(res_elevation_list)
        total_object_azimuth_list.append(0.0)
        total_object_elevation_list.append(0.0)

    for _ in range(equator_number):
        starting_rotation = gen_rotation_blender()
        object_rotation, object_azimuth_angle = gen_random_rotation_azimuth_only(
            start_angle=equator_rotation_angles["start"],
            end_angle=equator_rotation_angles["end"])
        object_rotation = starting_rotation @ object_rotation
        pose_result = pose_generation(azimuth_list, elevation_list, ortho_scale_list,
                                      aux_azimuth_start=aux_azimuth_start,
                                      aux_azimuth_end=aux_azimuth_end,
                                      aux_elevation_start=aux_elevation_start,
                                      aux_elevation_end=aux_elevation_end,
                                      aux_perturb_elevation_start=aux_perturb_elevation_start,
                                      aux_perturb_elevation_end=aux_perturb_elevation_end,
                                      fov_start=fov_start,
                                      fov_end=fov_end,
                                      ortho_radius=ortho_radius,
                                      object_rotation=object_rotation,
                                      persp_y_only_number=persp_y_only_number,
                                      ortho_y_only_number=ortho_y_only_number,
                                      aux_camera_number=y_aux_camera_number,
                                      size_image_percentage=size_ortho)

        result_c2w = pose_result[0]
        res_azimuth_list = pose_result[1]
        res_elevation_list = pose_result[2]
        res_fov_list = pose_result[3]
        res_scale_list = pose_result[4]
        res_camera_name_list = pose_result[5]

        for index in range(len(result_c2w)):
            final_data_generation(index,
                                  cam_names[total_index],
                                  result_c2w,
                                  res_fov_list,
                                  res_scale_list,
                                  res_camera_name_list,
                                  image_size, pose_data)
            total_index = total_index + 1
        total_azimuth_list.extend(res_azimuth_list)
        total_elevation_list.extend(res_elevation_list)
        total_object_azimuth_list.append(object_azimuth_angle)
        total_object_elevation_list.append(0.0)

    for f_index in range(all_direction_number):
        starting_rotation = gen_rotation_blender()
        object_rotation_result = gen_random_rotation_mostly_azimuth(z_start_angle=all_direction_range["z_start"],
                                                                    z_end_angle=all_direction_range["z_end"],
                                                                    x_start_angle=all_direction_range["xy_start"],
                                                                    x_end_angle=all_direction_range["xy_end"])

        object_rotation = object_rotation_result[0]
        object_azimuth_angle = object_rotation_result[1]
        object_elevation_angle = object_rotation_result[2]

        object_rotation = starting_rotation @ object_rotation
        pose_result = pose_generation(azimuth_list, elevation_list, ortho_scale_list,
                                      aux_azimuth_start=aux_azimuth_start,
                                      aux_azimuth_end=aux_azimuth_end,
                                      aux_elevation_start=aux_elevation_start,
                                      aux_elevation_end=aux_elevation_end,
                                      aux_perturb_elevation_start=aux_perturb_elevation_start,
                                      aux_perturb_elevation_end=aux_perturb_elevation_end,
                                      fov_start=fov_start,
                                      fov_end=fov_end,
                                      ortho_radius=ortho_radius,
                                      object_rotation=object_rotation,
                                      persp_y_only_number=persp_y_only_number,
                                      ortho_y_only_number=ortho_y_only_number,
                                      aux_camera_number=y_aux_camera_number,
                                      size_image_percentage=size_ortho)

        result_c2w = pose_result[0]
        res_azimuth_list = pose_result[1]
        res_elevation_list = pose_result[2]
        res_fov_list = pose_result[3]
        res_scale_list = pose_result[4]
        res_camera_name_list = pose_result[5]

        for index in range(len(result_c2w)):
            final_data_generation(index,
                                  cam_names[total_index],
                                  result_c2w,
                                  res_fov_list,
                                  res_scale_list,
                                  res_camera_name_list,
                                  image_size, pose_data)
            total_index = total_index + 1
        total_azimuth_list.extend(res_azimuth_list)
        total_elevation_list.extend(res_elevation_list)
        total_object_azimuth_list.append(object_azimuth_angle)
        total_object_elevation_list.append(0.0)

    online_config["real_azimuth"] = total_azimuth_list
    online_config["real_elevation"] = total_elevation_list
    online_config["real_object_azimuth"] = total_object_azimuth_list
    online_config["real_object_elevation"] = total_object_elevation_list

    t_date = time.time()
    start_date = time.localtime(t_date)
    start_date_str = time.strftime('%Y%m%d', start_date)

    print("Camera number is %i........" % (total_camera_number))

    camera_parameter_filename = "internal_cam_parameters.json"
    json_output_path = os.path.join(output_folder, camera_parameter_filename)
    write_json(json_output_path, camera_parameters_data)
