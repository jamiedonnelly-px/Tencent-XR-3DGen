#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import scipy.spatial
import torch


def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix.
    :param euler_angles: Euler angles in degrees.
    :returns: Rotation matrix as a numpy array of shape (3, 3).
    """
    r = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def opencv_to_blender(T):
    """
    change transformation materix in opencv coordinates to blender coordinates
    transform a point like: new_point = np.matmul(output_transform, old_point)
    :param T: transformation matrix in opencv coordinate system, 4 * 4 numpy array
    :returns: transformation matrix in blender coordinate system, 4 * 4 numpy array
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
    :param azimuth: Array of azimuth angles in degrees. Shape: (n,)
    :param elevation: Array of elevation angles in degrees. Shape: (n,)
    :return: numpy.ndarray of unit vector coordinates corresponding to the angles. Shape: (n, 3)
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
    :param camera_positions: An array of camera positions. Shape: (n_views, 3),
                            where n_views is the number of camera views.
                            Each row should be [x, y, z] coordinates of a camera position.
    :param target: A 3D point that each camera is looking at. Shape: (3,),
                    representing the [x, y, z] coordinates of the target point.
    :param world_up: The up direction in world coordinates. Shape: (3,),
                        representing the [x, y, z] coordinates of the world's up vector.
    :return: numpy.ndarray of 4x4 transformation matrices corresponding to each camera position.
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
    """
    Returns the distance between camera and coordinate center (radius on actual sphere)
    Distance is determined by camera fov and percentage of object on image
    :param fov: camera fov in rad
    :param bbox_size: scalar or 1d array, size of the tight bounding box that covers unit sphere
            projected onto image plane, relative to image dimensions
    :returns: radius as 2d numpy ndarray
    """
    half_fov_rads = np.radians(fov / 2)
    radius = 1 / np.sin(np.arctan(np.tan(half_fov_rads) * bbox_size))
    return radius


def gen_rotation_blender():
    """
    Returns a rotation matrix that rotates a vector by 180 degrees around the x-axis,
    which is exactly the same as Blender's internal rotation.
    """
    return np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def generate_cam2world(radius, cams_azimuth, cams_elevation):
    """
    Generates a list of camera poses.
    Camera poses are calculated like this:
    1. calculate the direction of the camera by sampling points on unit sphere,
    the camera is represented by azimuth and elevation on unit sphere;
    2. calculate the radius of the camera (distance to coordinate center) on actual sphere using get_radius;
    3. generate pose on unit sphere by calculating look-at matrix;
    4. use the radius and unit sphere pose to calculate final pose.
    :param radius: array of camera distance to coordinate system center
    :param cams_azimuth: Array of azimuth angles on unit sphere
    :param cams_elevation: Array of elevation angles on unit sphere
    """
    # calculate location on unit sphere
    cam_locations = angles_to_unit_vectors(cams_azimuth, cams_elevation) * radius.reshape(-1, 1)

    cam2worlds = look_at_opencv(cam_locations)  # [n_cams, 4, 4]

    return cam2worlds


def gen_random_rotation_azimuth_only(start_angle: float = 0,
                                     end_angle: float = 360.0):
    """
    Generates a random rotation matrix around z-axis for objects.
    This function does not directly calculate a matrix to rotate the object
    but rather returns a matrix that rotates the camera and make the final view similar as rotating the object.
    :param start_angle: start angle of the random
    :param end_angle: end angle of the random
    :returns z_matrix: numpy array of the random rotation matrix
    :returns rot_z_deg: random rotation angle of the random rotation matrix
    """
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
    """
    Generates a random rotation matrix around a random axis on x-y plane for objects.
    This function does not directly calculate a matrix to rotate the object
    but rather returns a matrix that rotates the camera and make the final view similar as rotating the object.
    :param xy_deg: rotation angle around the random axis on x-y plane
    :returns numpy array of the random rotation matrix
    """
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


def gen_random_rotation_mostly_azimuth(x_start_angle: float,
                                       x_end_angle: float,
                                       z_start_angle: float,
                                       z_end_angle: float):
    """
    Generates a random rotation matrix around z-axis and a random rotation matrix around a random axis on x-y plane.
    This function does not directly calculate a matrix to rotate the object
    but rather returns a matrix that rotates the camera and make the final view similar as rotating the object.
    :param x_start_angle: start angle of the random rotation around a random axis on x-y plane
    :param x_end_angle: end angle of the random rotation around a random axis on x-y plane
    :param z_start_angle: start angle of the random rotation around z-axis
    :param z_end_angle: end angle of the random rotation around z-axis
    :returns: ret: numpy array of the random rotation matrix
    :returns: rot_z_deg: random rotation angle around z-axis
    :returns: rot_xy_deg: random rotation angle around a random axis on x-y plane
    """
    # first rotates around z then rotate around random axis in x-y plane
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


def write_json(json_path: str, json_struct):
    """
    Write a json struct to a json file.
    :param json_path: path of the json file
    :param json_struct: json struct to write
    """
    try:
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(json_struct, f, indent=4, ensure_ascii=False)
    except (IOError, FileNotFoundError) as e:
        logging.error("Cannot write json file %s" % json_path)


if __name__ == '__main__':
    pass
