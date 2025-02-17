#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import h5py
import mesh_to_sdf
import miniball
import numpy as np
import scipy
import trimesh
from fast_intersection import check_mesh_contains


def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix.
    :param euler_angles: Euler angles in degrees.
    :returns Rotation matrix.
    """
    r = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def mesh_to_occupancy(mesh, points):
    """
    Compute occupancy (0-1 field of a group of points) of a mesh.
    To use this function install intersection in modules folder first.
    :param mesh: Input mesh.
    :param points: Points to calculate occupancy.
    :returns occupancy of these points
    """
    return check_mesh_contains(mesh, points)


def sample_surface_geo(mesh, num, weight):
    """
    Sample points and various geometric information on the surface of a mesh.
    :param mesh: Input mesh.
    :param num: Number of points to sample.
    :param weight: Weight of each vertex. If None, all vertices have the same weight.
    :returns v_points: Points on the surface.
    :returns v_normals: Normals of the points on the surface.
    :returns v_faces: Faces index of the points
    :returns face_centroid: Centroid of each face in v_faces
    :returns barycenter_points: Barycentric coordinates of each point on the surface.
    """
    v_points, v_tri = trimesh.sample.sample_surface(mesh, num, weight)
    v_normals = mesh.face_normals[v_tri]
    v_faces = mesh.faces[v_tri]
    face_vertices = mesh.vertices[v_faces]
    face_centroid = face_vertices.mean(axis=1)
    barycenter_points = trimesh.triangles.points_to_barycentric(face_vertices, v_points)
    return v_points, v_normals, v_faces, face_centroid, barycenter_points


def sample_point_near_surface(mesh, sample_number: int = 500000, points_sigma: float = 0.01):
    """
    Sample points and various geometric information in space near the surface of a mesh.
    :param mesh: Input mesh.
    :param sample_number: Number of points to sample.
    :param points_sigma: Standard deviation of Gaussian noise added to the sampled points.
    :returns near_surface_data_struct: struct containing near_surface points and normals
    """
    points_surface, normal_surface, _, _, _ = sample_surface_geo(mesh, sample_number, weight=None)
    points_surface += points_sigma * np.random.randn(sample_number, 3)
    near_surface_data_struct = {}
    near_surface_data_struct["near_surface_points"] = points_surface.astype(np.float32)
    near_surface_data_struct["near_surface_normals"] = normal_surface.astype(np.float32)
    return near_surface_data_struct


def sample_points_on_surface(mesh: trimesh.Trimesh, sample_number: int = 500000):
    """
    Sample points and various geometric information on the surface of a mesh.
    Wrapper of sample_surface_geo because we want to re-use sample_surface_geo in other places.
    :param mesh: Input mesh.
    :param sample_number: Number of points to sample.
    :returns surface_data_struct: struct containing surface points and geometric data
    """
    points, normals, faces, face_centroids, barycenter_points = sample_surface_geo(mesh, sample_number, weight=None)
    surface_data_struct = {}
    surface_data_struct["surface_points"] = points.astype(np.float32)
    surface_data_struct["surface_normals"] = normals.astype(np.float32)
    surface_data_struct["surface_faces"] = faces.astype(np.float32)
    surface_data_struct["surface_face-centroids"] = face_centroids.astype(np.float32)
    surface_data_struct["surface_points-barycentric"] = barycenter_points.astype(np.float32)
    return surface_data_struct


def sample_point_uniformly_in_volume(number_of_points: int = 500000, boxsize: float = 2.0):
    """
    Sample points randomly (uniform distribution) in a unit volume.
    Size of the volume is [-boxsize/2, +boxsize/2]^3.
    :param number_of_points: Number of points to sample.
    :param boxsize: Box size.
    :returns points_uniform: Points sampled uniformly in a unit volume.
    """
    # points uniformly distributed within [0,1]
    points_uniform = np.random.rand(number_of_points, 3)
    # [0,1] -> [-0.5, +0.5] -> [-boxsize/2, +boxsize/2]
    points_uniform = boxsize * (points_uniform - 0.5)
    return points_uniform.astype(np.float32)


def generate_geo_data_structs(mesh,
                              space_sample_number: int = 500000,
                              near_surface_sample_number: int = 500000,
                              surface_sample_number: int = 500000):
    """
    Generate data structures containing geometric information of a mesh for 3D diffusion training.
    :param mesh: Input mesh.
    :param space_sample_number: Number of points to sample in the space.
    :param near_surface_sample_number: Number of points to sample near the surface.
    :param surface_sample_number: Number of points to sample on the surface.
    :returns space_data: struct containing space points and geometric data
    :returns near_surface_data: struct containing near_surface points and geometric data
    :returns surface_data: struct containing surface points and geometric data
    """
    space_points = sample_point_uniformly_in_volume(space_sample_number)
    space_sdf = mesh_to_sdf.mesh_to_sdf(mesh, space_points, surface_point_method='sample')
    space_occupancy = mesh_to_occupancy(mesh, space_points)

    space_data = {}
    space_data["space_points"] = space_points
    space_data["space_sdf"] = np.expand_dims(space_sdf, axis=-1)
    space_data["space_occupancy"] = np.expand_dims(space_occupancy, axis=-1)

    near_surface_data = sample_point_near_surface(mesh, near_surface_sample_number)

    near_surface_sdf = mesh_to_sdf.mesh_to_sdf(mesh, near_surface_data["near_surface_points"],
                                               surface_point_method='sample')
    near_surface_occupancy = mesh_to_occupancy(mesh, near_surface_data["near_surface_points"])
    near_surface_data["near_surface_sdf"] = np.expand_dims(near_surface_sdf, axis=-1)
    near_surface_data["near_surface_occupancy"] = np.expand_dims(near_surface_occupancy, axis=-1)

    surface_data = sample_points_on_surface(mesh, sample_number=surface_sample_number)

    return space_data, near_surface_data, surface_data


def shuffle_numpy_array(data_length: int, data_struct: dict):
    """
    Shuffle numpy array data.
    :param data_length: length of data.
    :param data_struct: data struct to be shuffled.
    :returns result: shuffled data struct.
    """
    indices = np.arange(data_length)
    np.random.shuffle(indices)
    result = {}
    for data_name in data_struct:
        data_shuffled = data_struct[data_name][indices]
        result[data_name] = data_shuffled
    return result


def save_struct_as_numpy(data_struct: dict, output_folder: str, data_length: int):
    """
    Save data from a struct as numpy array.
    NPY file name is the same as data_key in data_struct.
    :param data_struct: data struct to be saved.
    :param output_folder: output folder of saved data.
    :param data_length: length of data.
    """
    for data_name in data_struct:
        npy_filename = os.path.join(output_folder, (data_name + ("_%i.npy" % data_length)))
        np.save(npy_filename, data_struct[data_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to manifold obj")
    parser.add_argument("--output_folder", type=str, default="",
                        help="output folder of sampled points")
    parser.add_argument("--transform_path", type=str, default="",
                        help="input transformation txt path")
    parser.add_argument("--z_transform_path", type=str, default="",
                        help="input transformation txt path for z_up (blender) coordinate system")
    parser.add_argument("--standard_height", type=float, default=1.92,
                        help="standard height of the mesh")
    parser.add_argument("--space_sample_number", type=int, default=500000,
                        help="number of sdf sampled in all spaces")
    parser.add_argument("--near_surface_sample_number", type=int, default=500000,
                        help="number of sdf sampled points near surface")
    parser.add_argument("--surface_sample_number", type=int, default=500000,
                        help="number of sdf sampled points on surface")
    parser.add_argument('--sample_format', type=str, default="h5_chunk",
                        help='output format, choose between h5,h5_chunk,npy')
    parser.add_argument('--chunk_size', type=int, default=4096,
                        help='chunked storage size, only used when sample_format=h5_chunk')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle output data sequence')
    args = parser.parse_args()

    mesh_path = args.mesh_path
    output_folder = args.output_folder
    transform_path = args.transform_path
    z_transform_path = args.z_transform_path
    standard_height = args.standard_height
    space_sample_number = args.space_sample_number
    near_surface_sample_number = args.near_surface_sample_number
    surface_sample_number = args.surface_sample_number
    sample_format = args.sample_format
    chunk_size = args.chunk_size
    shuffle = args.shuffle

    internal_rotation = euler_to_rotation_matrix(np.array([90, 0.0, 0.0]))
    inverse_internal_rotation = np.linalg.inv(internal_rotation)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    geometry_sample_folder = os.path.join(output_folder, "geometry")
    if not os.path.exists(geometry_sample_folder):
        os.mkdir(geometry_sample_folder)

    # we first calculate a tight bounding sphere of the mesh
    # and scale the diameter of this sphere to standard_height
    # we don't choose AABB because
    # (1) is not as thight as a bounding sphere
    # (2) if you rotate an object scaled within a AABB, one part of the object may be out of the unit volume

    # calculate convex hull of the mesh
    # some meshes have many vertices and will be very slow when calculating bounding sphere
    original_mesh = trimesh.load(mesh_path)
    hull_vertices = original_mesh.convex_hull.vertices
    # library miniball may fail to calculate bounding sphere
    # but we cannot stably reproduce the failure, as it disappears if we re-run the calculate code
    # so calculate three times is a temperory workaround
    try:
        bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
    except:
        # block the program is necessary here
        time.sleep(0.1)
        logging.error("Miniball failed. Retry once...............")
        try:
            bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
        except:
            time.sleep(0.1)
            logging.error("Miniball failed. Retry second time...............")
            try:
                bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
            except:
                time.sleep(0.1)
    obj_center = bounding_sphere_C
    length = 2 * math.sqrt(bounding_sphere_r2)
    scale = standard_height / length
    translation = -1 * obj_center
    z_up_translation = internal_rotation @ translation

    T = np.array(
        [[scale, 0, 0, scale * translation[0]],
         [0, scale, 0, scale * translation[1]],
         [0, 0, scale, scale * translation[2]],
         [0, 0, 0, 1]]
    )

    # transformation for blender coordinate system
    Z_up_T = np.array(
        [[scale, 0, 0, scale * z_up_translation[0]],
         [0, scale, 0, scale * z_up_translation[1]],
         [0, 0, scale, scale * z_up_translation[2]],
         [0, 0, 0, 1]]
    )

    np.savetxt(transform_path, T)
    np.savetxt(z_transform_path, Z_up_T)
    logging.info("Inside transformation calculation; default is y_up")
    logging.info("T is %s" % str(T))
    logging.info("z_up_T is %s" % str(Z_up_T))

    original_mesh.apply_transform(T)

    sdf_results = generate_geo_data_structs(original_mesh,
                                            space_sample_number=space_sample_number,
                                            near_surface_sample_number=near_surface_sample_number,
                                            surface_sample_number=surface_sample_number)
    space_data_struct = sdf_results[0]
    near_surface_data_struct = sdf_results[1]
    surface_data_struct = sdf_results[2]

    # Shuffle is necessary if you use sample_format h5_chunk.
    # Since h5_chunk will automatically slice the data into pieces and you can access one piece without reading all data,
    # shuffle will add randomness to each slice, preventing that you only read points that are all geometrically close,
    # this helps reduce the time of random shuffling in dataloader of training script
    if shuffle:
        space_data_struct = shuffle_numpy_array(space_sample_number, space_data_struct)
        near_surface_data_struct = shuffle_numpy_array(near_surface_sample_number, near_surface_data_struct)
        surface_data_struct = shuffle_numpy_array(surface_sample_number, surface_data_struct)

    if sample_format == "h5" or sample_format == "h5_chunk":
        h5_path = os.path.join(geometry_sample_folder, "sample.h5")
        with h5py.File(h5_path, "w") as f:
            for data_name in space_data_struct.keys():
                if sample_format == "h5_chunk":
                    if len(space_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = space_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip")

            for data_name in near_surface_data_struct.keys():
                if sample_format == "h5_chunk":
                    if len(near_surface_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = near_surface_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip")

            for data_name in surface_data_struct.keys():
                if sample_format == "h5_chunk":
                    if len(surface_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = surface_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip")
    else:
        save_struct_as_numpy(surface_data_struct, output_folder, surface_sample_number)
        save_struct_as_numpy(near_surface_data_struct, output_folder, near_surface_sample_number)
        save_struct_as_numpy(space_data_struct, output_folder, space_sample_number)
