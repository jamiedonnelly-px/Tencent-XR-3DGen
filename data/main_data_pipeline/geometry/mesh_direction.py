import argparse
import time

import numpy as np
import scipy
import trimesh
from scipy.spatial.transform import Rotation as R


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def euler_to_rotation_matrix(euler_angles):
    r = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def rotation_matrix_between(axis1: np.array, axis2: np.array):
    # 计算旋转轴
    rotation_axis = np.cross(axis1, axis2)
    axis1_norm = normalize(axis1)
    axis2_norm = normalize(axis2)
    rotation_angle = np.arccos(np.clip(np.dot(axis1_norm, axis2_norm), -1.0, 1.0))
    print(rotation_angle)

    rotation_struct = R.from_rotvec(rotation_angle * rotation_axis)
    rotation_matrix = rotation_struct.as_matrix()
    return rotation_matrix


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("SDF calculation starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to input mesh obj")
    args = parser.parse_args()

    mesh_path = args.mesh_path

    gravity_direction = np.array([0, -1.0, 0])

    original_mesh = trimesh.load(mesh_path)
    the_mesh = as_mesh(original_mesh)
    mesh_verts = the_mesh.vertices

    centeroid = np.mean(mesh_verts, axis=0)
    verts_m = mesh_verts - centeroid
    Cov = np.cov(verts_m.T)
    eigen_value, eigen_vector = np.linalg.eig(Cov.T)

    largest_value_index = np.argmax(eigen_value)
    main_direction = eigen_vector[largest_value_index, :]
    print(main_direction)

    print(rotation_matrix_between(main_direction, gravity_direction))
