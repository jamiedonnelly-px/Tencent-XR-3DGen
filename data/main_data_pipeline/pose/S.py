import argparse
import json
import os
import time

import numpy as np
import torch
import trimesh


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


def generate_pose(r, theta, phi, cam2world=True):
    '''
    generate cam2world poses in Front-Rright-Up world system
    '''

    if abs(abs(theta) - 90.0) < 0.001:
        theta = 89.99
    phi = torch.Tensor([phi])
    theta = torch.Tensor([theta])
    dx = torch.cos(theta) * torch.cos(phi)
    dy = torch.cos(theta) * torch.sin(phi)
    dz = torch.sin(theta)

    T = torch.stack((dx * r, dy * r, dz * r), dim=-1)  # [N,3]
    targets = torch.zeros_like(T)

    v_forward = torch.nn.functional.normalize(targets - T)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    if len(up.shape) == 1:
        up = up.unsqueeze(0)
    v_right = torch.cross(v_forward, up, dim=-1)  # [N, 3]
    v_down = torch.cross(v_forward, v_right, dim=-1)  # [N, 3]

    R = torch.stack((v_right, v_down, v_forward), dim=-1)
    R = torch.nn.functional.normalize(R, p=2, dim=-2, eps=1e-10)  # [N,3,3]

    ret = torch.eye(4) * 1.0
    ret[:3, :3] = R
    ret[:3, 3] = T

    if not cam2world:
        ret = torch.linalg.inv(ret)

    return ret


def generate_S_cam_poses(theta: float,
                         phi: float,
                         fov: float,
                         width: int,
                         height: int,
                         margin: float):
    '''
    generate intrinsic and extrinsic matrices where a ball of given radius
    would be exactly confined in image through camera projection
    '''

    focal = width / 2 / np.tan(fov / 2)
    intrinsic = torch.tensor([
        focal, 0, (width - 1) / 2,
        0, focal, (height - 1) / 2,
        0, 0, 1,
    ]).reshape(3, 3).float()

    span_horizontal = np.tan(fov / 2) * (1 - 2 * margin)
    span_vertical = np.tan(fov / 2) * (1 - 2 * margin) * height / width

    cam2world = generate_pose(1, theta, phi, cam2world=True).float()

    sphere_verts = torch.from_numpy(np.asarray(trimesh.primitives.Sphere(subdivisions=4).vertices)).float()
    sphere_verts = torch.cat((sphere_verts, torch.ones_like(sphere_verts[:, :1])), dim=-1)

    sphere_verts_cam = sphere_verts @ torch.linalg.inv(cam2world).t()
    sphere_verts_cam = sphere_verts_cam[:, :3] / sphere_verts_cam[:, 3:]

    # first push back so everything has positive depth of at least 0.1 unit
    push_back = max(-sphere_verts_cam[:, -1].min(), 0) + 1.1
    cam2world[:3, -1] *= push_back

    sphere_verts_cam = sphere_verts @ torch.linalg.inv(cam2world).t()
    sphere_verts_cam = sphere_verts_cam[:, :3] / sphere_verts_cam[:, 3:]

    # next push back so everything spans exactly to margin
    push_back = sphere_verts_cam[:, :2].abs() / torch.tensor([span_horizontal, span_vertical]) - sphere_verts_cam[:, 2:]
    cam2world[:3, -1] += max(0, push_back.max()) * torch.nn.functional.normalize(cam2world[:3, -1], dim=0)

    return intrinsic, cam2world


def generate_S_cam_poses_ortho(theta: float,
                               phi: float,
                               fov: float,
                               width: int,
                               height: int,
                               margin: float):
    '''
    generate intrinsic and extrinsic matrices where a ball of given radius
    would be exactly confined in image through camera projection
    '''

    focal = width / 2 / np.tan(fov / 2)
    intrinsic = torch.tensor([
        focal, 0, (width - 1) / 2,
        0, focal, (height - 1) / 2,
        0, 0, 1,
    ]).reshape(3, 3).float()

    span_horizontal = np.tan(fov / 2) * (1 - 2 * margin)
    span_vertical = np.tan(fov / 2) * (1 - 2 * margin) * height / width

    cam2world = generate_pose(1, theta, phi, cam2world=True).float()

    sphere_verts = torch.from_numpy(np.asarray(trimesh.primitives.Sphere(subdivisions=4).vertices)).float()
    sphere_verts = torch.cat((sphere_verts, torch.ones_like(sphere_verts[:, :1])), dim=-1)

    sphere_verts_cam = sphere_verts @ torch.linalg.inv(cam2world).t()
    sphere_verts_cam = sphere_verts_cam[:, :3] / sphere_verts_cam[:, 3:]

    # first push back so everything has positive depth of at least 0.1 unit
    push_back = max(-sphere_verts_cam[:, -1].min(), 0) + 1.1
    cam2world[:3, -1] *= push_back

    sphere_verts_cam = sphere_verts @ torch.linalg.inv(cam2world).t()
    sphere_verts_cam = sphere_verts_cam[:, :3] / sphere_verts_cam[:, 3:]

    # next push back so everything spans exactly to margin
    push_back = sphere_verts_cam[:, :2].abs() / torch.tensor([span_horizontal, span_vertical]) - sphere_verts_cam[:, 2:]
    cam2world[:3, -1] += max(0, push_back.max()) * torch.nn.functional.normalize(cam2world[:3, -1], dim=0)

    return intrinsic, cam2world


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
    config_struct = all_camera_generation_config["RSVC"]
    s_config = config_struct["S"]
    elevation = s_config["elevation"]
    latitude_number = s_config["latitude_number"]
    n_views = s_config["n_views"]
    phi_random_bias = float(s_config["phi_bias"]) * np.random.random()

    K = torch.empty(all_camera_number, 3, 3).float()
    C2W = torch.empty(all_camera_number, 4, 4).float()

    assert elevation[0] > - 89.0 and elevation[1] < 89.0, "allowed range is (-89, 89) to avoid gimbal lock"
    if latitude_number > 1:
        interval = (elevation[1] - elevation[0]) / (float(latitude_number - 1))
        theta_list = np.arange(elevation[0], elevation[1] + interval, interval).tolist()
    else:
        theta_list = [0.0]

    longtitude_number = int(n_views / latitude_number)
    phi_index = np.arange(0, longtitude_number).astype(
        np.float32) / float(longtitude_number)
    phi_index = phi_index.tolist()

    for i in range(latitude_number):
        for j in range(int(longtitude_number)):
            point_index = i * longtitude_number + j
            final_margin = (margin[0] - margin[1]) * np.random.rand() + margin[1]
            theta = theta_list[i % latitude_number]
            phi = phi_index[j % longtitude_number] * 360.0 + phi_random_bias
            final_fov = (fov[0] - fov[1]) * np.random.rand() + fov[1]

            print(i, j, point_index, theta, phi, final_fov, fov)

            K[point_index], C2W[point_index] = generate_S_cam_poses(theta * np.pi / 180,
                                                                    phi * np.pi / 180,
                                                                    final_fov * np.pi / 180,
                                                                    width=image_size[0],
                                                                    height=image_size[1],
                                                                    margin=final_margin)
