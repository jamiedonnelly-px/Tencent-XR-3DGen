import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


def current_time():
    """
    Get current time string.
    :return: current time string
    """
    t_current = time.time()
    local_time = time.localtime(t_current)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    return local_time_str


def h5_once(render_folder: str, mesh_instance_name: str, save_path: str):
    try:
        rgb_path = os.path.join(render_folder, "color")
        emission_path = os.path.join(render_folder, "emission/color")
        depth_path = os.path.join(render_folder, "depth")
        normal_path = os.path.join(render_folder, "view_normal")
        cam_pose_path = os.path.join(render_folder, "cam_parameters.json")

        with open(cam_pose_path, 'r') as fr:
            cam_pose = json.load(fr)

        name_list = os.listdir(rgb_path)
        name_list.sort()

        save_name_h5 = os.path.join(save_path, mesh_instance_name)
        save_name_h5 = save_name_h5 + ".h5"

        rgbs = []
        normals = []
        depths = []
        albedos = []
        masks = []
        camera_embeddings = []

        for i in range(len(name_list)):
            name = name_list[i]
            cam_name = name[0:len(name) - 4]

            #### camera embedding ######
            cam_pose_each = cam_pose[cam_name]["pose"]
            cam_pose_each = torch.tensor(cam_pose_each)
            cam_xyz_each = cam_pose_each[:, 3][:3]
            camera_embedding = cartesian_to_spherical_xyz(cam_xyz_each)
            camera_embeddings.append(camera_embedding)

            #### rgb/normal/depth/albedo ######
            rgb_name = os.path.join(rgb_path, name)
            depth_name = os.path.join(depth_path, name)
            normal_name = os.path.join(normal_path, name)
            emission_name = os.path.join(emission_path, name)

            rgb = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
            normal = cv2.imread(normal_name, cv2.IMREAD_UNCHANGED)
            emission = cv2.imread(emission_name, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)

            new_size = (1024, 1024)
            # Resize the image
            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_CUBIC)
            normal = cv2.resize(normal, new_size, interpolation=cv2.INTER_NEAREST)
            emission = cv2.resize(emission, new_size, interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)

            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal = normal.astype('float32')
            normal = normal.transpose(2, 0, 1) / 255.0

            depth = depth.astype('float32') / 1000.0

            mask = emission[:, :, 3:]
            mask = mask.astype('float32')
            mask = mask / 255.0
            mask = mask.squeeze(2)

            emission = emission[:, :, :3]
            emission = cv2.cvtColor(emission, cv2.COLOR_BGR2RGB)
            emission = emission.astype('float32')
            emission = emission.transpose(2, 0, 1) / 255.0

            rgb = rgb[:, :, :3]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype('float32')
            rgb = rgb.transpose(2, 0, 1) / 255.0

            emission = emission[:3, :, :]
            rgb = rgb[:3, :, :]

            emission[0, :, :][mask == 0] = 1.0
            emission[1, :, :][mask == 0] = 1.0
            emission[2, :, :][mask == 0] = 1.0

            rgb[0, :, :][mask == 0] = 1.0
            rgb[1, :, :][mask == 0] = 1.0
            rgb[2, :, :][mask == 0] = 1.0

            normal[0, :, :][mask == 0] = 1.0
            normal[1, :, :][mask == 0] = 1.0
            normal[2, :, :][mask == 0] = 1.0

            depth[:, :][mask == 0] = 0

            rgbs.append(rgb)
            albedos.append(emission)
            depths.append(depth)
            normals.append(normal)
            masks.append(mask)
        albedos = np.stack(albedos, axis=0)
        depths = np.stack(depths, axis=0)
        normals = np.stack(normals, axis=0)
        rgbs = np.stack(rgbs, axis=0)
        masks = np.stack(masks, axis=0)
        camera_embeddings = np.stack(camera_embeddings, axis=0)

        print("Save h5 to %s" % save_name_h5)

        with h5py.File(save_name_h5, 'w') as f:
            f.create_dataset('camera_embeddings', data=camera_embeddings,
                             compression='gzip', compression_opts=9)
            f.create_dataset('rgbs', data=rgbs,
                             compression='gzip', compression_opts=9)
            f.create_dataset('albedos', data=albedos,
                             compression='gzip', compression_opts=9)
            f.create_dataset('depths', data=depths,
                             compression='gzip', compression_opts=9)
            f.create_dataset('normals', data=normals,
                             compression='gzip', compression_opts=9)
            f.create_dataset('masks', data=masks,
                             compression='gzip', compression_opts=9)

        time.sleep(0.1)
    except:
        print("error occur!!!!")


def cartesian_to_spherical_xyz(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    z = np.sqrt(xy + xyz[2] ** 2)
    # for elevation angle defined from Z-axis down
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[1], xyz[0])
    return np.array([theta, azimuth, z, 1, 1, 1, 1])


def to_rgb_image_alpha(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(
            254, 255, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        mask_alpha = rgba.getchannel('A')
        img.paste(rgba, mask=mask_alpha)
        return img, mask_alpha
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def to_depth_image(depth_path, resolution):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (resolution, resolution),
                       interpolation=cv2.INTER_NEAREST)

    # 检查图像是否正确加载
    if depth is None:
        raise ValueError("图像加载失败，请检查路径和图像格式")
    # 将图像从 HWC 格式转换为 CHW 格式并转换为浮点数
    depth = depth.astype('float32')

    # 将图像转换为 PyTorch 张量
    depth_tensor = torch.from_numpy(depth)
    depth_tensor[depth_tensor > 20000] = 65535
    depth_tensor = depth_tensor / 1000
    return depth_tensor


def to_normal_image(normal_path, resolution):
    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    normal = cv2.resize(normal, (resolution, resolution),
                        interpolation=cv2.INTER_NEAREST)
    # 检查图像是否正确加载
    if normal is None:
        raise ValueError("图像加载失败，请检查路径和图像格式")
    # 将图像从 HWC 格式转换为 CHW 格式并转换为浮点数

    normal = normal.astype('float32')
    normal = normal.transpose(2, 0, 1)

    # 将图像转换为 PyTorch 张量
    normal_tensor = torch.from_numpy(normal)
    return normal_tensor


def to_normal_image_mask(normal_path, mask, resolution):
    # print("depth_path: ", depth_path)
    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    normal = cv2.resize(normal, (resolution, resolution),
                        interpolation=cv2.INTER_NEAREST)
    # 检查图像是否正确加载
    if normal is None:
        raise ValueError("图像加载失败，请检查路径和图像格式")
    # 将图像从 HWC 格式转换为 CHW 格式并转换为浮点数
    normal = normal.astype('float32')
    normal = normal.transpose(2, 0, 1)
    normal[0, :, :][mask == 0] = 255
    normal[1, :, :][mask == 0] = 255
    normal[2, :, :][mask == 0] = 255

    # 将图像转换为 PyTorch 张量
    normal_tensor = torch.from_numpy(normal)

    return normal_tensor


def read_json_to_list(json_name: str):
    with open(json_name, 'r') as fr:
        data = json.load(fr)
    data = data['data']['objaverse']
    render_folder_list = []
    mesh_instance_name_list = []
    for mesh_name in data.keys():
        mesh_instance_name_list.append(mesh_name)
        render_folder_list.append(data[mesh_name]["ImgDir"])
    return render_folder_list, mesh_instance_name_list


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--json_path", "-j",
                            required=True, help="json path.")
    arg_parser.add_argument("--save_path", "-s",
                            help="Path where h5 file saved!")
    arg_parser.add_argument('--pod_id', type=int, default=-1,
                            help='index of pods used in cluster')
    arg_parser.add_argument('--pod_num', type=int, default=-1,
                            help='total number of pods of cluster')
    arg_parser.add_argument('--pool_cnt', type=int, default=8,
                            help='multiprocessing pool cnt')
    args = arg_parser.parse_args()

    json_name = args.json_path
    save_path = args.save_path
    render_folder_list, mesh_instance_name_list = read_json_to_list(json_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pool = ThreadPoolExecutor(max_workers=args.pool_cnt, thread_name_prefix='h5_')

    if args.pod_num >= 0 and args.pod_id >= 0:
        mesh_path_len = len(render_folder_list)
        per_pod_len = mesh_path_len // args.pod_num
        idx_start = args.pod_id * per_pod_len
        if args.pod_id == args.pod_num - 1:
            idx_end = mesh_path_len
        else:
            idx_end = (args.pod_id + 1) * per_pod_len

        render_folder_list = render_folder_list[idx_start:idx_end]
        mesh_instance_name_list = mesh_instance_name_list[idx_start:idx_end]

        save_path = os.path.join(save_path, "pod_{}".format(args.pod_id))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        print("Rendering started using cluster with %i pods" % args.pod_num)
        print("Number of mesh on this pod: %i, (start: %i, end: %i)" %
              (len(render_folder_list), idx_start, idx_end))

    for index in range(len(render_folder_list)):
        render_folder = render_folder_list[index]
        mesh_instance_name = mesh_instance_name_list[index]
        print("Convert render folder at %s to h5 file %s.h5" %
              (render_folder, mesh_instance_name))

        pool.submit(h5_once, render_folder, mesh_instance_name, save_path)

    pool.shutdown()
    current_time_str = current_time()
    print("H5 conversion finished at %s" % (current_time_str))
