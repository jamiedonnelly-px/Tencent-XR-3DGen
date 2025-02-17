#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import OpenEXR
import cv2
import numpy as np
import tifffile
from PIL import Image

# change unit of depth from meter to millimeter
DEPTH_SCALE = 1000.0

# normal conversion: unit16_normal = (float_normal+1.0) * 30000.0
# use this scale to convert normal from [0,2] to [0,60000], which fits in uint16
NORMAL_SCALE = 30000.0
XYZ_SCALE = 30000.0


def __view_normal(normal_world, pose, world_system_bug_is_fixed=True):
    """
    convert normal from blender world coordinate system to camera coordinate system (view space)
    :param normal_world: normal in blender world coordinate system
    :param pose: camera pose, [3x4] or [4x4]
    :param world_system_bug_is_fixed: please keep TRUE; only set to FALSE when necessary; fix the bug that y and z axis are flipped
    :return: normal in camera coordinate system (view space)
    """
    R_c2w = np.array(pose)[:3, :3]
    if not world_system_bug_is_fixed:
        # flip y and z due to rendering bug, should be fixed now
        normal_world = normal_world * np.array([1.0, -1.0, -1.0])
    # convert to GL view space normal
    normal_view = normal_world @ R_c2w
    # view_normal_img = (normal_view.clip(-1, 1) + 1) * 125.0
    return normal_view


def to_srgb(v):
    """
    convert sRGB color value to linear space
    :param v: sRGB value
    :return: linear space value
    """
    thr = 0.0031308
    v_ = v.copy()
    v_[v <= thr] = v[v <= thr] * 12.92
    v_[v > thr] = 1.055 * (v[v > thr] ** .41667) - 0.055
    return v_.clip(0, 1)


def read_channel(exr_data, c, dtype=np.float32):
    """
    read a channel's data from an exr file
    :param exr_data: exr data struct
    :param c: channel name
    :param dtype: data type
    :return: channel data as numpy array
    """
    b = exr_data.header()['dataWindow']
    shape = [b.max.y - b.min.y + 1, b.max.x - b.min.x + 1]
    data = exr_data.channel(c)
    data = np.frombuffer(data, dtype=dtype).reshape(shape)
    return data


def __read_color(exr_data, blend_flag=False):
    """
    read rendered image data from an exr file
    :param exr_data: exr data struct
    :param blend_flag: toggle between using data in RenderLayer or ViewLayer; useful when parsing render results when rendering from third-party blend file
    :return: color data as a numpy array
    """
    if blend_flag:
        rgb = [read_channel(exr_data, f'RenderLayer.Combined.{c}', dtype=np.float16) for c in [
            'R', 'G', 'B', 'A']]
    else:
        rgb = [read_channel(exr_data, f'ViewLayer.Combined.{c}', dtype=np.float16) for c in [
            'R', 'G', 'B', 'A']]
    rgb = to_srgb(np.stack(rgb, 2))
    return (rgb * 255).astype(np.uint8)


def __read_depth(exr_data, blend_flag=False):
    """
    read rendered depth data from an exr file
    :param exr_data: exr data struct
    :param blend_flag: toggle between using data in RenderLayer or ViewLayer; useful when parsing render results when rendering from third-party blend file
    :return: depth data as a numpy array
    """
    if blend_flag:
        depth_z = read_channel(exr_data, 'RenderLayer.Depth.Z')
        rgb_a = read_channel(exr_data, 'RenderLayer.Combined.A', dtype=np.float16)
    else:
        depth_z = read_channel(exr_data, 'ViewLayer.Depth.Z')
        rgb_a = read_channel(exr_data, 'ViewLayer.Combined.A', dtype=np.float16)
    depth = depth_z.copy()
    invalid_depth = depth > 1e2
    depth[invalid_depth] = 0  # set the depth in invalid region (i.e. bachground) to 0
    return depth


def __read_position(exr_data, blend_flag=False):
    """
    read rendered xyz data from an exr file
    :param exr_data: exr data struct
    :param blend_flag: toggle between using data in RenderLayer or ViewLayer; useful when parsing render results when rendering from third-party blend file
    :return: xyz data as a numpy array
    """
    if blend_flag:
        xyz_data = [read_channel(exr_data, f'RenderLayer.Position.{c}') for c in ['X', 'Y', 'Z']]
    else:
        xyz_data = [read_channel(exr_data, f'ViewLayer.Position.{c}') for c in ['X', 'Y', 'Z']]
    xyz = np.stack(xyz_data, 2)
    return xyz


def __read_normal(exr_data, blend_flag=False):
    """
    read rendered normal data from an exr file
    :param exr_data: exr data struct
    :param blend_flag: toggle between using data in RenderLayer or ViewLayer; useful when parsing render results when rendering from third-party blend file
    :return: normal data as a numpy array
    """
    if blend_flag:
        normal = [read_channel(exr_data, f'RenderLayer.Normal.{c}') for c in ['X', 'Y', 'Z']]
    else:
        normal = [read_channel(exr_data, f'ViewLayer.Normal.{c}') for c in ['X', 'Y', 'Z']]

    normal_raw = np.stack(normal, 2)
    normal = normal_raw
    H, W, _ = normal.shape
    normal = normal.reshape((-1, 3)).T
    # change normal from blender to opencv please uncomment this two lines
    # T = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    # normal = np.dot(T, normal)
    normal = normal.T.reshape((H, W, 3))
    return normal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse exr files.')
    parser.add_argument('--exr_file', type=str,
                        help='path to exr file to be parsed')
    parser.add_argument('--camera_info_path', type=str, default="",
                        help='path to external camera info json')
    parser.add_argument('--blend', action='store_true',
                        help='use this when rendering .blend files')
    parser.add_argument('--remove_exr', action='store_true',
                        help='remove exr files after parse')
    parser.add_argument('--format', type=str, default="png",
                        help='file format for depth or normal, choose between png/tiff/png_16bit')
    args = parser.parse_args()

    exr_file = args.exr_file
    camera_info_path = args.camera_info_path
    blend_flag = args.blend
    exr_filename = os.path.split(exr_file)[1]
    exr_file_basename = os.path.splitext(exr_filename)[0]
    camera_name = exr_filename[:-4]

    exr_dir_path = Path(exr_file).parent.absolute()
    dir_path = Path(exr_dir_path).parent.absolute()
    color_dir_path = os.path.join(dir_path, "color")
    if not os.path.exists(color_dir_path):
        os.mkdir(color_dir_path)

    depth_dir_path = os.path.join(dir_path, "depth")
    if not os.path.exists(depth_dir_path):
        os.mkdir(depth_dir_path)

    normal_dir_path = os.path.join(dir_path, "normal")
    if not os.path.exists(normal_dir_path):
        os.mkdir(normal_dir_path)

    view_normal_dir_path = os.path.join(dir_path, "view_normal")
    if not os.path.exists(view_normal_dir_path):
        os.mkdir(view_normal_dir_path)

    xyz_dir_path = os.path.join(dir_path, "xyz")
    if not os.path.exists(xyz_dir_path):
        os.mkdir(xyz_dir_path)

    # pose info used in view normal generation
    if len(camera_info_path) < 1 or not os.path.exists(camera_info_path):
        camera_info_path = os.path.join(dir_path, "cam_parameters.json")
    with open(camera_info_path, "r", encoding='utf-8') as f:
        cam_info = json.load(f)

    exr = OpenEXR.InputFile(exr_file)

    color = __read_color(exr, blend_flag=blend_flag)
    depth = __read_depth(exr, blend_flag=blend_flag)
    normal = __read_normal(exr, blend_flag=blend_flag)
    normal_view = __view_normal(normal, cam_info["poses"][camera_name]["pose"])
    xyz = __read_position(exr, blend_flag=blend_flag)

    image_format = args.format
    im = Image.fromarray(color)
    im.save(os.path.join(color_dir_path, camera_name + ".png"))

    # tiff format supprts float16, so it's merely lossless when storing normal and xyz
    # LZMA has higher compression ratio comparing with zstd
    if image_format == "tif":
        depth_image_path = os.path.join(depth_dir_path, camera_name + "." + image_format)
        tifffile.imwrite(depth_image_path, depth.astype(np.float16), compression='LZMA')
    else:
        depth_image_path = os.path.join(depth_dir_path, camera_name + ".png")
        depth_16bit = (depth * DEPTH_SCALE).astype(np.uint16)
        im = Image.fromarray(depth_16bit)
        im.save(depth_image_path)

    if image_format == "tif":
        normal_image_path = os.path.join(normal_dir_path, camera_name + "." + image_format)
        tifffile.imwrite(normal_image_path, normal.astype(np.float16), compression='LZMA')
    elif image_format == "png_16bit":
        normal_image_path = os.path.join(normal_dir_path, camera_name + ".png")
        normal_16bit = ((normal + 1) * NORMAL_SCALE).astype(np.uint16)
        cv2.imwrite(normal_image_path, normal_16bit)
    else:
        normal_image_path = os.path.join(normal_dir_path, camera_name + "." + image_format)
        normal_viz = ((normal.clip(-1, 1) + 1) * 125).astype(np.uint8)
        im = Image.fromarray(normal_viz)
        im.save(normal_image_path)

    if image_format == "tif":
        view_normal_image_path = os.path.join(view_normal_dir_path, camera_name + "." + image_format)
        tifffile.imwrite(view_normal_image_path, normal_view.astype(np.float16), compression='LZMA')
    elif image_format == "png_16bit":
        view_normal_image_path = os.path.join(view_normal_dir_path, camera_name + ".png")
        view_normal_16bit = ((normal_view.clip(-1, 1) + 1) * NORMAL_SCALE).astype(np.uint16)
        cv2.imwrite(view_normal_image_path, view_normal_16bit)
    else:
        view_normal_image_path = os.path.join(view_normal_dir_path, camera_name + "." + image_format)
        normal_view_viz = ((normal_view.clip(-1, 1) + 1) * 125.0).astype(np.uint8)
        view_normal_im = Image.fromarray(normal_view_viz)
        view_normal_im.save(view_normal_image_path)

    if image_format == "tif":
        xyz_image_path = os.path.join(xyz_dir_path, camera_name + "." + image_format)
        tifffile.imwrite(xyz_image_path, xyz.astype(np.float16), compression='LZMA')
    elif image_format == "png_16bit":
        xyz_image_path = os.path.join(xyz_dir_path, camera_name + ".png")
        xyz_16bit = ((xyz.clip(-1, 1) + 1) * NORMAL_SCALE).astype(np.uint16)
        cv2.imwrite(xyz_image_path, xyz_16bit)
    else:
        xyz_image_path = os.path.join(xyz_dir_path, camera_name + "." + image_format)
        xyz_8bit = ((xyz.clip(-1, 1) + 1) * 125.0).astype(np.uint8)
        xyz_im = Image.fromarray(xyz_8bit)
        xyz_im.save(xyz_image_path)

    if args.remove_exr:
        os.remove(exr_file)
