"""
use data from senbo with images and pcd h5 for train diffusion model
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import itertools
from PIL import Image
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
from torchvision.utils import make_grid
from tqdm import tqdm
import cv2
import random
import json
import os
import shutil
import h5py
import imgaug.augmenters as iaa
from transformers import AutoImageProcessor
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets_diffusion.image_aug import gray_scketch_aug

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_train_path(origin_path, train_path, force_copy=False):
    """
    params:
        origin_path: path save in other machine
        train_path: path save in the training machine
        force_copy: if force copy
    return:
        if train_path exists, return train_path; else return origin_path
    """
    if os.path.exists(train_path) and (not force_copy):
        return train_path
    else:
        if train_path.endswith("sample.h5"):
            with h5py.File(origin_path, 'r') as h5file:
                surface_points = np.asarray(h5file["surface_points"])
                surface_normals = np.asarray(h5file["surface_normals"])
            surface_normal = np.concatenate(
                [surface_points, surface_normals], axis=1)
            chunk_size_y = surface_normal.shape[1]
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            with h5py.File(train_path, 'w') as h5file:
                h5file.create_dataset(name="surface_points_normals", data=surface_normal,
                                      compression='gzip', chunks=(50000, chunk_size_y))
        else:
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            shutil.copy(origin_path, train_path)
        return origin_path


def repadding_rgba_image(image, rescale=True, ratio=0.8, bg_color=255, center=True, aug_resize=True, aug_resize_prob=0.5):
    """repadding rgba image with augmentation

    Args:
        image (str| PIL.Image): image path or PIL.Image
        rescale (bool, optional): if rescale. Defaults to True.
        ratio (float, optional): foreground ratio. Defaults to 0.8.
        bg_color (int, optional): backgroud color. Defaults to 255.
        center (bool, optional): if centerize the fore ground obj. Defaults to True.
        aug_resize (bool, optional): if random resize for aug. Defaults to True.
        aug_resize_prob (float, optional): if aug_resize, the aug_resize_prob will work. Defaults to 0.5.

    Returns:
        PIL.Image: image after padding and aug
    """
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except:
            return -1
    resize_scale_min = 0.5
    in_w = image.width
    in_h = image.height
    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])

    white_bg_image = Image.new('RGB', size=(
        in_w, in_h), color=(bg_color, bg_color, bg_color))
    white_bg_image.paste(image, (0, 0), mask=image)
    if not center:
        rgba_image = np.concatenate(
            [np.array(white_bg_image), np.array(image)[..., -1:]], axis=-1)
        rgba_image = rgba_image.astype(np.uint8)
        return rgba_image
    max_size = max(w, h)
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.ones((side_len, side_len, 3), dtype=np.uint8) * bg_color
    mask = np.zeros((side_len, side_len, 1), dtype=np.uint8)
    center = side_len // 2

    padded_image[
        center - h // 2: center - h // 2 + h, center - w // 2: center - w // 2 + w
    ] = np.array(white_bg_image)[y: y + h, x: x + w]

    mask[
        center - h // 2: center - h // 2 + h, center - w // 2: center - w // 2 + w
    ] = np.array(image)[..., -1:][y: y + h, x: x + w]

    rgba_image = np.concatenate([padded_image, mask], axis=-1)
    rgba_image = Image.fromarray(rgba_image)

    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])
    max_edge = max(w, h)
    area_ratio = max_edge / max(in_w, in_h)
    if aug_resize and (random.random() < aug_resize_prob) and (area_ratio > 0.5):
        downscale = 1 - random.random() * resize_scale_min
        rgba_w = rgba_image.width
        rgba_h = rgba_image.height
        rgba_image = rgba_image.resize(
            (int(rgba_w*downscale), int(rgba_h*downscale)))
        rgba_image = rgba_image.resize((rgba_w, rgba_h))

    return rgba_image


def get_brightness_scale(cond_image, target_image):
    """get brightness scale based on cond_image for matching target_image

    Args:
        cond_image (PIL.Image): image need matching target_image
        target_image (PIL.Image): reference brightness image

    Returns:
        float: brightness scale
    """
    cond_image_a = np.array(cond_image)
    target_image_a = np.array(target_image)

    cond_image_a_gray = cv2.cvtColor(cond_image_a, cv2.COLOR_RGBA2GRAY)
    target_image_a_gray = cv2.cvtColor(target_image_a, cv2.COLOR_RGBA2GRAY)

    cond_brightness = np.mean(
        cond_image_a_gray, where=cond_image_a[..., -1] > 0)
    target_brightness = np.mean(
        target_image_a_gray, where=target_image_a[..., -1] > 0)

    brightness_scale = cond_brightness / (target_brightness + 0.000001)
    return min(brightness_scale, 1.0)


def lighting_fast(img, light, mask_img=None):
    """
        img: rgb order, shape:[h, w, 3], range:[0, 255]
        light: [-100, 100]
        mask_img: shape:[h, w], range:[0, 255]
    """
    assert -100 <= light <= 100
    max_v = 4
    bright = (light/100.0)/max_v
    mid = 1.0+max_v*bright
    # print('bright: ', bright, 'mid: ', mid)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    thresh = gray * gray * (mask_img.astype(np.float32) / 255.0)
    t = np.mean(thresh, where=(thresh > 0.1))

    mask = np.where(thresh > t, 255, 0).astype(np.float32)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    mask[mask_img == 0] = 0
    # cv2.imwrite("mask4.png", mask)
    brightrate = np.where(mask == 255.0, bright, (1.0/t*thresh)*bright)
    mask = np.where(mask == 255.0, mid, (mid-1.0)/t*thresh+1.0)
    img_float = img/255.0
    img_float = np.power(
        img_float, 1.0/mask[:, :, np.newaxis])*(1.0/(1.0-brightrate[:, :, np.newaxis]))
    img_float = np.clip(img_float, 0, 1.0)*255.0
    return img_float.astype(np.uint8)


def load_and_aug_image(image_path, aug_bg=False, color_bg=127, imageaug_seq=None):
    """load image and augmentation

    Args:
        image_path (str): image path
        aug_bg (bool, optional): if aug background. Defaults to False.
        color_bg (int, optional): background color. Defaults to 127.
        imageaug_seq (imgaug.augmenters, optional): imageaug function. Defaults to None.

    Returns:
        PIL.Image: image after augmentation
    """
    image = Image.open(image_path)
    image = image.resize(size=(512, 512))
    width = image.width
    height = image.height
    if aug_bg:
        color_center = random.randint(color_bg - 10, color_bg + 10)
        shift_max = 3
        color_random_shift = [random.randint(-shift_max, shift_max),
                              random.randint(-shift_max, shift_max),
                              random.randint(-shift_max, shift_max)]
        color = [x + color_center for x in color_random_shift]

        rgb_shift_max = 3
        image_shift_rgb = np.random.randint(-rgb_shift_max,
                                            rgb_shift_max, (height, width, 3))
        bg_color_img = np.concatenate([np.ones((height, width, 1)) * color[0],
                                       np.ones((height, width, 1)) * color[1],
                                       np.ones((height, width, 1)) * color[2]], axis=-1)
        bg_color_img += image_shift_rgb
        img2 = Image.fromarray(bg_color_img.astype(np.uint8))
        img2.paste(image, (0, 0), mask=image)
        image = img2
    else:
        img2 = Image.new('RGB', size=(width, height),
                         color=(color_bg, color_bg, color_bg))
        img2.paste(image, (0, 0), mask=image)
        image = img2

    if imageaug_seq is not None:
        image = Image.fromarray(imageaug_seq(
            images=np.array(image)[None, ...])[0])

    image = image.convert("RGB")
    return image


def to_rgb_image(maybe_rgba: Image.Image, bg_color=127, edge_aug_threshold=0, bright_scale=None):
    """convert image to rgb image

    Args:
        maybe_rgba (Image.Image): image for conversion
        bg_color (int, optional): backgroud color. Defaults to 127.
        edge_aug_threshold (int, optional): edge aug threshold. Defaults to 0.
        bright_scale (float, optional): bright scale. Defaults to None.

    Raises:
        ValueError: if image mode is not RGBA or RGB, raise error

    Returns:
        PIL.Image: rgb image after conversion and augmentation
    """
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # img = np.random.randint(random_grey_low, random_grey_high, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.ones([rgba.size[1], rgba.size[0], 3],
                      dtype=np.uint8) * bg_color
        img = Image.fromarray(img, 'RGB')

        # bright adapt
        if bright_scale is not None:
            rgba_array = np.array(rgba)
            rgb = cv2.convertScaleAbs(
                rgba_array[..., :3], alpha=bright_scale, beta=0)
            rgb = Image.fromarray(rgb)
            img.paste(rgb, mask=rgba.getchannel('A'))
        else:
            img.paste(rgba, mask=rgba.getchannel('A'))

        # edge augmentation
        if edge_aug_threshold > 0 and (random.random() < edge_aug_threshold):
            mask_img = np.array(rgba.getchannel('A'))
            mask_img[mask_img > 0] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            iterration_num = random.randint(1, 2)
            mask_img_small = cv2.erode(
                mask_img, kernel, iterations=iterration_num)
            mask_img_edge = mask_img - mask_img_small
            mask_img_edge = np.concatenate(
                [mask_img_edge[..., None]]*3, axis=-1) / 255.0
            rand_color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            img_array = np.array(img) * (1 - mask_img_edge) + \
                rand_color * mask_img_edge
            img = Image.fromarray(img_array.astype(np.uint8))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def get_stdpose_idx(image_azimuth):
    """get std pose idx

    Args:
        image_azimuth (float): azimuth of cond image

    Returns:
        int: idx of target pose
    """
    if (image_azimuth >= 45 and image_azimuth < 135):
        return 1
    elif (image_azimuth >= 135 and image_azimuth < 225):
        return 2
    elif (image_azimuth >= 225 and image_azimuth < 315):
        return 3
    else:
        return 0


def rotate(pcd, pcd_transpose_matrix):
    """rotate pointcloud with transpose matrix

    Args:
        pcd (np.ndarray): point cloud with shape (N, 3)
        pcd_transpose_matrix (np.ndarray): transpose matrix with shape (4, 4)

    Returns:
        np.ndarray: rotated point cloud with shape (N, 3)
    """
    pcd_norm = np.ones((pcd.shape[0], 4), dtype=np.float32)
    pcd_norm[:, :3] = pcd
    pcd_norm = pcd_norm @ pcd_transpose_matrix.T
    pcd_rotate = pcd_norm[:, :3]
    return pcd_rotate


def rotate_pcd_normal(pcd, cam_dict, img_idx,
                      azimuth_list=None,
                      align_type="relative"):
    """
    params:
        pcd: point clouds
        cam_dict: cam_dicts save cam info
        img_idx: image index
        azimuth_list: azimuth list
        align_type:  "relative" | "nearest_azimuth" | "zero_elevation" | "absolute"
    return:
        the rotated point clouds
    """

    if align_type == "absolute":
        return pcd, None, None

    geo_pcd_points = pcd[:, :3]
    geo_pcd_normal = pcd[:, 3:]

    x_rotate90 = np.array([[1, 0, 0, 0],
                           [0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],])

    x_rotate_inv90 = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1],])

    def get_y_rotate(alpha):
        alpha_pi = (alpha / 180) * math.pi
        sin_alpha = math.sin(alpha_pi)
        cos_alpha = math.cos(alpha_pi)
        y_rotate_matrix = np.array([[cos_alpha, 0, -sin_alpha, 0],
                                    [0,        1,     0,       0],
                                    [sin_alpha, 0, cos_alpha, 0],
                                    [0,        0,     0,      1],])
        return y_rotate_matrix

    # cam_pose0 = np.array(cam_dict["cam-0000"]["pose"]).astype(np.float32)
    cam_pose0 = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, -4.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    cam_pose1 = np.array([[0.0, 0.0, -1.0, 4.0],
                          [1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    cam_pose2 = np.array([[-1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 4.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    cam_pose3 = np.array([[0.0, 0.0, 1.0, -4],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
    cam_stdpose_list = [cam_pose0, cam_pose1, cam_pose2, cam_pose3]

    camk = np.array(cam_dict[f"cam-{str(img_idx).zfill(4)}"]["k"])
    camk = camk.astype(np.float32)

    if align_type == "nearest_azimuth":
        image_azimuth = azimuth_list[img_idx]
        # print(image_azimuth)
        cam_stdpose_idx = get_stdpose_idx(image_azimuth)
        # print(cam_stdpose_idx)
        cam_pose = cam_stdpose_list[cam_stdpose_idx]
        pcd_transpose_matrix = cam_pose0 @ np.linalg.inv(cam_pose)
        pcd_transpose_matrix = x_rotate_inv90 @ pcd_transpose_matrix @ x_rotate90

        geo_pcd_points_rotate = rotate(geo_pcd_points, pcd_transpose_matrix)
        geo_pcd_normal_rotate = rotate(geo_pcd_normal, pcd_transpose_matrix)
    elif align_type == "relative":
        cam_pose = np.array(
            cam_dict[f"cam-{str(img_idx).zfill(4)}"]["pose"]).astype(np.float32)
        pcd_transpose_matrix = cam_pose0 @ np.linalg.inv(cam_pose)
        pcd_transpose_matrix = x_rotate_inv90 @ pcd_transpose_matrix @ x_rotate90

        geo_pcd_points_rotate = rotate(geo_pcd_points, pcd_transpose_matrix)
        geo_pcd_normal_rotate = rotate(geo_pcd_normal, pcd_transpose_matrix)
    elif align_type == "zero_elevation":
        image_azimuth = azimuth_list[img_idx]
        azim_rotate_matrix = get_y_rotate(image_azimuth)
        geo_pcd_points_rotate = rotate(geo_pcd_points, azim_rotate_matrix)
        geo_pcd_normal_rotate = rotate(geo_pcd_normal, azim_rotate_matrix)
    else:
        exit("wrong align_type, align_type should be (relative, zero_elevation, nearest_azimuth)")

    pcd_normal = np.concatenate(
        [geo_pcd_points_rotate, geo_pcd_normal_rotate], axis=-1)
    return pcd_normal.astype(np.float32), cam_pose0, camk


def get_cam(cam_json_path):
    with open(cam_json_path, 'r') as fr:
        cam_dict = json.load(fr)
    return cam_dict


def get_cam_info(cam_info_json_path):
    with open(cam_info_json_path, 'r') as fr:
        cam_dict = json.load(fr)
    """
    SDS_RANDOM: RANDOM
    RSVC: RSVC
    """
    return cam_dict["config"]["RC"]["real_azimuth"]


def transform_target(images, scale=None, **kwargs):
    """
    Apply optional translation, scaling, and elastic transformation to a batch of images in that order.
    Transformed images are padded with its border values.

    Parameters:
    - images: Tensor of shape [batch, channel, H, W], the batch of images to transform.
    - mask: mask  for caculate scale rigion
    - translate: Optional; Tensor of shape [batch, 2] or [1, 2] for broadcasting. The translations should be in the range [-1, 1], representing the fraction of translation relative to the image dimensions.
    - scale: Optional; Tensor of shape [batch, 1] or [1, 1] for broadcasting. The scale factors; a value greater than 1 means zooming in (making objects larger and cropping), less than 1 means zooming out (fitting more into the view).

    Returns:
    - Transformed images of shape [batch, channel, H, W], where each transformation has been applied considering the backward warping flow, ensuring correct sampling and transformation of the image data.
    """

    H, W, _ = images.shape
    if scale is not None:
        images_scale = np.zeros_like(images)
        images1 = cv2.resize(images, dsize=None, fx=scale,
                             fy=scale, interpolation=cv2.INTER_LINEAR)
        rescale_h, rescale_w, _ = images1.shape
        if scale >= 1.0:
            start_h = (rescale_h - H) // 2
            start_w = (rescale_w - W) // 2
            end_h = start_h + H
            end_w = start_w + W
            images_scale = images1[start_h:end_h, start_w:end_w]
        else:
            start_h = (H - rescale_h) // 2
            start_w = (W - rescale_w) // 2
            end_h = start_h + rescale_h
            end_w = start_w + rescale_w
            images_scale[start_h:end_h, start_w:end_w] = images1
            return images_scale
    else:
        return images


class Diffusion3D_V6_1Cond(Dataset):
    def __init__(self,
                 configs,
                 data_type="train",
                 num_validation_samples=None
                 ) -> None:

        exp_dir = configs.get("exp_dir", None)
        assert exp_dir is not None
        self.exp_dir = exp_dir
        print(f"exp_dir: {exp_dir}")
        data_config = configs["data_config"]
        self.image_json_list = data_config["image_json_list"]
        self.pcd_json_list = data_config["pcd_json_list"]
        self.caption_json = data_config["caption_json"]
        self.cond_idx_list = data_config["cond_idx_list"]
        self.images_num_per_group = data_config["images_num_per_group"]
        self.group_idx_list = data_config["group_idx_list"]
        self.load_from_cache_last = data_config.get(
            "load_from_cache_last", False)
        self.sample_points_num = data_config.get("sample_points_num")
        floor_data_list_path = data_config.get("floor_data_list_path", None)
        self.shuffle = data_config.get("shuffle", False)
        self.validation = (data_config.get("data_type", data_type) == "test")
        self.target_align_type = data_config.get("target_align_type")
        self.exclude_class_list = data_config.get("exclude_class_list", None)
        self.include_class_list = data_config.get("include_class_list", None)

        self.num_view_perobj = data_config.get("num_view_perobj")
        if self.exclude_class_list is not None:
            print(f"exclude class names: {self.exclude_class_list}")

        self.exclude_class_list = data_config.get("exclude_class_list", None)
        if self.exclude_class_list is not None:
            print(f"exclude_class_list names: {self.exclude_class_list}")
        if self.include_class_list is not None:
            print(f"include_class_list names: {self.include_class_list}")

        self.tail_json_path = data_config.get("tail_json_path", None)
        if self.tail_json_path is not None:
            with open(self.tail_json_path, 'r') as fr:
                self.tail_dict = json.load(fr)["data"]
            print(f"tail_classnum: {len(list(self.tail_dict.keys()))}")
        else:
            self.tail_dict = None
            print(f"no tail_json_path input")

        self.exclude_objaverse_list_path = data_config.get(
            "exclude_objaverse_list_path", None)

        self.cloth_classname_list = None

        self.sample_points_num = data_config.get("sample_points_num")
        self.points_chunk_num = 500000 // self.sample_points_num - 1

        train_json_save_path = os.path.join(exp_dir, "train.json")
        test_json_save_path = os.path.join(exp_dir, "test.json")

        if floor_data_list_path is not None:
            with open(floor_data_list_path, "r") as fr:
                self.floor_data_list = json.load(fr)
        else:
            self.floor_data_list = []

        if not self.load_from_cache_last:
            print("rechecking data... ")
            all_data_list = self.read_data()
            data_train_list, data_test_list = self.__split_train_test(
                all_data_list)

            dataset_list_train = list(itertools.chain(*data_train_list))
            dataset_list_test = list(itertools.chain(*data_test_list))
            dataset_list_train.sort()
            dataset_list_test.sort()

            print("writing load cache")
            with open(train_json_save_path, "w") as fw:
                json.dump(dataset_list_train, fw, indent=2)
            with open(test_json_save_path, "w") as fw:
                json.dump(dataset_list_test, fw, indent=2)
        else:
            print("load from cache last")
            with open(train_json_save_path, 'r') as fr:
                dataset_list_train = json.load(fr)
            with open(test_json_save_path, 'r') as fr:
                dataset_list_test = json.load(fr)

        if not self.validation:
            self.all_objects = dataset_list_train
        else:
            self.all_objects = dataset_list_test
            if num_validation_samples is not None:
                self.all_objects = self.all_objects[:num_validation_samples]

        self.read_cloth_json(self.image_json_list)

        self.dino_processor = AutoImageProcessor.from_pretrained(
            configs["dino_config"]["pretrain_dir"], local_files_only=True)
        self.clip_processor = AutoImageProcessor.from_pretrained(
            configs["clip_config"]["pretrain_dir"], local_files_only=True)
        self.imgaug_seq = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.00, 0.03),
                                nb_rows=(2, 4), nb_cols=(2, 4))
        ])

        if data_type == "train":
            print("dataset for training, shuffling")
            random.shuffle(self.all_objects)
        print("loading", len(self.all_objects), " objects in the dataset")

    def __len__(self):
        return len(self.all_objects)

    def read_cloth_json(self, json_path_list):
        for json_path in json_path_list:
            if "cloth" not in json_path:
                continue
            with open(json_path, 'r') as fr:
                sub_dict = json.load(fr)
        self.cloth_classname_list = list(sub_dict["data"].keys())
        print(f"cloth classname list: {self.cloth_classname_list}")
    

    def read_json_list(self, json_path_list):
        json_dict = None
        for json_path in json_path_list:
            with open(json_path, 'r') as fr:
                sub_dict = json.load(fr)
            if json_dict is None:
                json_dict = sub_dict
            else:
                for classname, classdict in sub_dict["data"].items():
                    if classname in json_dict["data"].keys():
                        json_dict["data"][classname].update(
                            sub_dict["data"][classname])
                    else:
                        json_dict["data"].update(sub_dict["data"])
            if "clothes" in json_path:
                self.cloth_classname_list = list(sub_dict["data"].keys())
                print(f"cloth classname list: {self.cloth_classname_list}")
        return json_dict

    def read_data(self):
        image_data_dict = self.read_json_list(self.image_json_list)["data"]
        num = 0
        for classname, classdict in image_data_dict.items():
            num = num + len(list(classdict.keys()))
        print(f"image_dict item num: {num}")
        print(f"all image class name: {image_data_dict.keys()}")

        pcd_data_dict = self.read_json_list(self.pcd_json_list)["data"]
        num = 0
        for classname, classdict in pcd_data_dict.items():
            num = num + len(list(classdict.keys()))
        print(f"pcd_data_dict item num: {num}")

        with open(self.caption_json, 'r') as fr:
            caption_all_dict = json.load(fr)
        num = 0
        for classname, classdict in caption_all_dict.items():
            num = num + len(list(classdict.keys()))
        print(f"caption_all_dict item num: {num}")

        if self.exclude_objaverse_list_path is None:
            exclude_objaverse_id_list = []
        else:
            with open(self.exclude_objaverse_list_path, 'r') as fr:
                exclude_objaverse_id_list = json.load(fr)
        print(f"exclude objaverse num: {len(exclude_objaverse_id_list)}")

        all_data_list = []
        for classname, classdict in tqdm(image_data_dict.items()):
            if self.exclude_class_list is not None and classname in self.exclude_class_list:
                continue
            if self.include_class_list is not None and classname not in self.include_class_list:
                continue
            class_data_list = []
            for objname, objdict in tqdm(classdict.items()):
                if classname == "objaverse" and objname in exclude_objaverse_id_list:
                    continue
                if "ImgDir" not in objdict:
                    continue
                if classname not in pcd_data_dict:
                    continue
                if objname not in pcd_data_dict[classname]:
                    continue
                if "GeoPcd" not in pcd_data_dict[classname][objname]:
                    continue
                if self.tail_dict is not None and classname in self.tail_dict.keys() and objname in self.tail_dict[classname]:
                    continue

                image_dir = objdict["ImgDir"]
                image_dir_train = objdict["ImgDir_train"]
                pcd_dir = pcd_data_dict[classname][objname]["GeoPcd"]
                pcd_dir_train = pcd_data_dict[classname][objname]["GeoPcd_train"]

                for groupi in self.group_idx_list:
                    class_data_list.append([classname, objname, image_dir, pcd_dir, groupi,
                                            image_dir_train, pcd_dir_train])

            all_data_list.append(class_data_list)
        return all_data_list

    def __split_train_test(self, dataset_list, test_threshold=0.000001, test_min_num=1):
        train_list, test_list = [], []
        for i, class_dataset_list in enumerate(dataset_list):
            if len(class_dataset_list) == 0:
                print("dataset objs num is 0")
                continue
            class_name = class_dataset_list[0][0]
            num = len(class_dataset_list)
            if num < test_min_num*3:
                print(
                    f"{class_name} dataset objs num is little than test_min_num*3, all {num} for train")
                continue
            test_num = int(max(num * test_threshold, test_min_num))
            test_list.append(class_dataset_list[0:test_num])
            train_list.append(class_dataset_list[test_num:])
            print(
                f"class {class_name} split {num-test_num} for train and {test_num} for test")
        return train_list, test_list

    def __getitem__(self, index):
        classname, objname, image_dir, pcd_dir, groupi, \
            image_dir_train, pcd_dir_train = self.all_objects[index]

        # 0.1 for cloth data
        if self.cloth_classname_list is not None and classname in self.cloth_classname_list and random.random() < 0.95:
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

        # get image conditions
        condi = random.choice(self.cond_idx_list)
        image_sub_idx_cond_list = [
            self.images_num_per_group * groupi + x for x in condi]
        random.shuffle(image_sub_idx_cond_list)
        image_cond_path_list = []
        image_cond_list = []
        cond_images = []

        for image_idx in image_sub_idx_cond_list[:self.num_view_perobj]:
            img_cond_train_path = os.path.join(
                image_dir_train, "color", f"cam-{str(image_idx).zfill(4)}.png")
            img_cond_path = os.path.join(
                image_dir, "color", f"cam-{str(image_idx).zfill(4)}.png")
            img_cond_path_final = get_train_path(
                img_cond_path, img_cond_train_path)

            image_cond_path_list.append(img_cond_path_final)
            try:
                cond_image_alpha = Image.open(img_cond_path_final)
                cond_image_alpha_image = repadding_rgba_image(cond_image_alpha, rescale=True,
                                                              ratio=0.9, bg_color=127, center=True,
                                                              aug_resize=True, aug_resize_prob=0.5)
                cond_image_alpha_rescale = cond_image_alpha_image.resize(
                    (512, 512))
                cond_image_alpha_rescale = gray_scketch_aug(
                    cond_image_alpha_rescale, prob=0.04)
                rand_bright_scale = random.random() * (1.2 - 0.8) + 0.4
                image_cond = to_rgb_image(
                    cond_image_alpha_rescale, 127, bright_scale=rand_bright_scale)

            except:
                get_train_path(
                    img_cond_path, img_cond_train_path, force_copy=True)
                return self.__getitem__(np.random.randint(0, self.__len__() - 1))
            image_cond_list.append(image_cond)

        image_dino_conds = self.dino_processor(
            image_cond_list, return_tensors="pt")["pixel_values"]
        image_clip_conds = self.clip_processor(
            image_cond_list, return_tensors="pt")["pixel_values"]

        # get pointcloud
        pcd_train_path = os.path.join(pcd_dir_train, "sample.h5")
        pcd_path = os.path.join(pcd_dir, "sample.h5")

        try:
            pcd_path_final = get_train_path(pcd_path, pcd_train_path)
            if pcd_path_final == pcd_train_path:
                with h5py.File(pcd_path_final, 'r') as h5file:
                    rand_idx = np.random.randint(0, 4)
                    surface_origin = np.asanyarray(
                        h5file["surface_points_normals"][rand_idx*100000:(rand_idx+1)*100000])
                    # random sampling
                    rng = np.random.default_rng()
                    ind = rng.choice(
                        surface_origin.shape[0], self.sample_points_num, replace=False)
                    surface_origin = surface_origin[ind]
            else:
                with h5py.File(pcd_path_final, 'r') as h5file:
                    rand_idx = np.random.randint(0, self.points_chunk_num)
                    surface_points = np.asanyarray(h5file["surface_points"][
                        rand_idx*self.sample_points_num:(rand_idx+1)*self.sample_points_num])
                    surface_normals = np.asanyarray(h5file["surface_normals"][
                        rand_idx*self.sample_points_num:(rand_idx+1)*self.sample_points_num])
                surface_origin = np.concatenate(
                    [surface_points, surface_normals], axis=1)

        except:
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

        cam_json_path = os.path.join(image_dir, "cam_parameters.json")
        cam_json_train_path = os.path.join(
            image_dir_train, "cam_parameters.json")
        cam_json_path_final = get_train_path(
            cam_json_path, cam_json_train_path)

        cam_info_json_path = os.path.join(
            image_dir, "internal_cam_parameters.json")
        cam_info_json_train_path = os.path.join(
            image_dir_train, "internal_cam_parameters.json")
        cam_info_json_path_final = get_train_path(
            cam_info_json_path, cam_info_json_train_path)

        try:
            cam_dict = get_cam(cam_json_path_final)
            azimuth_list = get_cam_info(cam_info_json_path_final)
        except Exception as e:
            print(e)
            get_train_path(cam_json_path, cam_json_train_path, force_copy=True)
            get_train_path(cam_info_json_path,
                           cam_info_json_train_path, force_copy=True)
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

        # pcd_surface_normal, cam_pose0, camk = rotate_pcd_normal(surface_origin, cam_dict, image_sub_idx_cond_list[0], azimuth_list=azimuth_list, align_type=self.target_align_type)
        pcd_surface_normal = surface_origin
        max_pcd = np.abs(pcd_surface_normal[:, :3]).max()
        pcd_surface_normal[:, :3] = pcd_surface_normal[:, :3] * 0.95 / max_pcd

        return {
            "image_cond_path_list": image_cond_path_list,
            "image_dino_conds": image_dino_conds,
            "image_clip_conds": image_clip_conds,
            "pcd_surface_normal": pcd_surface_normal,
        }


# for test
if __name__ == "__main__":
    # from render_pcd import render_pcd_with_pytorch3d
    import open3d as o3d
    exp_dir = "configs/1view_gray_2048_flow"
    train_configs_path = os.path.join(exp_dir, "train_configs.json")
    with open(train_configs_path, "r") as fr:
        train_configs = json.load(fr)
    train_configs["exp_dir"] = exp_dir

    train_dataset = Diffusion3D_V6_1Cond(train_configs, data_type="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=True)

    idx = 0
    for data in train_dataloader:
        idx += 1
        # print(idx)
        image_cond_path_list = data["image_cond_path_list"]
        image_clip_conds = data["image_clip_conds"]
        image_dino_conds = data["image_dino_conds"]
        pcd_surface_normal = data["pcd_surface_normal"]
        print(image_cond_path_list[0][0])
        print(f"image_clip_conds shape: {image_clip_conds.shape}")
        print(f"image_dino_conds shape: {image_dino_conds.shape}")
        print(f"pcd_surface_normal shape: {pcd_surface_normal.shape}")

        # print(image_cond_path_list)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            pcd_surface_normal[0][:, :3].cpu().data.numpy())
        o3d.io.write_point_cloud("clay_cond_pcd.ply", point_cloud)

        # os.system(f"cp {image_cond_path_list[0][0]} cond_image0.png")
        # os.system(f"cp {image_cond_path_list[1][0]} cond_image1.png")
        # os.system(f"cp {image_cond_path_list[2][0]} cond_image2.png")
        # os.system(f"cp {image_cond_path_list[3][0]} cond_image3.png")
        # render_pcd_with_pytorch3d(camk[0], cam_pose0[0], pcd_surface_normal[0][:, :3], "cond_pcd_render.png", color=None, ortho_cam=True)

        from torchvision.utils import make_grid, save_image
        image_clip_conds = make_grid(
            image_clip_conds[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(image_clip_conds, "clay_image_clip_conds.png")
        image_dino_conds = make_grid(
            image_dino_conds[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(image_dino_conds, "clay_image_dino_conds.png")

        print(f"pcd_surface.min(): {pcd_surface_normal[:, :, :3].min()}")
        print(f"pcd_surface.max(): {pcd_surface_normal[:, :, :3].max()}")
        breakpoint()
