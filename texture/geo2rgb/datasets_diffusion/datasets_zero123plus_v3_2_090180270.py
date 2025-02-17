import numpy as np
from regex import F
import torch
from torch.utils.data import Dataset
import json
import itertools
from PIL import Image, ImageOps
from torchvision import transforms
from typing import  Optional
from torchvision.utils import make_grid 
from tqdm import tqdm
import cv2
import random
import json
import os
import math
import time
import shutil
from transformers import CLIPImageProcessor

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def get_train_path(origin_path, train_path, force_copy=False):
    if os.path.exists(train_path) and (not force_copy):
        return train_path
    else:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        shutil.copy(origin_path, train_path)
        return origin_path

def vertical_grad(src, color_start=1, color_end=0):
    x, y, w, h = cv2.boundingRect(np.array(src)[:, :, -1])
    imgh, imgw, c = src.shape
    # 创建一幅与原图片一样大小的透明图片
    grad_back_img = np.zeros((imgh, imgw, 3))

    w_start = random.randint(0, int(x))
    w_end = random.randint(int(w_start), int(imgw/2))
    w_change_length = w_end - w_start

    grad_img = np.ndarray((imgh, w_change_length, 3))

    # opencv 默认采用 BGR 格式而非 RGB 格式
    grad = float(color_start - color_end) / w_change_length

    for i in range(w_change_length):
        grad_img[:, i] = np.array([[[color_start - i * grad]]])
    grad_back_img[:, w_start:w_end] = grad_img
    return grad_back_img

def add_shadow(img):
    """
    img: np.array [h, w, 4]
    """
    mask_left = vertical_grad(img, color_start=1.0, color_end=0)
    mask_right = vertical_grad(img[:, ::-1], color_start=1.0, color_end=0)[:, ::-1]
    mask = mask_left + mask_right
    # cv2.imwrite("mask_left.png", (mask_left*255).clip(0, 255).astype(np.uint8))
    # cv2.imwrite("mask_right.png", (mask_right*255).clip(0, 255).astype(np.uint8))
    # cv2.imwrite("mask.png", (mask*255).clip(0, 255).astype(np.uint8))
    # breakpoint()
    img[:, :, :3] = img[:, :, :3] * (1 - mask)
    
    return img.clip(0, 255).astype(np.uint8)

def add_shadow_mask(img):
    mask = img[:, :, 3]
    random_kernel_size = random.randint(4, 20) * 2 + 1
    mask = cv2.GaussianBlur(mask, (random_kernel_size, random_kernel_size), 0) / 255.0
    img[:, :, :3] = img[:, :, :3] * np.stack([mask]*3, -1)
    return img.clip(0, 255).astype(np.uint8)

def aug_shadow(cond_image_alpha_rescale, aug_prob = 0.2):
    if random.random() > aug_prob:
        return cond_image_alpha_rescale
    cond_image_alpha_rescale_array = np.array(cond_image_alpha_rescale)
    if random.random() < 0.5:
        return Image.fromarray(add_shadow_mask(cond_image_alpha_rescale_array))
    else:
        return Image.fromarray(add_shadow_mask(cond_image_alpha_rescale_array))

def repadding_rgba_image(image, rescale=True, ratio=0.8, bg_color=255, center=True, aug_resize=True, aug_resize_prob=0.5):
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except:
            return -1
    resize_scale_min = 0.5
    in_w = image.width
    in_h = image.height
    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])

    white_bg_image = Image.new('RGB', size=(in_w, in_h), color=(bg_color, bg_color, bg_color))
    white_bg_image.paste(image, (0, 0), mask=image)
    if not center:
        rgba_image = np.concatenate([np.array(white_bg_image), np.array(image)[..., -1:]], axis=-1)
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
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = np.array(white_bg_image)[y : y + h, x : x + w]

    mask[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = np.array(image)[..., -1:][y : y + h, x : x + w]

    rgba_image = np.concatenate([padded_image, mask], axis=-1)
    rgba_image = Image.fromarray(rgba_image)

    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])
    max_edge = max(w, h)
    area_ratio = max_edge / max(in_w, in_h)
    if aug_resize and (random.random() < aug_resize_prob) and (area_ratio > 0.5):
        downscale = 1 - random.random() * resize_scale_min
        rgba_w = rgba_image.width
        rgba_h = rgba_image.height
        rgba_image = rgba_image.resize((int(rgba_w*downscale), int(rgba_h*downscale)))
        rgba_image = rgba_image.resize((rgba_w, rgba_h))

    return rgba_image


def get_brightness_scale(cond_image, target_image):
    cond_image_a = np.array(cond_image)
    target_image_a = np.array(target_image)

    cond_image_a_gray = cv2.cvtColor(cond_image_a, cv2.COLOR_RGBA2GRAY)
    target_image_a_gray = cv2.cvtColor(target_image_a, cv2.COLOR_RGBA2GRAY)

    cond_brightness = np.mean(cond_image_a_gray, where=cond_image_a[..., -1] > 0)
    target_brightness = np.mean(target_image_a_gray, where=target_image_a[..., -1] > 0)
    # print(f"cond_brightness: {cond_brightness}; target_brightness: {target_brightness}")
    # if cond_brightness <= 65:
    #     brightness_diff = cond_brightness - target_brightness
    # else:
    #     brightness_diff = cond_brightness - target_brightness - 10

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
    mask[mask_img==0] = 0
    # cv2.imwrite("mask4.png", mask)
    brightrate = np.where(mask == 255.0, bright, (1.0/t*thresh)*bright)
    mask = np.where(mask == 255.0, mid, (mid-1.0)/t*thresh+1.0)
    img_float = img/255.0
    img_float = np.power(img_float, 1.0/mask[:, :, np.newaxis])*(1.0/(1.0-brightrate[:, :, np.newaxis]))
    img_float = np.clip(img_float, 0, 1.0)*255.0
    return img_float.astype(np.uint8)

def to_rgb_image(maybe_rgba: Image.Image, bg_color=127, edge_aug_threshold=0, bright_scale=None, aug_downscale=False, aug_downscale_prob=0.5):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # img = np.random.randint(random_grey_low, random_grey_high, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.ones([rgba.size[1], rgba.size[0], 3], dtype=np.uint8) * bg_color
        img = Image.fromarray(img, 'RGB')

        #### bright adapt
        if bright_scale is not None:
            rgba_array = np.array(rgba)
            rgb = cv2.convertScaleAbs(rgba_array[..., :3], alpha=bright_scale, beta=0)
            rgb = Image.fromarray(rgb)
            img.paste(rgb, mask=rgba.getchannel('A'))
        else:
            img.paste(rgba, mask=rgba.getchannel('A'))

        #### edge augmentation
        if edge_aug_threshold > 0 and (random.random() < edge_aug_threshold):
            mask_img = np.array(rgba.getchannel('A'))
            mask_img[mask_img > 0] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            iterration_num = random.randint(1, 2)
            mask_img_small = cv2.erode(mask_img, kernel, iterations=iterration_num)
            mask_img_edge = mask_img - mask_img_small
            mask_img_edge = np.concatenate([mask_img_edge[..., None]]*3, axis=-1) / 255.0
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random.random() < 0.5:
                rand_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            img_array = np.array(img) * (1 - mask_img_edge) + rand_color * mask_img_edge
            img = Image.fromarray(img_array.astype(np.uint8))

        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=(255, 255, 255))


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image



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
        images1 = cv2.resize(images, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
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


class TargetAugmentor:
    def __init__(self, FLAGS):
        self.flags = FLAGS.get('target_augmentation', {})

    def __call__(self, image, scale=None, **kwargs):
        '''
            image: image with shape [h, w, c]
            scale: if scale is not None, scale=scale
        '''

        # generate random transform parameters
        params = dict()
        if "scale_adjustment" in self.flags:
            if scale is not None:
                params["scale"] = scale
            else:
                min_scale, max_scale = self.flags["scale_adjustment"]
                params["scale"] = (max_scale - min_scale) * np.random.rand() + min_scale

        # apply random transform
        image[:] = transform_target(image, **params)

        return image, params

def get_stdpose_idx(image_azimuth):
    if (image_azimuth >= 45 and image_azimuth < 135):
        return 1
    elif (image_azimuth >= 135 and image_azimuth < 225):
        return 2
    elif (image_azimuth >= 225 and image_azimuth < 315):
        return 3
    else:
        return 0


class ObjaverseDatasetV3_2_090180270(Dataset):
    def __init__(self,
        configs,
        data_type = "train",
        img_out_resolution: int = 320,
        bg_color="gray",
        load_from_cache_last=True,
        groups_num: int=1,
        validation: bool = False,
        num_views=4,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        read_normal: bool = False,
        read_color: bool = True,
        read_depth: bool = False,
        read_mask: bool = True,
        mix_color_normal: bool = False,
        suffix: str = 'png',
        subscene_tag: int = 3,
        backup_scene=None,
        num_validation_samples=None
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        exp_dir = configs.get("exp_dir", None)
        assert exp_dir is not None
        self.exp_dir = exp_dir
        print(f"exp_dir: {exp_dir}")
        data_config = configs["data_config"]
        self.dataset_json = data_config["dataset_json"]
        self.pretrained_model_name_or_path = data_config["pretrained_model_name_or_path"]
        self.target_img_type = data_config["target_img_type"]
        self.image_list_path_list = data_config["image_list_path_list"]
        self.group_idx_list = data_config["group_idx_list"]
        self.view_idx_list = data_config["view_idx_list"]
        self.cond_idx_list = data_config["cond_idx_list"]
        self.images_num_per_group = data_config["images_num_per_group"]
        self.load_from_cache_last = data_config.get("load_from_cache_last", load_from_cache_last)
        self.bg_color = bg_color
        if self.bg_color == 'white':
            self.bg_color_num = 255
        elif self.bg_color == 'black':
            self.bg_color_num = 0
        elif self.bg_color == 'gray':
            self.bg_color_num = 127


        self.validation = (data_config.get("data_type", data_type) == "test")
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        self.invalid_list = invalid_list
        self.groups_num = groups_num
        self.img_out_resolution = data_config.get("img_out_resolution", img_out_resolution)

        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.mix_color_normal = mix_color_normal  # mix load color and normal maps
        self.suffix = suffix
        self.subscene_tag = subscene_tag
        self.image_tensor_shape = None
        num_validation_samples = configs.get("num_validation_samples", num_validation_samples)
        

        self.image_path_list = []
        for image_list_path in self.image_list_path_list:
            with open(image_list_path, 'r') as fr:
                self.image_path_list += json.load(fr)
        print(f"image num: {len(self.image_path_list)}")

        with open(self.dataset_json, 'r') as fr:
            json_dict = json.load(fr)
        
        data_dict = json_dict["data"]

        train_json_save_path = os.path.join(exp_dir, "train.json")
        test_json_save_path = os.path.join(exp_dir, "test.json")

        if self.load_from_cache_last:
            print("load from cache last")
            with open(train_json_save_path, 'r') as fr:
                data_train_list = json.load(fr)
            with open(test_json_save_path, 'r') as fr:
                data_test_list = json.load(fr)
        else:
            print("rechecking data... ")
            all_data_list = self.read_data(data_dict)
            data_train_list, data_test_list = self.__split_train_test(all_data_list)
            print("writing load cache")
            with open(train_json_save_path, "w") as fw:
                json.dump(data_train_list, fw, indent=2)
            with open(test_json_save_path, "w") as fw:
                json.dump(data_test_list, fw, indent=2)

        dataset_list_train = list(itertools.chain(*data_train_list))
        dataset_list_test = list(itertools.chain(*data_test_list))
        dataset_list_train.sort()
        dataset_list_test.sort()

        if not self.validation:
            self.all_objects = dataset_list_train
        else:
            self.all_objects = dataset_list_test
            if num_validation_samples is not None:
                self.all_objects = self.all_objects[:num_validation_samples]

        print("loading", len(self.all_objects), " objects in the dataset")

        self.feature_extractor_vae = CLIPImageProcessor.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="feature_extractor_vae")

        self.feature_extractor_clip = CLIPImageProcessor.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="feature_extractor_clip")

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.img_out_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.target_augmentation = None
        if data_config.get("target_augmentation", None):
            print("use target augmentation")
            self.target_augmentation = TargetAugmentor(data_config)

        self.start_time = time.time()
        self.data_load_record = []

    def __len__(self):
        return len(self.all_objects)

    def read_data(self, data_dict):
        all_data_list = []
        all_num = 0
        for classname, classdict in tqdm(data_dict.items()):
            class_data_list = []
            for objname, objdict in tqdm(classdict.items()):
                image_dir = objdict["ImgDir"]
                image_dir_train = objdict["ImgDir_train"]
                # if not os.path.exists(image_dir):
                #     continue
                for groupi in self.group_idx_list:
                    for condi in self.cond_idx_list:
                        # if os.path.exists(os.path.join(image_dir, "color", f"cam-{str(groupi*self.images_num_per_group + condi).zfill(4)}.png")):
                            class_data_list.append([classname, objname, image_dir, image_dir_train, groupi, condi])
                #             all_num += 1
                # if all_num >= 100:
                #     break

            all_data_list.append(class_data_list)
        return all_data_list

    def __split_train_test(self, dataset_list, test_threshold=0.0000001, test_min_num=10):
        train_list, test_list = [], []
        for i, class_dataset_list in enumerate(dataset_list):
            if len(class_dataset_list) == 0:
                print("dataset objs num is 0")
                continue
            class_name = class_dataset_list[0][0]
            num = len(class_dataset_list)
            if num < test_min_num*3:
                print(f"{class_name} dataset objs num is little than test_min_num*3, all {num} for train")
                continue
            test_num = int(max(num * test_threshold, test_min_num))
            test_list.append(class_dataset_list[0:test_num])
            train_list.append(class_dataset_list[test_num:])
            print(f"class {class_name} split {num-test_num} for train and {test_num} for test")
        return train_list, test_list 

        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    
    def load_image(self, img_path, bg_color, alpha=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32)
        # img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 3 or img.shape[-1] == 4 # RGB or RGBA

        if alpha is None and img.shape[-1] == 4:
            alpha = img[:, :, 3:]
            img = img[:, :, :3]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha/255.0 + bg_color * (1 - alpha/255.0)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def get_single_item(self, index=None):
        try:
            single_image_path = random.choice(self.image_path_list)
            image_single = Image.open(single_image_path).convert("RGB")
        except:
            return self.get_single_item()
        image_single = to_rgb_image(image_single, self.bg_color_num)
        image_single = resize_with_padding(image_single, expected_size=(512, 512))

        cond_image_vae = self.feature_extractor_vae(images=image_single, return_tensors="pt").pixel_values
        cond_image_vae = cond_image_vae.squeeze()
        cond_images_vae = cond_image_vae[None, ...]

        cond_image_clip = self.feature_extractor_clip(images=image_single, return_tensors="pt").pixel_values
        cond_image_clip = cond_image_clip.squeeze()

        images_out = self.train_transforms(image_single)[None, ...]

        return {
            "cond_images": np.array(image_single).transpose((2, 0, 1))[None, ...], #### debug
            'cond_images_vae': cond_images_vae,
            'cond_image_clip': cond_image_clip,
            'images_out': images_out
        }

    def get_z123_item(self, index, only_first_prob=0.3):
        classname, objname, image_dir, image_dir_train, group_idx, cond_idxes = self.all_objects[index]

        if random.random() < only_first_prob: ##### probility of only first image as cond image.
            cond_idxes_true = [cond_idxes[0]]
        else:
            if len(cond_idxes) > 1:
                zero_cond_list = [x for x in range(len(cond_idxes))]
                other_cond_idxes_num = random.choice(zero_cond_list[1:])
                other_idxes = random.sample(cond_idxes[1:], other_cond_idxes_num)
                cond_idxes_true = [cond_idxes[0]] + other_idxes
            else:
                cond_idxes_true = [cond_idxes[0]]
        
        image_sub_idx_cond_list = [self.images_num_per_group * group_idx + x for x in cond_idxes_true]
        
        azimuth_elevation_json_path = os.path.join(image_dir, "RSVC-cam_parameters.json")
        azimuth_elevation_json_path_train = os.path.join(image_dir_train, "RSVC-cam_parameters.json")
        azimuth_elevation_json_path = get_train_path(azimuth_elevation_json_path, azimuth_elevation_json_path_train)
        try:
            with open(azimuth_elevation_json_path, 'r') as fr:
                azimuth_list = json.load(fr)["config"]["RSVC"]["real_azimuth"]
            cond_azimuth = azimuth_list[image_sub_idx_cond_list[0]]
            target_idx = get_stdpose_idx(cond_azimuth)
            image_sub_idx_target_list = self.view_idx_list[target_idx]
            # print(f"azimuth: {cond_azimuth}; target_idx: {target_idx}")
        except:
            return self.get_z123_item(np.random.randint(0, self.__len__() - 1))
        # print(f"image_sub_idx_cond_list: {image_sub_idx_cond_list}")
        rgb_dir = os.path.join(image_dir, "color")
        rgb_dir_train = os.path.join(image_dir_train, "color")
        target_img_dir = rgb_dir


        if not os.path.exists(target_img_dir):
            print(f'not exists target_img_dir: {target_img_dir}')
            return self.get_z123_item(np.random.randint(0, self.__len__() - 1))

        cond_images_vae = []
        cond_image_clip = None
        cond_images = [] ###  debug
        for image_idx in image_sub_idx_cond_list:
            img_cond_path = os.path.join(target_img_dir, f"cam-{str(image_idx).zfill(4)}.png")
            # print(f"cond_image_path: {img_cond_path}")
            img_cond_train_path = os.path.join(rgb_dir_train, f"cam-{str(image_idx).zfill(4)}.png")
            img_cond_path = get_train_path(img_cond_path, img_cond_train_path)
            try:
                cond_image_alpha = Image.open(img_cond_path)
                cond_image_alpha_image = repadding_rgba_image(cond_image_alpha, rescale=True, 
                                                    ratio=0.9, bg_color=self.bg_color_num, center=True)
            except:
                return self.get_z123_item(np.random.randint(0, self.__len__() - 1))
            
            # print(f"cond_image_alpha shape: {cond_image_alpha.size}")
            cond_image_alpha_rescale = cond_image_alpha_image.resize((512, 512))
            # print(f"cond_image_alpha_array shape: {cond_image_alpha_rescale.size}")
            ### random cond background
            cond_image_alpha_rescale = aug_shadow(cond_image_alpha_rescale)
            cond_image = to_rgb_image(cond_image_alpha_rescale, self.bg_color_num, edge_aug_threshold=0.4)
            # print(f"cond_image shape: {cond_image.size}")
            cond_images.append(np.array(cond_image).transpose((2, 0, 1))) ###  debug

            cond_image_vae = self.feature_extractor_vae(images=cond_image, return_tensors="pt").pixel_values
            cond_image_vae = cond_image_vae.squeeze()
            cond_images_vae.append(cond_image_vae)

            if cond_image_clip is None:
                cond_image_clip = self.feature_extractor_clip(images=cond_image, return_tensors="pt").pixel_values
                cond_image_clip = cond_image_clip.squeeze()
                cond_image0 = cond_image_alpha.copy()

        images_path_list = []
        images_out = []
        bright_scale = None
        for image_idx in image_sub_idx_target_list:
            img_path = os.path.join(target_img_dir, f"cam-{str(image_idx).zfill(4)}.png")
            img_train_path = os.path.join(rgb_dir_train, f"cam-{str(image_idx).zfill(4)}.png")
            img_path = get_train_path(img_path, img_train_path)
            if not os.path.exists(img_path):
                print(f'not exists img_path: {img_path}')
                return self.get_z123_item(np.random.randint(0, self.__len__() - 1))
            images_path_list.append(img_path)
            try:
                target_image = Image.open(img_path)
                bright_scale = get_brightness_scale(cond_image0, target_image)
                target_image = to_rgb_image(target_image, self.bg_color_num, bright_scale=bright_scale)
                target_image = self.train_transforms(target_image)
                images_out.append(target_image.squeeze())
            except:
                return self.get_z123_item(np.random.randint(0, self.__len__() - 1))

        cond_images = np.stack(cond_images, axis=0) # debug
        cond_images_vae = torch.stack(cond_images_vae, dim=0)
        images_out = torch.stack(images_out, dim=0)

        return {
            "cond_path": img_cond_path,
            "cond_images": cond_images,  ### debug
            'cond_images_vae': cond_images_vae,
            'cond_image_clip': cond_image_clip,
            'images_out': images_out
        }

    def __getitem__(self, index):
        # if random.random() <= 0.1:
        #     return self.get_single_item(index)
        # else:
        return self.get_z123_item(index, only_first_prob=0.3)


if __name__ == "__main__":
    configs = {
                "exp_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v27_1cond_4views_abs_512",
                "data_config": {
                    "dataset_name" : "ObjaverseDatasetV3_2_090180270",
                    "dataset_json" : "/aigc_cfs_2/neoshang/data/data_list/20241029_1view_condition/part1_16_proprietas_clothes_remake_h20.json",
                    "pretrained_model_name_or_path": "/aigc_cfs_2/neoshang/models/zero123plus-v1.2",
                    "image_list_path_list": ["/apdcephfs_cq10/share_1615605/neoshang/data/coco_train2017_img_list.json",
                                            "/apdcephfs_cq10/share_1615605/neoshang/data/animal_img_list.json",
                                            "/apdcephfs_cq10/share_1615605/neoshang/data/imagenet_2012/images_path.json",
                                            "/apdcephfs_cq10/share_1615605/neoshang/data/winter21_whole/images_path.json"],
                    "load_from_cache_last": False,
                    "target_img_type": "shading",
                    "img_out_resolution": 512,
                    "view_idx_list": [[0, 1, 2, 3, 4, 8], [1, 2, 3, 0, 5, 9], [2, 3, 0, 1, 6, 10], [3, 0, 1, 2, 7, 11]],
                    "group_idx_list": [0],
                    "cond_idx_list": [[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40],[41],[42],[43],[44],[45],[46],[47],[48],[49],[50],[51],[52]],
                    "images_num_per_group": 53,
                    "num_validation_samples": 16
                }
            }

    train_dataset = ObjaverseDatasetV3_2_090180270(configs, data_type="train", load_from_cache_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 1, num_workers=0, pin_memory=True, shuffle=True)

    for data in train_dataloader:
        cond_images = data["cond_images"]
        cond_images_vae = data["cond_images_vae"]
        cond_image_clip = data["cond_image_clip"]
        images_out = data["images_out"]

        from torchvision import utils as vutils
        from torchvision.utils import make_grid, save_image
        print(cond_images[0].shape)
        cond_images = make_grid(cond_images[0], nrow=2, padding=0) / 255.0

        save_image(cond_images, "z123_cond_images.png")

        images_out = make_grid(images_out[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(images_out, "z123_images_out.png")

        cond_images_vae = make_grid(cond_images_vae[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(cond_images_vae, "z123_cond_images_vae.png")

        save_image(cond_image_clip, "z123_cond_image_clip.png")

        breakpoint()
