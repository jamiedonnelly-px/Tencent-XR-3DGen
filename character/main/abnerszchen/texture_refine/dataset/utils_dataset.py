import os
import torch
import json
import PIL
import shutil
import copy
from itertools import cycle
import random
import numpy as np

DATA_MCWY1_KEYS = ['MCWY_1_Top', 'MCWY_1_Outfit', 'MCWY_1_Bottom', 'MCWY_1_Shoe']
DATA_MCWY2_KEYS = ['MCWY_2_Top', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe']
DATA_READY_KEYS = ['readyplayerme_top', 'readyplayerme_footwear', 'readyplayerme_bottom']
DATA_DAZ_KEYS = ['DAZ_DAZ_Bottom', 'DAZ_DAZ_Outfit', 'DAZ_DAZ_Shoe', 'DAZ_DAZ_Top']
DATA_VROID_KEYS = ['VRoid_VRoid_Top', 'VRoid_VRoid_Bottom', 'VRoid_VRoid_Shoe']
DATA_KEYS_DICT = {
    "mcwy1": DATA_MCWY1_KEYS,
    "mcwy2": DATA_MCWY2_KEYS,
    "ready": DATA_READY_KEYS,
    "daz": DATA_DAZ_KEYS,
    "vroid": DATA_VROID_KEYS
}

CAM_KEYS = ['0089', '0011', '0100', '0132', '0167', '0286', '0318']
HUMAN_TRAIN_KEYS = ['0011', '0048', '0089', '0100', '0132', '0167', '0286', '0318']
HUMAN_CONDI_KEYS = ['0037', '0089', '0121']
HUMAN_CAP_KEYS = ['0037', '0121']

SRENDER_TRAIN_KEYS = ['0009', '0011', '0013', '0015', '0032', '0034', '0036', '0038']
SRENDER_CONDI_KEYS = ['0024', '0026', '0033']

SRENDER_DAO_TRAIN_KEYS = ['0026', '0033', '0030', '0031', '0037', '0038']
SRENDER_DAO_CONDI_KEYS = ['0030', '0031', '0037']
SRENDER_DAO_CAP_KEYS = ['0030', '0037']

HUMAN_9_TRAIN_KEYS = ['0007', '0011', '0100', '0131', '0145', '0167', '0245', '0286', '0318']

VR100_TRAIN_KEYS = ['0006', '0007', '0011', '0015', '0016', '0032', '0074']
VR100_CONDI_KEYS = ['0006', '0032']
VR100_CAP_KEYS = ['0006', '0032']

def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(json_data, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return

def read_lines(txt):
    lines = [line.strip() for line in open(txt, "r").readlines()]
    return lines

def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)

# ***** dataset img

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
clip_mean = np.array(OPENAI_CLIP_MEAN, dtype=np.float32)
clip_std = np.array(OPENAI_CLIP_STD, dtype=np.float32)

def load_rgba_as_rgb(img_path, res=None):
    """load with RGBA and convert to RGB with white backgroud, if is RGB just return

    Args:
        img_path: _description_

    Returns:
        PIL.Image [h, w, 3]
    """
    img = PIL.Image.open(img_path)
    if img.mode == "RGBA":
        background = PIL.Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = PIL.Image.alpha_composite(background, img).convert("RGB")
    if res is not None and isinstance(res, int):
        img = img.resize((res, res))
    return img


# def convert_to_np(image, resolution):
#     image = image.convert("RGB").resize((resolution, resolution))
#     return np.array(image).transpose(2, 0, 1)

def load_img_rgb(path, resolution=512, as_clip_input=False):
    """load image and convert RGB, as tensor [-1, 1], [c, h, w]

    Args:
        path: image path, (skip check now for speed up)

    Returns:
        image: tensor [-1, 1], [c, h, w]
    """
    image = load_rgba_as_rgb(path, resolution)
    image = np.array(image).transpose(2, 0, 1)
    image = 2 * (torch.from_numpy(image) / 255.) - 1
    # # image = convert_to_np(PIL.Image.open(path), resolution)
    # if as_clip_input:
    #     image = ((image.T  / 255 - clip_mean) / clip_std).T
    #     image = torch.from_numpy(image)
    # else:
    #     image = 2 * (torch.from_numpy(image) / 255.) - 1
    return image

# def load_img(path):
#     image = np.array(PIL.Image.open(path).convert("RGB")).transpose(2, 0, 1)
#     image = 2 * (torch.from_numpy(image) / 255.) - 1
#     return image

def load_condi_img(path, resolution=224):
    """load with RGBA and convert to RGB with white backgroud, then resize to 224, 224

    Args:
        path: _description_
        resolution: _description_. Defaults to 224.

    Returns:
        image tensor [3, 224, 224]
    """
    img = load_rgba_as_rgb(path).resize((resolution, resolution))
    image = np.array(img)
    image = ((image  / 255 - clip_mean) / clip_std).transpose(2, 0, 1)
    image = torch.from_numpy(image)
    return image


def load_depth(path, resolution=None, rescale=True, min_range=1e-2):
    """load uint16 depth and rescale to tensor [0, 1]

    Args:
        path: png path
        resolution: new res
        rescale: _description_. Defaults to True

    Returns:
        depth: [1, h, w] tensor
    """
    depth_image = PIL.Image.open(path)
    if resolution is not None:
        # depth_image = depth_image.resize((resolution, resolution), PIL.Image.BICUBIC)   # TODO check.
        depth_image = depth_image.resize((resolution, resolution), PIL.Image.BILINEAR)   # TODO check.
        # depth_image = depth_image.resize((resolution, resolution), PIL.Image.NEAREST)   # TODO check.
    depth_array = np.array(depth_image)
    if rescale:
        dmin, dmax = np.min(depth_array), np.max(depth_array)
        if (dmax - dmin) < min_range:    # invalid, It's usually rendered in full black(see nothing)
            depth_array = np.zeros_like(depth_array).astype(np.float32)
        else:
            depth_array = (depth_array - dmin) / (dmax - dmin)
    depth = torch.from_numpy(depth_array).unsqueeze(0)
    return depth

def depth_normalize(depth_np):
    dmin, dmax = np.min(depth_np), np.max(depth_np)
    depth = (depth_np - dmin) / (dmax - dmin)
    depth = np.clip(np.rint(depth * 255.0), 0, 255).astype(np.uint8)
    return depth

def depth_normalize_tensor(depth_tensor):
    dmin, dmax = torch.min(depth_tensor), torch.max(depth_tensor)
    depth = (depth_tensor - dmin) / (dmax - dmin)
    return depth

def scale_depth_pil_255(depth_mm_path):
    """load pil with uint16 mm and rescale to [0, 255] pil

    Args:
        depth_mm_path: png path

    Returns:
        [0, 255] pil
    """
    depth_np = np.array(PIL.Image.open(depth_mm_path))
    depth = depth_normalize(depth_np)
    return PIL.Image.fromarray(depth)

def save_depth_255(depth_np, path):
    """save depth numpy as uint8

    Args:
        depth_np: [h, w]
        path: png path
    """
    depth = depth_normalize(depth_np)
    PIL.Image.fromarray(depth).save(path)
    return

def save_depth(depth_np, path, scale=1000.0):
    """save depth numpy as uint16 with scale=1000. in mm

    Args:
        depth_np: [h, w]
        path: png path
        scale: _description_. Defaults to 1000.0.
    """
    depth_np_scaled = (depth_np * scale).astype(np.uint16)
    depth_image = PIL.Image.fromarray(depth_np_scaled)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_image.save(path)
    return

# ****
def split_list_avg(raw_list, num):
    """split one list to num list

    Args:
        raw_list: list
        num: N

    Returns:
        2d list
    """
    if not raw_list or len(raw_list) < num:
        raise ValueError("no need split")
    splited_lists = [[] for _ in range(num)]
    for key_pair, temp_list in zip(raw_list, cycle(splited_lists)):
        temp_list.append(key_pair)
    return splited_lists

def parse_objs_json(objs_json):
    """ parse standard json to dict and pair list
    return: objs_dict: dict
    key_pair_list: list of pair ('data', dtype, oname)
    """
    if not os.path.exists(objs_json):
        print('[Error] can not find objs_json '.format(objs_json))
        return dict(), []
    objs_dict = load_json(objs_json)
    if 'data' not in objs_dict:
        print('[Error] not standard json '.format(objs_json))
        return dict(), []
    key_pair_list = []
    for dataset, dataset_dict in objs_dict['data'].items():
        key_pair_list += [('data', dataset, obj_name) for obj_name in list(dataset_dict.keys())]

    return objs_dict, key_pair_list

def split_pod_json(objs_json, tasks_pair_pod, out_pod_json):
    """split from objs_json by tasks_pair_pod as keys, then save to out_pod_json

    Args:
        objs_json: _description_
        tasks_pair_pod: _description_
        out_pod_json: _description_
    Return:
        pod_dict: splited dict
    """
    if not os.path.exists(objs_json):
        print('[Error] can not find objs_json '.format(objs_json))
        return
    objs_dict = load_json(objs_json)
    pod_dict = dict()

    for task_pair in tasks_pair_pod:
        if task_pair[0] not in pod_dict:
            pod_dict[task_pair[0]] = {}
        if task_pair[1] not in pod_dict[task_pair[0]]:
            pod_dict[task_pair[0]][task_pair[1]] = {}
        pod_dict[task_pair[0]][task_pair[1]][task_pair[2]] = objs_dict[task_pair[0]][task_pair[1]][task_pair[2]]

    save_json(pod_dict, out_pod_json)
    return pod_dict

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dst_item = os.path.join(dst_folder, item)

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)


def concatenate_images_horizontally(image_list, out_img_path=None):
    """concatenate images horizontally

    Args:
        image_list: list of PIL.Image 
        out_img_path: save if not None

    Returns:
        output_image PIL.Image 
    """
    total_width = sum([img.width for img in image_list])
    max_height = max([img.height for img in image_list])

    output_image = PIL.Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in image_list:
        output_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if out_img_path is not None:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        output_image.save(out_img_path)
    return output_image

def concatenate_images_vertically(image_list, out_img_path=None):
    """Concatenate images vertically

    Args:
        image_list: list of PIL.Image 
        out_img_path: save if not None

    Returns:
        output_image PIL.Image 
    """
    max_width = max([img.width for img in image_list])
    total_height = sum([img.height for img in image_list])

    output_image = PIL.Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in image_list:
        output_image.paste(img, (0, y_offset))
        y_offset += img.height

    if out_img_path is not None:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        output_image.save(out_img_path)
    return output_image


def concatenate_images_2d(image_list_2d, out_img_path=None):
    """concat 2d PIL.Image to one image

    Args:
        image_list_2d: 2d list of PIL.Image 
        out_img_path: save if not None
    Returns:
        output_image PIL.Image 
    """

    total_width = sum([img.width for img in image_list_2d[0]])
    total_height = sum([image_list_2d[i][0].height for i in range(len(image_list_2d))])

    output_image = PIL.Image.new("RGB", (total_width, total_height))

    y_offset = 0
    for row in image_list_2d:
        x_offset = 0
        max_height = max([img.height for img in row])
        for img in row:
            output_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += max_height

    if out_img_path is not None:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        output_image.save(out_img_path)
    return output_image



def split_train_val_dict(raw_dict, test_ratio, min_cnt=5):
    train_dict = copy.deepcopy(raw_dict)
    val_dict = copy.deepcopy(raw_dict)

    data_dict = raw_dict['data']

    train_data_dict = {}
    val_data_dict = {}
    val_cnt = 0
    all_cnt = 0
    for dname, d_metas in data_dict.items():
        keys = list(d_metas.keys())
        d_cnt = len(keys)
        sample_cnt = min(max(min_cnt, int(d_cnt * test_ratio)), d_cnt)
        print(f'sample {sample_cnt}/{d_cnt} from {dname}')

        random.shuffle(keys)
        train_data_dict[dname] = {key: d_metas[key] for key in keys[sample_cnt:]}
        val_data_dict[dname] = {key: d_metas[key] for key in keys[:sample_cnt]}
        val_cnt += sample_cnt
        all_cnt += d_cnt
    train_dict['data'] =  train_data_dict
    val_dict['data'] =  val_data_dict
    print(f' val_cnt:{val_cnt}/{all_cnt} ')
    return train_dict, val_dict

def split_jsons(all_data_json, train_pairs, test_pairs, out_dir, prefix='tex_creator'):
    """split train/val

    Args:
        all_data_json: _description_
        train_pairs: _description_
        test_pairs: _description_
        out_dir: _description_
        prefix: _description_. Defaults to 'tex_creator'.
    """
    diffusion_train_dict = split_pod_json(all_data_json, train_pairs, os.path.join(out_dir, f'{prefix}_diffusion_train.json'))
    split_pod_json(all_data_json, test_pairs, os.path.join(out_dir, f'{prefix}_test.json'))
    train_dict, val_dict = split_train_val_dict(diffusion_train_dict, 0.05, 5)
    save_json(train_dict, os.path.join(out_dir, f'{prefix}_train.json'))
    save_json(val_dict, os.path.join(out_dir, f'{prefix}_val.json'))
    print(f'test: {len(test_pairs)} for prefix {prefix}')



def zip_folder(folder_path, zip_path):
    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

import time
import torch

class TimerSZ:
    def __init__(self, disable=False):
        self.times = {}
        self.order = []
        torch.cuda.synchronize()
        self.prev_time = time.time()
        self.disable = disable

    def tick(self, name):
        if self.disable:
            return

        torch.cuda.synchronize()
        elapsed_time = time.time() - self.prev_time
        if name in self.times:
            self.times[name]['total_time'] += elapsed_time
            self.times[name]['count'] += 1
        else:
            self.times[name] = {'total_time': elapsed_time, 'count': 1}
            self.order.append(name)
        self.prev_time = time.time()

    def get_times(self, verbose=True):
        if self.disable:
            return []
        avg_times = []
        total_times = []
        for name in self.order:
            time_data = self.times[name]
            avg_time = time_data['total_time'] / time_data['count']
            avg_times.append((name, avg_time))
            total_times.append((name, time_data['total_time']))
        if verbose:
            print("avg_times: ", avg_times)
            print("total_times: ", total_times)
        return avg_times, total_times
