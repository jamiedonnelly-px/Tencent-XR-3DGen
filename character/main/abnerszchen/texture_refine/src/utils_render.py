import os
import torch
import numpy as np
from PIL import Image
import time
import subprocess
src_dir = os.path.dirname(os.path.abspath(__file__))

def load_images(path_list):

    return [Image.open(path) for path in path_list]

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

    output_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in image_list:
        output_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if out_img_path is not None:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        output_image.save(out_img_path)
    return output_image

def save_rgba_geom_images(rgba: torch.Tensor, output_png: str):
    """save batch rgba tensor in [-1, 1] to pils

    Args:
        rgba (torch.Tensor):  [N, H, W, 4]  RGBA tensor，in [-1, 1]
        output_png (str): 
    """
    rgba_normalized = (rgba + 1) / 2

    pils = []
    for i in range(rgba_normalized.shape[0]):
        img = rgba_normalized[i]

        img = (img * 255).byte()

        img_pil = Image.fromarray(img.cpu().numpy(), 'RGBA')

        pils.append(img_pil)
    concatenate_images_horizontally(pils, output_png)

@torch.no_grad()
def decode_normalized_depth(depths, batched_norm=False):
    """Normalize absolute depth to inverse depth

    Args:
        depths: [n,h,w,2] in real meter, 2=d + alhpa, from py3d renderer
        batched_norm: _description_. Defaults to False.

    Returns:
        [n,h,w,3] in [0, 1] tensor
    """
    view_z, mask = depths.unbind(-1)
    view_z = view_z * mask + 100 * (1 - mask)
    inv_z = 1 / view_z
    inv_z_min = inv_z * mask + 100 * (1 - mask)
    if not batched_norm:
        max_ = torch.max(inv_z, 1, keepdim=True)
        max_ = torch.max(max_[0], 2, keepdim=True)[0]

        min_ = torch.min(inv_z_min, 1, keepdim=True)
        min_ = torch.min(min_[0], 2, keepdim=True)[0]
    else:
        max_ = torch.max(inv_z)
        min_ = torch.min(inv_z_min)
    inv_z = (inv_z - min_) / (max_ - min_)
    inv_z = inv_z.clamp(0, 1)
    inv_z = inv_z[..., None].repeat(1, 1, 1, 3)

    return inv_z

def save_normalized_geom(depth_maps, output_image_path):
    """_summary_

    Args:
        depth_maps: [n,h,w,3] in [0, 1] tensor
        output_image_path: _description_
    """
    depth_maps = depth_maps.cpu().numpy()
    depth_maps *= 255.0
    depth_maps = depth_maps.astype(np.uint8)    # 4, 256, 256, 3
    depth_maps_row = np.hstack(depth_maps)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    Image.fromarray(depth_maps_row).save(output_image_path)


def save_batch_imgs(imgs, out_path):
    """save list of [h w c] or [b h w c]

    Args:
        imgs: [b,h, w, 3] numpy in [0, 1]
        out_path: _description_
    """
    results = []
    for img in imgs:
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., -1]
        if isinstance(img, np.ndarray):
            in_pil = Image.fromarray(img)
        elif torch.is_tensor(img):
            in_pil = Image.fromarray((img * 255.0).cpu().numpy().round().astype("uint8"))
        else:
            print('ERROR invalid type')
            return
        results.append(in_pil)

    concatenate_images_horizontally(results, out_path)

    return


def save_batch_render_rgba(views_rgba, out_path, res=1024):
    """_summary_

    Args:
        views_rgba: list of tensor [4, 1024, 1024]
        out_path: _description_
    """
    results = [
        Image.fromarray((rgba * 255.0).permute(1, 2, 0).cpu().numpy().round().astype("uint8"), mode="RGBA").resize((res, res))
        for rgba in views_rgba
    ]
    concatenate_images_horizontally(results, out_path)
    return

def save_tensor_chw(tensor_chw, out_path):
    """_summary_

    Args:
        tensor_chw: tensor [3, 1024, 1024]
        out_path: _description_
    """
    min_, max_ = tensor_chw.min(), tensor_chw.max()
    tensor_chw = (tensor_chw - min_) / (max_ - min_)
    np_ = (tensor_chw * 255.0).permute(1, 2, 0).cpu().numpy().round().astype("uint8")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(np_).save(out_path)

    return


def save_render_masked_views(result_views, render_masks, out_png, see_res=None):
    out_view_pils = []
    # [nc, image_size, image_size, c+1]
    for idx in range(len(result_views)):
        render_mask = (render_masks[idx][..., -1] > 0).detach().cpu().numpy()
        out_view = result_views[idx].detach().cpu().numpy()
        out_view[~render_mask] = (0.6, 0, 0)
        out_view_pil = Image.fromarray((out_view * 255.0).round().astype("uint8"))
        if see_res is not None and isinstance(see_res, int):
            out_view_pil = out_view_pil.resize((see_res, see_res))
        out_view_pils.append(out_view_pil.convert("RGB"))
    make_image_grid(out_view_pils, 1, len(out_view_pils)).save(out_png)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


from typing import List
def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def blender_smart_uv(mesh_path, output_mesh_path):
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    cmd = f"/usr/blender-3.6.2-linux-x64/blender -b -P {src_dir}/obj_convert.py -- --mesh_path '{mesh_path}' --output_mesh_path '{output_mesh_path}' --process_stages 'smart_uv+add_image'"
    subprocess.run(cmd, shell=True)
    if os.path.exists(output_mesh_path):
        return output_mesh_path
    else:
        print('ERROR blender_smart_uv failed!')
        return None

def check_mesh_uv(mesh_path):
    ext = os.path.splitext(mesh_path)[-1]
    if ext == ".glb":
        # TODO need convert
        return mesh_path

    elif ext == ".obj":
        if not os.path.exists(mesh_path.replace(".obj", ".mtl")):
            # without uv, need blender smart uv
            output_mesh_path = os.path.join(os.path.dirname(mesh_path), "smart_uv.obj")
            ts = time.time()
            output_mesh_path = blender_smart_uv(mesh_path, output_mesh_path)
            print(f"blender_smart_uv use time = {time.time() - ts}")
            if output_mesh_path:
                return output_mesh_path
            else:
                raise ValueError(f"blender_smart_uv failed to {output_mesh_path}")
    else:
        raise NotImplementedError(f"invalid ext={ext}")
    return mesh_path



def split_image(image, rows, cols):
    width, height = image.size
    block_width = width // cols
    block_height = height // rows

    images = []
    for i in range(rows):
        for j in range(cols):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            sub_image = image.crop((left, upper, right, lower))
            images.append(sub_image)

    return images


def tensor_to_pils(data, vis_res=512):
    pils = [
        Image.fromarray(
            np.clip(np.rint(data[idx].detach().cpu().numpy() * 255.0), 0,
                    255).astype(np.uint8)).resize((vis_res, vis_res))
        for idx in range(data.shape[0])
    ]
    return pils


def save_bhwc_tensor(data, out_path, vis_res=512):
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    concatenate_images_horizontally(tensor_to_pils(data, vis_res=vis_res), out_path)
    return

def util_print_times(t_list, prename=""):
    if not t_list or len(t_list) <= 1:
        print('invalid t_list ', t_list)
        return
    for i, (time, name) in enumerate(t_list):
        if i > 0:
            t_cost = time - t_list[i - 1][0]
            print(f"|{prename}||{name} use time = {t_cost}")

    total_t = t_list[-1][0] - t_list[0][0]
    print(f'|{prename}|total cost time = {total_t}')
    return
