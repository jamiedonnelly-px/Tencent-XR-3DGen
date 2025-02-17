import os
import torch
from PIL import Image

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

def save_rgba_normals_images(rgba: torch.Tensor, output_png: str):
    """save batch rgba tensor in [-1, 1] to pils

    Args:
        rgba (torch.Tensor):  [N, H, W, 4]  RGBA tensor，in [-1, 1]
        output_png (str): 
    """
    rgba_normalized = (rgba + 1) / 2
    rgba_normalized = (rgba_normalized * 125) / 255.0
    
    pils = []
    for i in range(rgba_normalized.shape[0]):
        img = rgba_normalized[i]

        img = (img * 255).byte()

        img_pil = Image.fromarray(img.cpu().numpy(), 'RGBA')

        pils.append(img_pil)
    concatenate_images_horizontally(pils, output_png)

def save_rgba_depth_images(rgba: torch.Tensor, output_png: str):
    """save batch rgba tensor in [-1, 1] to pils

    Args:
        rgba (torch.Tensor):  [N, H, W, 4]  RGBA tensor，in [-1, 1]
        output_png (str): 
    """
    # rgba_normalized = (rgba + 1) / 2
    rgba_normalized = rgba / 20.0
    rgba_normalized = rgba_normalized[:, :, :,0]
    print(rgba_normalized.shape)
    rgba_normalized = torch.cat([rgba_normalized.unsqueeze(3), rgba_normalized.unsqueeze(3), rgba_normalized.unsqueeze(3)], dim=3)
    
    pils = []
    for i in range(rgba_normalized.shape[0]):
        img = rgba_normalized[i]

        img = (img * 255).byte()

        img_pil = Image.fromarray(img.cpu().numpy(), 'RGB')

        pils.append(img_pil)
    concatenate_images_horizontally(pils, output_png)