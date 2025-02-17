from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))
import time
import cv2
# import gradio as gr
import numpy as np
import torch
import PIL
from PIL import Image
import rembg
from rembg import remove
rembg_session = rembg.new_session()
from segment_anything import sam_model_registry, SamPredictor

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# def check_input_image(input_image):
#     if input_image is None:
#         raise gr.Error("No image uploaded!")

class RMBG(object):
    def __init__(self, device):
        # sam = sam_model_registry["vit_h"](checkpoint="/aigc_cfs_gdp/weixuan/mv2mesh_weights/4views_512_20240823/sam_vit_h_4b8939.pth").to(device)
        sam = sam_model_registry["vit_h"](checkpoint="/aigc_cfs_gdp/xibin/models/mv2mesh_seg_rmbg/sam_vit_h_4b8939.pth").to(device)
        # sam = seg_model.to(device)
        self.predictor = SamPredictor(sam)
        
    def rmbg_sam(self, input_image):
        def _sam_segment(predictor, input_image, *bbox_coords):
            bbox = np.array(bbox_coords)
            image = np.asarray(input_image)

            start_time = time.time()
            predictor.set_image(image)

            masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

            print(f"SAM Time: {time.time() - start_time:.3f}s")
            out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            out_image[:, :, :3] = image
            out_image_bbox = out_image.copy()
            out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
            torch.cuda.empty_cache()
            return Image.fromarray(out_image_bbox, mode='RGBA')

        RES = 1024
        input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)

        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        return _sam_segment(self.predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
        
    def rmbg_rembg(self, input_image):
        def _rembg_remove(
            image: PIL.Image.Image,
            rembg_session = None,
            force: bool = False,
            **rembg_kwargs,
        ) -> PIL.Image.Image:
            do_remove = True
            
            # breakpoint()
            if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
                # explain why current do not rm bg
                print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
                background = Image.new("RGBA", image.size, (0, 0, 0, 0))
                image = Image.alpha_composite(background, image)
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            return image
        return _rembg_remove(input_image, rembg_session, force_remove=True)

    def run(self, rm_type, image, foreground_ratio, background_choice, backgroud_color,):
        # image = cv2.resize(np.array(image), (crop_size, crop_size))
        # image = Image.fromarray(image)

        if background_choice == "Alpha as mask":
            image = do_resize_content(image, foreground_ratio)
            image = expand_to_square(image)
            image = add_background(image, backgroud_color)
            
            return image
        
        elif "Remove" in background_choice:
            
            if rm_type.upper() == "SAM":
                image = self.rmbg_sam(image)
            elif rm_type.upper() == "REMBG":
                image = self.rmbg_rembg(image)
            else:
                return -1

            # breakpoint()
            image = do_resize_content(image, foreground_ratio)
            image = expand_to_square(image)
            image = add_background(image, backgroud_color)
            return image
    
        elif "Original" in background_choice:
            return image
        else:
            return -1
        
    
    def run_and_resize(self, rm_type, image, foreground_ratio, background_choice, backgroud_color, obj_size=512):
        # image = cv2.resize(np.array(image), (crop_size, crop_size))
        # image = Image.fromarray(image)

   
        if rm_type.upper() == "SAM":
            image = self.rmbg_sam(image)
        elif rm_type.upper() == "REMBG":
            image = self.rmbg_rembg(image)
        else:
            return -1
        
        image = np.array(image)
        B = np.argwhere(image)
        (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
        image = image[ystart:ystop, xstart:xstop, :]
        image = Image.fromarray(image)
        
        # breakpoint()
        # image = image.resize((obj_size,obj_size))
        
        # image = do_resize_content(image, foreground_ratio)
        # image = expand_to_square(image)
        # image = add_background(image, backgroud_color)
        
        return image
    
        
def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def add_background(image, bg_color=(255, 255, 255, 0)):
    # given an RGBA image, alpha channel is used as mask to add background color
    # breakpoint()
    background = Image.new("RGBA", image.size, bg_color)
    
    return Image.alpha_composite(background, image)

def repadding_rgba_image(image, rescale=True, ratio=0.8, bg_color=255):
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except:
            return -1
    in_w = image.width
    in_h = image.height

    print("in_w: ", in_w)
    print("in_h: ", in_h)

    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])

    white_bg_image = Image.new('RGB', size=(in_w, in_h), color=(bg_color, bg_color, bg_color))
    white_bg_image.paste(image, (0, 0), mask=image)

    max_size = max(w, h)
    if rescale:
        side_len = int(max_size / ratio)
    else:
        # side_len = in_w
        side_len = max_size
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

    return rgba_image

if __name__ == "__main__":
    # print("running here !!!")
    device = torch.device(f"cuda:{'0'}" if torch.cuda.is_available() else "cpu")
    rmbg = RMBG(device)
    img_path = "/aigc_cfs_gdp/sz/result/pipe_test/639c3cee-085f-4e2d-acfa-fce1150d36b6/mesh2image.png"

    img = Image.open(img_path)

    img = rmbg.run_and_resize("rembg", img, 0.8, 'Remove', (255, 255, 255, 255), )
    # mvimg_0 = rmbg.run_and_resize("rembg", mvimg_0, 0.9, 'Remove', (127, 127, 127, 255), )

    img = repadding_rgba_image(img, rescale=True, ratio=0.9, bg_color=255)
    img = Image.fromarray(img)

    background = Image.new("RGBA", img.size, (127, 127, 127, 255))
    img = Image.alpha_composite(background, img).convert("RGB")

    img.save("test_seg.png")

    # img = np.array(img)
    # print("max: ", np.max(img))
    # print("min: ", np.min(img))
    # mask = img[:,:,3]
    # img[:,:,0][mask == 0] = 127
    # img[:,:,1][mask == 0] = 127
    # img[:,:,2][mask == 0] = 127

    # img_save = img[:,:,:3]
    # img_save = Image.fromarray(img_save)
    # print(img_save.size)
    # # breakpoint()
    # img_save.save("test_seg.png")
    # breakpoint()
