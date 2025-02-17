import os
import argparse
from PIL import Image
import numpy as np
from diffusers.utils import load_image, make_image_grid


def split_image_grid(image):
    width, height = image.size

    single_width = width // 2
    try_cnt = height // single_width // 2

    images = []
    for i in range(try_cnt):
        upper = 2 * i * single_width
        lower = (2 * i + 2) * single_width
        sub_image = image.crop((0, upper, 2 * single_width, lower))
        images.append(sub_image)

    return images

def main():
    parser = argparse.ArgumentParser(
        description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_png', type=str, help='path of  ')
    parser.add_argument('out_png', type=str, help='path of ')
    parser.add_argument('--out_res', type=int, default=1024)
    parser.add_argument('--sr_py', type=str, default="/aigc_cfs_2/sz/proj/Real-ESRGAN/inference_realesrgan.py")
 
    args = parser.parse_args()
    in_png = args.in_png
    out_png = args.out_png
    out_res = args.out_res
    sr_py = args.sr_py
    py_dir = os.path.dirname(sr_py)
    
    in_pil = Image.open(in_png)
    out_dir = os.path.dirname(out_png)
    os.makedirs(out_dir, exist_ok=True)

    grid_list = split_image_grid(in_pil)
    sr_pils = []
    for i, image_grid in enumerate(grid_list):
        split_raw_path = os.path.join(out_dir, f"debug_{i}.png")
        sr_dir = os.path.join(out_dir, "infer_sr")
        os.makedirs(sr_dir, exist_ok=True)
        image_grid.save(split_raw_path)
        s = int(out_res // image_grid.size[0] * 2)
        cmd = f"cd {py_dir} && python {sr_py} -n RealESRGAN_x4plus -i {split_raw_path} -o {sr_dir} -s {s}"
        print('cmd ', cmd)
        # breakpoint()
        os.system(cmd)
        
        sr_pils.append(Image.open(os.path.join(sr_dir, f"debug_{i}_out.png")))
    
    make_image_grid(sr_pils, len(grid_list), 1).save(out_png)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()