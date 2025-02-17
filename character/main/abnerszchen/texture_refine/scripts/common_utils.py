import os
import glob
from PIL import Image

def make_gif_dir(in_dir, out_gif_path, image_key='epoch_*.jpg', duration=500):
    os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)
    
    filenames = glob.glob(os.path.join(in_dir, image_key))
    if not filenames or len(filenames) < 1:
        print(f'can not find valid filenames with image_key {image_key} in in_dir {in_dir}')
        return
    images = [Image.open(fn) for fn in filenames]
    images[0].save(out_gif_path, save_all=True, append_images=images[1:], loop=0, duration=duration)