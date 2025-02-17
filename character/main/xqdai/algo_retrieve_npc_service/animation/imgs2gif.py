import os
import glob
from pathlib import Path
from numpy import arange, pi, sin, cos, arccos
import numpy as np
from PIL import Image
import cv2

def set_transparent_background_to_white(input_image_path, output_image_path):
    image = Image.open(input_image_path).convert("RGBA")
    width, height = image.size
    white_background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    white_background.paste(image, (0, 0), image)
    white_background.convert("RGB").save(output_image_path)

if __name__ == '__main__':
    import sys
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    fbx_path = argv[0]
    gif_file = argv[1]

    current_file_path = os.path.abspath(__file__)
    current_file_directory = os.path.dirname(current_file_path)

    output_folder = current_file_directory+"/save/output_"
    output_folder = output_folder+fbx_path.split("/")[-4]+"_"+fbx_path.split("/")[-3]

    parent_dir = os.path.dirname(output_folder)

    png_files = sorted(glob.glob(os.path.join(output_folder, '*.png')))

    for png_file in png_files:
        set_transparent_background_to_white(png_file,png_file)


    frames = [Image.open(png_file) for png_file in png_files]
    frames[0].save(gif_file, save_all=True, append_images=frames[1:], loop=0,duration=1000 / 20, optimize=False, lossless=True)

    os.system("rm -rf "+output_folder)