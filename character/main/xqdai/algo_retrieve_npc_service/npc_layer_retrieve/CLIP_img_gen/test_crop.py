from PIL import Image
import argparse
import cv2
import os
from pathlib import Path
import pandas as pd
import numpy as np
from filelock import FileLock
import random
import json


def find_bounding_box(image_path, alpha_threshold=100):
    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    left, top, right, bottom = None, None, None, None

    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = image.getpixel((x, y))
            if a > alpha_threshold:
                if left is None or x < left:
                    left = x
                if top is None or y < top:
                    top = y
                if right is None or x > right:
                    right = x
                if bottom is None or y > bottom:
                    bottom = y

    if left is None:
        return None

    delta = 10

    left = np.max([0,left-delta])
    top = np.max([0,top-delta])
    right = np.min([image.width-1,right+delta])
    bottom = np.min([image.height-1,bottom+delta])
   
    return left, top, right, bottom

def save_cropped_image(image_path, bounding_box,output_path):
    image = Image.open(image_path)
    cropped_image = image.crop(bounding_box)
    cropped_image.save(output_path)
    

if __name__ == "__main__":
    
    img_path = "/mnt/aigc_cfs_gz/layer_avatar_data/readplayerMe/render/render_data/bottom/1_output_512_MightyWSB/color/cam-0000.png"
    output_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/test_crop.png"
    bounding_box = find_bounding_box(img_path)
    save_cropped_image(img_path,bounding_box,output_path)