import cv2
import numpy as np
import torch
import os
import glob
import argparse
def create_mask(image, black_threshold=15, white_threshold=240):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray_image, 0, black_threshold)
    white_mask = cv2.inRange(gray_image, white_threshold, 255)
    mask = cv2.bitwise_or(black_mask, white_mask)
    return mask

def fill_holes(image):
    mask = create_mask(image)
    cv2.imwrite('debug_mask.png', mask)
    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    filled_image = cv2.inpaint(image, mask, 15, cv2.INPAINT_TELEA)
    return filled_image

def refine_tex(raw_path):
    numpy_image = cv2.imread(raw_path)
    filled_image = fill_holes(numpy_image)
    return filled_image


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_tex', type=str)
    parser.add_argument('out_tex', type=str)
    args = parser.parse_args()

    
    filled_image = refine_tex(args.in_tex)
    cv2.imwrite(args.out_tex, filled_image)
    return

if __name__ == "__main__":
    main()

