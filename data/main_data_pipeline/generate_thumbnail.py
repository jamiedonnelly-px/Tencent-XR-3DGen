import argparse
import os

import numpy as np
from PIL import Image

thumbnail_names = ['0022', '0041', '0121', '0268', '0085', '0291', '0100', '0068', '0091']


def picture_needed_by_thumbnail(path: str):
    image_name = os.path.split(path)[1]
    for name in thumbnail_names:
        if str(name) in image_name:
            return True
    return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render data script.')
    parser.add_argument('--color_folder_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--save_thumbnail_path', type=str,
                        help='path to mesh to be rendered')
    args = parser.parse_args()

    color_folder_path = args.color_folder_path
    save_thumbnail_path = args.save_thumbnail_path

    picture_names = os.listdir(color_folder_path)
    picture_names.sort()
    picture_data = [[], [], []]
    picture_one_row = 3
    picture_counter = 0
    for index in range(len(picture_names)):
        picture_name = picture_names[index]
        picture_full_path = os.path.join(color_folder_path, picture_name)
        if picture_needed_by_thumbnail(picture_full_path):
            ori_img = Image.open(picture_full_path)
            img_data = np.array(ori_img)
            picture_data[int(picture_counter / picture_one_row)
            ].append(img_data)
            picture_counter = picture_counter + 1

    stacked_data = []
    for data in picture_data:
        stacked_data.append(np.vstack(data))
    thumbnail = np.hstack(stacked_data)
    im = Image.fromarray(thumbnail)
    im.save(save_thumbnail_path)
