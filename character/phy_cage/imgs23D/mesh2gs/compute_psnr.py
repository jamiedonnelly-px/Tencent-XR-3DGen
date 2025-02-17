import os, pdb
import numpy as np 
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread, imsave
from PIL import Image

basedirs = ["axe0", "babyyoda0", "bottle0", "cup0", "foxcape0", "frog0", "inkquill0", "plant0", "rabbit0", "shiba0", "strawhat0"]


def get_first_img(x):
    return np.array(x)[:,:256] / 255.


for basedir in basedirs:
    try:
        path1 = os.path.join("output_focus0", basedir)
        # path2 = os.path.join("output_regs", basedir)
        path3 = os.path.join("output", basedir)

        gt = np.array(imread(os.path.join("../../testdata", f"{basedir}.png"))) / 255.
        if gt.shape[-1] == 4:
            gt = gt[..., 3:] * gt[..., :3] + (1 - gt[..., 3:]) * 1.

        img1 = get_first_img(imread(os.path.join(path1, "save_image.png")))
        # img2 = get_first_img(imread(os.path.join(path2, "save_image.png")))
        img3 = get_first_img(imread(os.path.join(path3, "save_image.png")))

        sub1 = get_first_img(imread(os.path.join(path1, "sem_1/save_image.png")))
        # sub2 = get_first_img(imread(os.path.join(path2, "sem_1/save_image.png")))
        sub3 = get_first_img(imread(os.path.join(path3, "sem_1/save_image.png")))

        res1 = peak_signal_noise_ratio(gt, img1)
        # res2 = peak_signal_noise_ratio(gt, img2)
        res3 = peak_signal_noise_ratio(gt, img3)

        print(basedir, ": ", res1, res3)
    except:
        pass


