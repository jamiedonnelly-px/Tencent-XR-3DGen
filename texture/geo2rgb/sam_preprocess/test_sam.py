import os
from PIL import Image
import json
import cv2
import copy
import numpy as np
from run_sam import  process_image_path, process_image_path_list


# json_path = "/aigc_cfs_2/neoshang/data/test_pipe/test_prompts_images.json"
# save_dir = "/aigc_cfs_2/neoshang/code/zero123plus/sam_preprocess/test_out1"
# os.makedirs(save_dir, exist_ok=True)

# image_path = "/aigc_cfs_gdp/neoshang/data/selected_highres_may/image-103.png"
# image_cond, image_mask = process_image_path(image_path, bg_color=127, wh_ratio=0.9, rmbg_type="1.4")
# image_cond.save(os.path.join(save_dir, "test.png"))
# image_mask_array = np.array(image_mask)
# _, binary = cv2.threshold(image_mask_array, 1, 255, cv2.THRESH_BINARY)
# Image.fromarray(binary.astype(np.uint8)).save(os.path.join(save_dir, "binary.png"))
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# for i, stat in enumerate(stats):
#     label = copy.deepcopy(labels)
#     label[label == i] = 255
#     label[label != 255] = 0
#     Image.fromarray(label.astype(np.uint8)).save(os.path.join(save_dir, f"label{i}.png"))

# for i, stat in enumerate(stats):
#     if stat[-1] < 25:
#         labels[labels == i] = -1
# labels[labels != -1] = 1
# labels[labels == -1] = 0

# Image.fromarray((labels * 255).astype(np.uint8)).save(os.path.join(save_dir, "labels.png"))

# image_mask_array = image_mask_array * labels

# print("number of labels:", num_labels)

# Image.fromarray((np.array(image_mask) * 255).astype(np.uint8)).save(os.path.join(save_dir, "test_mask.png"))

# with open(json_path, "r") as fr:
#     json_dict = json.load(fr)
# num = 0

# for classname, classdict in json_dict.items():
#     for objname, objdict in classdict.items():
#         image_path = objdict["image_path"]
#         print(image_path)
#         image_cond, image_mask = process_image_path(image_path, bg_color=127, wh_ratio=0.9, rmbg_type="1.4")
#         num += 1
#         image_cond.save(os.path.join(save_dir, str(num).zfill(4)+".png"))


# images_path_list = ["/aigc_cfs_2/neoshang/code/diffusers_triplane/data/images_mvcond/mv10/p1.png",
#                     "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/images_mvcond/mv10/p2.png",
#                     "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/images_mvcond/mv10/p3.png"]
# images_list = process_image_path_list(images_path_list, bg_color=127, wh_ratio=0.9, rmbg_type="1.4")
# for i, image in enumerate(images_list):
#     image[0].save(f"/aigc_cfs_2/neoshang/code/zero123plus/sam_preprocess/test_out1/{i}.png")
# breakpoint()


image_path = "/aigc_cfs_2/neoshang/code/zero123plus/sam_preprocess/Billiard _ Pool Table.G03.shadowless.2k.png"
image_rgb, image_mask = process_image_path(image_path, bg_color=255, wh_ratio=0.8, rmbg_type="1.4", rmbg_force=False)
image_rgb.save("/aigc_cfs_2/neoshang/code/zero123plus/sam_preprocess/test_out1/image_rgb.png")
image_mask.save("/aigc_cfs_2/neoshang/code/zero123plus/sam_preprocess/test_out1/image_mask.png")