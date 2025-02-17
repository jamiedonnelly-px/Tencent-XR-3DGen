import os

from PIL import Image

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.uv_dataset.interface_dataset import load_uv_dataset

dataset_root = "/aigc_cfs_3/layer_tex/uv_datasets/debug/mcwy2"
dataset = load_uv_dataset(dataset_root)
print('dataset ', dataset)

max_train_samples = 10
train_dataset = dataset["train"].shuffle()
if max_train_samples is not None:
    train_dataset = train_dataset.select(range(max_train_samples))

print('train_dataset ', train_dataset)

image_column = "image"
conditioning_image_column = "conditioning_image"

def preprocess_train(examples):
    images = [Image.open(image).convert("RGB") for image in examples[image_column]]

    conditioning_images = [Image.open(image).convert("RGB") for image in examples[conditioning_image_column]]

    examples["pixel_values"] = images
    examples["conditioning_pixel_values"] = conditioning_images

    return examples

train_dataset = train_dataset.with_transform(preprocess_train)
print('new train_dataset ', train_dataset)
print('exp ', train_dataset[0])

    
