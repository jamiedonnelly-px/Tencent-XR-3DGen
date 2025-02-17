import os
import argparse
import json
import glob
import shutil
from tqdm import tqdm
import random
import copy

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(os.path.join(project_root, "dataset"))
from utils_dataset import load_json, save_json, parse_objs_json, split_pod_json, split_jsons

def get_split(in_dir):
    keys = ['train', 'val', 'test']
    split_dict = {}
    for key in keys:
        _, key_pairs = parse_objs_json(os.path.join(in_dir, f'tex_creator_{key}.json'))
        split_dict[key] = [key_pair[-1] for key_pair in key_pairs]
    os.makedirs(in_dir, exist_ok=True)
    save_json(split_dict, os.path.join(in_dir, 'split.json'))
    
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='make dataset json, split train/val/test json')
    parser.add_argument('in_dir', type=str, help='from diffusion', default='/aigc_cfs/neoshang/data/json_for_traintest/objaverse/latent_geotri_Transformer_v20_128_obj_20231219_neo_20231219_add_condition_sort_images.json')
    args = parser.parse_args()

    # Run.
    get_split(args.in_dir)
    return

if __name__ == "__main__":
    main()
