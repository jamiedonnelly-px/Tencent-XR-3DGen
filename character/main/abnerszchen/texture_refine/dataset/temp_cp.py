import os
import shutil
import os
import argparse
import json
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, set_start_method
import shutil

from utils_dataset import parse_objs_json
json_file_path = "/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/tex_creator.json"

raw_dict, key_pair_list = parse_objs_json(json_file_path)

src_folder_tex = "/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug"
dst_folder_tex = "/aigc_cfs/sz/data/tex/first_300"

src_folder_condi = "/apdcephfs_cq8/share_2909871/shenzhou/result/gen/b1/sz_diffusion_4096_v0_test/first_2k"
dst_folder_condi = "/aigc_cfs/sz/data/tex/first_300/condi"

def copy_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

def process_key_pair(key_pair):
    value = raw_dict[key_pair[0]][key_pair[1]][key_pair[2]]
    tex_pairs = value["tex_pairs"]

    for tex_pair in tex_pairs:
        for img_path in tex_pair:
            relative_path = os.path.relpath(img_path, src_folder_tex)
            dst_path = os.path.join(dst_folder_tex, relative_path)
            copy_file(img_path, dst_path)

    condi_img_path = value["Condition_img"]
    relative_path_condi = os.path.relpath(condi_img_path, src_folder_condi)
    dst_path_condi = os.path.join(dst_folder_condi, relative_path_condi)
    copy_file(condi_img_path, dst_path_condi)

num_threads = 8

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(tqdm(executor.map(process_key_pair, key_pair_list), total=len(key_pair_list)))

shutil.copy2(json_file_path, os.path.join(dst_folder_tex, 'tex_creator.json'))


