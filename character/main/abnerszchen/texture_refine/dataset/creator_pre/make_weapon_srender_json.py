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

cams=[9, 11, 13, 15,  32, 34, 36, 38]   # 1/4, 0~8
# condi_cams = [24]   # [3, 0] [3, 2], [4, 0]
condi_cams = [24, 26, 33]   # [3, 0] [3, 2], [4, 1]

        
def make_dataset_json_srender(in_raw_json, split_json, render_root, train_view_cnt=10, split_test300=True):
    objs_dict, key_pair_list = parse_objs_json(in_raw_json)
    split_dict = load_json(split_json)
    new_data_dict = dict()
    valid_cnt = 0
    train_pairs, test_pairs = [], []
    for key_pair in key_pair_list:
        d_, dname, oname = key_pair
        ImgDir = objs_dict[d_][dname][oname]['ImgDir']
        
        imgs = [os.path.join(ImgDir, f'color/cam-{cam:04d}.png') for cam in cams]
        depths = [os.path.join(ImgDir, f'depth/cam-{cam:04d}.png') for cam in cams]
        pose_json = os.path.join(ImgDir, 'cam_parameters.json')
        
        meta_dict = {}
        meta_dict['pose_json'] = pose_json
        meta_dict['condition_imgs'] = [os.path.join(ImgDir, f'color/cam-{cam:04d}.png') for cam in condi_cams]
        meta_dict['tex_pairs'] = [[img, depth] for img, depth in zip(imgs, depths)]
        
        # TODO check
        
        if oname in split_dict['train'] or oname in split_dict['val']:
            train_pairs.append(key_pair)
        elif oname in split_dict['test']:
            test_pairs.append(key_pair)
        else:
            print('invalid ', key_pair)
            continue


        if dname not in new_data_dict:
            new_data_dict[dname] = {}
        new_data_dict[dname][oname] = meta_dict
        valid_cnt += valid_cnt
                    
    out_dict = {'data': new_data_dict}
    os.makedirs(render_root, exist_ok=True)
    out_json = os.path.join(render_root, 'tex_creator.json')
    save_json(out_dict, out_json)
    
    split_jsons(out_json, train_pairs, test_pairs, render_root, prefix='tex_creator')
    
    print(f'save done to {out_json} with valid_cnt {valid_cnt}/{len(key_pair_list)}')

    if split_test300:
        min_cnt = min(len(train_pairs), 300)
        random.shuffle(train_pairs)
        min_train_pairs = train_pairs[:min_cnt]
        
        min_cnt_test = min(len(test_pairs), 30)
        random.shuffle(test_pairs)
        min_test_pairs = test_pairs[:min_cnt_test]
        split_jsons(out_json, min_train_pairs, min_test_pairs, render_root, prefix='sample300_tex_creator')
        
        print()
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='make dataset json, split train/val/test json')
    parser.add_argument('in_raw_json', type=str, help='from diffusion', default='/aigc_cfs/neoshang/data/json_for_traintest/objaverse/latent_geotri_Transformer_v20_128_obj_20231219_neo_20231219_add_condition_sort_images.json')
    parser.add_argument('split_json', type=str)
    parser.add_argument('render_root', type=str)
    args = parser.parse_args()

    # Run.
    make_dataset_json_srender(args.in_raw_json, args.split_json, args.render_root)
    return

if __name__ == "__main__":
    main()
