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

        
def make_dataset_json_lrm(in_raw_json, render_root, train_view_cnt=10, split_test300=True):
    objs_dict, key_pair_list = parse_objs_json(in_raw_json)
   
    render_done_json = os.path.join(render_root, 'render_done.json')
    assert os.path.exists(render_done_json)
    
    render_done_dict = load_json(render_done_json)
    
    valid_cnt = 0
    new_data_dict = {}
    train_pairs, test_pairs = [], []
    for diffusion_obj, render_obj_dir in tqdm(render_done_dict.items()):
        if not os.path.exists(diffusion_obj):
            print('error train_id_list ', train_id_list)
            continue        
        dname, oname = render_obj_dir.split('/')[-2:]
        meta_dict = objs_dict['data'][dname][oname]
        
        train_id_list = meta_dict['image_sort_list'][:train_view_cnt]
        temp = glob.glob(os.path.join(os.path.dirname(diffusion_obj), f'cam-*.png'))
        if not temp:
            continue
        Condition_img = temp[0]
        
        if not os.path.exists(Condition_img):
            print(f'Error: invalid condition imgs of {Condition_img}')
            print('error train_id_list ', train_id_list)
            continue
        
        gt_img_dir = meta_dict['ImgDir']
        pose_json = os.path.join(gt_img_dir, 'cam_parameters.json')
        img_pair_list = []
        for train_id in train_id_list:
            cam_name = f'cam-{train_id:04d}.png'
            gt_img = os.path.join(gt_img_dir, f'color/{cam_name}')
            render_depth_img = os.path.join(render_obj_dir, f'depth_cam-{train_id:04d}.png')
            if os.path.exists(gt_img) and os.path.exists(render_depth_img):
                img_pair_list.append([gt_img, render_depth_img])

        if len(img_pair_list) > 0 and os.path.exists(pose_json):
            new_meta_dict = {'Condition_img':Condition_img, 'pose_json':pose_json,
                             'diffusion_obj':diffusion_obj, 'tex_pairs':img_pair_list}
            if not dname in new_data_dict:
                new_data_dict[dname] = dict()
            new_data_dict[dname][oname] = new_meta_dict
            valid_cnt += 1
            
            is_diffusion_train = 'latent' in meta_dict
            pair = ('data', dname, oname)
            if is_diffusion_train:
                train_pairs.append(pair)
            else:
                test_pairs.append(pair)
        else:
            print('invalid ', dname, oname)
            
    out_dict = {'data': new_data_dict}
    out_json = os.path.join(render_root, 'tex_creator.json')
    save_json(out_dict, out_json)
    
    split_jsons(out_json, train_pairs, test_pairs, render_root, prefix='tex_creator')
    
    print(f'save done to {out_json} with valid_cnt {valid_cnt}/{len(list(render_done_dict.keys()))}')

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
    parser.add_argument('render_root', type=str)
    args = parser.parse_args()

    # Run.
    make_dataset_json_lrm(args.in_raw_json, args.render_root)
    return

if __name__ == "__main__":
    main()
