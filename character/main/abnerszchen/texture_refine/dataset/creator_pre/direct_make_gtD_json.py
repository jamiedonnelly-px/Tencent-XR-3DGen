import os
import argparse
import random
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(os.path.join(project_root, "dataset"))

from utils_dataset import read_lines, save_json, parse_objs_json, split_jsons


def process_est_obj(est_obj, objs_dict):
    dname, oname = est_obj.split('/')[-4:-2]
    meta_dict = objs_dict['data'][dname][oname]
    ImgDir = meta_dict["ImgDir"]
    pose_json = os.path.join(ImgDir, 'cam_parameters.json')

    temp = glob.glob(os.path.join(os.path.dirname(est_obj), f'cam-*.png'))
    if not temp:
        return None

    Condition_img = temp[0]

    train_id_list = meta_dict['image_sort_list']
    img_pair_list = []
    for train_id in train_id_list:
        cam_name = f'cam-{train_id:04d}.png'
        gt_img = os.path.join(ImgDir, f'color/{cam_name}')
        render_depth_img = os.path.join(ImgDir, f'depth/{cam_name}')
        if os.path.exists(gt_img) and os.path.exists(render_depth_img):
            img_pair_list.append([gt_img, render_depth_img])

    if len(img_pair_list) > 0 and os.path.exists(pose_json):
        new_meta_dict = {'Condition_img': Condition_img, 'pose_json': pose_json,
                         'diffusion_obj': est_obj, 'tex_pairs': img_pair_list}
        is_diffusion_train = 'latent' in meta_dict
        return dname, oname, new_meta_dict, is_diffusion_train
    else:
        return None



def make_gtD_json(in_raw_json, in_est_objs_txt, out_dir, pool_cnt=8, train_view_cnt=10, split_test300=True):
    assert os.path.exists(in_raw_json)
    objs_dict, key_pair_list = parse_objs_json(in_raw_json)
    
    # dname/oname
    est_objs = read_lines(in_est_objs_txt)
    if not est_objs or len(est_objs) < 1:
        print('can not fin any obj in ', in_est_objs_txt)
        return

    os.makedirs(out_dir, exist_ok=True)

    new_data_dict = {}
    train_pairs, test_pairs = [], []
    valid_cnt = 0

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda est_obj: process_est_obj(est_obj, objs_dict), est_objs), total=len(est_objs)))

    for result in results:
        if result is None:
            continue

        dname, oname, new_meta_dict, is_diffusion_train = result
        if not dname in new_data_dict:
            new_data_dict[dname] = dict()
        new_data_dict[dname][oname] = new_meta_dict
        valid_cnt += 1

        pair = ('data', dname, oname)
        if is_diffusion_train:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
                  
    out_dict = {'data': new_data_dict}
    out_json = os.path.join(out_dir, 'tex_creator.json')
    save_json(out_dict, out_json)
    
    # split train/val
    split_jsons(out_json, train_pairs, test_pairs, out_dir, prefix='tex_creator')
    print(f'save done to {out_json} with valid_cnt {valid_cnt}/{len(est_objs)}')

    if split_test300:
        min_cnt = min(len(train_pairs), 300)
        random.shuffle(train_pairs)
        min_train_pairs = train_pairs[:min_cnt]
        
        min_cnt_test = min(len(test_pairs), 30)
        random.shuffle(test_pairs)
        min_test_pairs = test_pairs[:min_cnt_test]
        split_jsons(out_json, min_train_pairs, min_test_pairs, out_dir, prefix='sample300_tex_creator')
        
        print('split_test300')  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list (batch run render_obj_texture)')
    parser.add_argument('in_raw_json', type=str, help='need find pose json from raw json')
    parser.add_argument('in_est_objs_txt', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--pool_cnt', type=int, default=16)
    args = parser.parse_args()

    # Run.
    make_gtD_json(args.in_raw_json, args.in_est_objs_txt, args.out_dir, pool_cnt=args.pool_cnt)
    return

if __name__ == "__main__":
    main()
