import os
import shutil
import copy
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dataset'))

from utils_dataset import load_json, save_json, HUMAN_TRAIN_KEYS, HUMAN_9_TRAIN_KEYS, HUMAN_CONDI_KEYS

# dnames = ["mario"]
# dnames = []
## 1.5w
# dnames = ['daz', "hanfeng", "xenoblade2", "feihong", "mario", "gransaga", "guofenggame", "hok", "vroid", "Objaverse_Avatar"]
## all 9w
# dnames = ['vroid', 'daz', 'gransaga', 'hok', 'lol', 'mario', 'feihong', 'guofenggame', 'hanfeng', 'honkai3',
#           'xenoblade2', 'ACGFight', 'MHKT', 'pmx', 'onepiece', 'DragonBall', 'P5', 
#           'Designcenter_1', 'DragonOath', 'BladeKnights', 'Traha', 'Sanguo', 'Darkness', 'Objaverse_Avatar', 
#           'Designcenter_20231201', 'YYS', 'MIR4']
# human_cam_keys = list(set(HUMAN_TRAIN_KEYS + HUMAN_CONDI_KEYS))

dnames = ['vroid',
          'Designcenter_1', 
          'Designcenter_20231201']
human_cam_keys = list(set(HUMAN_9_TRAIN_KEYS + HUMAN_CONDI_KEYS))

cp_files = ['cam_parameters.json']
cp_view_dirs = ['color', 'depth', 'normal']


skip_exists = False

def copy_files(src_dir, dest_dir):
    if skip_exists and os.path.exists(dest_dir):
        return True
    
    os.makedirs(dest_dir, exist_ok=True)

    for cp_file in cp_files:
        shutil.copy(os.path.join(src_dir, cp_file), dest_dir)

    for folder in cp_view_dirs:
        src_folder = os.path.join(src_dir, folder)
        dest_folder = os.path.join(dest_dir, folder)

        os.makedirs(dest_folder, exist_ok=True)

        for cam_key in human_cam_keys:
            src_file = os.path.join(src_folder, f'cam-{cam_key}.png')
            dest_file = os.path.join(dest_folder, f'cam-{cam_key}.png')

            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
    return True

def copy_files_threaded(src_dest_pairs, max_workers=96):
    """batch cp dir files

    Args:
        src_dest_pairs: pair list of (src_dir, dest_dir)
        max_workers: _description_. Defaults to 96.

    Returns:
        results list of bool
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        suc_cnt = 0
        results = list(tqdm(executor.map(lambda src_dest_pair: copy_files(src_dest_pair[0], src_dest_pair[1]), src_dest_pairs), total=len(src_dest_pairs)))
        for result in results:
            if result:
                suc_cnt += 1
    
    return results         
            
def cp_views(in_json, out_root_dir):
    objs_dict = load_json(in_json)
    
    new_objs_dict = {'data':{}}
    # new_objs_dict = copy.deepcopy(objs_dict)
    
    data_dict = objs_dict['data']
    os.makedirs(out_root_dir, exist_ok=True)

    src_dest_pairs = []
    for dname, dataset_dict in data_dict.items():
        if len(dnames) > 0 and dname not in dnames:
            continue
        new_objs_dict['data'][dname] = {}
        
        for oname, meta_dict in dataset_dict.items():
            src_dir = meta_dict['ImgDir']
            dest_dir = os.path.join(out_root_dir, dname, oname)
            src_dest_pairs.append((src_dir, dest_dir))
            
            new_objs_dict['data'][dname][oname] = meta_dict
            new_objs_dict['data'][dname][oname]['ImgDir'] = dest_dir
    
    print(f'find {len(src_dest_pairs)} src_dest_pairs')
    new_json = os.path.join(out_root_dir, 'data_used.json')
    save_json(new_objs_dict, new_json)
        
    results = copy_files_threaded(src_dest_pairs)
    suc_cnt = 0
    for result in results:
        if result:
            suc_cnt += 1    
    print(f'suc_cnt:{suc_cnt}, cp from json {in_json} to {out_root_dir}')
    

def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_2/weizhe/code_clean/rendering_free_onetri/scripts/alldata_1222.json")
    parser.add_argument('out_root_dir', type=str)
    parser.add_argument('--pool_cnt', type=int, default=96)
    args = parser.parse_args()

    cp_views(args.in_json, args.out_root_dir)

if __name__ == "__main__":
    main()
