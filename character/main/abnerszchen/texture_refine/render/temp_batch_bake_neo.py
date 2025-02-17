import os
import random
import json
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)    
    return data

def save_json(json_data, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))    
    return
def parse_objs_json(objs_json):
    """ parse standard json to dict and pair list
    return: objs_dict: dict
    key_pair_list: list of pair ('data', dtype, oname)
    """
    if not os.path.exists(objs_json):
        print('[Error] can not find objs_json '.format(objs_json))
        return dict(), []
    objs_dict = load_json(objs_json)
    if 'data' not in objs_dict:
        print('[Error] not standard json '.format(objs_json))
        return dict(), []    
    key_pair_list = []
    for dataset, dataset_dict in objs_dict['data'].items():
        key_pair_list += [('data', dataset, obj_name) for obj_name in list(dataset_dict.keys())]
 
    return objs_dict, key_pair_list

def batch_bake(in_json, out_root_dir, cnt=100, temp_listcnt=1):
    objs_dict, key_pair_list = parse_objs_json(in_json)
    random.seed(1234)
    random.shuffle(key_pair_list)
    key_pair_list = key_pair_list[:cnt]
    os.makedirs(out_root_dir, exist_ok=True)

    outdirs = []
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        obj_path= meta["Manifold"]
        ImgDir = meta['ImgDir']
        color_dir = os.path.join(ImgDir, "color")
        in_pose_json = os.path.join(ImgDir, "cam_parameters.json")
        
        outdir = os.path.join(out_root_dir, dname, oname)
        if os.path.exists(os.path.join(outdir, "mesh.obj")):
            continue
        cmd = f"python temp_bake_neo_views.py {obj_path} {color_dir} {in_pose_json} {outdir} --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt {temp_listcnt}"
        
        os.system(cmd)
        outdirs.append(outdir)
      
    print(f'out cnt:{len(outdirs)}, from json {in_json} to {out_root_dir}')
    

def main():
    parser = argparse.ArgumentParser(description='BAKE obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_2/WSB/Data/debug/test_xcube/new_neo_data.json")
    parser.add_argument('out_root_dir', type=str)
    parser.add_argument('--temp_listcnt', type=int, default=1, help="if > 0, decimate mesh")
    args = parser.parse_args()

    batch_bake(args.in_json, args.out_root_dir, temp_listcnt=args.temp_listcnt)

if __name__ == "__main__":
    main()
