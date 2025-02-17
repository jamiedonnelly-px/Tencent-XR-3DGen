import os
import sys
import json
from tqdm import tqdm
import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json

def cp_once(meta, key, root_dir="/aigc_cfs_gdp/"):
    raw_path = meta[key]
    raw_path_dir = os.path.dirname(raw_path)
        
    match = re.match(r"^(/[^/]+)", raw_path_dir)
    raw_root_dir = match.group(1) + "/"

    new_path_dir = raw_path_dir.replace(raw_root_dir, root_dir)
    new_path = raw_path.replace(raw_root_dir, root_dir)
    
    if not os.path.exists(new_path_dir):
        os.makedirs(new_path_dir)  
         
    if not os.path.exists(new_path):
        os.system(f"cp -R {raw_path_dir}/* {new_path_dir}")         
    
    assert os.path.exists(new_path), new_path
    meta[key] = new_path
    return

def cp_meta_once(meta, keys, root_dir="/aigc_cfs_gdp/"):
    for key in keys:
        cp_once(meta, meta, root_dir)
    return
def cp_to_cfs(in_web_flatten_json, out_web_flatten_json, root_dir="/aigc_cfs_gdp/"):
    assert os.path.exists(in_web_flatten_json), in_web_flatten_json
    os.makedirs(os.path.dirname(out_web_flatten_json), exist_ok=True)
    
    web_flatten_dict = load_json(in_web_flatten_json)
  
    
    valid_cnt = 0
    for oname, meta in web_flatten_dict.items():
        cp_meta_once(meta, ["Mesh_obj_raw", "Obj_Mesh"], root_dir)
        valid_cnt += 1
        print(f"{oname} ok")
        
    save_json(web_flatten_dict, out_web_flatten_json)
    
    print(f"valid_cnt {valid_cnt} from {in_web_flatten_json} save to {out_web_flatten_json}")
            
# def cp_to_cfs(in_web_flatten_json, out_web_flatten_json, root_dir="/aigc_cfs_gdp/"):
#     assert os.path.exists(in_web_flatten_json), in_web_flatten_json
#     os.makedirs(os.path.dirname(out_web_flatten_json), exist_ok=True)
    
#     web_flatten_dict = load_json(in_web_flatten_json)
  
#     valid_cnt = 0
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         for oname, meta in web_flatten_dict.items():
#             executor.submit(cp_meta_once, meta, ["Mesh_obj_raw", "Obj_Mesh"], root_dir)
#             valid_cnt += 1
#             print(f"{oname} ok")
        
#     save_json(web_flatten_dict, out_web_flatten_json)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cp Obj_Mesh and Mesh_obj_raw to gdp cfs, inplace root to')
    parser.add_argument('in_web_flatten_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class_0406.json")
    parser.add_argument('out_web_flatten_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/layer_embedding_20240403_total.json")
    parser.add_argument('--root_dir', type=str, default="/aigc_cfs_gdp/", help="multi_kd_uv_filter___MCWY_2_Dress___JP_DR_8_M_A.jpg")
    args = parser.parse_args()
    
    assert os.path.exists(f"{args.root_dir}/this_cfs_gdp_0530_sz.txt"), f"have not mount {args.root_dir}"
    
    cp_to_cfs(args.in_web_flatten_json, args.out_web_flatten_json, args.root_dir)
