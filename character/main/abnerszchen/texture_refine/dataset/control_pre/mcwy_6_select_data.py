import os
import logging
import argparse
import sys
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
from scripts.utils_pool_cmds import run_commands_in_parallel

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='select data')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/image_caption_done.json")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/image_caption_done_select.json")
    parser.add_argument("--select_all", action="store_true", help="select all except error if True")
    parser.add_argument('--select_key_json', type=str, default="", help="example: /aigc_cfs_3/layer_tex/readyplayerme/vis_sort_select/select.json")
    args = parser.parse_args()
    
    only_right_mtl = not args.select_all
    only_select_dnames = []
    # only_select_dnames = ["Designcenter_outfit", "Designcenter_top_bottom"]
    
    in_json = args.in_json
    out_json = args.out_json
    select_key_json = args.select_key_json
    assert os.path.exists(in_json), in_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_json)
    select_key_dict = {}
    if select_key_json and os.path.exists(select_key_json):
        select_key_dict = load_json(select_key_json)
    
    out_dict = {"data":{}}
    
    cnt = 0
    for d_, dname, oname in key_pair_list:
        if only_select_dnames and dname not in only_select_dnames:
            continue
        meta = objs_dict[d_][dname][oname]
        if only_right_mtl and meta["mtl_type"] != "right":
            continue
        if meta["mtl_type"] == "error":
            continue
        if select_key_dict:
            if dname not in select_key_dict or (dname in select_key_dict and oname not in select_key_dict[dname]):
                print('not in select_key_dict, skip')
                continue
        
        # pass, select
        if dname not in out_dict[d_]:
            out_dict[d_][dname] = {}
        out_dict[d_][dname][oname] = meta
        cnt += 1
 
    save_json(out_dict, out_json)
    print(f'select data {cnt}/{len(key_pair_list)} done to {out_json}')
        
if __name__ == "__main__":
    main()

