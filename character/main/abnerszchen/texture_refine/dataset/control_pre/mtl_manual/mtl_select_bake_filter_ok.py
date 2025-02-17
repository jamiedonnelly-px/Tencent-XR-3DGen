import os
import sys
import json
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines

def parse_key(vis_jpg, magic_key="___"):
    """pase vis_jpg like multi_kd_uv_filter___MCWY_2_Dress___JP_DR_8_M_A to MCWY_2_Dress, JP_DR_8_M_A

    Args:
        vis_jpg (_type_): _description_
        magic_key (str, optional): _description_. Defaults to "___".
    """
    name = os.path.splitext(vis_jpg)[0]
    res = name.split(magic_key)
    dname, oname = res[-2], res[-1]
   
    return dname, oname

def select_bake_filter_ok(in_bake_json, in_txt, out_json, magic_key="___"):
    assert os.path.exists(in_bake_json), in_bake_json
    assert os.path.exists(in_txt), in_txt
    
    lines = read_lines(in_txt)
    if not lines:
        print(f'can not find lines from {in_txt}')
        return

    bake_dict = load_json(in_bake_json)
    manual_dict = {"data":{}}
    cnt = 0
    for vis_jpg in lines:
        try:
            dname, oname = parse_key(vis_jpg, magic_key=magic_key)
            if dname not in manual_dict["data"]:
                manual_dict["data"][dname] = {}
            manual_dict["data"][dname][oname] = {"obj_raw_type":"bake", "Mesh_obj_append":bake_dict["data"][dname][oname]["Bake"]}
            cnt += 1
        except Exception as e:
            print(e)
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(manual_dict, out_json)
    print(f"save {cnt}/{len(lines)} manual objs to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make maunal json from bake info json and select objs txt')
    parser.add_argument('in_bake_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/bake_info.json")
    parser.add_argument('in_txt', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/mcwy2_mtl_pack/one_kd_uv_before_filter_manual/filter_ok_temp0406.txt")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/bake_filter_ok_one_kd_temp0406.json")
    # parser.add_argument('--dir_type', type=str, default="magic", help="multi_kd_uv_filter___MCWY_2_Dress___JP_DR_8_M_A.jpg")
    args = parser.parse_args()
    
    select_bake_filter_ok(args.in_bake_json, args.in_txt, args.out_json)
