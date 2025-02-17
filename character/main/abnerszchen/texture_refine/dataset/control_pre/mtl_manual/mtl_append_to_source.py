import os
import sys
import json
from tqdm import tqdm
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json

def append_to_source(in_source_json, in_manual_jsons_txt, out_json):
    assert os.path.exists(in_source_json), in_source_json
    assert os.path.exists(in_manual_jsons_txt), in_manual_jsons_txt
    
    lines = read_lines(in_manual_jsons_txt)
    if not lines:
        print(f'can not find lines from {in_manual_jsons_txt}')
        return

    raw_source_dict = load_json(in_source_json)
    
    append_cnt = 0
    for in_json in lines:
        append_dict, key_pair_list = parse_objs_json(in_json)
        for d_, dname, oname in key_pair_list:
            meta = raw_source_dict[d_][dname][oname]
            append_meta = append_dict[d_][dname][oname]
            meta["append_type"] = append_meta["obj_raw_type"]
            meta["Mesh_obj_raw"] = append_meta["Mesh_obj_append"]
            append_cnt += 1
       
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(raw_source_dict, out_json)
    print(f"append {append_cnt} with {len(lines)} manual json to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make maunal json from bake info json and select objs txt')
    parser.add_argument('in_source_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/source_mcwy2_4class.json")
    parser.add_argument('in_manual_jsons_txt', type=str, default="/aigc_cfs_2/sz/proj/tex_cq/dataset/control_pre/mtl_manual/append_jsons.txt")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class.json")
    # parser.add_argument('--dir_type', type=str, default="magic", help="multi_kd_uv_filter___MCWY_2_Dress___JP_DR_8_M_A.jpg")
    args = parser.parse_args()
    
    append_to_source(args.in_source_json, args.in_manual_jsons_txt, args.out_json)
