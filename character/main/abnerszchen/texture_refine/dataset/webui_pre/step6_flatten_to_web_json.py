import os
import sys
import json
from tqdm import tqdm
import argparse
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)


from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json


def flatten_to_web_json(in_standard_json, out_json):
    assert os.path.exists(in_standard_json), in_standard_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_standard_json)
                
    valid_cnt = 0
    web_flatten_dict = {}

    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        assert "Mesh_obj_raw" in meta
        if "Mesh_obj_pro" in meta:
            if meta["Mesh_obj_pro"] != meta["Mesh_obj_raw"]:
                meta["bak_Mesh_obj_pro"] = meta["Mesh_obj_pro"]
                meta["Mesh_obj_pro"] = meta["Mesh_obj_raw"]
        
        assert meta["Mesh_obj_raw"] == meta["Obj_Mesh"]
        
        meta["dname"] = dname
        web_flatten_dict[oname] = meta
        valid_cnt += 1
       
        
    save_json(web_flatten_dict, out_json)

    print(f"preprocessed_cnt {valid_cnt}  raw_pairs, save to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('--in_standard_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace/20241029_mesh_single_kd_source.json")
    parser.add_argument('--out_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace/web_flatten_gdp.json")
    args = parser.parse_args()
    
    flatten_to_web_json(args.in_standard_json, args.out_json)
