import os
import sys
import json
from tqdm import tqdm
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json


        
def web_json_to_standard(in_web_json, out_json):
    assert os.path.exists(in_web_json), in_web_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    in_dict = load_json(in_web_json)
    
    standard_dict = {"data": {}}
    for oname, meta in in_dict.items():
        dname = meta["dname"]
        if dname not in standard_dict["data"]:
            standard_dict["data"][dname] = {}
        standard_dict["data"][dname][oname] = meta

    save_json(standard_dict, out_json)

    print(f"save to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_web_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/gdp_manual_standard.json")
    args = parser.parse_args()
    
    web_json_to_standard(args.in_web_json, args.out_json)
