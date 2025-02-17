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


        
def uv_pack(in_web_json, in_uv_failed_json, out_dir):
    assert os.path.exists(in_web_json), in_web_json
    assert os.path.exists(in_uv_failed_json), in_uv_failed_json
    os.makedirs(out_dir, exist_ok=True)
    
    in_dict = load_json(in_web_json)
    failed_dict = load_json(in_uv_failed_json)
    
    for oname, cmd in failed_dict.items():
        meta = in_dict[oname]
        Mesh_obj_raw = meta["Mesh_obj_raw"]
        cmd_cp = f"cp -r {os.path.dirname(Mesh_obj_raw)} {os.path.join(out_dir, oname)}"
        os.system(cmd_cp)
        
    print(f"pack {len(failed_dict)} to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_web_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json")
    parser.add_argument('in_uv_failed_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv/generate_uv_failed_cmds.json")
    parser.add_argument('out_dir', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/bad_uv_pack")
    args = parser.parse_args()
    
    uv_pack(args.in_web_json, args.in_uv_failed_json, args.out_dir)
