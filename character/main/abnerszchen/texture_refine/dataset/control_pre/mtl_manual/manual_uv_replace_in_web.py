import os
import sys
import json
from tqdm import tqdm
import argparse
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json


        
def replace_in_web(in_web_json, manual_uv_dir, out_web_json):
    assert os.path.exists(in_web_json), in_web_json
    assert os.path.exists(manual_uv_dir), manual_uv_dir
    os.makedirs(os.path.dirname(out_web_json), exist_ok=True)
    
    in_dict = load_json(in_web_json)
    
    manual_objs = glob.glob(os.path.join(manual_uv_dir, "*/manual_*.obj"))
    for manual_obj in manual_objs:
        oname = os.path.basename(os.path.dirname(manual_obj))
        assert oname in in_dict, oname
        meta = in_dict[oname]
        meta["Mesh_obj_backup"] = meta["Mesh_obj_raw"]
        meta["Mesh_obj_raw"] = manual_obj
    
    save_json(in_dict, out_web_json)
    print(f"replace {len(manual_objs)} from {manual_uv_dir} to {out_web_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_web_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json")
    parser.add_argument('manual_uv_dir', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/manual_move_uv")
    parser.add_argument('out_web_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual_fixuv.json")
    args = parser.parse_args()
    
    replace_in_web(args.in_web_json, args.manual_uv_dir, args.out_web_json)
