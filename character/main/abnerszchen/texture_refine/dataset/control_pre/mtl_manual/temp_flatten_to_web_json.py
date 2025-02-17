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

def standard_to_flatten():
    
    return

def process_key_pair(key_pair_chunk, objs_dict):
    web_flatten_dict = {}
    valid_cnt = 0
    invalid_pairs = []

    for d_, dname, oname in key_pair_chunk:
        meta = objs_dict[d_][dname][oname]
        if "Mesh_obj_raw" in meta:
            my_obj = meta["Mesh_obj_raw"]
        else:
            meta["Mesh_obj_raw"] = meta["Obj_Mesh"]

        if os.path.exists(meta["Mesh_obj_raw"]):
            meta["dname"] = dname
            web_flatten_dict[oname] = meta
            valid_cnt += 1
        else:
            print('invalid', invalid_pair)
            invalid_pair = (d_, dname, oname)
            invalid_pairs.append(invalid_pair)

    return web_flatten_dict, valid_cnt, invalid_pairs



        
def flatten_to_web_json(in_standard_json, out_json):
    assert os.path.exists(in_standard_json), in_standard_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_standard_json)
        
    thread_count = 10
    chunk_size = len(key_pair_list) // thread_count
    key_pair_chunks = [key_pair_list[i * chunk_size:(i + 1) * chunk_size] for i in range(thread_count)]

                
    valid_cnt = 0
    invalid_pairs = []
    web_flatten_dict = {}

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = list(tqdm(executor.map(process_key_pair, key_pair_chunks, [objs_dict] * thread_count), total=thread_count))

    for result in results:
        web_flatten_dict.update(result[0])
        valid_cnt += result[1]
        invalid_pairs.extend(result[2])
        
    save_json(web_flatten_dict, out_json)

    print(f"preprocessed_cnt {valid_cnt} and {len(invalid_pairs)} raw_pairs, save to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_standard_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/20240711_ruku_ok_gdp_tex_need.json")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp.json")
    args = parser.parse_args()
    
    flatten_to_web_json(args.in_standard_json, args.out_json)
