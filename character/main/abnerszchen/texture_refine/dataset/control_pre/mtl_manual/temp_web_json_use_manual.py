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


def process_meta(oname, meta, raw_dict):
    if oname not in raw_dict:
        return 0, oname

    raw_meta = raw_dict[oname]
    Mesh_obj_raw = meta["Mesh_obj_raw"]
    # replace
    if os.path.exists(Mesh_obj_raw):
        raw_meta["Mesh_obj_raw"] = Mesh_obj_raw
    else:
        print(f'[ERROR] find invalid oname={oname}')
        return 0, oname

    if "append_type" in meta:
        raw_meta["append_type"] = meta["append_type"]

    return 1, oname


def web_use_manual(in_web_raw_json, in_web_manual_json, out_json):
    assert os.path.exists(in_web_raw_json), in_web_raw_json
    assert os.path.exists(in_web_manual_json), in_web_manual_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    raw_dict = load_json(in_web_raw_json)
    manual_dict = load_json(in_web_manual_json)

    thread_count = 10
    valid_cnt = 0
    invalid_onames = []
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = {
            executor.submit(process_meta, oname, meta, raw_dict): oname
            for oname, meta in manual_dict.items()
        }
        pbar = tqdm(total=len(futures), desc="Processing", ncols=70)
        for future in as_completed(futures):
            flag, oname = future.result()
            valid_cnt += flag
            if not flag:
                invalid_onames.append(oname)
            pbar.update()
        pbar.close()

    save_json(raw_dict, out_json)
    save_lines(invalid_onames, out_json.replace(".json", "_invalids.txt"))

    print(f"preprocessed_cnt {valid_cnt} / {len(manual_dict)} , save to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_web_raw_json',
                        type=str,
                        default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp.json")
    parser.add_argument('in_web_manual_json',
                        type=str,
                        default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_0507.json")
    parser.add_argument('out_json',
                        type=str,
                        default="/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json")
    args = parser.parse_args()

    web_use_manual(args.in_web_raw_json, args.in_web_manual_json, args.out_json)
