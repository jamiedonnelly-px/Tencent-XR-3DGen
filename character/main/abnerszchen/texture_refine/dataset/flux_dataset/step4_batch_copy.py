import argparse
import json
import os
import shutil
import random
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_file(src, dst, skip_exists=True):
    if skip_exists and os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


def cp_one_item(meta, imgs_root):
    oname = meta["oname"]
    out_meta_dir = os.path.join(imgs_root, oname)
    try:
        for key in ["image", "conditioning_image"]:
            src_path = meta[key]
            _, fname = os.path.split(os.path.abspath(src_path))
            dst_path = os.path.join(out_meta_dir, fname)
            copy_file(src_path, dst_path)
            meta[key] = dst_path
    except:
        traceback.print_exc()
        return False, oname
    return True, oname

def batch_cp_uv_dataset(input_json, new_root):
    assert os.path.exists(input_json), input_json
    with open(input_json, "r") as f:
        data = json.load(f)
    
    max_threads = os.cpu_count() - 6
    count = 0
    failed_onames = []
    imgs_root = os.path.join(new_root, "imgs")
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []
        for meta in tqdm(data, desc="submit Copying files"):
            futures.append(executor.submit(cp_one_item, meta, imgs_root))
            
        for future in as_completed(futures):
            suc_flag, oname = future.result()
            count += int(suc_flag)
            if not suc_flag:
                failed_onames.append(oname)    


    output_json = os.path.join(new_root, "all_cp_done.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"copy from {input_json} to {new_root}, with json {output_json}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files and update JSON data.")
    parser.add_argument("--input_json", type=str, default="/aigc_cfs/sz/dataset/flux_layer1010/cp/all_cp_done.json")
    parser.add_argument("--new_root", type=str, default="/data1/sz/dataset/flux_layer1010/")
    # parser.add_argument("--input_json", type=str, default="/aigc_cfs/sz/dataset/flux_layer1010/all.json")
    # parser.add_argument("--new_root", type=str, default="/aigc_cfs/sz/dataset/flux_layer1010/cp")

    args = parser.parse_args()

    input_json = args.input_json
    new_root = args.new_root
        
    batch_cp_uv_dataset(input_json, new_root)