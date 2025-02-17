import argparse
import json
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def copy_file(src, dst, skip_exists=False):
    # if skip_exists and os.path.exists(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

def replace_dir_in_dict(data, src_root, new_root):
    for item in data:
        for key in ["image", "conditioning_image"]:
            item[key] = item[key].replace(src_root, new_root)    
    return data

def cp_uv_dataset(input_json, src_root, new_root):
    assert os.path.exists(input_json), input_json
    with open(input_json, "r") as f:
        data = json.load(f)
    
    max_threads = os.cpu_count() - 6
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []
        for item in tqdm(data, desc="Copying files"):
            for key in ["image", "conditioning_image"]:
                src_path = item[key]
                dst_path = src_path.replace(src_root, new_root)
                futures.append(executor.submit(copy_file, src_path, dst_path))

        for future in tqdm(futures, desc="Waiting for copy tasks to finish"):
            future.result()
                    

    data = replace_dir_in_dict(data, src_root, new_root)

    output_json = os.path.join(new_root, os.path.basename(input_json))
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"copy from {input_json} to {new_root}, with json {output_json}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files and update JSON data.")
    parser.add_argument("input_json", type=str, help="Input JSON file.", default="/aigc_cfs_3/layer_tex/uv_datasets/ready_right_3class/all.json")
    parser.add_argument("src_root", type=str, help="Source root directory.")
    parser.add_argument("new_root", type=str,help="New root directory.")

    args = parser.parse_args()

    input_json = args.input_json
    src_root = args.src_root
    new_root = args.new_root
        
    cp_uv_dataset(input_json, src_root, new_root)