import os
import shutil
import argparse
import sys
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, save_lines, split_pod_json, split_list_avg
from scripts.utils_pool_cmds import run_commands_in_parallel

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/find_image_done.json")
    parser.add_argument('out_dir', type=str)
    parser.add_argument(
        "--blip_model", type=str, default="/aigc_cfs/model/instructblip-flan-t5-xl"
    )    
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs')
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    blip_model = args.blip_model
    num_gpus = args.num_gpus
    if num_gpus < 0:
        num_gpus = torch.cuda.device_count()
    print(f"use num_gpus {num_gpus}")
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)

    run_py = os.path.join(os.path.dirname(current_script_path), "mcwy_5_image_caption.py")
    assert os.path.exists(run_py), run_py
        
    objs_dict, key_pair_list = parse_objs_json(in_json)
    key_pair_lists = split_list_avg(key_pair_list, num_gpus)
    cmds, split_out_jsons = [], []
    for i, tasks_pair_pod in enumerate(key_pair_lists):
        split_json = os.path.join(out_dir, f'split_pod_g{i}.json')
        
        split_pod_json(in_json, tasks_pair_pod, split_json)
        print('split_json ', split_json, len(tasks_pair_pod))
        
        out_json_name = f"image_caption_g{i}.json"
        cmd = f"CUDA_VISIBLE_DEVICES={i} python {run_py} {split_json} {out_dir} --blip_model {blip_model} --out_json_name {out_json_name}"
        cmds.append(cmd)
        split_out_jsons.append(os.path.join(out_dir, out_json_name))
    
    cmds_txt = os.path.join(out_dir, 'multi_gpus_caption_cmds.txt')
    save_lines(cmds, cmds_txt)
    
    print(f'save {len(cmds)} cmds to cmds_txt {cmds_txt}, begin run..')
    
    run_commands_in_parallel(cmds, pool_count=num_gpus)
    
    merge_dict = {"data":{}}
    for split_out_json in split_out_jsons:
        if not os.path.exists(split_out_json):
            print(f'ERROR can not find split_out_json {split_out_json}')
            continue
        
        split_objs_dict, split_key_pair_list = parse_objs_json(split_out_json)
        for d_, dname, oname in split_key_pair_list:
            if dname not in merge_dict[d_]:
                merge_dict[d_][dname] = {}
            merge_dict[d_][dname][oname] = split_objs_dict[d_][dname][oname]
    
    merge_json = os.path.join(out_dir, "image_caption_done.json")
    save_json(merge_dict, merge_json)
    print(f"save done to {merge_json}")
    
    # check and merge
    # out_dict = os.path.join(out_dir, 'image_caption_done.json')
    # save_json(objs_dict, out_dict)
    # print(f'image_caption done {len(key_pair_list)}, save to {out_dict}')
        
if __name__ == "__main__":
    main()

