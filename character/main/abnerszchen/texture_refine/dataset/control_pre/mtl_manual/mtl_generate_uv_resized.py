import os
import logging
import argparse
import sys
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines, concatenate_images_horizontally
from scripts.utils_pool_cmds import run_commands_in_parallel
from diffusers.utils import load_image

def get_magic_name(obj, key="___"):
    obj_dir, obj_name = os.path.split(obj)
    obj_dir_dir, oname = os.path.split(obj_dir)
    dname = os.path.basename(obj_dir_dir)
    
    new_dir_name = key.join([dname, oname])
    return new_dir_name

def check_uv(key_pair, out_dir, objs_dict : dict):
    d_, dname, oname = key_pair
    meta = objs_dict[d_][dname][oname]
    out_uv_dir = os.path.join(out_dir, dname, oname, 'uv_condition')
    uv_kd = os.path.join(out_uv_dir, "texture_kd.png")
    uv_pos = os.path.join(out_uv_dir, "uv_pos.png")
    # uv_normal = os.path.join(out_uv_dir, "uv_normal.png")
    # if os.path.exists(uv_kd) and os.path.exists(uv_pos) and os.path.exists(uv_normal):
    if os.path.exists(uv_kd) and os.path.exists(uv_pos):
    # if os.path.exists(uv_kd):
        meta["uv_kd"] = uv_kd
        # meta["uv_pos"] = uv_pos
        # meta["uv_normal"] = uv_normal
        return d_, dname, oname, 1
    else:
        return d_, dname, oname, 0

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='generate uv obj, keep_raw, not check normal geom uv')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/source_mcwy2_4class.json")
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--gpu_num', type=int, default=-1)
    args = parser.parse_args()
    

    in_json = args.in_json
    out_dir = args.out_dir
    gpu_num = args.gpu_num
    if gpu_num < 0:
        gpu_num = torch.cuda.device_count()
    print(f"gpu_num , {gpu_num}")
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_json)
    uv_py = os.path.join(project_root, "dataset/control_pre/generate_uv_conditions.py")
    assert os.path.exists(uv_py), uv_py
    
    cmds = []
    out_uv_dirs = []
    cnt = 0
    cmds_dict = {}
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        mesh_in = meta["Mesh_obj_raw"]
        out_uv_dir = os.path.join(out_dir, dname, oname, 'uv_condition')
        gpu_id = cnt % gpu_num
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {uv_py} '{mesh_in}' '{out_uv_dir}' --res 1024\n"
        # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {uv_py} '{mesh_in}' '{out_uv_dir}' --res 1024 --keep_raw\n"
        cmds.append(cmd)
        cnt += 1
        out_uv_dirs.append(out_uv_dir)
        cmds_dict[oname] = cmd

    
    print(f'cmds cnt {len(cmds)}')
    cmds_txt = os.path.join(out_dir, 'generate_uv_cmds.txt')
    save_lines(cmds, cmds_txt)
    print(f'save to cmds_txt {cmds_txt}')
    
    # run_commands_in_parallel(cmds, pool_count=2 * gpu_num)
    # print(f'run done')
    
    ## check
    valid_cnt = 0
    failed_cmds = []
    with ThreadPoolExecutor() as executor: 
        results = list(tqdm(executor.map(lambda pair: check_uv(pair, out_dir, objs_dict), key_pair_list), total=len(key_pair_list)))
        for d_, dname, oname, flag in results:
            valid_cnt += flag
            if not flag:
                objs_dict[d_][dname].pop(oname)
                failed_cmds.append(cmds_dict[oname])
    
    save_json(objs_dict, os.path.join(out_dir, 'generate_uv_done.json'))
    save_lines(failed_cmds, os.path.join(out_dir, 'failed_cmds.sh'))
    print(f'check done {valid_cnt}/{len(key_pair_list)}')

    for out_uv_dir in out_uv_dirs:
        texture_kd = os.path.join(out_uv_dir, "texture_kd.png")
        uv_pos = os.path.join(out_uv_dir, "uv_pos.png")

        if not os.path.exists(texture_kd) or not os.path.exists(uv_pos):
            continue
        out_img = os.path.join(out_dir, "vis_uv", get_magic_name(out_uv_dir) + ".jpg")
        concatenate_images_horizontally(
            [
                load_image(texture_kd).resize((512, 512)),
                load_image(uv_pos).resize((512, 512)),
            ],
            out_img,
        )
 
if __name__ == "__main__":
    main()

