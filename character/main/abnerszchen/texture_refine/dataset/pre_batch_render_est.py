import os
import argparse
import json
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
import shutil

from utils_dataset import load_json, read_lines, save_lines
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'render'))
# from render.render_obj import render_obj_texture

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(os.path.join(project_root, "render"))
from render_obj import render_obj_texture

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dst_item = os.path.join(dst_folder, item)

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)
            
def render_task(est_obj, out_dir, in_pose_json, render_res, max_mip_level):
    dname, o_name = est_obj.split('/')[-4:-2]
    render_out_dir = os.path.join(out_dir, f'render_est/{dname}/{o_name}')
    render_obj_texture(est_obj, in_pose_json, render_out_dir, render_res=render_res, max_mip_level=max_mip_level)
    
    copy_folder(os.path.dirname(os.path.dirname(est_obj)), os.path.join(render_out_dir, 'est'))
    
    return est_obj

def batch_render_est(in_est_objs_txt, in_pose_json, out_dir, pool_cnt=8, render_res=1024, max_mip_level=None, save_res=512):
    
    # dname/oname
    est_objs = read_lines(in_est_objs_txt)
    if not est_objs or len(est_objs) < 1:
        print('can not fin any obj in ', in_est_objs_txt)
        return

    os.makedirs(out_dir, exist_ok=True)

    set_start_method('spawn', force=True)
    num_cores = pool_cnt or cpu_count()
    with Pool(num_cores) as pool:
        async_results = [pool.apply_async(render_task, (est_obj, out_dir, in_pose_json, render_res, max_mip_level)) for est_obj in est_objs]
        done_list = [r.get() for r in async_results]
                
    print(f'render done to {out_dir}')
  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list (batch run render_obj_texture)')
    parser.add_argument('in_est_objs_txt', type=str)
    parser.add_argument('in_pose_json', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--pool_cnt', type=int, default=8)
    args = parser.parse_args()

    # Run.
    batch_render_est(args.in_est_objs_txt, args.in_pose_json, args.out_dir, pool_cnt=args.pool_cnt)
    return

if __name__ == "__main__":
    main()
