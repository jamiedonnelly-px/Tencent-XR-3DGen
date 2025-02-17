import os
import argparse
import json
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method


import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(os.path.join(project_root, "dataset"))

from utils_dataset import read_lines, save_json, parse_objs_json, copy_folder

def render_task(task_pair, run_py, render_res, max_mip_level):
    est_obj, in_pose_json, render_out_dir = task_pair
    dest_dir = os.path.join(render_out_dir, 'est')
    render_in_obj = os.path.join(dest_dir, '0000/mesh.obj')
    
    if os.path.exists(render_in_obj) and os.path.exists(os.path.join(render_out_dir, 'depth_cam-0031.png')):
        # skip exists
        print(f'skip {render_in_obj}')
        return (render_in_obj, render_out_dir)
            
    os.system(f"python {run_py} {est_obj} {in_pose_json} {render_out_dir} --lrm --render_res {render_res} ")

    if not os.path.exists(dest_dir):
        copy_folder(os.path.dirname(os.path.dirname(est_obj)), dest_dir)
    
    if os.path.exists(render_in_obj):
        return (render_in_obj, render_out_dir)
    else:
        return None

def match_obj_pose_out_list(est_objs, objs_dict, out_dir):
    task_pairs = []
    for est_obj in est_objs:
        dname, oname = est_obj.split('/')[-4:-2]
        ImgDir = objs_dict['data'][dname][oname]["ImgDir"]
        pose_json = os.path.join(ImgDir, 'cam_parameters.json')
        render_out_dir = os.path.join(out_dir, f'render_est/{dname}/{oname}')
        task_pairs.append((est_obj, pose_json, render_out_dir))
    return task_pairs

def batch_render_est_lrm(in_raw_json, in_est_objs_txt, out_dir, pool_cnt=8, render_res=512, max_mip_level=None, save_res=512):
    assert os.path.exists(in_raw_json)
    run_py = os.path.join(project_root, "run_render_obj.py")
    assert os.path.exists(run_py)
    objs_dict, key_pair_list = parse_objs_json(in_raw_json)
    
    # dname/oname
    est_objs = read_lines(in_est_objs_txt)
    if not est_objs or len(est_objs) < 1:
        print('can not fin any obj in ', in_est_objs_txt)
        return

    os.makedirs(out_dir, exist_ok=True)
    
    task_pairs = match_obj_pose_out_list(est_objs, objs_dict, out_dir)

    print(f'Begin render {len(task_pairs)} task_pairs')
    set_start_method('spawn', force=True)
    num_cores = pool_cnt or cpu_count()
    with Pool(num_cores) as pool:
        async_results = [pool.apply_async(render_task, (task_pair, run_py, render_res, max_mip_level)) for task_pair in task_pairs]
        done_list = [r.get() for r in async_results]
    
    render_done_dict = {}
    for done_pair in done_list:
        if not done_pair:
            continue
        render_in_obj, render_out_dir = done_pair
        render_done_dict[render_in_obj] = render_out_dir
    render_done_json = os.path.join(out_dir, 'render_done.json')
    save_json(render_done_dict, render_done_json)
    print(f'render done to {out_dir}, save dict in {render_done_json}')
  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list (batch run render_obj_texture)')
    parser.add_argument('in_raw_json', type=str, help='need find pose json from raw json')
    parser.add_argument('in_est_objs_txt', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--pool_cnt', type=int, default=16)
    args = parser.parse_args()

    # Run.
    batch_render_est_lrm(args.in_raw_json, args.in_est_objs_txt, args.out_dir, pool_cnt=args.pool_cnt)
    return

if __name__ == "__main__":
    main()
