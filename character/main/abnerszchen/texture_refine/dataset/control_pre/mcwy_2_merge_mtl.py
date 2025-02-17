import os
import json
import argparse
import sys
import subprocess
import multiprocessing
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
from dataset.control_pre.filter_merge_mat import batch_filter_mtl_type

def merge_mtl(blender_root, mtl_merge_py, meta : dict, only_right=False):
    Mesh_obj_raw = meta["Mesh_obj_raw"]
    if not Mesh_obj_raw:
        return None
    if meta.get('mtl_type', None) == 'right':
        return Mesh_obj_raw if os.path.exists(Mesh_obj_raw) else None
    
    if only_right:
        return None
    out_dir = os.path.join(os.path.dirname(Mesh_obj_raw), "merge_mtl")
    cmd = f"{blender_root} -b -P  {mtl_merge_py} -- --input_mesh_path {Mesh_obj_raw} --output_mesh_folder {out_dir}\n"
    result = subprocess.run(cmd, shell=True)
    out_obj = os.path.join(out_dir, "combined.obj")
    return out_obj if os.path.exists(out_obj) else None

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/source.json")
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--blender_root', type=str, default='/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender')
    parser.add_argument('--addon_path', type=str, default="/aigc_cfs/sz/software/material-combiner-addon-master.zip",
                        help='path of addon zip file to be installed')    
    args = parser.parse_args()
    

    in_json = args.in_json
    out_dir = args.out_dir
    blender_root = args.blender_root
    addon_path = args.addon_path
    assert os.path.exists(in_json), in_json
    assert os.path.exists(blender_root), blender_root
    assert os.path.exists(addon_path), addon_path
    os.makedirs(out_dir, exist_ok=True)
    
    addon_py = os.path.join(os.path.dirname(current_script_path), "blender_addon_install.py")
    assert os.path.exists(addon_py), addon_py
    result = subprocess.run(f"{blender_root} -b -P {addon_py} -- --addon_path  {addon_path}", shell=True, check=True)
    assert result.returncode == 0
    print('1. install blender addon done.')

    filter_mtl_json = os.path.join(out_dir, 'filter_mtl.json')
    filter_dict = batch_filter_mtl_type(in_json, filter_mtl_json)
    assert os.path.exists(filter_mtl_json), filter_mtl_json
    print('2. filter mtl done.')
    
    objs_dict, key_pair_list = parse_objs_json(in_json)

    mtl_merge_py = os.path.join(os.path.dirname(current_script_path), "blender_material_merge.py")
    assert os.path.exists(mtl_merge_py), mtl_merge_py
    tasks = []
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        meta['mtl_type'] = None
        for mtl_type, type_dict in filter_dict.items():
            if oname in type_dict:
                meta['mtl_type'] = mtl_type
        tasks.append([blender_root, mtl_merge_py, meta])

    ## batch run merge mtl
    with multiprocessing.Pool() as pool:
        results = pool.starmap(merge_mtl, tasks)
    
    valid_cnt = 0
    invalid_pairs = []
    for out_obj, key_pair in zip(results, key_pair_list):
        d_, dname, oname = key_pair
        if out_obj is not None:
            valid_cnt += 1
            objs_dict[d_][dname][oname]["Mesh_obj_pro"] = out_obj
        else:
            invalid_pairs.append(",".join(key_pair))
            objs_dict[d_][dname][oname]["Mesh_obj_pro"] = None
        
    
    print(f'valid_cnt, all_cnt {valid_cnt}/{len(results)}')
    
    save_lines(invalid_pairs, os.path.join(out_dir, 'failed_merge_mtl.txt'))
    merge_mtl_done_json = os.path.join(out_dir, 'merge_mtl_done.json')
    save_json(objs_dict, merge_mtl_done_json)
    print(f'save merge_mtl_done_json {merge_mtl_done_json}')
    print('3. merge mtl done.')
    
if __name__ == "__main__":
    main()

