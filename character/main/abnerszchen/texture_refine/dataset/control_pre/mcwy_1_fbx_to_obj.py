import os
import json
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
from scripts.utils_pool_cmds import run_commands_in_parallel

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs/Asset/designcenter/clothes/mcwy_data.json")
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--blender_root', type=str, default='/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender')
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    blender_root = args.blender_root
    assert os.path.exists(in_json), in_json
    assert os.path.exists(blender_root), blender_root
    os.makedirs(out_dir, exist_ok=True)
    
    py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fbx_to_obj.py")
    assert os.path.exists(py_path), py_path
    
    objs_dict, key_pair_list = parse_objs_json(in_json)
    in_dict = objs_dict['data']
    
    # TODO
    need_keys = ['MCWY_2_Top', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe']
    select_keys = [key for key in list(in_dict.keys()) if key in need_keys]
    # select_keys = list(in_dict.keys())
    
    cmds = []
    cvt_dict = {}
    cmds_dict = {}
    run_cnt, all_cnt = 0, 0
    for select_key in select_keys:
        for oname, meta in in_dict[select_key].items():
            all_cnt += 1
            fbx_path = meta['Mesh']
            if not os.path.exists(fbx_path):
                print(f'warn: can not find fbx_path {fbx_path}')
                continue
            run_cnt += 1
            
            output_fullpath = os.path.join(out_dir, select_key, oname, 'mesh.obj')
            cmd = f"{blender_root} -b -P {py_path} -- --mesh_path '{fbx_path}' --output_fullpath '{output_fullpath}' --copy_texture\n"
            cmds.append(cmd)
            
            cmds_dict[oname] = cmd
            cvt_dict[fbx_path] = output_fullpath
            meta['Mesh_obj_raw'] = output_fullpath
    
    print(f'run_cnt, all_cnt {run_cnt}/{all_cnt}')
    cmds_txt = os.path.join(out_dir, 'fbx_to_obj_cmds.txt')
    cvt_json = os.path.join(out_dir, 'need_fbx_obj_map.json')
    save_lines(cmds, cmds_txt)
    save_json(cvt_dict, cvt_json)
    print(f'save {len(cmds)} cmds to cmds_txt {cmds_txt}, cvt_json {cvt_json}')
    
    run_commands_in_parallel(cmds, pool_count=12)
    print('run_commands_in_parallel done')
    
    new_dict = {"data":{}}
    failed_cmds = []
    for select_key in select_keys:
        for oname, meta in in_dict[select_key].items():
            Mesh_obj_raw = meta['Mesh_obj_raw']
            if not os.path.exists(Mesh_obj_raw):
                meta['Mesh_obj_raw'] = None
                failed_cmds.append(cmds_dict[oname])
            
            if select_key not in new_dict['data']:
                new_dict['data'][select_key] = {}
            new_dict['data'][select_key][oname] = meta
    
    failed_cmds_txt = os.path.join(out_dir, 'failed_fbx_to_obj_cmds.txt')
    cvt_obj_done_json = os.path.join(out_dir, 'cvt_obj_done.json')
    save_lines(failed_cmds, failed_cmds_txt)
    save_json(new_dict, cvt_obj_done_json)
    print(f'save {len(failed_cmds)} cmds to failed_cmds_txt {failed_cmds_txt}, cvt_obj_done_json {cvt_obj_done_json}')
        
if __name__ == "__main__":
    main()

