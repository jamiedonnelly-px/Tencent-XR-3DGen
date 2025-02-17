import os
import json
import argparse

def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)
def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)    
    return data
def save_json(json_data, out_file):
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return
#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--blender_root', type=str, default='/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender')
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    blender_root = args.blender_root
    assert os.path.exists(in_json)
    assert os.path.exists(blender_root)
    os.makedirs(out_dir, exist_ok=True)
    
    in_dict = load_json(in_json)['data']
    # need_keys = ['MCWY_2_Top', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe']
    # select_keys = [key for key in list(in_dict.keys()) if key in need_keys]
    
    select_keys = list(in_dict.keys())
    
    cmds = []
    cvt_dict = {}
    for select_key in select_keys:
        for oname, meta in in_dict[select_key].items():
            fbx_path = meta['Mesh']
            if not os.path.exists(fbx_path):
                continue
            output_fullpath = os.path.join(out_dir, select_key, oname, 'mesh.obj')
            cmd = f"{blender_root} -b -P /aigc_cfs_2/sz/proj/tex_cq/dataset/control_pre/fbx_to_obj.py -- --mesh_path '{fbx_path}' --output_fullpath '{output_fullpath}' --copy_texture\n"
            cmds.append(cmd)
            
            cvt_dict[fbx_path] = output_fullpath
    
    cmds_txt = os.path.join(out_dir, 'fbx_to_obj_cmds.txt')
    cvt_json = os.path.join(out_dir, 'need_fbx_obj_map.json')
    save_lines(cmds, cmds_txt)
    save_json(cvt_dict, cvt_json)
    print(f'save {len(cmds)} cmds to cmds_txt {cmds_txt}, cvt_json {cvt_json}')
    
    
if __name__ == "__main__":
    main()

