import os
import sys
import argparse

def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_glb_list', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--blender_root', type=str, default='/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender')
    args = parser.parse_args()

    in_glb_list = args.in_glb_list
    out_dir = args.out_dir
    blender_root = args.blender_root
    assert os.path.exists(in_glb_list)
    assert os.path.exists(blender_root)
    os.makedirs(out_dir, exist_ok=True)
    
    lines = [line.strip() for line in open(in_glb_list, "r").readlines()]
    glbs = [line for line in lines if os.path.exists(line)]
    
    cmds = []
    for glb in glbs:
        type_dir, name = os.path.split(glb)
        assert_type = os.path.basename(type_dir)
        name_id = os.path.splitext(name)[0]
        out_obj_path = os.path.join(out_dir, assert_type, f'obj_{name_id}/mesh.obj')
        cmd = f"{blender_root} -b -P /aigc_cfs_2/sz/proj/tex_cq/dataset/control_pre/glb_to_obj.py -- --mesh_path {glb} --output_obj_path {out_obj_path}\n"
        cmds.append(cmd)
    
    cmds_txt = os.path.join(out_dir, 'glb_to_obj_cmds.txt')
    save_lines(cmds, cmds_txt)
    print(f'save to cmds_txt {cmds_txt}')
    
    
if __name__ == "__main__":
    main()

