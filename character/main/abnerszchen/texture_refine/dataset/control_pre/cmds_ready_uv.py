import os
import sys
import glob
import argparse

def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('obj_root_dir', type=str)
    args = parser.parse_args()

    obj_root_dir = args.obj_root_dir
    assert os.path.exists(obj_root_dir)
    
    objs = glob.glob(os.path.join(obj_root_dir, "*/obj_*/*.obj"))    
    
    cmds = []
    for obj in objs:
        type_dir, name = os.path.split(obj)
        out_dir = os.path.join(type_dir, 'uv_condition')
    
        cmd = f"python /aigc_cfs_2/sz/proj/tex_cq/dataset/control_pre/generate_uv_conditions.py {obj} {out_dir}\n"
        cmds.append(cmd)
    
    cmds_txt = os.path.join(obj_root_dir, 'generate_uv_cmds.txt')
    save_lines(cmds, cmds_txt)
    print(f'save to cmds_txt {cmds_txt}')
    
    
if __name__ == "__main__":
    main()

