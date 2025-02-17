import os
import sys
import argparse

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.uv_conditions import obj_xatlas


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='uv xatlas')
    parser.add_argument('in_obj_path', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()

    obj_xatlas(args.in_obj_path, args.out_dir)
    
if __name__ == "__main__":
    main()

