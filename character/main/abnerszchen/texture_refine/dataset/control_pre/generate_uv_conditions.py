import os
import sys
import argparse

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.uv_conditions import render_uv_condition


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_obj_path', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--res', type=int, default=1024)
    parser.add_argument("--keep_raw", action="store_true", help="not normalized mesh if True")
    args = parser.parse_args()

    print('in_obj_path ', args.in_obj_path, args.keep_raw)
    render_uv_condition(args.in_obj_path, args.res, args.out_dir, use_normalized=not args.keep_raw)
    
if __name__ == "__main__":
    main()

