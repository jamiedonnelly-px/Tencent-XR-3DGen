import os
import json
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set generate_uv_done.json')
    parser.add_argument('--in_new_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/s1_merge_flatten_gdp_new_need_check.json")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925")
    args = parser.parse_args()

    step_2_py = os.path.join(project_root, "dataset/control_pre/mcwy_2_merge_mtl.py")
    step_3_py = os.path.join(project_root, "dataset/control_pre/mcwy_3_generate_uv.py")
    in_new_json = args.in_new_json
    out_dir = args.out_dir
    assert os.path.exists(step_2_py), step_2_py
    assert os.path.exists(step_3_py), step_3_py
    assert os.path.exists(in_new_json), in_new_json
    
    cmd = f"python {step_2_py} {in_new_json} {out_dir}"
    os.system(cmd)
    
    merge_mtl_done_json = os.path.join(out_dir, "merge_mtl_done.json")
    if not os.path.exists(merge_mtl_done_json):
        print(f'error, step2 failed, not find merge_mtl_done_json={merge_mtl_done_json}')
        exit()
        
    cmd = f"python {step_3_py} {merge_mtl_done_json} {out_dir}"
    os.system(cmd)

    generate_uv_done_json = os.path.join(out_dir, "generate_uv_done.json")
    if not os.path.exists(generate_uv_done_json):
        print(f'error, step3 failed, not find generate_uv_done_json={generate_uv_done_json}')
        exit()
            
# python mcwy_3_generate_uv.py ${mcwy_out_dir}/merge_mtl_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_3_uv.txt
# python mcwy_4_find_image.py ${mcwy_out_dir}/generate_uv_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_4_find_image.txt

if __name__ == "__main__":
    main()

