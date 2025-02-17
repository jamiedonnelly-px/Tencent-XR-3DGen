import os
import sys
import argparse

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_condi_json', type=str)
    parser.add_argument('obj_root_dir', type=str)
    args = parser.parse_args()

    in_condi_json = args.in_condi_json
    obj_root_dir = args.obj_root_dir
    assert os.path.exists(in_condi_json)
    assert os.path.exists(obj_root_dir)

    objs_dict, key_pair_list = parse_objs_json(in_condi_json)

    valid_cnt, failed_lines, failed_uv_dirs = 0, [], []
    data_dict = {}
    for d_, dname, oname in key_pair_list:
        mesh_raw = objs_dict[d_][dname][oname]['Mesh']
        uv_dir = os.path.join(obj_root_dir, dname, oname, 'uv_condition')
        uv_normal = os.path.join(uv_dir, 'uv_normal.png')
        uv_kd = os.path.join(uv_dir, 'texture_kd.png')
        
        if not os.path.exists(uv_normal) or not os.path.exists(uv_kd):
            failed_lines.append(oname)
            failed_uv_dirs.append(uv_dir)
            continue
        if not dname in data_dict:
            data_dict[dname] = {}
        data_dict[dname][oname] = {'uv_kd': uv_kd, 'uv_normal':uv_normal, 'Mesh': mesh_raw, 'caption':[""]}
        valid_cnt += 1

  
    out_json = os.path.join(obj_root_dir, 'valid.json')
    failed_onames_txt = os.path.join(obj_root_dir, 'failed_onames.txt')
    save_json({'data': data_dict}, out_json)
    save_lines(failed_lines, failed_onames_txt)
    print(f'save {valid_cnt} to out_json {out_json}, and {len(failed_lines)} failed cmds failed_cmds_txt to {failed_onames_txt}')


if __name__ == "__main__":
    main()
