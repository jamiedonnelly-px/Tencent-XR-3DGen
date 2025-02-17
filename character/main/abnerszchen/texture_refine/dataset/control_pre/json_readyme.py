import os
import sys
import json
import glob
import argparse
def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)

def save_json(json_data, out_file):
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('obj_root_dir', type=str)
    args = parser.parse_args()

    obj_root_dir = args.obj_root_dir
    assert os.path.exists(obj_root_dir)

    blender_cmds_txt = os.path.join(obj_root_dir, "glb_to_obj_cmds.txt")
    uv_cmds_txt = os.path.join(obj_root_dir, "generate_uv_cmds.txt")
    assert os.path.exists(blender_cmds_txt)
    assert os.path.exists(uv_cmds_txt)

    valid_cnt, failed_lines, failed_uv_dirs = 0, [], []
    dataset_dict = {}
    for line in open(blender_cmds_txt, "r").readlines():
        data = line.strip().split(' ')
        obj_dir, glb_path = data[-1], data[-3]
        uv_dir = os.path.join(obj_dir, 'uv_condition')
        uv_normal = os.path.join(uv_dir, 'uv_normal.png')
        uv_kd = os.path.join(uv_dir, 'texture_kd.png')
        if not os.path.exists(obj_dir) or not os.path.exists(glb_path) or not os.path.exists(uv_normal) or not os.path.exists(uv_kd):
            failed_lines.append(line)
            failed_uv_dirs.append(uv_dir)
            continue

        oname = "_".join(obj_dir.split('/')[-2:])
        dataset_dict[oname] = {'uv_kd': uv_kd, 'uv_normal':uv_normal, 'Mesh': glb_path, 'caption':[""]}
        valid_cnt += 1

    uv_cmds = [line.strip() for line in open(uv_cmds_txt, "r").readlines()]
    failed_uv_lines = [uv_cmd for uv_cmd in uv_cmds if uv_cmd.split(' ')[-1] in failed_uv_dirs]
    
    out_data_dict = {}
    out_data_dict['readplayerMe'] = dataset_dict
    out_json = os.path.join(obj_root_dir, 'valid.json')
    failed_cmds_txt = os.path.join(obj_root_dir, 'failed_blender_cmds.txt')
    failed_uv_cmds_txt = os.path.join(obj_root_dir, 'failed_uv_cmds.txt')
    save_json({'data': out_data_dict}, out_json)
    save_lines(failed_lines, failed_cmds_txt)
    save_lines(failed_uv_lines, failed_uv_cmds_txt)
    print(f'save {valid_cnt} to out_json {out_json}, and {len(failed_lines)} failed cmds failed_cmds_txt to {failed_cmds_txt} / {failed_uv_cmds_txt}')


if __name__ == "__main__":
    main()
