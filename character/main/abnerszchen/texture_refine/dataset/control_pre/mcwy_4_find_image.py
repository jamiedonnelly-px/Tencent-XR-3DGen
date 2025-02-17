import os
import shutil
import argparse
import sys
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import (
    parse_objs_json,
    save_json,
    SRENDER_DAO_CONDI_KEYS,
    SRENDER_DAO_CAP_KEYS,
    HUMAN_CONDI_KEYS,
    HUMAN_CAP_KEYS,
)

def find_image(key_pair, out_dir, objs_dict : dict, data_type="mcwy"):
    d_, dname, oname = key_pair
    meta = objs_dict[d_][dname][oname]
    out_image_dir = os.path.join(out_dir, dname, oname, 'render_emission')
    ImgDir = meta["ImgDir"]
    if data_type == "mcwy":
        color_dir = os.path.join(ImgDir, 'emission/color')
        condi_keys, cap_keys = SRENDER_DAO_CONDI_KEYS, SRENDER_DAO_CAP_KEYS
        cp_keys = list(set(condi_keys + cap_keys))
    elif data_type == "human":  # 380 pose
        color_dir = os.path.join(ImgDir, 'emission/color')
        condi_keys, cap_keys = HUMAN_CONDI_KEYS, HUMAN_CAP_KEYS
        cp_keys = list(set(condi_keys + cap_keys))
    else:
        raise ValueError(f"invalid data_type {data_type}")
        
    if not os.path.exists(color_dir):
        print('can not find color_dir ', color_dir)
        return d_, dname, oname, 0
    os.makedirs(out_image_dir, exist_ok=True)
    
    condi_imgs_train, condi_imgs_in = [], []
    for key in cp_keys:
        src_img = os.path.join(color_dir, f"cam-{key}.png")
        if not os.path.exists(src_img):
            print('can not find src_img ', src_img)
            return d_, dname, oname, 0
        
        dst_img = os.path.join(out_image_dir, f"cam-{key}.png")
        shutil.copyfile(src_img, dst_img)
        
        if key in condi_keys:
            condi_imgs_train.append(dst_img)
        if key in cap_keys:
            condi_imgs_in.append(dst_img)
    
    meta["condi_imgs_train"] = condi_imgs_train
    meta["condi_imgs_in"] = condi_imgs_in
    return d_, dname, oname, 1

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/generate_uv_done.json")
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--data_type', type=str, default="mcwy", help="mcwy, human(vroid, designcenter..), ..")
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    data_type = args.data_type
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_json)

    valid_cnt = 0
    with ThreadPoolExecutor() as executor: 
        results = list(tqdm(executor.map(lambda pair: find_image(pair, out_dir, objs_dict, data_type), key_pair_list), total=len(key_pair_list)))
        for d_, dname, oname, flag in results:
            valid_cnt += flag
            if not flag:
                objs_dict[d_][dname].pop(oname)

    out_dict = os.path.join(out_dir, 'find_image_done.json')
    save_json(objs_dict, out_dict)
    print(f'find_image done {valid_cnt}/{len(key_pair_list)}, save to {out_dict}')

if __name__ == "__main__":
    main()
