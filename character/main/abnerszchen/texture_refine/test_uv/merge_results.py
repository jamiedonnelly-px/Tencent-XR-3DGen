import argparse
import time
import json
import os
import random
from PIL import Image
from diffusers.utils import load_image
import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import parse_objs_json, load_json, concatenate_images_2d, save_json, load_rgba_as_rgb


def run_merge_results(in_jsons_txt, out_dir):
    assert os.path.exists(in_jsons_txt), in_jsons_txt
    in_jsons = [line.strip() for line in open(in_jsons_txt, "r").readlines()]
    os.makedirs(out_dir, exist_ok=True)

    objs_dict_list = []
    infer_keys = []
    for in_json in in_jsons:
        objs_dict, key_pair_list = parse_objs_json(in_json)
        d_, dname, oname = key_pair_list[0]
        meta = objs_dict[d_][dname][oname]
        keys = list(meta.keys())
        result = [s for s in keys if s.startswith("infer_") and not s.endswith("_in")]
        objs_dict_list.append(objs_dict)
        infer_keys.append(result[0])
        
    
    random.seed(1234)
    random.shuffle(key_pair_list)

    print(f"need run{len(key_pair_list)} of len json={len(in_jsons)}")
    print('infer_keys ', infer_keys)
    batch_size = 10
    cnt = 0
    img_pairs = []
    vis_size = 256
    
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]

        in_render_path = meta["condi_imgs_train"][0]
        img_pair = [load_rgba_as_rgb(in_render_path, res=vis_size)]
        for one_dict, infer_key in zip(objs_dict_list, infer_keys):
            path = one_dict[d_][dname][oname][infer_key]
            img_pair.append(load_image(path).resize((vis_size, vis_size)))
        img_pairs.append(img_pair)
        cnt += 1
    
    img_pairs_list = [img_pairs[i:i + batch_size] for i in range(0, len(img_pairs), batch_size)]
    print('img_pairs_list ', len(img_pairs_list))
    transposed_list = [list(i) for i in zip(*img_pairs_list)]
    
    for i, pil_2d in enumerate(img_pairs_list):
        transposed_list = [list(i) for i in zip(*pil_2d)]
        concatenate_images_2d(transposed_list, os.path.join(out_dir, f"merge_{i:03d}.png"))
    
    print(f"batch infer {cnt}/{len(key_pair_list)} done, save to {out_dir}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_jsons_txt', type=str, default='/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test/merge_jsons.txt', help='select model. can be uv_mcwy, control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly')
    parser.add_argument('out_dir', type=str, default='/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test/vis', help='select model. can be uv_mcwy, control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly')
    args = parser.parse_args()

    run_merge_results(args.in_jsons_txt, args.out_dir)
