from PIL import Image
import argparse

import os
import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import parse_objs_json, concatenate_images_vertically, load_rgba_as_rgb


def parse_paths(in_json, select_key = "infer_uv_sdxl"):
    objs_dict, key_pair_list = parse_objs_json(in_json)
    image_paths = []
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        if select_key not in meta:
            continue
        image_paths.append(meta[select_key])
    return image_paths

def vis_merge(image_paths, batch_size, out_dir):
    img_pairs_list = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    print(f"will save {len(image_paths)}/{batch_size} = {len(img_pairs_list)}")
    
    for i, img_pairs in enumerate(img_pairs_list):
        pils = [load_rgba_as_rgb(img_path) for img_path in img_pairs]
        concatenate_images_vertically(pils, os.path.join(out_dir, f"vis_{i:03d}.png"))
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json',
                        type=str,
                        default='/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test/my/output.json')
    parser.add_argument('out_dir',
                        type=str,
                        help='')
    parser.add_argument('--select_key',
                        type=str, default="infer_uv_sdxl",
                        help='')
    parser.add_argument('--batch_size',
                        type=int, default=6,
                        help='')
    args = parser.parse_args()

    image_paths = parse_paths(args.in_json, select_key = args.select_key)
    vis_merge(image_paths, args.batch_size, args.out_dir)
    print(f"save imgs from {args.in_json} to {args.out_dir}")
