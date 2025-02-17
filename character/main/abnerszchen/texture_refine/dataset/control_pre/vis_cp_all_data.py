import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_rgba_as_rgb, concatenate_images_horizontally

def cat_data(uv_normal, uv_kd, render, out_image, res=512):
    try:
        vis = [load_rgba_as_rgb(uv_normal, res), load_rgba_as_rgb(uv_kd, res), load_rgba_as_rgb(render, res)]
        concatenate_images_horizontally(vis, out_img_path=out_image)
    except:
        print('faild ', uv_normal, uv_kd, out_image)
        return False
    return True

def cat_data_threaded(normalkd_dest_pairs, max_workers=96):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        suc_cnt = 0
        results = list(tqdm(executor.map(lambda pair: cat_data(pair[0], pair[1], pair[2], pair[3]), normalkd_dest_pairs), total=len(normalkd_dest_pairs)))
        for result in results:
            if result:
                suc_cnt += 1
    
    return results   

def cp_all_data(in_json, out_dir):
    assert os.path.exists(in_json)
    os.makedirs(out_dir, exist_ok=True)
    
    raw_dict, key_pair_list = parse_objs_json(in_json)
    pair_list = []
    for d_, dname, oname in key_pair_list:
        meta = raw_dict[d_][dname][oname]
        uv_normal = meta['uv_normal']
        uv_kd = meta['uv_kd']
        render = meta['condi_imgs_in'][0]
        
        short_dname = dname.split('_')[-1]
        out_image = os.path.join(out_dir, f'{short_dname}/{oname}.png')
        pair_list.append((uv_normal, uv_kd, render, out_image))
    
    cat_data_threaded(pair_list)
    print(f'cp {len(pair_list)} to {out_dir}')
    return


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='cp all data to one dir for vis')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    cp_all_data(in_json, out_dir)
    


if __name__ == "__main__":
    main()
