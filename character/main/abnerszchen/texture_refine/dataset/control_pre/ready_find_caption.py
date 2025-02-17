import os
import sys
import argparse
from tqdm import tqdm

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dataset"))

from dataset.utils_dataset import load_json, save_json, parse_objs_json

CAPTION_KEYS = ['0011', '0048', '0089', '0100', '0132', '0167', '0286']

def find_caption(in_json, in_render_dir, out_json):
    assert os.path.exists(in_json)
    
    raw_dict, key_pair_list = parse_objs_json(in_json)
    for d_, oname, dname in tqdm(key_pair_list):
        body_part, obj_id = dname.split('_obj_')[0], dname.split('_obj_')[-1]
        
        ImgDir = os.path.join(in_render_dir, body_part, f'{obj_id}_output_512_MightyWSB')
        color_dir = os.path.join(ImgDir, 'color')
        color_json_dir = os.path.join(ImgDir, 'color_json')
        
        meta = raw_dict[d_][oname][dname]
        captions = set()
        render_imgs = []
        for key in CAPTION_KEYS:
            # cap_json = os.path.join(color_dir, f'cam-{key}.json')
            cap_json = os.path.join(color_json_dir, f'cam-{key}.json')
            if not os.path.exists(cap_json):
                continue
            cap = load_json(cap_json)['caption']
            if cap and len(cap) > 3:
                captions.add(cap)
            
            render_img = os.path.join(color_dir, f'cam-{key}.png')
            if os.path.exists(render_img):
                render_imgs.append(render_img)
        if not captions:
            print(f'can not find valid caption of {dname}')
            # continue
        meta['ImgDir'] = ImgDir
        meta['caption'] = list(captions)
        meta['render_imgs'] = list(render_imgs)
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(raw_dict, out_json)
    print(f'merge caption from {in_json} and {in_render_dir} to {out_json}')
    
    return


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str)
    parser.add_argument('in_render_dir', type=str)
    parser.add_argument('out_json', type=str)
    args = parser.parse_args()

    find_caption(args.in_json, args.in_render_dir, args.out_json)
    
if __name__ == "__main__":
    main()

