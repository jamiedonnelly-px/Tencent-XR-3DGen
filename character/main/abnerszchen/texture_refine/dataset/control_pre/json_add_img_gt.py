import os
import sys
import argparse
from tqdm import tqdm

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dataset"))

from dataset.utils_dataset import load_json, save_json, parse_objs_json, \
    VR100_TRAIN_KEYS, VR100_CONDI_KEYS, VR100_CAP_KEYS, \
    SRENDER_DAO_TRAIN_KEYS, SRENDER_DAO_CONDI_KEYS

def gather_imgs(in_dir, keys):
    imgs = []
    for key in keys:
        render_img = os.path.join(in_dir, f'cam-{key}.png')
        if os.path.exists(render_img):
            imgs.append(render_img)
    return imgs

def add_img_gt(in_valid_json, in_render_json, out_json, render_type='srender'):
    assert os.path.exists(in_valid_json)
    assert os.path.exists(in_render_json)
    
    train_keys, condi_keys, cap_keys = [], [], []
    if render_type == 'srender':
        train_keys, condi_keys, cap_keys = SRENDER_DAO_TRAIN_KEYS, SRENDER_DAO_CONDI_KEYS, []
    elif render_type == 'vrender100':
        train_keys, condi_keys, cap_keys = VR100_TRAIN_KEYS, VR100_CONDI_KEYS, VR100_CAP_KEYS
    else:
        raise ValueError(f"invalid render_type {render_type}")
    

    raw_dict, key_pair_list = parse_objs_json(in_valid_json)
    render_dict, _ = parse_objs_json(in_render_json)
    new_data_dict = {}
    valid_cnt = 0
    for d_, oname, dname in tqdm(key_pair_list):
        try:
            meta = raw_dict[d_][oname][dname]

            ImgDir = render_dict[d_][oname][dname]['ImgDir']
            color_dir = os.path.join(ImgDir, 'color')
            color_json_dir = os.path.join(ImgDir, 'color_json')

            condi_imgs_train, condi_imgs_in = gather_imgs(color_dir, train_keys), gather_imgs(
                color_dir, condi_keys)

            if not condi_imgs_train:
                print(f'can not find valid image of {dname}')
                # continue
            if os.path.exists(color_json_dir) and len(cap_keys) > 0:
                captions = set()
                for key in cap_keys:
                    cap_json = os.path.join(color_json_dir, f'cam-{key}.json')
                    cap = load_json(cap_json)['caption']
                    if cap and len(cap) > 3:
                        captions.add(cap)                    
                meta['caption'] = list(captions)
            else:
                meta['caption'] = [""]
                
            meta['ImgDir'] = ImgDir
            meta['condi_imgs_train'] = list(condi_imgs_train)
            meta['condi_imgs_in'] = list(condi_imgs_in)
            if len(meta['caption']) > 1 and len(meta['condi_imgs_train']) > 1:
                valid_cnt += 1

            if not oname in new_data_dict:
                new_data_dict[oname] = {}
            new_data_dict[oname][dname] = meta

        except Exception as e:
            print('ERROR skip ', dname, e)
            continue

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json({'data':new_data_dict}, out_json)
    print(f'merge {valid_cnt }/{len(key_pair_list)} text img caption from {in_valid_json} and {in_render_json} to {out_json}')

    return


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_valid_json', type=str, help='/aigc_cfs/sz/data/tex/lowpoly/valid.json')
    parser.add_argument('in_render_json', type=str, help='/aigc_cfs/Asset/lists/diff_json/alldata_20240126_lowpoly_only.json')
    parser.add_argument('out_json', type=str)
    parser.add_argument('--render_type', type=str, default='srender', help='srender or vrender100 or vrender300..')
    args = parser.parse_args()

    add_img_gt(args.in_valid_json, args.in_render_json, args.out_json, args.render_type)

if __name__ == "__main__":
    main()

    """bak vrender100
python dataset/control_pre/json_add_img_gt.py /aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right.json \
/aigc_cfs/Asset/designcenter/clothes/mcwy_data.json \
/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_cap_img.json \
    --render_type vrender100
    """