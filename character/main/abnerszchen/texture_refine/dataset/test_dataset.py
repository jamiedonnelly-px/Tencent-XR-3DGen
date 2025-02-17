import os
import argparse
import json
from utils_dataset import save_json, load_json, parse_objs_json

def generate_meta(meta_out_dir):
    need_cam_keys = ['0089', '0011', '0100', '0132', '0167', '0286', '0318']
    pbr_keys = ['kd', 'ks', 'normal']

    # from one obj
    gt_png_paths = []
    for pbr_key in pbr_keys:
        for need_cam_key in need_cam_keys:
            png_path = os.path.join(meta_out_dir, f"gt_{pbr_key}_{need_cam_key}.png")
            gt_png_paths.append(png_path)
                
    est_keys = [f'est_{i:03}' for i in range(2)]
    
    est_png_paths = []
    for type_key in est_keys:
        # one obj
        for pbr_key in pbr_keys:
            for need_cam_key in need_cam_keys:
                png_path = os.path.join(meta_out_dir, f"{type_key}_{pbr_key}_{need_cam_key}.png")
                est_png_paths.append(png_path)
    
    return {'GT_imgs':gt_png_paths, 'Est_imgs':est_png_paths}
    
def get_gt_dict(objs_dict, key_pair, out_dir, magic_key='++'):
    first, dtype, oname = key_pair
    
    
    raw_meta = objs_dict[first][dtype][oname]
    meta_name = dtype + magic_key + oname
    meta_out_dir = os.path.join(out_dir, meta_name)
    
    pngs_dict = generate_meta(meta_out_dir)
    
    if not raw_meta['ImgDir']:
        print('error ', raw_meta)
        return None, None
    
    new_meta_dict = {"Mesh": raw_meta["Mesh"], 
     "Condition_img": os.path.join(raw_meta['ImgDir'], 'color/cam-0100.png'),
     }
    for i in pngs_dict.keys():
        new_meta_dict[i] = pngs_dict[i]
        
    return meta_name, new_meta_dict

def flatten_dataset():
    # pair (est, gt, condition)
    
    ('/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/vroid++1008310613509133290/est_000_kd_0089.png',
     '/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/vroid++1008310613509133290/gt_kd_0089.png',
     '/apdcephfs_cq3/share_2909871/3dAsset_artcenter/render_vae_latent/render/1008310613509133290/vroid_obj_0_1008310613509133290_manifold_full.obj_output_512_MightyWSB/color/cam-0100.png')
    
    return

def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()

    # Run.
    objs_dict, key_pair_list = parse_objs_json(args.in_json)
    key_pair_list = key_pair_list[:100]
    
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print('key_pair_list ', len(key_pair_list), key_pair_list[0])
    new_data_dict = {}
    for key_pair in key_pair_list:
        meta_name, new_meta_dict = get_gt_dict(objs_dict, key_pair, out_dir)
        new_data_dict[meta_name] = new_meta_dict
        
    # Done.
    out_json = os.path.join(out_dir, 'debug.json')
    save_json(new_data_dict, out_json)
    print(f"Done., save to {out_json}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
