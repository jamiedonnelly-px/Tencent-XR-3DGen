import os
import argparse
import json
import glob
import shutil
from tqdm import tqdm

from utils_dataset import load_json, save_json, read_lines, save_lines, CAM_KEYS


def make_dataset_json(in_gt_json, in_render_root, out_dir):
    gt_data_dict = load_json(in_gt_json)['data']
    cnt_gt = 0
    for key in list(gt_data_dict.keys()):
        cnt_gt += len(list(gt_data_dict[key].keys()))
    print('cnt_gt ', cnt_gt)
    est_objs = read_lines(os.path.join(in_render_root, 'est_objs.txt'))
    render_est_dir = os.path.join(in_render_root, 'render_est')
    
    
    os.makedirs(out_dir, exist_ok=True)

    valid_cnt = 0
    new_data_dict = {}
    for est_obj in tqdm(est_objs):
        Condition_img = os.path.join(os.path.dirname(os.path.dirname(est_obj)), 'cam-0100.png')
        dname, oname = est_obj.split('/')[-4:-2]
        est_img_dir = os.path.join(render_est_dir, dname, oname)
        
        # obj_out = os.path.join(out_dir, dname, oname)
        # os.makedirs(obj_out, exist_ok=True)
        
        img_pair_list = []
        if dname in gt_data_dict and oname in gt_data_dict[dname]:
            meta_dict = gt_data_dict[dname][oname]
            gt_img_dir = meta_dict['ImgDir']
            GT_mesh = meta_dict['Mesh']
            
            for CAM_KEY in CAM_KEYS:
                gt_img = os.path.join(gt_img_dir, f'color/cam-{CAM_KEY}.png') # TODO kd, ks, normal
                est_img = os.path.join(est_img_dir, f'kd_cam-{CAM_KEY}.png')# TODO kd, ks, normal
                if os.path.exists(gt_img) and os.path.exists(est_img):
                    img_pair_list.append([gt_img, est_img])
        
        if os.path.exists(Condition_img) and os.path.exists(GT_mesh) and len(img_pair_list) > 0:
            new_meta_dict = {'Condition_img':Condition_img, 'GT_mesh':GT_mesh, 'tex_pairs':img_pair_list}
            if not dname in new_data_dict:
                new_data_dict[dname] = dict()
            new_data_dict[dname][oname] = new_meta_dict
            valid_cnt += 1
        else:
            print('invalid ', dname, oname)
            
    out_dict = {'data': new_data_dict}
    out_json = os.path.join(out_dir, 'tex_refine.json')
    save_json(out_dict, out_json)
    print(f'save done to {out_json} with valid_cnt {valid_cnt}/{len(est_objs)}')
  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='merge gt and est, make dataset json')
    parser.add_argument('in_gt_json', type=str)
    parser.add_argument('in_render_root', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()

    # Run.
    make_dataset_json(args.in_gt_json, args.in_render_root, args.out_dir)
    return

if __name__ == "__main__":
    main()
