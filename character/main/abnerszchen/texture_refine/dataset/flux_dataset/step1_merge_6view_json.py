import os
import argparse
import sys
import traceback
import copy
from tqdm import tqdm
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, save_lines

def split_list(lst, n):
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='select data with Category and get 6view dir')
    parser.add_argument('--in_uv_source_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace/mesh_single_kd_source.json")
    parser.add_argument('--in_6view_json', type=str, default="/aigc_cfs_11/Asset/active_list/3d_diffusion/multiview_diffusion/single_view_condition/clothes_color_only.json")
    parser.add_argument('--out_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/train_flux/source.json")
    args = parser.parse_args()

    in_uv_source_json = args.in_uv_source_json
    in_6view_json = args.in_6view_json
    out_json = args.out_json
    need_categorys = ["top", "outfit", "trousers", "shoe"]
    
    assert os.path.exists(in_uv_source_json), in_uv_source_json
    assert os.path.exists(in_6view_json), in_6view_json
    
    objs_dict, key_pair_list = parse_objs_json(in_uv_source_json)
    objs_6view_dict, _ = parse_objs_json(in_6view_json)
    
    failed_pairs, failed_onames = [], []
    all_cnt, select_cnt, suc_cnt = 0, 0, 0
    new_dict = {"data":{}}
    for d_, dname, oname in tqdm((key_pair_list)):
        meta = objs_dict[d_][dname][oname]
        all_cnt += 1
        Category = meta["Category"]
        try:
            if Category in need_categorys:
                select_cnt += 1
                if dname not in new_dict[d_]:
                    new_dict[d_][dname] = {}
                new_dict[d_][dname][oname] = meta
                ImgDir = objs_6view_dict[d_][dname][oname]["ImgDir"]
                if "ImgDir" in meta:
                    new_dict[d_][dname][oname]["bak_ImgDir"] = meta["ImgDir"]
                    new_dict[d_][dname][oname]["ImgDir"] = ImgDir
                    suc_cnt += 1
                
        except Exception as e:
            failed_pairs.append((d_, dname, oname))
            failed_onames.append(oname)
            traceback.print_exc()
        
    
    save_json(new_dict, out_json)
    save_lines(failed_onames, os.path.join(os.path.dirname(out_json), "failed_merge_6view.txt"))
    
    print('failed_pairs ', failed_pairs)
    print(f"suc_cnt/select_cnt/all_cnt = {suc_cnt}/{select_cnt}/{all_cnt}, save to {out_json}")

if __name__ == "__main__":
    main()    