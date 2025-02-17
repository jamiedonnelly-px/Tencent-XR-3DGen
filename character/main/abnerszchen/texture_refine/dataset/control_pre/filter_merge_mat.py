import os
import sys
import json
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, read_lines


# ----------------------------------------------------------------------------

fbx_type = ['right', 'multi_mtl', 'error', 'have_single_d']

def check_cnt(data, key="newmtl", return_names=False):
    cnt = 0
    name_set = set()
    for line in data:
        if key in line:
            cnt += 1
            name = line.split(" ")[-1]
            name_set.add(name)
    if return_names:
        return cnt, list(name_set)
    return cnt


def check_type_threaded(normalkd_dest_pairs, max_workers=96):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        suc_cnt = 0
        results = list(tqdm(executor.map(lambda pair: check_type(pair[0], pair[1], pair[2], pair[3]), normalkd_dest_pairs), total=len(normalkd_dest_pairs)))
        for result in results:
            if result:
                suc_cnt += 1
    
    return results   

def check_type(obj_path):
    mtl_path = obj_path.replace('.obj', '.mtl')
    if not os.path.exists(mtl_path):
        return 'error_nomtl'
    
    data = read_lines(mtl_path)
    cnt_mtl = check_cnt(data, key='newmtl', return_names=False)
    cnt_kd, kd_paths = check_cnt(data, key='map_Kd', return_names=True)
    if not cnt_mtl or not cnt_kd:
        return 'only_single_d'
    
    if cnt_mtl == 1 and cnt_kd == 1:
        return 'right'

    
    # no kd
    if cnt_mtl == cnt_kd:
        if len(kd_paths) == 1:
            return 'multi_mtl_one_kd'
        if len(kd_paths) > 1:
            return "multi_mtl_multi_kd"
        return "unknown_multi_mtl"
    # with kd
    else:
        if cnt_mtl > 1 or cnt_kd > 1:
            if len(kd_paths) == 1:
                return 'multi_mtl_one_kd_d'
            if len(kd_paths) > 1:
                return 'multi_mtl_multi_kd_d'
            return 'unknown_multi_mtl_multi_kd' 
        return 'error_unknown'
        
    return 'error_unknown'


def main_mcwy():
    parser = argparse.ArgumentParser(description='filter fbx for mtl merge')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_json', type=str)
    args = parser.parse_args()

    in_json = args.in_json
    out_json = args.out_json
    assert os.path.exists(in_json)

    fbx_obj_dict = load_json(in_json)
    filter_dict = {}
    for fbx, obj in fbx_obj_dict.items():
        fbx_type = check_type(obj)
        if fbx_type not in filter_dict:
            filter_dict[fbx_type] = {}
        filter_dict[fbx_type][fbx] = obj
        
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(filter_dict, out_json)
    right_dict = filter_dict.get('right', {})
    print('type cnt', len(filter_dict))
    for fbx_type, data in filter_dict.items():
        print(f"{fbx_type}: {len(data)}")
        
    print(f'filter {in_json} to out_json {out_json}, valid {len(right_dict)} / {len(list(fbx_obj_dict.keys()))}')


def process_key_pair(key_pair, objs_dict):
    d_, dname, oname = key_pair
    meta = objs_dict[d_][dname][oname]
    obj = meta['Mesh_obj_raw']
    filter_dict = {}
    if not obj:
        fbx_type = "error_notfind"
    else:
        # obj = obj.replace('_manifold_full.obj', '_resize.obj')
        fbx_type = check_type(obj)
    if fbx_type not in filter_dict:
        filter_dict[fbx_type] = {}
    filter_dict[fbx_type][oname] = {"Mesh":obj, "data_type":dname, "ImgDir":meta["ImgDir"]}
    return filter_dict

def batch_filter_mtl_type(in_json, out_json):
    assert os.path.exists(in_json)

    objs_dict, key_pair_list = parse_objs_json(in_json)
    filter_dict = {}

    with ThreadPoolExecutor() as executor:
        filter_dicts = list(tqdm(executor.map(process_key_pair, key_pair_list, [objs_dict] * len(key_pair_list))))
    # Merge all filter_dicts
    filter_dict = {}
    for fd in filter_dicts:
        for key, value in fd.items():
            if key not in filter_dict:
                filter_dict[key] = value
            else:
                filter_dict[key].update(value)
        
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(filter_dict, out_json)
    print('type cnt', len(filter_dict))
    for fbx_type, data in filter_dict.items():
        print(f"{fbx_type}: {len(data)}")
            
    right_dict = filter_dict.get('right', {})
    print(f'filter {in_json} to out_json {out_json}, valid {len(right_dict)} / {len(key_pair_list)}')
    return filter_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='filter fbx for mtl merge')
    parser.add_argument('in_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/image_caption_done.json")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/filter_mtl_with_skin.json")
    args = parser.parse_args()
    
    batch_filter_mtl_type(args.in_json, args.out_json)
