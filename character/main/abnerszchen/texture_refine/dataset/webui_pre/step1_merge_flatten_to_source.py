import os
import copy
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines, split_pod_json

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set _setok.json and _new_need_check.json')
    parser.add_argument('--in_source_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/20240925_gdp.json")
    parser.add_argument('--in_flatten_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/web_flatten_gdp_manual_fixuv_0711.json")
    parser.add_argument('--out_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/s1_merge_flatten_gdp.json")
    args = parser.parse_args()

    in_source_json = args.in_source_json
    in_flatten_json = args.in_flatten_json
    out_json = args.out_json
    assert os.path.exists(in_source_json), in_source_json
    assert os.path.exists(in_flatten_json), in_flatten_json
    
    
    objs_dict, key_pair_list = parse_objs_json(in_source_json)
    in_dict = objs_dict['data']
    all_keys = list(in_dict.keys())
    print('all_keys ', all_keys)
    
    in_flatten_dict = load_json(in_flatten_json)
    
    setok_pairs = []
    new_need_check_dict = {"data": {}}
    need_key = "Mesh_obj_raw"
    set_cnt, all_cnt = 0, 0
    conflict_onames = []
    for d_, dname, oname in key_pair_list:
        all_cnt += 1
        if oname in in_flatten_dict:
            source_meta = objs_dict[d_][dname][oname]
            flatten_meta = in_flatten_dict[oname]
            set_mesh_value = flatten_meta[need_key]
            if need_key in source_meta:
                if set_mesh_value != source_meta[need_key]:
                    conflict_onames.append(oname)
            
            ## set
            source_meta[need_key] = set_mesh_value
            set_cnt += 1
            setok_pairs.append((d_, dname, oname))
        else:
            # need new
            source_meta = objs_dict[d_][dname][oname]
            if dname not in new_need_check_dict["data"]:
                new_need_check_dict["data"][dname] = {}
            new_need_check_dict["data"][dname][oname] = copy.deepcopy(source_meta)
            new_need_check_dict["data"][dname][oname][need_key] = source_meta["Obj_Mesh"]
            
    conflict_onames_txt = os.path.splitext(out_json)[0] + "_conflict.txt"
    backup_pre_json = os.path.splitext(out_json)[0] + "_backup_flatten.json"
    setok_json = os.path.splitext(out_json)[0] + "_setok.json"
    new_need_check_json = os.path.splitext(out_json)[0] + "_new_need_check.json"
    save_json(objs_dict, out_json)
    save_json(in_flatten_dict, backup_pre_json)
    split_pod_json(out_json, setok_pairs, setok_json)
    save_json(new_need_check_dict, new_need_check_json)
    save_lines(conflict_onames, conflict_onames_txt)
    print(f'set_cnt= {set_cnt} len_flatten={len(in_flatten_dict)} all_cnt= {all_cnt}, len_source= {len(key_pair_list)}')
        
if __name__ == "__main__":
    main()

