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
    parser.add_argument('--in_source_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/20241010_daz_decimate_add_ct.json")
    parser.add_argument('--in_flatten_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/web_flatten_gdp_0925.json")
    parser.add_argument('--out_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/web_flatten_gdp.json")
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
    redundant_pairs = []
    source_key = "Obj_Mesh"
    need_key = "Mesh_obj_raw"
    need_pro_key = "Mesh_obj_pro"
    backup_key = "backup_Mesh_obj_raw_1010"
    set_dnames = ['DAZ_DAZ_Bottom', 'DAZ_DAZ_Outfit', 'DAZ_DAZ_Top']
    set_cnt, all_cnt = 0, 0
    for d_, dname, oname in key_pair_list:
        all_cnt += 1
        if dname not in set_dnames:
            continue
        if oname in in_flatten_dict:
            
            source_meta = objs_dict[d_][dname][oname]
            flatten_meta = in_flatten_dict[oname]
            
            raw_mesh_value = flatten_meta[need_key]
            source_mesh_value = source_meta[source_key]

            if raw_mesh_value != source_mesh_value:
                # assert "DAZ" in dname or "Hair" in dname, (dname, oname)
                
                flatten_meta[backup_key] = raw_mesh_value
                flatten_meta[need_key] = source_mesh_value
                flatten_meta[need_pro_key] = source_mesh_value
                
                assert os.path.exists(flatten_meta[need_key]), flatten_meta[need_key]
                set_cnt += 1
                setok_pairs.append((d_, dname, oname))
        else:
            print('redundant_', oname)
            redundant_pairs.append((d_, dname, oname))
            

    save_json(in_flatten_dict, out_json)
   
    # save_json(new_need_check_dict, new_need_check_json)
    # save_lines(conflict_onames, conflict_onames_txt)
    print(f'set_cnt= {set_cnt} len_flatten={len(in_flatten_dict)}, redundant_pairs={len(redundant_pairs)} all_cnt= {all_cnt}, len_source= {len(key_pair_list)}')
        
if __name__ == "__main__":
    main()

