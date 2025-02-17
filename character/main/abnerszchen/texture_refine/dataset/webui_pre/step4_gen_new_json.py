import os
import json
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set ruku standard and flatten')
    parser.add_argument('--in_setok_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/s1_merge_flatten_gdp_setok.json")
    parser.add_argument('--in_gen_uv_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/generate_uv_done.json")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925")
    args = parser.parse_args()


    in_setok_json = args.in_setok_json
    in_gen_uv_json = args.in_gen_uv_json
    out_dir = args.out_dir

    assert os.path.exists(in_setok_json), in_setok_json
    assert os.path.exists(in_gen_uv_json), in_gen_uv_json
    
    objs_dict, key_pair_list = parse_objs_json(in_setok_json)
    gen_uv_dict, gen_uv_key_pair_list = parse_objs_json(in_gen_uv_json)
    need_key = "Mesh_obj_raw"
    uv_key = "Mesh_obj_pro"
    backup_key = "backup_Mesh_obj_raw"
    
    set_cnt = 0
    for d_, dname, oname in gen_uv_key_pair_list:
        uv_meta = gen_uv_dict[d_][dname][oname]
        assert uv_key in uv_meta
        new_mesh = uv_meta[uv_key]
        if need_key in uv_meta:
            uv_meta[backup_key] = uv_meta[need_key]
        uv_meta[need_key] = new_mesh
        
        # add to source
        if dname not in objs_dict[d_]:
            objs_dict[d_][dname] = {}
        objs_dict[d_][dname][oname] = uv_meta
        set_cnt += 1

    final_merge_json = os.path.join(out_dir, "final_merge.json")
   
    save_json(objs_dict, final_merge_json)
    re_dict, re_key_pair_list = parse_objs_json(final_merge_json)
    print(f'set_cnt= {set_cnt} raw len={len(key_pair_list)} add len= {len(gen_uv_key_pair_list)}, final len= {len(re_key_pair_list)}')
                
        
if __name__ == "__main__":
    main()

