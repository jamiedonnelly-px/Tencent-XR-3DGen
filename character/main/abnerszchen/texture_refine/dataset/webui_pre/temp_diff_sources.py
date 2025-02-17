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
    parser.add_argument('--in_source_json1', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/pre/generate_uv_done.json")
    parser.add_argument('--in_source_json2', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/20241010_daz_decimate_add_ct.json")
    parser.add_argument('--out_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/source_diff.json")
    args = parser.parse_args()

    in_source_json1 = args.in_source_json1
    in_source_json2 = args.in_source_json2
    out_json = args.out_json
    assert os.path.exists(in_source_json1), in_source_json1
    assert os.path.exists(in_source_json2), in_source_json2
    
    
    objs_dict1, key_pair_list1 = parse_objs_json(in_source_json1)
    objs_dict2, key_pair_list2 = parse_objs_json(in_source_json2)

    set1 = set(key_pair_list1)
    set2 = set(key_pair_list2)

    diff1 = set1 - set2  # 只在key_pair_list1中的元素
    diff2 = set2 - set1  # 只在key_pair_list2中的元素

    print("只在key_pair_list1中的元素:", diff1)
    print("只在key_pair_list2中的元素:", diff2)
    
if __name__ == "__main__":
    main()

