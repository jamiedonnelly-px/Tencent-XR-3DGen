import os
import copy
import argparse
import sys
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines, split_pod_json


def check_uv_single(in_obj):
    try:
        if not os.path.exists(in_obj):
            print(f"[ERROR] in_obj file '{in_obj}' not found")
            return -1

        mtl_filename = None
        with open(in_obj, "r") as obj_file:
            for line in obj_file:
                if line.startswith("mtllib "):
                    mtl_filename = line.split()[1]
                    break

        if mtl_filename is None:
            print(f"[ERROR] No mtllib found in {in_obj}")
            return -2

        mtl_path = os.path.join(os.path.dirname(in_obj), mtl_filename)
        if not os.path.exists(mtl_path):
            print(f"[ERROR] mtl file '{mtl_path}' not found")
            return -3

        with open(mtl_path, "r") as mtl_file:
            mtl_content = mtl_file.read()

        map_kd_lines = [line for line in mtl_content.splitlines() if line.startswith("map_Kd")]
        return len(map_kd_lines)
    except Exception as e:
        traceback.print_exc()
        return -10


cp_keys = ["mtl_type", "uv_pos", "dname", "mtl_type"]


def use_raw_obj(meta, meta_uv):
    meta["Mesh_obj_raw"] = meta["Obj_Mesh"]
    for cp_key in cp_keys:
        if cp_key in meta_uv:
            meta[cp_key] = meta_uv[cp_key]
    meta["replace_type"] = "keep"
    return 1


def process_obj(meta, meta_uv):
    in_obj = meta["Obj_Mesh"]
    kd_cnt = check_uv_single(in_obj)
    if kd_cnt == 1:
        use_raw_obj(meta, meta_uv)
        return 1
    else:
        # need replace
        new_obj = meta_uv['Mesh_obj_raw']
        new_kd_cnt = check_uv_single(new_obj)
        if new_kd_cnt != 1:
            new_obj = os.path.join(os.path.dirname(meta_uv['uv_pos']), "mesh.obj")
            force_cnt = check_uv_single(new_obj)
            assert force_cnt == 1, force_cnt

        meta['Obj_Mesh'] = new_obj
        meta["Mesh_obj_raw"] = new_obj
        for cp_key in cp_keys:
            if cp_key in meta_uv:
                meta[cp_key] = meta_uv[cp_key]
        meta["replace_type"] = "replace"
        meta['bak_Obj_Mesh'] = in_obj
        return 10


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set _setok.json and _new_need_check.json')
    parser.add_argument('--in_raw_source_json',
                        type=str,
                        default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/20241010_daz_decimate_add_ct.json")
    parser.add_argument('--in_uv_json',
                        type=str,
                        default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/pre/generate_uv_done.json")
    # parser.add_argument('--in_source_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/pre/generate_uv_done.json")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace")
    args = parser.parse_args()

    in_raw_source_json = args.in_raw_source_json
    in_uv_json = args.in_uv_json
    out_dir = args.out_dir
    out_check_json = os.path.join(out_dir, 'check_uv_one_kd_replace.json')
    out_source_json = os.path.join(out_dir, 'final_generate_uv_done.json')
    assert os.path.exists(in_raw_source_json), in_raw_source_json

    Category_list = ['trousers', 'outfit', 'top', 'shoe']
    objs_dict, key_pair_list = parse_objs_json(in_raw_source_json)
    objs_dict_uv, key_pair_list_uv = parse_objs_json(in_uv_json)

    def process_uv_pos(key_pair, replace_condition=False):
        d_, dname, oname = key_pair
        meta = objs_dict[d_][dname][oname]
        meta_uv = objs_dict_uv[d_][dname][oname]

        Category = meta["Category"]
        if Category not in Category_list:
            result = use_raw_obj(meta, meta_uv)
            return oname, -20
        else:
            result = process_obj(meta, meta_uv)

        return oname, result

    count_result_1 = 0
    cnt_dict = {}
    with tqdm(total=len(key_pair_list), desc="Processing UV Pos") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_uv_pos, (d_, dname, oname)) for d_, dname, oname in key_pair_list]
            for future in futures:
                oname, result = future.result()
                if result == 1:
                    count_result_1 += 1
                r_key = f"cnt_{result}"
                if r_key not in cnt_dict:
                    cnt_dict[r_key] = []
                cnt_dict[r_key].append(oname)
                pbar.update(1)

    for key, value in cnt_dict.items():
        print(f"{key}: {len(value)}")
    save_json(cnt_dict, out_check_json)
    save_json(objs_dict, out_source_json)
    print(
        f"check_uv_single = 1 cnt={count_result_1} / {len(key_pair_list)}, save to {out_check_json} and {out_source_json}"
    )

    ## check
    final_count_result_1 = 0
    need_dict = {}
    def check_uv_pos(key_pair):
        d_, dname, oname = key_pair
        meta = objs_dict[d_][dname][oname]
        if meta["Category"] in Category_list:
            result = check_uv_single(meta["Obj_Mesh"])
            need_dict[oname] = result
        else:
            result = -20
        return oname, result

    with tqdm(total=len(key_pair_list), desc="Checking UV Pos") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_uv_pos, (d_, dname, oname)) for d_, dname, oname in key_pair_list]
            for future in futures:
                oname, result = future.result()
                if result == 1:
                    final_count_result_1 += 1
                pbar.update(1)
    print(f'check final_count_result_1 = {final_count_result_1} / {len(need_dict)}')


if __name__ == "__main__":
    main()
