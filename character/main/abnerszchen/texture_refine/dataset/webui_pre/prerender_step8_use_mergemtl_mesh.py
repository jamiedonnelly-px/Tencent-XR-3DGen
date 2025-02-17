import os
import copy
import argparse
import sys
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import subprocess

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



def merge_mtl(blender_root, mtl_merge_py, in_mesh, out_dir=None, output_mesh_name="combined.obj"):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(in_mesh), "merge_mtl")
    cmd = f"{blender_root} -b -P  {mtl_merge_py} -- --input_mesh_path {in_mesh} --output_mesh_folder {out_dir} --output_mesh_name {output_mesh_name}\n"
    result = subprocess.run(cmd, shell=True)
    out_obj = os.path.join(out_dir, output_mesh_name)
    return out_obj if os.path.exists(out_obj) else None

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set _setok.json and _new_need_check.json')
    parser.add_argument('--in_source_json',
                        type=str,
                        default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace/final_generate_uv_done.json")
    parser.add_argument('--out_json',
                        type=str,
                        default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/replace/mesh_single_kd_source.json")
    parser.add_argument('--blender_root', type=str, default='/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender')
    parser.add_argument('--addon_path', type=str, default="/aigc_cfs/sz/software/material-combiner-addon-master.zip",
                        help='path of addon zip file to be installed')
    args = parser.parse_args()

    in_source_json = args.in_source_json
    out_json = args.out_json
    blender_root = args.blender_root
    addon_path = args.addon_path

    assert os.path.exists(in_source_json), in_source_json

    Category_list = ['trousers', 'outfit', 'top', 'shoe']
    objs_dict, key_pair_list = parse_objs_json(in_source_json)

    addon_py = os.path.join(project_root, "dataset/control_pre/blender_addon_install.py")
    assert os.path.exists(addon_py), addon_py
    result = subprocess.run(f"{blender_root} -b -P {addon_py} -- --addon_path  {addon_path}", shell=True, check=True)
    assert result.returncode == 0
    print('1. install blender addon done.')

    mtl_merge_py = os.path.join(project_root, "dataset/control_pre/blender_material_merge.py")
    assert os.path.exists(mtl_merge_py), mtl_merge_py

    def replace_mergemtl_mesh(key_pair):
        d_, dname, oname = key_pair
        meta = objs_dict[d_][dname][oname]

        Category = meta["Category"]
        if Category in Category_list and meta["replace_type"] == "replace":
            ## 鞋子特殊处理
            if Category == "shoe":
                print('oname Category ', oname)
                for lr in ["left", "right"]:
                    raw_obj = os.path.join(os.path.dirname(meta["bak_Obj_Mesh"]), f"{lr}/asset.obj")
                    assert os.path.exists(raw_obj), raw_obj
                    combined_obj = os.path.join(os.path.dirname(meta["bak_Obj_Mesh"]), f"merge_mtl/{lr}/asset.obj")
                    if not os.path.exists(combined_obj):
                        print('need merge shoe oname ', oname, lr)
                        out_dir = os.path.join(os.path.dirname(meta["bak_Obj_Mesh"]), f"merge_mtl/{lr}")
                        lr_combined_obj = merge_mtl(blender_root,
                                                 mtl_merge_py,
                                                 raw_obj,
                                                 out_dir=out_dir,
                                                 output_mesh_name="asset.obj")
                        temp_cnt = check_uv_single(lr_combined_obj)
                        assert temp_cnt == 1 and os.path.exists(combined_obj), (temp_cnt, combined_obj)
                        
            ## 包括鞋子在内都处理
            combined_obj = os.path.join(os.path.dirname(meta["bak_Obj_Mesh"]), "merge_mtl/combined.obj")
            if os.path.exists(combined_obj):
                meta["Obj_Mesh"] = combined_obj
            else:
                print('need merge oname ', oname)
                combined_obj = merge_mtl(blender_root, mtl_merge_py, meta["bak_Obj_Mesh"])
                temp_cnt = check_uv_single(combined_obj)
                if temp_cnt != 1:
                    print('combine failed oname ', oname)
                    return oname, -10
                meta["Obj_Mesh"] = combined_obj

            meta["bak_Mesh_obj_raw"] = meta["Mesh_obj_raw"]
            meta["Mesh_obj_raw"] = meta["Obj_Mesh"]
            new_kd_cnt = check_uv_single(combined_obj)
            return oname, new_kd_cnt
        else:
            return oname, -5


    count_result_1 = 0
    cnt_dict = {}
    with tqdm(total=len(key_pair_list), desc="Processing UV Pos") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(replace_mergemtl_mesh, (d_, dname, oname)) for d_, dname, oname in key_pair_list]
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
    save_json(objs_dict, out_json)
    print(
        f"check_uv_single = 1 cnt={count_result_1} / {len(key_pair_list)}, save to {out_json} "
    )

    ## check
    final_count_result_1 = 0
    need_dict = {}
    def check_uv_pos(key_pair):
        d_, dname, oname = key_pair
        meta = objs_dict[d_][dname][oname]
        if meta["Category"] in Category_list:
            result = int(check_uv_single(meta["Mesh_obj_raw"]) and check_uv_single(meta["Obj_Mesh"]))
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
