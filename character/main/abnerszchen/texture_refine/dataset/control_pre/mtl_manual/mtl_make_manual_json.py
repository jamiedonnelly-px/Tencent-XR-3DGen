import os
import sys
import json
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import save_json, read_lines, save_lines

def parse_key(obj_path, dir_type, magic_key="___"):
    """_summary_

    Args:
        obj_path (_type_): _description_
        dir_type (_type_): magic: */MCWY_2_Bottom___BTM_23/mergeuv/merge_BTM_23_fbx2020.obj, raw: */MCWY_2_Bottom/BTM_07/mergeuv/merge_BTM_07_fbx2020.obj
        magic_key (str, optional): _description_. Defaults to "___".
    """
    if dir_type == "magic":
        name = obj_path.split("/")[-3]
        res = name.split(magic_key)
        assert len(res) == 2, res
        dname, oname = res
    elif dir_type == "raw":
        dname, oname = obj_path.split("/")[-4:-2]
    else:
        assert False
    return dname, oname

def make_manual_json(in_txt, out_json, dir_type, magic_key="___"):
    assert os.path.exists(in_txt), in_txt
    obj_root_dir = os.path.dirname(in_txt)
    
    lines = read_lines(in_txt)
    objs = []
    for line in lines:
        obj = os.path.join(obj_root_dir, line)
        if os.path.exists(obj):
            objs.append(obj)
    if not objs:
        print(f'can not find objs from {in_txt}')
        return

    manual_dict = {"data":{}}
    for obj_path in objs:
        dname, oname = parse_key(obj_path, dir_type, magic_key=magic_key)
        if dname not in manual_dict["data"]:
            manual_dict["data"][dname] = {}
        manual_dict["data"][dname][oname] = {"obj_raw_type":"manual", "Mesh_obj_append":obj_path}
    
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(manual_dict, out_json)
    print(f"save {len(objs)} manual objs to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make maunal json from maunal objs txt')
    parser.add_argument('in_txt', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_01_03/manual_multi_kd.txt")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_01_03/manual_multi_kd.json")
    parser.add_argument('--dir_type', type=str, default="magic", help="magic or raw, means dname___oname/mergeuv/*.obj or dname/oname/mergeuv/*.obj")
    args = parser.parse_args()
    
    make_manual_json(args.in_txt, args.out_json, args.dir_type)
