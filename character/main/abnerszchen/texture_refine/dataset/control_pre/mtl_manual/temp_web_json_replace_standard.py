import os
import sys
import json
from tqdm import tqdm
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from dataset.utils_dataset import load_json, save_json, read_lines, save_lines, parse_objs_json

def cvt_embedding_to_standard(embedding_dict):
    """_summary_

    Args:
        embedding_dict: [data][Category][dname][oname]
    """
    new_data_dict = {}
    embedding_data_dict = embedding_dict["data"]
    for category, dname_metas in embedding_data_dict.items():
        for dname, oname_metas in dname_metas.items():
            new_data_dict[dname] = {}
            for oname, meta in oname_metas.items():
                new_data_dict[dname][oname] = meta
    
    standard_dict = {"data":new_data_dict}
    return standard_dict

def web_json_to_standard(in_standard_emb_json, in_web_json, out_json):
    assert os.path.exists(in_standard_emb_json), in_standard_emb_json
    assert os.path.exists(in_web_json), in_web_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    objs_dict, key_pair_list = parse_objs_json(in_standard_emb_json)
    in_dict = load_json(in_web_json)
    
    valid_cnt = 0
    invalid_pairs = []
    standard_dict = {"data": {}}
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        new_meta = in_dict[oname]
        Obj_Mesh = meta["Obj_Mesh"]
        Mesh_obj_raw = new_meta["Mesh_obj_raw"]
        meta["Obj_Mesh"] = Mesh_obj_raw
        meta["Obj_Mesh_raw"] = Obj_Mesh
           

    save_json(objs_dict, out_json)

    print(f"save to {out_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_standard_emb_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/web_0406/embedding.json")
    parser.add_argument('in_web_json', type=str, default="/aigc_cfs_2/sz/proj/tex_cq/configs/web_0507/web_flatten_gdp.json")
    parser.add_argument('out_json', type=str, default="/aigc_cfs_2/sz/proj/tex_cq/configs/web_0507/standard_fixuv_gdp.json")
    args = parser.parse_args()
    
    web_json_to_standard(args.in_standard_emb_json, args.in_web_json, args.out_json)
