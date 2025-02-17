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

def make_web_json(in_preprocess_json, in_embedding_json, out_dir):
    assert os.path.exists(in_preprocess_json), in_preprocess_json
    assert os.path.exists(in_embedding_json), in_embedding_json
    os.makedirs(out_dir, exist_ok=True)
    
    preprocess_dict = load_json(in_preprocess_json)
    embedding_dict = load_json(in_embedding_json)
    embedding_dict = cvt_embedding_to_standard(embedding_dict)

    embedding_json = os.path.join(out_dir, "embedding.json")
    save_json(embedding_dict, embedding_json)
    objs_dict, key_pair_list = parse_objs_json(embedding_json)
    
    valid_cnt = 0
    invalid_pairs = []
    web_flatten_dict = {}
    append_type_dict, append_cnt = {}, 0
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        try:
            # use preprocess mesh
            meta.update(preprocess_dict[d_][dname][oname])
            valid_cnt += 1
            web_flatten_dict[oname] = meta
            if "append_type" in meta:
                append_type = meta["append_type"]
                if append_type not in append_type_dict:
                    append_type_dict[append_type] = []
                append_type_dict[append_type].append(oname)
        
        except Exception as e:
            # keep raw mesh
            invalid_pair = (d_, dname, oname)
            invalid_pairs.append(invalid_pair)
            print('invalid', e, invalid_pair)
            
            meta["Mesh_obj_raw"] = meta["Obj_Mesh"]
            web_flatten_dict[oname] = meta
    
    for append_type, metas in append_type_dict.items():
        cnt = len(metas)
        print(f"append_type: {append_type} with cnt {cnt}")
        append_cnt += cnt
        
    new_source_json = os.path.join(out_dir, "new_source.json")
    save_json(objs_dict, new_source_json)

    web_flatten_json = os.path.join(out_dir, "web_flatten.json")
    save_json(web_flatten_dict, web_flatten_json)
    
    save_json(append_type_dict, os.path.join(out_dir, "append_type.json"))

    print(f"preprocessed_cnt {valid_cnt} and {len(invalid_pairs)} raw_pairs, save to {embedding_json} ,  {new_source_json}  and {web_flatten_json}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='update Mesh_obj_raw, make web json, merge preprocess source json and embedding json.')
    parser.add_argument('in_preprocess_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class_0406.json")
    parser.add_argument('in_embedding_json', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/layer_embedding_20240403_total.json")
    parser.add_argument('out_dir', type=str, default="/aigc_cfs_3/layer_tex/mcwy/merge/web_0406")
    # parser.add_argument('--dir_type', type=str, default="magic", help="multi_kd_uv_filter___MCWY_2_Dress___JP_DR_8_M_A.jpg")
    args = parser.parse_args()
    
    make_web_json(args.in_preprocess_json, args.in_embedding_json, args.out_dir)
