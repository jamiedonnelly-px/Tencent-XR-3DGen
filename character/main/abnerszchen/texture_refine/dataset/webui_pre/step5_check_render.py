import os
import json
import traceback
from tqdm import tqdm
import argparse
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj
from render.geom_utils import mesh_normalized
import render.texture as texture
from render.uv_conditions import render_geometry_uv, cvt_geom_to_pil
from PIL import Image
import PIL
import nvdiffrast.torch as dr
from concurrent.futures import ThreadPoolExecutor, as_completed
        
def process_item(d_, dname, oname, need_key, objs_dict, glctx, uv_res):
    try:
        meta = objs_dict[d_][dname][oname]
        in_obj = meta[need_key]
        raw_mesh = load_mesh(in_obj)

        try_cnt = 5
        for i in range(try_cnt):
            try:
                mesh_normalized(raw_mesh)
                break
            except Exception as e:
                print(f"warning mesh_normalized failed. skip mesh_normalized")
                print(f"Exception occurred: {e}. Retrying {i + 1}/{try_cnt}")
                if i == try_cnt - 1:
                    print("[ERROR] mesh_normalized failed!!!")
                    break

        raw_mesh = auto_normals(raw_mesh)
        gb_xyz, gb_normal, gb_mask = render_geometry_uv(glctx, raw_mesh, uv_res)
        in_geom_pil = cvt_geom_to_pil(gb_xyz, gb_mask)

        return True, None
    except Exception as e:
        print(f'[error] e={e}')
        return False, oname
    
#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='set generate_uv_done.json')
    parser.add_argument('--in_json', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925/final_merge.json")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925")
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    assert os.path.exists(in_json), in_json
    glctx = dr.RasterizeCudaContext()
    uv_res = 1024
    
    objs_dict, key_pair_list = parse_objs_json(in_json)
    need_key = "Mesh_obj_raw"
    suc_cnt = 0
    faild_onames = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, d_, dname, oname, need_key, objs_dict, glctx, uv_res) for d_, dname, oname in key_pair_list]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            success, oname = future.result()
            if success:
                suc_cnt += 1
            else:
                faild_onames.append(oname)
                d_, dname = next((d_, dname) for d_, dname, _ in key_pair_list if _ == oname)
                objs_dict[d_][dname].pop(oname)
        
    final_ok_json = os.path.join(out_dir, "final_ok.json")
    faild_onames_txt = os.path.join(out_dir, "faild_onames.txt")
    save_json(objs_dict, final_ok_json)
    save_lines(faild_onames, faild_onames_txt)
    print(f'suc_cnt= {suc_cnt} raw len={len(key_pair_list)} ')


if __name__ == "__main__":
    main()

    