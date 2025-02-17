import os
import sys
import json
import argparse
import logging

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from blender_utils.blender_interface import interface_anything_to_obj, interface_render_gif
from tex_frontal.src.step0_depth_text2image import DepthT2IPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='frontal image generation')
    parser.add_argument('--cfg_json', type=str, default='configs/frontal.json')
    parser.add_argument('--input_mesh', type=str, default='data/shield/obj_mesh_mesh.obj')
    parser.add_argument('--input_image', type=str, default='', help="in_image mode")
    parser.add_argument('--prompt', type=str, default='', help="in_text mode")
    parser.add_argument('--output_dir', type=str, default='out/shield')
    args = parser.parse_args()
    
    cfg_json = args.cfg_json
    input_mesh = args.input_mesh
    input_image = args.input_image
    prompt = args.prompt
    output_dir = args.output_dir

    assert os.path.exists(cfg_json), f"can not find valid cfg_json: {cfg_json}"
    with open(cfg_json, encoding='utf-8') as f:
        cfg_dict = json.load(f)
        print('load cfg_dict=', cfg_dict)
    
    # 1. convert anything to obj
    os.makedirs(output_dir, exist_ok=True)
    obj_mesh_path = os.path.join(output_dir, "format_mesh.obj")
    if not interface_anything_to_obj(input_mesh, obj_mesh_path, blender_path=cfg_dict["blender_path"]):
        raise ValueError('interface_anything_to_obj failed')
    print('anything_to_obj done')
    
    # 2. segment input image, generate frontal image
    dt2i_pipe = DepthT2IPipeline(seed=8986)

    mesh2image_extra_params = cfg_dict["mesh2image_extra_params"]
    out_image_path = os.path.join(output_dir, "mesh2image.png")
    suc_flag, result = dt2i_pipe.main_infer_mesh2image(obj_mesh_path,
                                    prompt=prompt,
                                    in_image_path_list=[input_image] if input_image else None,
                                    out_image_path=out_image_path,
                                    mesh2image_extra_params=mesh2image_extra_params)    
    out_image_path = result["out_path"]
    print(f'mesh2image done, save to {out_image_path}')
    