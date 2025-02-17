from sympy import var
import os
import sys
from collections import deque
import time
import json
import random
import string
import logging
import subprocess
# os.environ["GRADIO_EXAMPLES_CACHE"] = "/path/to/your/cache/folder"  # TODO
import gradio as gr
import argparse

import sys
web_ui_dir = os.path.dirname(os.path.abspath(__file__))
codedir = os.path.dirname(web_ui_dir)
sys.path.append(web_ui_dir)
sys.path.append(codedir)


from web_setup import ui_setup
from client_teximguv import TeximguvClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from dataset.utils_dataset import parse_objs_json, save_lines
from grpc_backend.run_obj_to_glb import cvt_obj_to_glb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_root', type=str)
    parser.add_argument('--select_cnt', type=int, default=10)
    args = parser.parse_args()

    in_json = args.in_json
    out_root = args.out_root
    select_cnt = args.select_cnt
    
    objs_dict, key_pair_list = parse_objs_json(in_json)
    random.shuffle(key_pair_list)
    
    pair_list = key_pair_list[:select_cnt]
    
    os.makedirs(out_root, exist_ok=True)
    glb_list = []
    for d_, dname, oname in pair_list:
        meta_dict = objs_dict[d_][dname][oname]
        uv_kd = meta_dict['uv_kd']
        obj_path = os.path.join(os.path.dirname(uv_kd), 'mesh.obj')
        out_glb = os.path.join(out_root, f"{oname}.glb")
        cvt_obj_to_glb(obj_path, out_glb)
        glb_list.append(out_glb)
    
    glbs_txt = os.path.join(out_root, 'out_glb_list.txt')
    save_lines(glb_list, glbs_txt)
    print(f'save to glbs_txt: {glbs_txt}')
        
    
    
