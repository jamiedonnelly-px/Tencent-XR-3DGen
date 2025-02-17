import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import random
import nvdiffrast.torch as dr
import torch.nn.functional as F
import sys
import subprocess


current_script_path = os.path.abspath(__file__)
project_root = (os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

# from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
# from scripts.utils_pool_cmds import run_commands_in_parallel
from render.render_mesh import  render_gif
from render.mesh import load_mesh, Mesh, auto_normals

def main_render_gif(
    in_obj_path,
    out_gif,
    lrm_mode=True
):
    in_mesh: Mesh = load_mesh(in_obj_path)
    render_gif(in_mesh, out_gif, lrm_mode=lrm_mode)
    
    return


def cvt_glb_to_obj(in_glb_path, output_obj_path, blender_root="/aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender"):
    """convert glb to obj with blender

    Args:
        in_glb_path: x.glb
        output_obj_path: xxx.obj
        blender_root: _description_. Defaults to /aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender or "/usr/blender-3.6.2-linux-x64/blender".

    Returns:
        True if cvt done, else False
    """
    if not os.path.exists(in_glb_path):
        return False
    assert os.path.exists(blender_root), f"can not find blender_root {blender_root}"
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    cmd = f"{blender_root} -b -P {project_root}/dataset/control_pre/glb_to_obj.py -- --mesh_path '{in_glb_path}' --output_obj_path '{output_obj_path}'"
    print('debug cmd ', cmd)
    subprocess.run(cmd, shell=True)
    return os.path.exists(output_obj_path)


def parse_obj_mesh(in_mesh_path):
    pre, ext = os.path.splitext(in_mesh_path)
    if ext == ".glb":
        output_obj_path = os.path.join(os.path.dirname(in_mesh_path), f"{os.path.basename(pre)}/cvt_mesh.obj")
        cvt_glb_to_obj(in_mesh_path, output_obj_path)
        return output_obj_path
    elif ext == ".obj":
        return in_mesh_path
    else:
        raise ValueError

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='bake to one uv tex')
    parser.add_argument('in_mesh_path', type=str)
    parser.add_argument('out_gif', type=str)
    
    # /aigc_cfs_2/sz/proj/tex_cq/scripts/utils_pool_cmds.py
    args = parser.parse_args()

    in_mesh_path = args.in_mesh_path
    
    in_obj_path = parse_obj_mesh(in_mesh_path)
    main_render_gif(in_obj_path, args.out_gif)


if __name__ == "__main__":
    main()
