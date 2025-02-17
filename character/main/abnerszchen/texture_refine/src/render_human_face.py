import os
import argparse
import torch
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
import time
import torch.nn.functional as F
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.mesh import load_mesh, Mesh, auto_center
from render.render_mesh import resize_render, transform_pos, auto_normals
from render.util import perspective, ortho_ndc, make_w2c_mats
from render.obj import write_obj
import render.texture as texture

from src.nv_diff_bake import NvdiffRender


def make_face_views_mvp_tensor(cam_type="perspective", zoom=0.9, dist=3.0, fovy=0.15, device="cuda"):
    """make mvp = proj @ w2c

    Args:
        cam_type: ortho / perspective. Defaults to "ortho".
        zoom: scale for ortho. Defaults to 0.9.
        dist: _description_. Defaults to 3.0.
        fovy: fov for perspective . Defaults to 0.7.
        device: _description_. Defaults to "cuda".

    Returns:
        mvp tensor [n, 4, 4]
        camera_centers tensor [n, 1, 3]
    """
    # intrinsic
    if cam_type == "perspective":
        proj = perspective(fovy, 1, 0.01, 100)
    elif cam_type == "ortho":
        proj = ortho_ndc(zoom, 0.01, 100)
    else:
        raise ValueError(f"invalid cam_type={cam_type}")

    # extrinsic
    y_face_center = 0.85 # 0.85
    
    azims = [0]
    elevs = [0] * len(azims)
    dists = [dist] * len(azims)
    w2c_mats, camera_centers = make_w2c_mats(elevs, azims, dists, center=[0, y_face_center, 0])
    
    mvp_list = []
    for w2c in w2c_mats:
        # Load modelview matrix.
        mv = torch.tensor(w2c, dtype=torch.float32)

        mvp_ = proj @ mv
        mvp_list.append(mvp_[None, ...])

    mvp = torch.cat(mvp_list, dim=0).to(device)
    camera_centers = torch.cat([torch.tensor(center, dtype=torch.float32).reshape(1, 1, 3) for center in camera_centers],
                               dim=0).to(device)
    return mvp, camera_centers



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='render human obj')
    parser.add_argument(
        '--in_obj',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/texall/textured.obj")
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/render_face_only")
    parser.add_argument('--render_res', type=int, default=2048)
    parser.add_argument('--tex_res', type=int, default=1024)
    parser.add_argument('--max_mip_level', type=int, default=4)
    args = parser.parse_args()

    in_obj = args.in_obj
    out_dir = args.out_dir
    render_res = args.render_res
    tex_res = args.tex_res
    max_mip_level = args.max_mip_level

    save_debug = True
    mvp, camera_centers = make_face_views_mvp_tensor()
    
    # run once
    nv_render = NvdiffRender(render_res=render_res, tex_res=tex_res)
    nv_render.prepare_mesh_geom_and_pose(in_obj,
                                         mvp=mvp, camera_centers=camera_centers,
                                         out_dir=out_dir if save_debug else None)

    # vis
    nv_render.render_views(out_dir, max_mip_level=max_mip_level)
    
    print(f'render {in_obj}, save to {out_dir}')