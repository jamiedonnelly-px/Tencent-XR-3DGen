import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

import nvdiffrast.torch as dr

import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

# from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
# from scripts.utils_pool_cmds import run_commands_in_parallel
from render.render_mesh import (
    refine_texture_knn,
    revert_texc_as_mask,
)
from render.uv_conditions import render_geometry_uv


def save_tensor_hwc(tensor_hwc, out_img):
    """_summary_

    Args:
        tensor_hwc: [h,w,c](rgb) or [h, w](gray)
        out_img: _description_
    """
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    Image.fromarray(
        np.clip(tensor_hwc.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    ).save(out_img)


def save_np_hwc(np_hwc, out_img):
    """_summary_

    Args:
        np_hwc: [h,w,c](rgb) or [h, w](gray)
        out_img: _description_
    """
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    Image.fromarray(np.clip(np_hwc * 255.0, 0, 255).astype(np.uint8)).save(out_img)


def pos_knn(glctx, imesh, uv_tex, see_mask, ssaa=-1, debug_dir=None):
    """Find the nearest visible point and inpaint the color of the invisible point

    Args:
        glctx: _description_
        imesh: class Mesh
        uv_tex: [h, w, 3/c]
        see_mask: [h, w, 1]  render-see mask in uv
        ssaa: _description_. Defaults to -1.

    Returns:
        uv_tex_new: [h, w, 3/c] tensor
    """
    device = uv_tex.device
    gb_xyz, _, gb_mask = render_geometry_uv(glctx, imesh, uv_tex.shape[0], ssaa=ssaa)
    gb_xyz = gb_xyz.cpu().numpy()

    # [h, w, 3]
    uv_tex_np = uv_tex.cpu().numpy()

    # [h, w] gb_mask: all need uv
    gb_mask = gb_mask.to(torch.bool).squeeze(-1).cpu().numpy()
    see_mask = see_mask.to(torch.bool).squeeze(-1).cpu().numpy()

    # [h, w] mask
    inpaint_mask = gb_mask & (~see_mask)
    # inpaint_mask = binary_dilation(inpaint_mask, iterations=2)

    # [N1, 3]
    inpaint_points = gb_xyz[inpaint_mask]
    # [N2, 3]
    see_points = gb_xyz[see_mask]
    see_colors = uv_tex_np[see_mask]

    knn = NearestNeighbors(n_neighbors=1).fit(see_points)
    _, indices = knn.kneighbors(inpaint_points)  #  indices [N1, 1], in [0, N2)

    inpaint_colors = see_colors[indices[:, 0]]

    # [N2, 2]
    inpaint_indices = np.stack(np.nonzero(inpaint_mask), axis=-1)
    uv_tex_np[inpaint_indices[:, 0], inpaint_indices[:, 1]] = inpaint_colors

    uv_tex_new = torch.from_numpy(uv_tex_np).to(device)

    if debug_dir is not None:
        print("indices", indices.shape)
        print("inpaint_colors", inpaint_colors.shape)
        save_np_hwc(see_mask, os.path.join(debug_dir, "see_mask.jpg"))
        save_np_hwc(gb_mask, os.path.join(debug_dir, "gb_mask.jpg"))
        save_np_hwc(inpaint_mask, os.path.join(debug_dir, "inpaint_mask.jpg"))
        inpaint_tex = np.zeros_like(uv_tex_np)
        inpaint_tex[inpaint_indices[:, 0], inpaint_indices[:, 1]] = inpaint_colors
        save_np_hwc(inpaint_tex, os.path.join(debug_dir, "inpaint_tex.jpg"))
        save_np_hwc(
            uv_tex_np * see_mask[..., None], os.path.join(debug_dir, "see_tex.jpg")
        )

    return uv_tex_new


def inpaint_refine_uv_tex(
    glctx,
    in_mesh,
    tex_data_opt,
    mvp,
    uv_refine_type="pos_knn",
    viwecos_mask_all=None,
    mask_sp=4,
    ssaa=-1,
    debug_dir=None,
):
    """update tex_data_opt with inpaint_knn

    Args:
        glctx: _description_
        in_mesh: Mesh class
        tex_data_opt: torch.nn.Parameter [1, h, w, 3]
        mvp: [nv, 4, 4]
        uv_refine_type: pos_knn or uv_knn. Defaults to "pos_knn".
        viwecos_mask_all: None or [nv, h, w, 1]. Defaults to None.
        mask_sp: _description_. Defaults to 4.
        ssaa: _description_. Defaults to -1.
        debug_dir: _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_
    """
    uv_tex = (tex_data_opt.data).clone().squeeze(0)  # [h,w,3]
    tex_res = uv_tex.shape[0]

    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx  # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]

    # [1, h, w, 1]
    uv_mask = revert_texc_as_mask(
        glctx,
        vtx_pos,
        pos_idx,
        vtx_uv,
        uv_idx,
        mvp,
        tex_res,
        up_scale=mask_sp,
        alpha_mask=viwecos_mask_all,
    )
    uv_mask = uv_mask.squeeze(0)

    if uv_refine_type == "pos_knn":
        uv_tex_new = pos_knn(glctx, in_mesh, uv_tex, uv_mask, ssaa=ssaa, debug_dir=debug_dir)

    elif uv_refine_type == "uv_knn":
        # [resolution, resolution]
        uv_mask = uv_mask.squeeze(-1)
        # [h,w,3]
        uv_tex_new_knn, inpaint_region = refine_texture_knn(uv_tex, uv_mask)
        uv_mask_hw1 = uv_mask.unsqueeze(-1)
        uv_tex_new = (
            uv_mask_hw1 * tex_data_opt.data.squeeze(0)
            + (1 - uv_mask_hw1) * uv_tex_new_knn
        )

        if debug_dir is not None:
            save_tensor_hwc(
                uv_tex_new_knn, os.path.join(debug_dir, f"debug_knn_tex.jpg")
            )
            Image.fromarray(inpaint_region).save(
                os.path.join(debug_dir, f"debug_inpaint.jpg")
            )
            save_tensor_hwc(uv_mask, os.path.join(debug_dir, f"debug_mask.jpg"))
            save_tensor_hwc(uv_tex, os.path.join(debug_dir, f"debug_optim_tex.jpg"))
            save_tensor_hwc(uv_tex_new, os.path.join(debug_dir, f"debug_mix_tex.jpg"))
    else:
        raise NotImplementedError(f"invalid uv_refine_type {uv_refine_type}")

    tex_data_opt.data = uv_tex_new.unsqueeze(0)
    
    return
