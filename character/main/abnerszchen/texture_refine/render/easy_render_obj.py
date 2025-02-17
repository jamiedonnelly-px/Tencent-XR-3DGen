import os
import argparse
import torch
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
import math

from render_obj import load_mesh, auto_center, Mesh
from render_mesh import render_texture_views
from util import make_4views_mvp_tensor

def tensor_to_pils(data, vis_res=512):
    pils = [
        Image.fromarray(
            np.clip(np.rint(data[idx].detach().cpu().numpy() * 255.0), 0,
                    255).astype(np.uint8)).resize((vis_res, vis_res))
        for idx in range(data.shape[0])
    ]
    return pils

def concatenate_images_horizontally(image_list, out_img_path=None):
    """concatenate images horizontally

    Args:
        image_list: list of PIL.Image 
        out_img_path: save if not None

    Returns:
        output_image PIL.Image 
    """
    total_width = sum([img.width for img in image_list])
    max_height = max([img.height for img in image_list])

    output_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in image_list:
        output_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if out_img_path is not None:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        output_image.save(out_img_path)
    return output_image


def main_render(in_obj, out_dir, render_res, cam_type="ortho", max_mip_level=4):

    os.makedirs(out_dir, exist_ok=True)
    # Render
    glctx = dr.RasterizeCudaContext()

    raw_mesh : Mesh = load_mesh(in_obj)
    raw_mesh = auto_center(raw_mesh)
    print('load done')
    
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx
    mvp, _ = make_4views_mvp_tensor(cam_type=cam_type)

    ### merge tex channels, use common rast, uv, interpolate.
    tex = raw_mesh.material  # class Material
    tex_data = tex['kd'].data

    color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data, render_res,
                                        max_mip_level)


    masked_color_pils = tensor_to_pils(color * alpha)
    concatenate_images_horizontally(masked_color_pils, os.path.join(out_dir, f"render_{cam_type}.png"))

    # # debug mask
    # py3d_depth_npy = "/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/meshortho_True_512.npy"
    # py3d_depth = np.load(py3d_depth_npy)     # 4, 512 512 3
    # print(py3d_depth.shape)
    
    # alpha_np = alpha.detach().cpu().numpy() > 0
    # py3d_depth_mask = py3d_depth > 0
    # print('py3d_depth_mask ', py3d_depth_mask.shape)
    # py3d_depth_mask = py3d_depth_mask[..., :1]
    
    # diff = (alpha_np & ~py3d_depth_mask).sum()
    # print('diff ', diff)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('--in_obj', type=str, default="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/mesh.obj")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/nv_r")
    parser.add_argument('--render_res', type=int, default=512)
    parser.add_argument('--max_mip_level', type=int, default=4)
    args = parser.parse_args()

    in_obj = args.in_obj
    out_dir = args.out_dir

    main_render(in_obj, out_dir, args.render_res, max_mip_level=args.max_mip_level)
