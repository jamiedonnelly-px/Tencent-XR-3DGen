import os
import numpy  as np
import torch
import torch.nn as nn
import cv2
import nvdiffrast.torch as dr

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.mesh import load_mesh, Mesh, auto_center
from render.material import Material
from render.render_mesh import parse_pose_json, render_texture_views, save_textures, util_merge_tex, mix_rgba_bg
from render.geom_utils import try_mesh_normalized
import render.texture as texture
from PIL import Image
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally


def render_mesh(glctx, raw_mesh, mvp, render_res=512, max_mip_level=None, need_texs=['kd']):
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx

    ### merge tex channels, use common rast, uv, interpolate.
    tex : Material = raw_mesh.material # class Material   
    tex_data_merge, key_channel_se_pairs = util_merge_tex(tex, need_texs)
    
    color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_merge, render_res, max_mip_level)
       
    return color, alpha

def load_obj_and_pose(in_obj, in_pose_json, lrm_mode=False, device='cuda'):
    """ load obj, read render pose json, get camera intr-extri, add to device
    Args:
        in_obj: path obj
        in_pose_json: like data/cams/cam_parameters_select.json
        lrm_mode: if lrm_mode, fx=fy=255.5, need rm in future!!!!!!!!!!!!!!!!!!!!!!!! TODO
        device: _description_. Defaults to 'cuda'.

    Returns:
        frames, dict {'cam_name_list':cam_name_list, 'mv': mv, 'mvp': mvp, 'campos': campos}
        # cam_name_list [nv]
        # w2c [nv, 4, 4] 
        # mv [nv, 4, 4] 
        # mvp [nv, 4, 4] = proj * mv, proj(intri) is relative about fovy and aspect, not about resolution
        # campos [nv, 3]   
    Args:
    """
    if not os.path.exists(in_obj):
        print(f'can not find obj {in_obj}')
        return None
    raw_mesh : Mesh = load_mesh(in_obj)
    
    frames = parse_pose_json(in_pose_json, lrm_mode=lrm_mode)
    frames = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in frames.items()}
        
    return raw_mesh, frames

def render_obj_texture(in_obj, in_pose_json, out_dir, lrm_mode=False, render_res=512, max_mip_level=None, 
                       use_normalized=False, save_res=None, device='cuda'):
    """render obj with texture

    Args:
        in_obj: obj path.
        in_pose_json: like data/cams/cam_parameters_select.json
        lrm_mode: if lrm_mode, fx=fy=255.5, need rm in future!!!!!!!!!!!!!!!!!!!!!!!! TODO
        out_dir: out dir
        render_res
        max_mip_level: render with mipmapping or not. if resolution small, do not use. TODO check
        save_res: if not None, force set save resolution

    Returns:
        _description_
    """
    # TODO(csz) ->cfg
    need_texs = ['kd', 'normal']
            
    raw_mesh, frames = load_obj_and_pose(in_obj, in_pose_json, lrm_mode=lrm_mode, device=device)
    print('load obj and pose done. begin render..')
    if use_normalized:
        tf = try_mesh_normalized(raw_mesh)
        print('tf', tf)
    
    glctx = dr.RasterizeCudaContext()
    
    ## render gt

    ### common geomtry 
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx
    mvp = frames['mvp']
    cam_name_list = frames['cam_name_list']

    ### merge tex channels, use common rast, uv, interpolate.
    tex : Material = raw_mesh.material # class Material   
    tex_data_merge, key_channel_se_pairs = util_merge_tex(tex, need_texs)
    
    color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_merge, render_res, max_mip_level)
   
    # save imgs
    print('debug begin save..')
    save_textures(color, alpha, key_channel_se_pairs, out_dir, cam_name_list=cam_name_list, save_res=save_res)

    # tex_psnr = mse_to_psnr(torch.mean((tex_data_merge[..., :4] - tex_data_opt[..., :4])**2).item())
    # mask = tex_data_merge[..., :4] > 0
    # tex_psnr_alpha = mse_to_psnr(torch.mean((tex_data_merge[..., :4][mask] - tex_data_opt[..., :4][mask])**2).item())
    # print('tex_psnr/tex_psnr_alpha ', tex_psnr, tex_psnr_alpha)
    
    
    return


def render_obj_with_in_kd(in_obj, in_kd_pil, in_pose_json, lrm_mode=False, render_res=512, max_mip_level=None, 
                       use_normalized=False, device='cuda'):
    raw_mesh, frames = load_obj_and_pose(in_obj, in_pose_json, lrm_mode=lrm_mode, device=device)
    print('load obj and pose done. begin render..')
    if use_normalized:
        tf = try_mesh_normalized(raw_mesh)
        print('tf', tf)
    
    uv_tex = torch.tensor(np.array(in_kd_pil) / 255., device=device).to(torch.float32)
    raw_mesh.material['kd'] = texture.Texture2D(uv_tex)

    glctx = dr.RasterizeCudaContext()
    
    ### common geomtry 
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx
    mvp = frames['mvp']

    vtx_pos = vtx_pos.to(torch.float32)
    mvp = mvp.to(torch.float32)
    vtx_uv = vtx_uv.to(torch.float32)
    print('vtx_pos ', vtx_pos.dtype)
    print('mvp ', mvp.dtype)
    print('vtx_uv ', vtx_uv.dtype)
    print('uv_tex ', uv_tex.dtype)
    tex_data = uv_tex.unsqueeze(0)
    color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data, render_res, max_mip_level)
         
    return color, alpha 

       
def save_render(color, alpha, out_path=None, bg_type="white", row=1, save_res=None):
    if save_res is None:
        save_res = color.shape[1]
    vis_pil = []
    if color is not None and torch.is_tensor(color):
        color_masked = mix_rgba_bg(color, alpha, bg_type=bg_type)
        opt_vis = [
            Image.fromarray(
                np.clip(np.rint(color_masked[idx].cpu().numpy() * 255.0), 0,
                        255).astype(np.uint8)).resize((save_res, save_res)).convert('RGB')
            for idx in range(color_masked.shape[0])
        ]
        vis_pil = [opt_vis[i:i + (len(opt_vis) // row)] for i in range(0, len(opt_vis), len(opt_vis) // row)]
        # new_lst = [lst[i:i + (len(lst) // r)] for i in range(0, len(lst), len(lst) // r)]

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output_image = concatenate_images_2d(vis_pil, out_img_path=out_path)    
    return output_image