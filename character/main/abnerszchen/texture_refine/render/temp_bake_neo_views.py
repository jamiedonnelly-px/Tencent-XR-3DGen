import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import random
import nvdiffrast.torch as dr

import sys

current_script_path = os.path.abspath(__file__)
project_root = (os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

# from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
# from scripts.utils_pool_cmds import run_commands_in_parallel
from render.render_mesh import (
    parse_pose_json,
    render_texture_views,
    render_normal_cos,
    render_depth_views,
    mix_rgba,
    resize_render,
)
from render.bake_utils import inpaint_refine_uv_tex
from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj
from render.uv_conditions import mesh_xatlas
import render.texture as texture
from render.geom_utils import mesh_normalized, mesh_normalized_by_txt, clean_decimate_mesh

from dataset.utils_dataset import concatenate_images_2d

def load_pose_json(pose_json, lrm_mode, device='cuda', select_view=[]):
    """load render pose from json

    Args:
        pose_json: data/cams/cam_parameters_select.json
        lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
        select_view: _description_. Defaults to [].

    Returns:
        mvp [nv, 4, 4] = proj * mv, proj(intri) is relative about fovy and aspect, not about resolution
        w2c [nv, 4, 4] 
    """
    frames = parse_pose_json(pose_json, lrm_mode=lrm_mode)
    mvp = frames['mvp'].to(device)
    w2c = frames['w2c'].to(device)
    if select_view and len(select_view) > 0:
        view_cnt = mvp.shape[0]
        filtered_list = [x for x in select_view if x < view_cnt]
        mvp, w2c = mvp[filtered_list], w2c[filtered_list]
    return mvp, w2c


def load_render_color(in_color_dir, device='cuda'):
    """load image tensor from dir, in [0, 1]

    Args:
        in_color_dir: _description_

    Returns:
        tensor [b, h, w, 3] in [0, 1]
    """
    imgs = glob.glob(os.path.join(in_color_dir, f'cam-*.png'))
    if not imgs:
        print(f'ERROR can not find any img in {in_color_dir} ')
        return None
    imgs = sorted(imgs)

    images_pil = [Image.open(img).convert("RGB") for img in imgs]
    numpy_arrays = [np.array(img) for img in images_pil]
    # tensor [b, h, w, 3] in [0, 1]
    images = torch.from_numpy(np.stack(numpy_arrays)).to(device)
    images = images.to(torch.float32) / 255.
    return images


def init_gray_tex(tex_res, out_img_path):
    numpy_array = np.ones((tex_res, tex_res, 3)) * 0.5
    numpy_array = (numpy_array * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    Image.fromarray(numpy_array).save(out_img_path)


def init_opt_tex(tex_res, background='gray', device='cuda'):
    if background == 'gray':
        tex = torch.ones((1, tex_res, tex_res, 3)).to(device) * 0.5
    elif background == 'black':
        tex = torch.zeros((1, tex_res, tex_res, 3)).to(device)
    elif background == 'white':
        tex = torch.ones((1, tex_res, tex_res, 3)).to(device)
    else:
        print('invalid init_opt_tex background ', background)
        return None
    tex_merge = torch.nn.Parameter(tex.clone(), requires_grad=True)
    return tex_merge


def tensor_to_pils(data, vis_res=512):
    pils = [
        Image.fromarray(
            np.clip(np.rint(data[idx].detach().cpu().numpy() * 255.0), 0,
                    255).astype(np.uint8)).resize((vis_res, vis_res))
        for idx in range(data.shape[0])
    ]
    return pils

def vis_ref_opt(images_ref, viwecos, alpha_opt, color_opt, out_img_path, vis_res=512):
    viwecos = viwecos.repeat_interleave(3, dim=-1)
    
    vis_ref = images_ref * alpha_opt
    vis_viwecos = viwecos * alpha_opt
    vis_opt = color_opt * alpha_opt
    
    opt_vis_ref = tensor_to_pils(vis_ref, vis_res)
    opt_vis_viwecos = tensor_to_pils(vis_viwecos, vis_res)
    opt_vis_opt = tensor_to_pils(vis_opt, vis_res)
    vis_pil = [opt_vis_ref, opt_vis_viwecos, opt_vis_opt]
    concatenate_images_2d(vis_pil, out_img_path)
    return

def bake_tex_from_images(in_mesh, images, mvp_all, w2c_all, tex_res=512, vis_dir=None, temp_listcnt=1):
    """_summary_

    Args:
        in_mesh: class Mesh
        images: [nv, h, w, 3] in [0, 1]
        mvp_all: [nv, 4, 4]
        w2c_all: _description_
        tex_res: _description_. Defaults to 512.

    Returns:
        _description_
    """
    optim_cfg = {
        "lr_base": 0.01,
        "lr_ramp": 0.1,
        "max_iter": 500,
        "batch_size": 10,
        "max_mip_level": 4,
        "mask_sp": 4,
        "optim_render_res": 1024,
        "uv_refine_type": "pos_knn",    # raw, uv_knn, or pos_knn
        "min_cos": -1,    # normal-view cos
    }

    if mvp_all.shape[0] != images.shape[0] or mvp_all.shape[0] != w2c_all.shape[0]:
        print(f'ERROR mvp_all.shape {mvp_all.shape[0]} != images {images.shape[0]} failed')
        return None
    print("optim_cfg ", optim_cfg)

    # 1. init
    tex_data_opt = init_opt_tex(tex_res, background='gray', device=images.device)
    optimizer = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x) / float(optim_cfg['max_iter'])))
    glctx = dr.RasterizeCudaContext()

    in_mesh = auto_normals(in_mesh)
    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx  # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]
    vtx_normal, normal_idx = in_mesh.v_nrm, in_mesh.t_nrm_idx # [Nv, 3], [Nf, 3]

    # 2. optim
    optim_render_res = optim_cfg['optim_render_res']
    if images.shape[1] != optim_render_res:
        images = resize_render(images, optim_render_res, optim_render_res)

    # [nv, resolution, resolution, 1]
    viwecos_all = render_normal_cos(glctx, vtx_pos, pos_idx, mvp_all, vtx_normal, w2c_all, optim_render_res)
    min_cos = optim_cfg.get("min_cos", 0.01)
    if min_cos > 0:
        viwecos_mask_all = (viwecos_all > min_cos).float()
    else:
        viwecos_mask_all = None
    
    print('temp_listcnt ', temp_listcnt)
    for iter in range(optim_cfg['max_iter']):
        batch = min(optim_cfg['batch_size'], mvp_all.shape[0])
        # views = random.sample(range(mvp_all.shape[0]), batch)
        if temp_listcnt == 0:
            views = [0, 1, 2, 3, 4, 5, 6, 7]
        elif temp_listcnt == 1:
            views = [0, 6, 20, 21, 22, 23, 24, 25]
        elif temp_listcnt == 2:
            views = [0, 6, 26, 21, 22, 23, 24, 27]
        elif temp_listcnt == 3:
            views = [0, 6, 28, 21, 22, 23, 24, 29]
        elif temp_listcnt == 4:
            views = [0, 6, 30, 21, 22, 23, 24, 31]
        
        mvp, w2c = mvp_all[views], w2c_all[views]
        images_iter = images[views]
        viwecos_iter = viwecos_all[views]

        # [b, optim_render_res, optim_render_res, 3/1]
        color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt,
                                                    optim_render_res, optim_cfg['max_mip_level'])

        if min_cos > 0:
            alpha_opt = viwecos_mask_all[views] * alpha_opt

        if vis_dir is not None and (iter == 0 or iter == (optim_cfg['max_iter']-1)):
            vis_ref_opt(images_iter, viwecos_iter, alpha_opt, color_opt, os.path.join(vis_dir, f"vis_{iter}.jpg"))
            vis_ref_opt(images_iter, viwecos_iter, torch.ones_like(alpha_opt), color_opt, os.path.join(vis_dir, f"aavis_{iter}.jpg"))

        weight = torch.ones_like(color_opt)
        loss = torch.mean(weight * ((images_iter * alpha_opt - color_opt * alpha_opt)**2))  # L2 pixel loss.
        optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # if iter == max(10, optim_cfg['max_iter'] - 20):
        #     # refine texture uv
        #     uv_refine_type = optim_cfg.get("uv_refine_type", "raw")
        #     if uv_refine_type != "raw":
        #         inpaint_refine_uv_tex(
        #             glctx,
        #             in_mesh,
        #             tex_data_opt,
        #             mvp_all,
        #             uv_refine_type=uv_refine_type,
        #             viwecos_mask_all=viwecos_mask_all,
        #             mask_sp=optim_cfg.get("mask_sp", 4),
        #             debug_dir=vis_dir,
        #         )
        #         print('debug update tex')

    # 4. generate new texture map and new obj
    new_mesh = in_mesh.clone()
    new_mesh.material = in_mesh.material

    uv_tex = (tex_data_opt.data).clone().squeeze(0) # [h,w,3]
    new_mesh.material['kd'] = texture.Texture2D(uv_tex)  
    return new_mesh


def main_bake(
    in_obj_path,
    in_color_dir,
    in_pose_json,
    out_obj_dir,
    re_atlas=True,
    tex_res=512,
    transformation_txt = "",
    decimate_target=10000,
    lrm_mode=False,
    keep_raw=False,
    temp_listcnt=1,
):
    if not os.path.exists(in_obj_path):
        print(f"ERROR can not find in_obj_path {in_obj_path}")
        return
    if not os.path.exists(in_pose_json):
        print(f"ERROR can not find in_pose_json {in_pose_json}")
        return

    images = load_render_color(in_color_dir)
    if images is None:
        print(f'ERROR load_render_color from {in_color_dir} failed')
        return
    mvp, w2c = load_pose_json(in_pose_json, lrm_mode=lrm_mode, device='cuda')

    raw_mesh = load_mesh(in_obj_path, mtl_override=None, skip_mtl=True)
    print('debug raw mtl ', raw_mesh.material['kd'].data.shape)

    if not keep_raw:
        if transformation_txt and os.path.exists(transformation_txt):
            print(f"use transformation_txt {transformation_txt}")
            mesh_normalized_by_txt(raw_mesh, transformation_txt)
        else:
            mesh_normalized(raw_mesh)

    if not decimate_target is None and decimate_target > 0:
        raw_mesh = clean_decimate_mesh(raw_mesh, decimate_target=decimate_target)

    if re_atlas:
        in_mesh = mesh_xatlas(raw_mesh)
        in_mesh.material = raw_mesh.material
    else:
        in_mesh = raw_mesh

    print('begin render optim')
    new_mesh = bake_tex_from_images(in_mesh, images, mvp, w2c, tex_res=tex_res, vis_dir=out_obj_dir, temp_listcnt=temp_listcnt)

    write_obj(out_obj_dir, new_mesh)
    return


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='bake to one uv tex')
    parser.add_argument('in_obj_path', type=str)
    parser.add_argument('in_color_dir', type=str)
    parser.add_argument('in_pose_json', type=str)
    parser.add_argument('out_obj_dir', type=str)
    parser.add_argument('--tex_res', type=int, default=1024)
    parser.add_argument('--transformation_txt', type=str, default="")
    parser.add_argument('--decimate_target', type=int, default=10000, help="if > 0, decimate mesh")
    parser.add_argument("--lrm_mode", action="store_true", default=False,
                        help="use lrm mode. temp. need remove in future!TODO")    
    parser.add_argument("--keep_raw", action="store_true", default=False,
                        help="if keep_raw, not mesh_normalized!TODO")    
    parser.add_argument('--temp_listcnt', type=int, default=1, help="if > 0, decimate mesh")
    # /aigc_cfs_2/sz/proj/tex_cq/scripts/utils_pool_cmds.py
    args = parser.parse_args()

    re_atlas = True    # TODO
    main_bake(args.in_obj_path,
              args.in_color_dir,
              args.in_pose_json,
              args.out_obj_dir,
              re_atlas=re_atlas,
              tex_res=args.tex_res,
              transformation_txt=args.transformation_txt,
              decimate_target=args.decimate_target,
              lrm_mode=args.lrm_mode,
              keep_raw=args.keep_raw,
              temp_listcnt=args.temp_listcnt)


if __name__ == "__main__":
    main()
