import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import random
import nvdiffrast.torch as dr
import cv2
import torch.nn.functional as F
import sys

current_script_path = os.path.abspath(__file__)
project_root = (os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

# from dataset.utils_dataset import parse_objs_json, load_json, save_json, save_lines
# from scripts.utils_pool_cmds import run_commands_in_parallel
from render.render_mesh import  parse_pose_json, render_texture_views, render_depth_views
from render.mesh import load_mesh, Mesh
from render.obj import write_obj
from render.uv_conditions import mesh_xatlas
import render.texture as texture
from render.geom_utils import mesh_normalized, clean_decimate_mesh
from render import util

from dataset.utils_dataset import concatenate_images_2d

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)  #T * origin

def load_pose_npy(pose_npy, device='cuda', select_view=[]):
    """load render pose from json

    Args:
        pose_npy: mesh_*_pose.json
        lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
        select_view: _description_. Defaults to [].

    Returns:
        mvp [nv, 4, 4] = proj * mv, proj(intri) is relative about fovy and aspect, not about resolution
        w2c [nv, 4, 4] 
    """
    pose_np = np.load(pose_npy)  # [8, 16=12+4]
    c2w_nv, intrinsic_nv = pose_np[:, :12], pose_np[:, 12:]
    mvp_list, w2c_list = [], []
    for idx in range(c2w_nv.shape[0]):
        c2w = c2w_nv[idx]
        intrinsic = intrinsic_nv[idx]
        
        c2w = (np.concatenate((c2w, [0, 0, 0, 1.]))).reshape(4, 4)
        
        # print('debug c2w', c2w)
        # print('debug intrinsic', intrinsic)
        fovy = 2 * np.arctan(0.5 / intrinsic[0])
        proj = util.perspective(fovy, 1, 0.01, 100)
        print('debug fovy ', fovy)

        w2c = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32))

        # Load modelview matrix
        # mv = torch.linalg.inv(torch.tensor((c2w), dtype=torch.float32))
        mv = torch.linalg.inv(torch.tensor(opencv_to_blender(c2w), dtype=torch.float32))
        
        mvp = proj @ mv
        
        mvp_list.append(mvp[None, ...])
        w2c_list.append(w2c[None, ...])
                        
    mvp = torch.cat(mvp_list, dim=0).to(device)
    w2c = torch.cat(w2c_list, dim=0).to(device)
    if select_view and len(select_view) > 0:
        view_cnt = mvp.shape[0]
        filtered_list = [x for x in select_view if x < view_cnt]
        mvp, w2c = mvp[filtered_list], w2c[filtered_list]
    return mvp, w2c

def round_to_multiple_of_8(x):
    return ((x + 7) // 8) * 8

def load_and_resize(img):
    image = Image.open(img).convert("RGB")
    width, height = image.size
    new_width = round_to_multiple_of_8(width)
    new_height = round_to_multiple_of_8(height)
    resized_image = image.resize((new_width, new_height))    
    return resized_image

def load_lrm_color(in_color_dir, device='cuda'):
    """load image tensor from dir, in [0, 1]

    Args:
        in_color_dir: _description_

    Returns:
        tensor [b, h, w, 3] in [0, 1]
    """
    imgs = glob.glob(os.path.join(in_color_dir, f'mesh_*_*.png'))
    if not imgs:
        print(f'ERROR can not find any img in {in_color_dir} ')
        return None
    imgs = sorted(imgs)
    print('debug imgs ', imgs)

    images_pil = [load_and_resize(img) for img in imgs]
    numpy_arrays = [np.array(img) for img in images_pil]
    # tensor [b, h, w, 3] in [0, 1]
    images = torch.from_numpy(np.stack(numpy_arrays)).to(device)
    images = images.to(torch.float32) / 255.
    return images


def resize_render(in_tensor, new_h, new_w):
    """resize bhwc tensor

    Args:
        in_tensor: b, h, w, c
        new_h, new_w

    Returns:
        b, h, w, c
    """
    in_tensor = in_tensor.permute(0, 3, 1, 2)
    in_tensor = F.interpolate(in_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    in_tensor = in_tensor.permute(0, 2, 3, 1).contiguous()
    return in_tensor

def tensor_to_pils(data, vis_res=512):
    """_summary_

    Args:
        data: [b, h, w, 3]
        vis_res: _description_. Defaults to 512.

    Returns:
        list of pil
    """
    pils = [
        Image.fromarray(
            np.clip(np.rint(data[idx].detach().cpu().numpy() * 255.0), 0,
                    255).astype(np.uint8)).resize((vis_res, vis_res))
        for idx in range(data.shape[0])
    ]
    return pils

def load_lrm_input(in_npy, device='cuda'):
    """load albedo, normal, xyz, as [b, h, w, 3] * 3

    Args:
        in_npy: _description_
        device: _description_. Defaults to 'cuda'.

    Returns:
        _description_
    """
    data_np = np.load(in_npy)  # (8, 3, 266, 266)
    data_tensor = torch.from_numpy(data_np).to(device)
    h, w = data_tensor.shape[-2:]
    assert h == w
    new_h = round_to_multiple_of_8(h)
    
    debug_vis = False
    if debug_vis:
        vis_h = h
        # data_tensor = F.interpolate(data_tensor, size=(vis_h, vis_h), mode="bilinear", align_corners=False)
        vis_tensor = data_tensor.permute(0, 2, 3, 1).contiguous() # [8, h, w, 9]
        albedo, normal, xyz = torch.chunk(vis_tensor, chunks=3, dim=-1)
        mask = (xyz < 1).float()
        masked_rgb = albedo * mask  # [8, 3, h, w]
        out_dir = "/aigc_cfs/weizhe/code/check/MV_LRM/LRM/outputs/mvlrm/output_val_images_step_61000/mesh_26/bake"
        pils = tensor_to_pils(masked_rgb, vis_res=vis_h)
        albedo_pils = tensor_to_pils(albedo, vis_res=vis_h)
        xyz_pils = tensor_to_pils(xyz, vis_res=vis_h)
        concatenate_images_2d([albedo_pils, xyz_pils, pils], os.path.join(out_dir, "mask_rgb.jpg"))
        
    data_tensor = F.interpolate(data_tensor, size=(new_h, new_h), mode="bilinear", align_corners=False)
    data_tensor = data_tensor.permute(0, 2, 3, 1).contiguous() # [8, 272, 272, 9]
    return data_tensor
    # albedo, normal, xyz = torch.chunk(data_tensor, chunks=3, dim=-1)
    
    # return albedo, normal, xyz

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



def vis_ref_opt(images_ref, alpha_opt, color_opt, out_img_path, vis_res=512):
    vis_ref = images_ref * alpha_opt
    vis_opt = color_opt * alpha_opt
    
    opt_vis_ref = tensor_to_pils(vis_ref, vis_res)
    opt_vis_opt = tensor_to_pils(vis_opt, vis_res)
    vis_pil = [opt_vis_ref, opt_vis_opt]
    concatenate_images_2d(vis_pil, out_img_path)
    return


def mask_refine(depth, infer_images, raw_alpha, thres=180, k=20):
    """dilate and erode, Reduce white edge

    Args:
        depth: tensor [b, w, h, 1]
        infer_images: tensor [b, w, h, 3]
        raw_alpha: tensor [b, w, h, 1]
        thres: binary threshold. Defaults to 180.
        k: kernel size. Defaults to 20.

    Returns:
        infer_images_tmp: masked tensor [b, h, w, 3]
        raw_alpha: new alpha tensor [b, w, h, 1]
    """
    print('depth', depth.shape)
    [b, w, h, c] = depth.shape

    kernel = np.ones((k, k), dtype=np.uint8)
    mask_list_torch = []
    for i in range(b):
        depth_tmp = depth[i, :, :, :]
        depth_tmp = depth_tmp * 100
        depth_tmp = depth_tmp.cpu().numpy()
        depth_tmp.astype(np.uint8)
        ret, depth_tmp = cv2.threshold(depth_tmp, 0, 255, cv2.THRESH_BINARY)
        depth_dilate = cv2.dilate(depth_tmp, kernel, 1)
        depth_erode = cv2.erode(depth_tmp, kernel, 1)

        mask = np.abs(depth_dilate - depth_erode)
        mask_list_torch.append(torch.from_numpy(mask))
    mask_torch = torch.stack(mask_list_torch).cuda()
    print('mask_torch', mask_torch.shape)
    print('infer_images', infer_images.shape)
    print('raw_alpha', raw_alpha.shape)

    # [b, h, w, 3]
    infer_images_tmp = infer_images * raw_alpha

    scale_value = thres / 255.0
    infer_000 = infer_images_tmp[:, :, :, 0]
    infer_001 = infer_images_tmp[:, :, :, 1]
    infer_002 = infer_images_tmp[:, :, :, 2]

    infer_mask_0 = infer_000 > scale_value
    infer_mask_1 = infer_001 > scale_value
    infer_mask_2 = infer_002 > scale_value

    # [b, h, w]
    mask = infer_mask_0 & infer_mask_1 & infer_mask_2 & (mask_torch > 0)

    infer_images_tmp[mask] = 0
    raw_alpha[mask] = 0

    return infer_images_tmp, raw_alpha


def bake_tex_from_images(
    in_mesh,
    images,
    mvp_all,
    w2c_all,
    tex_res=512,
    vis_dir=None,
    use_raw_tex=True,
    use_erode=True,
    xyz_maps=None,
):
    """bake texture with optimization

    Args:
        in_mesh: class Mesh
        images: [nv, h, w, 3] in [0, 1], gt ref image
        mvp_all: [nv, 4, 4]
        w2c_all: _description_
        tex_res: _description_. Defaults to 512.
        vis_dir
        use_raw_tex: if True, use raw texture, else init tex with gray
        use_erode: if True, dilate and erode the ref image. because geom not match rgb
        xyz_maps: if valid and same shape as images, use this mask

    Returns:
        _description_
    """
    optim_cfg = {"lr_base": 0.01, "lr_ramp": 0.1, "max_iter": 500, "batch_size": 8, "max_mip_level": 4}

    if mvp_all.shape[0] != images.shape[0] or mvp_all.shape[0] != w2c_all.shape[0]:
        print(f'ERROR mvp_all.shape {mvp_all.shape[0]} != images {images.shape[0]} failed')
        return None

    # 1. init
    if use_raw_tex:
        raw_kd = in_mesh.material['kd'].data.clone().detach()
        print('debug raw_kd ', raw_kd.shape)
        resized_kd = resize_render(raw_kd, tex_res, tex_res)
        print('debug resized_kd ', resized_kd.shape)
        tex_data_opt = torch.nn.Parameter(resized_kd, requires_grad=True)
    else:  
        tex_data_opt = init_opt_tex(tex_res, background='gray', device=images.device)
    optimizer = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x) / float(optim_cfg['max_iter'])))
    glctx = dr.RasterizeCudaContext()
    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx  # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]

    # 2. optim
    optim_render_res = images.shape[1]
    depth_bhwc = render_depth_views(glctx, vtx_pos, pos_idx, mvp_all, w2c_all, optim_render_res)
    raw_alpha = depth_bhwc > 0   # [b, render_res, render_res, 1]
    if xyz_maps is not None and xyz_maps.shape == images.shape:
        raw_alpha = xyz_maps < 1
        print('debug use xyz_maps as mask')
    elif use_erode:
        _, raw_alpha = mask_refine(depth_bhwc, images, raw_alpha, thres=180, k=100)
    raw_alpha = raw_alpha.float()

    for iter in range(optim_cfg['max_iter']):
        batch = min(optim_cfg['batch_size'], mvp_all.shape[0])
        views = random.sample(range(mvp_all.shape[0]), batch)
        mvp, w2c = mvp_all[views], w2c_all[views]
        images_iter = images[views]
        alpha = raw_alpha[views]

        # [b, optim_render_res, optim_render_res, 3/1]
        color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt,
                                                    optim_render_res, optim_cfg['max_mip_level'])

        if vis_dir is not None and (iter == 0 or iter == (optim_cfg['max_iter']-1)):
            vis_ref_opt(images_iter, alpha, color_opt, os.path.join(vis_dir, f"vis_{iter}.jpg"))
            vis_ref_opt(images_iter, torch.ones_like(alpha), color_opt, os.path.join(vis_dir, f"vis_debug_{iter}.jpg"))

        weight = torch.ones_like(color_opt)
        loss = torch.mean(weight * ((images_iter * alpha - color_opt * alpha)**2))  # L2 pixel loss.
        optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

    # 4. generate new texture map and new obj
    new_mesh = in_mesh.clone()
    new_mesh.material = in_mesh.material
    uv_tex = (tex_data_opt.data).clone().squeeze(0)
    new_mesh.material['kd'] = texture.Texture2D(uv_tex)

    return new_mesh


def main_bake(
    in_obj_path,
    out_obj_dir,
    re_atlas=True,
    tex_res=512,
    decimate_target=10000,
    keep_raw=False,
):
    if not os.path.exists(in_obj_path):
        print(f"ERROR can not find in_obj_path {in_obj_path}")
        return

    in_dir = os.path.dirname(in_obj_path)

    in_pose_npy = os.path.join(in_dir, "mesh_pose.npy")
    in_data_npy = os.path.join(in_dir, "mesh_input.npy")
  
    if not os.path.exists(in_pose_npy) or not os.path.exists(in_data_npy):
        print(f"ERROR can not find in_pose_npy {in_pose_npy} or in_data_npy {in_data_npy}")
        return
    mvp, w2c = load_pose_npy(in_pose_npy, device='cuda')
    images = load_lrm_input(in_data_npy, device='cuda')
    print('debug images ', images.shape)
    
    raw_mesh = load_mesh(in_obj_path, mtl_override=None, skip_mtl=False)
    print('debug raw mtl ', raw_mesh.material['kd'].data.shape)

    if not keep_raw:
        mesh_normalized(raw_mesh)

    if not decimate_target is None and decimate_target > 0:
        raw_mesh = clean_decimate_mesh(raw_mesh, decimate_target=decimate_target)

    if re_atlas:
        in_mesh = mesh_xatlas(raw_mesh)
        in_mesh.material = raw_mesh.material
    else:
        in_mesh = raw_mesh

    print('begin render optim')
    new_mesh = bake_tex_from_images(in_mesh, images, mvp, w2c, tex_res=tex_res, vis_dir=out_obj_dir)

    write_obj(out_obj_dir, new_mesh)
    return


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='bake to one uv tex')
    parser.add_argument('in_obj_path', type=str)
    parser.add_argument('out_obj_dir', type=str)
    parser.add_argument('--tex_res', type=int, default=1024)
    parser.add_argument('--decimate_target', type=int, default=10000, help="if > 0, decimate mesh")
    parser.add_argument("--keep_raw", action="store_true", default=False,
                        help="if keep_raw, not mesh_normalized!TODO")    
    # /aigc_cfs_2/sz/proj/tex_cq/scripts/utils_pool_cmds.py
    args = parser.parse_args()

    re_atlas = False
    main_bake(args.in_obj_path,
              args.out_obj_dir,
              re_atlas=re_atlas,
              tex_res=args.tex_res,
              decimate_target=args.decimate_target,
              keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
