
import os
import torch
import math
import numpy as np
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    HardFlatShader,
    TexturesVertex
)
from PIL import Image
import nvdiffrast.torch as dr

import sys
current_script_path = os.path.abspath(__file__)
project_root = (os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

import camera_util
from render.mesh import load_mesh, Mesh
from render.render_mesh import  render_texture_views, render_depth_views
from render.mesh import load_mesh, Mesh
from render.obj import write_obj
import render.texture as texture
from dataset.utils_dataset import concatenate_images_2d

obj_file = "/aigc_cfs_2/sz/proj/other/zero123plus/data/face_ly/render/normalized_blender.obj"
out_dir = "/aigc_cfs_2/sz/proj/other/zero123plus/data/face_ly/render/debug_pose"
os.makedirs(out_dir, exist_ok=True)

def save_pose_npy(pose_npy, device="cuda"):
    """_summary_

    Args:
        obj_file: _description_
        image_size: _description_. Defaults to 320.
        device: _description_. Defaults to "cuda".

    Returns:
        images, depths, [b, h, w, 3/1] tensor
    """
    # elevations = torch.tensor([0, 60, 0])
    # azimuths = torch.tensor([0, 30, 90])
    
    elevations = torch.tensor([30, -20, 30, -20, 30, -20])
    azimuths = torch.tensor([30, 90, 150, 210, 270, 330]) 
    
    dists = torch.tensor([1.6] * elevations.shape[0])
    
    R, T = look_at_view_transform(dist=dists, elev=elevations, azim=azimuths)
    fov = np.rad2deg((math.atan(16 / 35) * 2))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    # camera_util.make_pose_npy_as_instant(elevations, azimuths, dists, fov, pose_npy)
    camera_util.make_pose_npy_as_instant_from_cameras(cameras, pose_npy)
    return cameras



def load_pose_npy(pose_npy, device='cuda', is_opencv_pose=False, select_view=[]):
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
        proj = camera_util.perspective(fovy, 1, 0.01, 100)
        print('debug fovy ', fovy * 180. / np.pi)

        w2c = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32))

        # Load modelview matrix
        if is_opencv_pose:
            mv = torch.linalg.inv(torch.tensor(camera_util.opencv_to_blender(c2w), dtype=torch.float32))
        else:
            mv = torch.linalg.inv(torch.tensor((c2w), dtype=torch.float32))
        
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

def vis_ref_opt(images_ref, alpha_opt, color_opt, out_img_path, vis_res=512):
    vis_ref = images_ref * alpha_opt
    vis_opt = color_opt * alpha_opt
    
    opt_vis_ref = tensor_to_pils(vis_ref, vis_res)
    opt_vis_opt = tensor_to_pils(vis_opt, vis_res)
    vis_pil = [opt_vis_ref, opt_vis_opt]
    return concatenate_images_2d(vis_pil, out_img_path)


def vis_bhwc(rgb, alpha, out_img_path, vis_res=512):
    """_summary_

    Args:
        rgb: [b, c, h, 3]
        alpha: [b, c, h, 1]
        out_img_path: _description_
        vis_res: _description_. Defaults to 512.

    Returns:
        _description_
    """
    vis_ref = rgb * alpha
    
    opt_vis_ref = tensor_to_pils(vis_ref, vis_res)
    vis_pil = [opt_vis_ref]
    return concatenate_images_2d(vis_pil, out_img_path)
    

def render_py3d(obj_file, cameras, out_img_dir, image_size=512, device="cuda"):
    mesh = load_objs_as_meshes([obj_file], device=torch.device(device))
    raster_settings = RasterizationSettings(image_size=image_size)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardFlatShader(device=torch.device(device), cameras=cameras)
    )
        
    lights = PointLights(device=device, location=[[1.0, 1.0, 1.0]])

    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
        # blur_radius=0.0,
        # bin_size=0
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardFlatShader(device=torch.device(device), cameras=cameras)
    )
    
    # b, h, w, 4(3+1alpha)
    images = renderer(mesh.extend(len(cameras.R)), cameras=cameras, lights=lights)
    # b, h, w, 1
    depths = renderer.rasterizer(mesh.extend(len(cameras.R)), cameras=cameras).zbuf
    print('py3d images ', images.shape)
    print('py3d depths ', depths.shape)
    os.makedirs(out_img_dir, exist_ok=True)
    cat_pil = vis_bhwc(images[..., :3], images[..., -1:], os.path.join(out_img_dir, f"rgb.jpg"))
    # vis_ref_opt(images, torch.ones_like(alpha_opt), color_opt, os.path.join(out_img_dir, f"vis_a1.jpg"))   
    return cat_pil

def render_with_pose_npy(obj_file, pose_npy, out_img_dir, render_res=320):
    mvp, w2c = load_pose_npy(pose_npy)
    in_mesh = load_mesh(obj_file, mtl_override=None, skip_mtl=False)

    glctx = dr.RasterizeCudaContext()
    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx  # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]

    # ##### z up
    # R_debug = torch.tensor([[0, 1, 0],
    #                          [0, 0, 1],
    #                          [1, 0, 0]]).to(vtx_pos.device).float()
    # vtx_pos = torch.matmul(vtx_pos, R_debug)
    
    # 2. optim
    depth_bhwc = render_depth_views(glctx, vtx_pos, pos_idx, mvp, w2c, render_res)
    print('depth_bhwc ', depth_bhwc.shape)

    tex_data_opt = torch.nn.Parameter(in_mesh.material['kd'].data.clone().detach(), requires_grad=True)    
    color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp,
                                                tex_data_opt,
                                                render_res, max_mip_level=2)    
    print('color_opt ', color_opt.shape)
    os.makedirs(out_img_dir, exist_ok=True)
    cat_pil = vis_ref_opt(color_opt, alpha_opt, color_opt, os.path.join(out_img_dir, f"vis.jpg"))
    vis_ref_opt(color_opt, torch.ones_like(alpha_opt), color_opt, os.path.join(out_img_dir, f"vis_a1.jpg"))    
    return cat_pil

def debug_pose_render(obj_file, out_dir, render_res=320):
    os.makedirs(out_dir, exist_ok=True)
    pose_npy = os.path.join(out_dir, "mesh_pose.npy")

    cameras = save_pose_npy(pose_npy)
    
    out_img_dir = os.path.join(out_dir, "render_img")
    nv_pil = render_with_pose_npy(obj_file, pose_npy, out_img_dir, render_res=render_res)
    
    out_img_dir_py3d = os.path.join(out_dir, "render_py3d")
    py3d_pil = render_py3d(obj_file, cameras, out_img_dir_py3d, image_size=render_res)
    
    concatenate_images_2d([[nv_pil], [py3d_pil]], os.path.join(out_dir, "merge.png"))
    print(f"save merge img to {os.path.join(out_dir, 'merge.png')}")




debug_pose_render(obj_file, out_dir)

