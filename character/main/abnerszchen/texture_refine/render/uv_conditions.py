import os
import sys
import torch
import xatlas
import numpy as np
import nvdiffrast.torch as dr
import time
import PIL

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj
from render.geom_utils import try_mesh_normalized

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

def resize_render_tensor(in_tensor, new_size):
    """resize bhwc tensor

    Args:
        in_tensor: b, h, w, c
        new_size: [h, w]

    Returns:
        b, h, w, c
    """
    in_tensor = in_tensor.permute(0, 3, 1, 2)
    in_tensor = torch.nn.functional.interpolate(in_tensor, new_size, mode="bilinear", align_corners=False)
    in_tensor = in_tensor.permute(0, 2, 3, 1).contiguous()
    return in_tensor

def render_geometry_uv(ctx, mesh : Mesh, resolution, ssaa=-1):
    """render geometry uv

    Args:
        ctx: RasterizeCudaContext
        mesh(mesh.Mesh): class Mesh
        resolution(int): render uv res
        ssaa: super-sample anti-a

    Returns:
        gb_geom(xyz), gb_normal([-1, 1]), gb_mask(0 or 1). all are [h, w, x=3/3/1] tensor
    """
    render_res = ssaa * resolution if ssaa > 1 else resolution
    render_res = min(render_res, 2048)
    
    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), (render_res, render_res))

    # Interpolate world space position [1, h, w, 3] and mask [1, h, w, 1]
    gb_geom, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    if isinstance(mesh.v_nrm, torch.Tensor):
        pass
    elif mesh.v_nrm is None:
        mesh = auto_normals(mesh)

    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast, mesh.t_pos_idx.int())
    # gb_normal = (gb_normal + 1.0)*0.5   # [-1, 1] -> [0, 1] as img    
    gb_mask = (rast[..., 3:4] > 0)
    
    if ssaa > 1:
        gb_geom = resize_render_tensor(gb_geom, [resolution, resolution])
        gb_normal = resize_render_tensor(gb_normal, [resolution, resolution])
        gb_mask = resize_render_tensor(gb_mask.float(), [resolution, resolution])
    
    gb_mask = gb_mask.to(torch.uint8)
    return gb_geom.squeeze(0), gb_normal.squeeze(0), gb_mask.squeeze(0)

def cvt_xyz_to_disp(gb_xyz, gb_mask):
    
    return gb_disp

def print_object_attributes(obj):
    for attr_name in dir(obj):
        if not callable(getattr(obj, attr_name)) and not attr_name.startswith("__"):
            print(f"{attr_name}: {getattr(obj, attr_name)}")

def mesh_xatlas(imesh, vmap=False):
    """_summary_

    Args:
        imesh: Mesh class
    Return:
        new_mesh Mesh class with v_tex and t_tex_idx
   
    """
    device = imesh.v_pos.device
    v_pos = imesh.v_pos.detach().cpu().numpy()            # [N, 3]
    t_pos_idx = imesh.t_pos_idx.detach().cpu().numpy()    # [M, 3]     
        
    ### unwrap uvs
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_pos, t_pos_idx)
    chart_options = xatlas.ChartOptions()
    # print('debug chart_options ', chart_options)
    # print_object_attributes(chart_options)
    # chart_options.max_iterations = 5  # 0 disable merge_chart for faster unwrap...
    # chart_options.max_iterations = 10  # 0 disable merge_chart for faster unwrap...
    # chart_options.fix_winding = True
    # chart_options.max_cost = 50
    # chart_options.roundness_weight = 0.001
    # chart_options.normal_seam_weight = 2000
    pack_options = xatlas.PackOptions()
    # print('debug pack_options ', pack_options)
    # print_object_attributes(pack_options)
    # pack_options.max_chart_size = 0
    # # pack_options.blockAlign = True
    # pack_options.bruteForce = False
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    # atlas.generate(chart_options=chart_options)
    # `vmapping` contains the original vertex index for each new vertex (shape N_new, type uint32).
    # `indices` contains the vertex indices of the new triangles (shape F*3, type uint32)
    # `uvs` contains texture coordinates of the new vertices (shape N_new*2, type float32)    
    vmapping, indices, uvs = atlas[0]  
    
    ### Convert to tensors, numpy->torch
    # v_pos = v_pos[vmapping]
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    faces = torch.tensor(indices_int64, dtype=torch.int64, device=device)
    new_mesh = Mesh(torch.tensor(v_pos, dtype=torch.float32, device=device).contiguous(), 
                        torch.tensor(t_pos_idx, device=device).to(torch.long).contiguous(), 
                        v_tex=uvs, t_tex_idx=faces, material=None)    


    # vmap: remap geom v/f to texture vt/ft, make each v correspond to a unique vt
    if vmap:
        vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(device)
        new_mesh.v_pos = new_mesh.v_pos[vmapping]
        new_mesh.t_pos_idx = new_mesh.t_tex_idx
        
    # new_mesh.material = copy.deepcopy(imesh.material)
    return new_mesh

def obj_xatlas(in_obj, out_dir=None):
    imesh = load_mesh(in_obj, skip_mtl=True)
    
    new_mesh = mesh_xatlas(imesh)
    if out_dir is not None:
        write_obj(out_dir, new_mesh)
    return new_mesh

def save_mask_img(gb_mask : torch.Tensor, out_mask_img):
    os.makedirs(os.path.dirname(out_mask_img), exist_ok=True)
    
    mask_np = (gb_mask * 255).squeeze(-1).detach().cpu().numpy()
    PIL.Image.fromarray(mask_np, mode='L').save(out_mask_img)
    return

def cvt_geom_to_pil(gb_geom : torch.Tensor, gb_mask : torch.Tensor):
    """save geom [-1, 1] to [0, 1]

    Args:
        gb_geom (torch.Tensor): tensor geometry [H, W, 3] in [-1, 1], position or normal
        gb_mask (torch.Tensor): tensor [H, W, 1]

    Returns:
        PIL.Image: pil in [0, 255] h,w,3
    """
    mask_expanded = gb_mask.expand(-1, -1, 3)
    selected_pixels = torch.masked_select(gb_geom, mask_expanded == 1)
    gmin, gmax = torch.min(selected_pixels).item(), torch.max(selected_pixels).item()
    assert -1.1 <= gmin <= gmax <= 1.1, f"geom need normalized to [-1, 1]. xyz in [-1, 1] cube, normal in [-1, 1], but get [{gmin}, {gmax}]"
    gb_geom_pro = (gb_geom + 1.) / 2.
        
    gb_vis = (gb_geom_pro * gb_mask).cpu().numpy()   # [0, 1]
    vis_255 = np.clip(np.rint(gb_vis * 255.0), 0, 255).astype(np.uint8)
    pil = PIL.Image.fromarray(vis_255)
    return pil

def save_geom(gb_geom : torch.Tensor, gb_mask : torch.Tensor, out_dir, prefix='uv_pos'):
    """save tensor geometry([-1, 1]) to npy and [0, 1] pil

    Args:
        gb_geom: tensor geometry [H, W, 3] in [-1, 1], position or normal
        gb_mask: tensor [H, W, 1]
        out_dir: save dir
        prefix: save to out_dir/prefix.npy&png
    """
    os.makedirs(out_dir, exist_ok=True)
    pil = cvt_geom_to_pil(gb_geom, gb_mask)
    pil.save(os.path.join(out_dir, f'{prefix}.png'))
    
    # mask_expanded = gb_mask.expand(-1, -1, 3)
    # selected_pixels = torch.masked_select(gb_geom, mask_expanded == 1)
    # gmin, gmax = torch.min(selected_pixels).item(), torch.max(selected_pixels).item()
    # assert -1.1 <= gmin <= gmax <= 1.1, f"geom need normalized to [-1, 1]. xyz in [-1, 1] cube, normal in [-1, 1], but get [{gmin}, {gmax}]"
        
    # gb_vis = (((gb_geom + 1.) / 2.) * gb_mask).cpu().numpy()   # [0, 1]
    # vis_255 = np.clip(np.rint(gb_vis * 255.0), 0, 255).astype(np.uint8)
    # PIL.Image.fromarray(vis_255).save(os.path.join(out_dir, f'{prefix}.png'))
    
    gb_np = gb_geom.cpu().numpy()
    np.save(os.path.join(out_dir, f'{prefix}.npy'), gb_np)
        
    return


            
def render_uv_condition(in_obj_path, resolution, out_dir, use_normalized=True, ssaa=-1):
    if not os.path.exists(in_obj_path):
        print(f'can not find obj {in_obj_path}')
        return
    os.makedirs(out_dir, exist_ok=True)
    imesh : Mesh = load_mesh(in_obj_path)
    
    if use_normalized:
        transformation = try_mesh_normalized(imesh)
        np.save(os.path.join(out_dir, 'transformation.npy'), transformation.cpu().numpy())
        print('use_normalized')
    
    imesh = auto_normals(imesh)
    
    # TODO
    kd_data = imesh.material['kd'].data # [1, h, w, 3]
    # print('debug kd_data raw ', kd_data.shape)
    # TODO
    if resolution > -1:
        imesh.material['kd'].data = torch.nn.Parameter(resize_render_tensor(kd_data, [resolution, resolution]))
    else:
        resolution = 1024 # TODO
    # print('debug kd_data ', kd_data.shape)
    # print('debug imesh kd_data ', imesh.material['kd'].data.shape, imesh.material['kd'].data.device)
    write_obj(out_dir, imesh)

    glctx = dr.RasterizeCudaContext()
    
    gb_xyz, gb_normal, gb_mask = render_geometry_uv(glctx, imesh, resolution, ssaa=ssaa)
    
    # gb_disp = cvt_xyz_to_disp(gb_xyz, gb_mask)
    
    save_mask_img(gb_mask, os.path.join(out_dir, 'uv_mask.png'))
    save_geom(gb_xyz, gb_mask, out_dir, prefix='uv_pos')
    save_geom(gb_normal, gb_mask, out_dir, prefix='uv_normal')
    # save_geom(gb_xyz * 0.1 + gb_normal , gb_mask, out_dir, prefix='uv_sum')
    
    
    return

