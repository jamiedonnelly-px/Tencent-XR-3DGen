import os
import json

import torch
import numpy  as np
import nvdiffrast.torch as dr
import cv2
import torch.nn.functional as F
from render import util
from render.mesh import Mesh, auto_normals
from render import geom_utils
from render.grid_put import mipmap_linear_grid_put_2d

def get_tensor_range(data):
    return torch.min(data).item(), torch.max(data).item()

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)  #T * origin

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

def parse_pose_json(in_pose_json, lrm_mode=False):
    """read render pose json, get camera intr-extri

    Args:
        in_pose_json: like data/cams/cam_parameters_select.json
        lrm_mode: if lrm_mode, fx=fy=255.5, need rm in future!!!!!!!!!!!!!!!!!!!!!!!!

    Returns:
        frames, dict {'cam_name_list':cam_name_list, 'mv': mv, 'mvp': mvp, 'campos': campos}
        # cam_name_list [nv]
        # w2c [nv, 4, 4] 
        # mv [nv, 4, 4] 
        # mvp [nv, 4, 4] = proj * mv, proj(intri) is relative about fovy and aspect, not about resolution
        # campos [nv, 3]         
    """
    with open(os.path.join(in_pose_json), encoding='utf-8') as f:
        cam_dict = json.load(f)

    frames = []
    cam_name_list, w2c_list, mv_list, mvp_list, campos_list = [], [], [], [], []
    for cam_name, frame in cam_dict.items():
        k, c2w = np.array(frame['k']), np.array(frame['pose'])

        if lrm_mode:
            fovy = util.focal_length_to_fovy(k[0, 0], int((k[0, 2] + 0.5) * 2))
        else:
            fovy = util.focal_length_to_fovy(k[0, 0], int(k[0, 2] * 2))

        proj = util.perspective(fovy, 1, 0.01, 100)

        w2c_old = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32))  ##### TODO

        # Load modelview matrix. from json with opencv format
        mv = torch.linalg.inv(torch.tensor(opencv_to_blender(c2w), dtype=torch.float32))
        if lrm_mode:    # obja manifold mesh is y-up. render img and lrm out is z-up. rot to z-up
            mv = mv @ util.rotate_x(-np.pi / 2)  #TODO(csz) only about obvaverse, need rm in future!!!!!
        w2c = mv.clone()    #####
        campos = torch.linalg.inv(mv)[:3, 3]
        
        mvp = proj @ mv

        cam_name_list.append(cam_name)
        w2c_list.append(w2c[None, ...])
        mv_list.append(mv[None, ...])
        mvp_list.append(mvp[None, ...])
        campos_list.append(campos[None, ...])

    frames = {'cam_name_list':cam_name_list, 'w2c': torch.cat(w2c_list, dim=0), 'mv': torch.cat(mv_list, dim=0),
              'mvp': torch.cat(mvp_list, dim=0), 'campos': torch.cat(campos_list, dim=0)}

    return frames


def rotate_scene_mvp(frame_k, itr_all=30, cam_radius=2., lrm_mode=True, rot_x=-0.5):
    """only for eval, render rotate circle imgs
    """
    # Smooth rotation for display.
    ### debug face

    if lrm_mode:
        fovy = util.focal_length_to_fovy(frame_k[0, 0], int((frame_k[0, 2] + 0.5) * 2))
    else:
        fovy = util.focal_length_to_fovy(frame_k[0, 0], int(frame_k[0, 2] * 2))

    proj = util.perspective(fovy, 1, 0.01, 100)

    mvp_list = []
    for itr in range(itr_all):
        ang    = (itr / itr_all) * np.pi * 2
        mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(rot_x) @ util.rotate_y(ang))

        if not lrm_mode:
            mv = mv @ util.rotate_x(np.pi / 2)  #TODO(csz) only about obvaverse, need rm in future!!!!!
            # mv = mv @ util.rotate_x(-np.pi / 2)  #TODO(csz) only about obvaverse, need rm in future!!!!!
        mvp    = proj @ mv
        mvp_list.append(mvp[None, ...])

    return {
        'mvp': torch.cat(mvp_list, dim=0).cuda(),
    }


def transform_pos(v_pos, mtx_in):
    """pos_clip = mtx_in * p_w

    Args:
        v_pos: p_w [N, 3] or [1, N, 3] in cuda
        mtx_in: [b, 4, 4] in cuda

    Returns:
        pos_clip [b, N, 4]
    """
    if v_pos.dim() == 2:
        v_pos = v_pos[None, ...]
    posw = torch.nn.functional.pad(v_pos, pad=(0,1), mode='constant', value=1.0)
    return torch.matmul(posw, torch.transpose(mtx_in, 1, 2))

def util_merge_tex(tex, need_texs=['kd', 'normal']):
    """merge all type tex to one data

    Args:
        tex: Material or dict
        need_texs: _description_. Defaults to ['kd', 'normal'].
    Return:
        tex_data_merge: [1, ht, wt, x=3/9...]
        key_channel_se_pairs: list of (need_tex, channel_start, channel_end)
    """
    tex_datas, key_channel_se_pairs, channel_start = [], [], 0
    for need_tex in need_texs:
        if (need_tex == 'kd' or need_tex == 'ks' or need_tex == 'normal') and need_tex in tex:
            texture = tex[need_tex].data    # [1, ht, wt, x]
            tex_datas.append(texture)

            channel_end = channel_start + texture.shape[-1]
            key_channel_se_pairs.append((need_tex, channel_start, channel_end))
            channel_start = channel_end
    tex_data_merge = torch.cat(tex_datas, dim=-1)
    return tex_data_merge, key_channel_se_pairs

def mix_rgba(rgb, alpha, background):
    """mix rgba from rgb and alpha
    [alpha * rgb + (1-alpha) * bg, alpha]
    
    Args:
        rgb: [nv, h, w, 3]
        alpha: [nv, h, w, 1]
        background: [nv, h, w, 3]

    Returns:
        rgba [nv, h, w, 4]
    """
    background_rgba = torch.cat((background, torch.zeros_like(background[..., -1:])), dim=-1)
    rgba = torch.lerp(background_rgba, torch.cat((rgb, torch.ones_like(rgb[..., -1:])), dim=-1), alpha.float())
    return rgba

def mix_rgba_bg(rgb, alpha, bg_type="white"):
    if bg_type == 'black':
        background = torch.zeros_like(rgb)
    elif bg_type == 'white':
        background = torch.ones_like(rgb)
    elif bg_type == 'random':
        background = torch.rand_like(rgb)
    else:
        assert False, "Unknown background type %s" % bg_type
    rgba = mix_rgba(rgb, alpha, background)
    return rgba

def render_depth_views(glctx, vtx_pos, pos_idx, mvp, w2c, resolution):
    """render depth with multi views 
    Args:
        glctx: dr.RasterizeCudaContext()
        vtx_pos: [Nv, 3] xyz
        pos_idx: [Nf, 3] face
        mvp: [nv, 4, 4] 
        w2c: [nv, 4, 4] 
        resolution: int
    Return:
        depth: [nv, resolution, resolution, 1], in meter
    """
    ## 1. rast
    pos_clip = transform_pos(vtx_pos, mvp)
    pos_idx_int = pos_idx.int()

    # [nv, resolution, resolution, 4] (u, v, z/w, id)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx_int, resolution=[resolution, resolution])

    # [nv, resolution, resolution, 3]
    gb_pos, _ = dr.interpolate(vtx_pos[None, ...], rast_out, pos_idx_int)
    nv, h, w, _ = gb_pos.shape
    gb_pos = gb_pos.reshape(nv, h*w, -1)
    # pw, [nv, resolution * resolution, 4]
    world_points = torch.cat([gb_pos, torch.ones_like(gb_pos[..., 0:1])], dim=-1)

    # pc = Tcw * pw [nv, resolution * resolution, 3]
    camera_points = torch.bmm(w2c, world_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
    depth = torch.norm(camera_points, dim=-1, p=1)
    # [nv, resolution, resolution, 1]
    depth = depth.reshape(nv, h, w, -1)
    hard_mask = torch.clamp(rast_out[..., -1:], 0, 1)
    # antialias_mask = dr.antialias(
    #     hard_mask.clone().contiguous(), rast_out, pos_clip, pos_idx_int)
    depth = depth * hard_mask

    return depth



def render_normal_cos(glctx, vtx_pos, pos_idx, mvp, vtx_normal, w2c, resolution):
    """render normal viewcos with multi views 
    Args:
        glctx: dr.RasterizeCudaContext()
        vtx_pos: [Nv, 3] xyz
        pos_idx: [Nf, 3] face
        mvp: [nv, 4, 4] 
        vtx_normal: [Nv, 3] 
        w2c: [nv, 4, 4] 
        resolution: int
    Return:
        viewcos: [nv, resolution, resolution, 1], in abs(cos)
    """
    ## 1. rast
    pos_clip = transform_pos(vtx_pos, mvp)
    pos_idx_int = pos_idx.int()

    # [nv, resolution, resolution, 4] (u, v, z/w, id)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx_int, resolution=[resolution, resolution])
    
    c2w = torch.linalg.inv(torch.tensor(w2c, dtype=torch.float32))
    cam_center = c2w[:, :3, -1:].permute(0, 2, 1)   # [nv, 1, 3]

    view_dir = cam_center - vtx_pos[None, ...]
    view_dir = util.safe_normalize(view_dir)    # [nv, Nv, 3]

    cosine = F.cosine_similarity(vtx_normal[None, ...], view_dir, dim=-1)
    cosine = (cosine[..., None]).abs()

    viewcos, _ = dr.interpolate(cosine.contiguous(), rast_out, pos_idx_int)

    # # [nv, resolution, resolution, 3]
    # gb_normal, _ = dr.interpolate(vtx_normal[None, ...].contiguous(), rast_out, pos_idx_int)
    # gb_normal = util.safe_normalize(gb_normal)
    
    # nv, h, w, _ = gb_normal.shape
    # # nw, [nv, resolution * resolution, 3]
    # gb_normal = gb_normal.reshape(nv, h*w, -1)

    # print('gb_normal ', gb_normal.shape)
    # print('w2c[:, :3, :3] ', w2c[:, :3, :3].T.shape)
    # cam_normal = torch.bmm(gb_normal, torch.linalg.inv(w2c[:, :3, :3]))
    # # cam_normal = gb_normal @ (w2c[:, :3, :3].T)
    # viewcos = cam_normal[..., [2]].abs()
    
    # # [nv, resolution, resolution, 1]
    # viewcos = viewcos.reshape(nv, h, w, -1)
    
    hard_mask = torch.clamp(rast_out[..., -1:], 0, 1)
    # antialias_mask = dr.antialias(
    #     hard_mask.clone().contiguous(), rast_out, pos_clip, pos_idx_int)
    viewcos = viewcos * hard_mask

    return viewcos


def save_depths(depth, out_dir, cam_name_list=None, key='depth'):
    """save depth as uint16 png with mm

    Args:
        depth: [nv, resolution, resolution, 1] tensor in meter
        out_dir: _description_
        cam_name_list: _description_. Defaults to None.
    """
    for idx in range(depth.shape[0]):
        img = depth[idx]
        if cam_name_list and len(cam_name_list) == depth.shape[0]:
            out_path = os.path.join(out_dir, f'{key}_{cam_name_list[idx]}.png')
        else:
            out_path = os.path.join(out_dir, f'{key}_{idx:03d}.png')
        img_cpu = img.detach().cpu().numpy()

        h, w = img_cpu.shape[:2]
        img_cpu = img_cpu.reshape(h, w)

        util.save_depth(img_cpu, out_path)
    return


def render_uvcoord_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, resolution):
    """render with multi views in frames, only render uvcoord.
    Args:
        glctx: dr.RasterizeCudaContext()
        vtx_pos: [Nv, 3] xyz
        pos_idx: [Nf, 3] face
        vtx_uv: [Nv, 2] uv coords
        uv_idx: [Nf, 3] uv face
        mvp: [nv, 4, 4] 
        resolution: int
    Return:
        color: [nv, resolution, resolution, x]
        alpha: [nv, resolution, resolution, 1]
    """
    ## 1. rast
    pos_clip = transform_pos(vtx_pos, mvp)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx.int(), resolution=[resolution, resolution])

    ## 2. interpolate, fill uv of res^2. uv coords [b, res, res, 2] range:[0, 1] or texture will auto mod to [0, 1]
    # texc, _ = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int())
    texc, _ = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int(), rast_db=rast_out_db, diff_attrs='all')

    return texc

def get_all_4_locations(values_y, values_x):
    """_summary_

    Args:
        values_y: _description_
        values_x: _description_

    Returns:
        [N*4], [N*4]
    """
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


def revert_texc_as_mask(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, resolution, up_scale=4, alpha_mask=None):
    """_summary_

    Args:
        glctx: _description_
        vtx_pos: _description_
        pos_idx: _description_
        vtx_uv: _description_
        uv_idx: _description_
        mvp: _description_
        resolution: _description_
        up_scale: _description_. Defaults to 4.
        alpha_mask: if not None(nv, h, w, 1), use texc & alpha_mask

    Returns:
        [1, resolution, resolution, 1]
    """
    uv_res = min(2048, resolution * up_scale)
    # texc:[nv, uv_res, uv_res, 2]
    texc = render_uvcoord_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, uv_res)
    uv_coords = texc.reshape(-1, 2)
    if alpha_mask is not None:
        assert alpha_mask.shape[0] == mvp.shape[0], f"{alpha_mask.shape}, {mvp.shape}"
        if alpha_mask.shape[-1] != uv_res:
            alpha_mask = resize_render(alpha_mask, uv_res, uv_res)
        alpha_mask = alpha_mask.reshape(-1)
        uv_coords = uv_coords[alpha_mask > 0]
    print('uv_coords ', uv_coords.shape)
    
    # texture_locations_y: [N]
    texture_locations_y, texture_locations_x = get_all_4_locations(
        uv_coords[:, 1].reshape(-1) * (uv_res - 1),
        uv_coords[:, 0].reshape(-1) * (uv_res - 1)
    )

    # mask_tensor [uv_res, uv_res,1]
    mask_tensor = torch.zeros((uv_res, uv_res, 1), device=vtx_pos.device)
    mask_tensor[texture_locations_y, texture_locations_x, :] = 1

    # re-scale to [1, resolution, resolution, 1]
    mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).permute(0, 3, 1, 2),
                              scale_factor=(resolution / uv_res),
                              mode="bilinear",
                              align_corners=False)
    mask_tensor = mask_tensor.permute(0, 2, 3, 1).contiguous()

    return mask_tensor


def render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data, resolution, max_mip_level=None):
    """render with multi views in frames, only render texture map, without light shading.
    resolution is render res, ht/wt is texture map res,  resolution != ht
    Args:
        glctx: dr.RasterizeCudaContext()
        vtx_pos: [Nv, 3] xyz
        pos_idx: [Nf, 3] face
        vtx_uv: [Nv, 2] uv coords
        uv_idx: [Nf, 3] uv face
        mvp: [nv, 4, 4] 
        tex_data: [1, ht, wt, x], x can be 3/4/9 or so on. =texture channel cnt
        resolution: int
        max_mip_level: if not None and > 0, use mipmap when dr.texture
    Return:
        color: [nv, resolution, resolution, x]
        alpha: [nv, resolution, resolution, 1]
    """
    ## 1. rast
    pos_clip = transform_pos(vtx_pos, mvp)
    # pos_clip = pos_clip.to(torch.float32)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx.int(), resolution=[resolution, resolution])

    ## 2. interpolate, fill uv of res^2. uv coords [b, res, res, 2] range:[0, 1] or texture will auto mod to [0, 1]
    # texc, _ = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int())
    texc, texd = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int(), rast_db=rast_out_db, diff_attrs='all')

    ## 3. sample texture
    if max_mip_level and max_mip_level > 0:
        color = dr.texture(tex_data, texc, texd, filter_mode='auto', max_mip_level=max_mip_level)
        # color = dr.texture(tex_data, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        color = dr.texture(tex_data, texc, filter_mode='linear')

    color = dr.antialias(color, rast_out, pos_clip, pos_idx.int())

    alpha = torch.clamp(rast_out[..., -1:], 0, 1)
    color = color * alpha   # Mask out background
    return color, alpha

def refine_texture_knn(uv_tex_, uv_mask_):
    """_summary_

    Args:
        uv_tex: [h, w, 3] tensor float 
        uv_mask: [h, w] tensor 0/1
    """
    device = uv_tex_.device
    uv_tex = uv_tex_.detach().cpu().numpy()
    uv_mask = (uv_mask_ > 0).detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(uv_mask, iterations=32)
    # inpaint_region = np.ones_like(uv_mask)
    inpaint_region[uv_mask] = 0

    search_region = uv_mask.copy()
    larger_structuring_element = np.ones((5, 5), dtype=bool)
    not_search_region = binary_erosion(search_region, iterations=3, structure=larger_structuring_element)
    # not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    # knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
    # knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", radius=10).fit(
    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", radius=100).fit(
        search_coords
    )
    _, indices = knn.kneighbors(inpaint_coords)

    uv_tex[tuple(inpaint_coords.T)] = uv_tex[tuple(search_coords[indices[:, 0]].T)]

    uv_tex_new = torch.from_numpy(uv_tex).to(device)
    return uv_tex_new, inpaint_region


def recover_texture_grid(infer_images : torch.Tensor, in_mesh : Mesh, glctx, mvp, w2c, tex_res, min_view_cos=0.5):
    """
    infer_images [b, render_res, render_res, 3] in [0, 1]
    """
    in_mesh = auto_normals(in_mesh)
    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx   # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]
    vtx_normal, normal_idx = in_mesh.v_nrm, in_mesh.t_nrm_idx # [Nv, 3], [Nf, 3]
    device = infer_images.device
    
    if infer_images.dim() == 3:
        infer_images = infer_images.unsqueeze(0)
    
    render_res = infer_images.shape[-2]
    if infer_images.shape[0] != mvp.shape[0] or infer_images.shape[0] != w2c.shape[0]:
        print('ERROR recover_texture need equal batch image size ', infer_images.shape, mvp.shape, w2c.shape)
        return None
    
    # 1. rast, alpha
    ## 1. rast
    pos_clip = transform_pos(vtx_pos, mvp)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx.int(), resolution=[render_res, render_res])

    # 2. render uvs, normal
    ## rendered uv coord [b, render_res=512, 512, 2] in [0, 1], x and y means col and row from left-top corner
    uvs, _ = dr.interpolate(vtx_uv.unsqueeze(0), rast_out, uv_idx.int())
    ## rendered normal [b, 512, 512, 3] in [0, 1] in world
    normal, _ = dr.interpolate(vtx_normal.unsqueeze(0).contiguous(), rast_out, normal_idx.int())
    normal = util.safe_normalize(normal)

    print('debug normal ', normal.shape)
    print('debug w2c ', w2c.shape)
    ## rot normal from world to cam
    nv, h, w, _ = normal.shape
    normal = normal.reshape(nv, h*w, -1)
    rot_normal = torch.bmm(w2c[:, :3, :3], normal.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
    rot_normal = rot_normal.reshape(nv, h, w, -1)
    # rot_normal = normal @ w2c[:, :3, :3]
    viewcos = rot_normal[..., [2]]  # [b, 512, 512, 3]
    print('debug viewcos ', viewcos.shape)
    # TODO(csz)


    # 3. revert to texture map one by one
    texture_map = torch.zeros((tex_res, tex_res, 3), device=device, dtype=torch.float32)
    cnt_map = torch.zeros((tex_res, tex_res, 1), device=device, dtype=torch.float32)
    skip_views = [7]    # TODO
    main_views = [3]
    main_weight = 2
    for idx in range(infer_images.shape[0]):
        if idx in skip_views:
            continue
        cur_alpha = (rast_out[idx, ..., 3:] > 0).float() # [512, 512, 1]
        cur_viewcos = viewcos[idx]  # [512, 512, 3]
        print('debug cur_viewcos ', cur_viewcos.min(), cur_viewcos.max(), cur_viewcos.median())
        
        mask = (cur_alpha > 0) & (cur_viewcos > min_view_cos)  # [b, H, W, 1]
        mask = mask.view(-1)    
        print('debug mask ', mask.shape)        
        
        rgbs = infer_images[idx].view(-1, 3)[mask].contiguous()
        cur_uvs = uvs[idx].view(-1, 2).clamp(0, 1)[mask]
        coords = cur_uvs[..., [1, 0]] * 2 - 1   # -> row, col in [-1, 1]
        print('debug rgbs ', rgbs.shape)
        print('debug coords ', coords.shape)
        cur_texture, cur_cnt = mipmap_linear_grid_put_2d(tex_res, tex_res, coords, 
                                                        rgbs, min_resolution=128, return_count=True)
        print('debug cur_texture ', cur_texture.shape)
        print('debug cur_cnt ', cur_cnt.shape)
        
        if idx in main_views:
            cur_cnt *= main_weight
            
        mask = cnt_map.squeeze(-1) < 0.1    # unseen area
        texture_map[mask] += cur_texture[mask]
        cnt_map[mask] += cur_cnt[mask]
        
    texture_mask = cnt_map.squeeze(-1) > 0
    texture_map[texture_mask] = texture_map[texture_mask] / cnt_map[texture_mask].repeat(1, 3)

    texture_mask = texture_mask.view(tex_res, tex_res)

    uv_tex_new, inpaint_region = refine_texture_knn(texture_map, texture_mask)
    texture_mask = texture_mask.cpu().numpy()
    return uv_tex_new, inpaint_region, texture_mask

    
def recover_texture(rendered_images, alphas, uv_maps, tex_data):
    """recover texture from batch render images

    Args:
        rendered_images: [b, h, w, x]
        alphas: [b, h, w, 1]
        uv_maps: [b, h, w, 2]
        tex_data: [1, h_t, w_t, x]

    Returns:
        _description_
    """
    recovered_texture = torch.zeros_like(tex_data)
    cnt = torch.zeros_like(tex_data[..., :-1])
    h_t, w_t = tex_data.shape[1:3]

    scaled_uv = uv_maps / torch.max(uv_maps)   # scale when  > h,
    print('debug recovered_texture ', recovered_texture.shape, h_t, w_t)
    print('scaled_uv ', scaled_uv.shape, tex_data.shape, torch.min(scaled_uv).item(), torch.max(scaled_uv).item())

    for b in range(rendered_images.shape[0]):
        rendered_image = rendered_images[b]
        uv_map = scaled_uv[b]
        alpha = alphas[b]

        for y in range(rendered_image.shape[0]):
            for x in range(rendered_image.shape[1]):
                if not alpha[y, x] > 0:
                    continue
                color = rendered_image[y, x]

                u, v = uv_map[y, x]

                tex_x = int(u * (w_t - 1))
                tex_y = int(v * (h_t - 1))

                recovered_texture[0, tex_y, tex_x, :] += color
                cnt[0, tex_y, tex_x, :] += 1

    # recovered_texture = recovered_texture / torch.max(recovered_texture)
    return recovered_texture

    # b, h, w, x = rendered_images.shape
    # _, h_t, w_t, _ = tex_data.shape
    # recovered_texture = torch.zeros_like(tex_data)

    # tex_coords = (uv_maps * (torch.tensor([w - 1, h - 1], device=uv_maps.device))).long()  # scale when h_t  > h,
    # print('tex_coords ', tex_coords.shape)

    # # 创建一个索引矩阵，表示每个像素在 batch 中的位置
    # batch_indices = torch.arange(b, device=uv_maps.device).view(b, 1, 1).expand(b, h, w)

    # # 应用 alpha 遮罩
    # masked_rendered_images = rendered_images * alphas.expand_as(rendered_images)
    # print('masked_rendered_images ', masked_rendered_images.shape, alphas.shape, alphas.expand(b, h, w, x).shape)
    # valid_data =  masked_rendered_images[alphas.expand(b, h, w, x) > 0]
    # print('valid_data ', valid_data.shape, batch_indices.shape)

    # # recovered_texture[0].index_put_((tex_coords[..., 1].flatten(), tex_coords[..., 0].flatten(), batch_indices.flatten()), valid_data, accumulate=True)

    # for c in range(x):
    #     recovered_texture[0, :, :, c].index_put_((tex_coords[..., 1].flatten(), tex_coords[..., 0].flatten(), batch_indices.flatten()), masked_rendered_images[..., c].flatten(), accumulate=True)

    # # 归一化 recovered_texture
    # recovered_texture = recovered_texture / torch.max(recovered_texture)

    # return recovered_texture

def save_textures(color, alpha, key_channel_se_pairs, out_dir, cam_name_list=None, save_res=None, bg_type='white'):
    """mix backgroud and save rendered colors to out_dir

    Args:
        color: [nv, h, w, x] x can be 3/4/9 and so on. kd + ks + normal = 9(3+3+3) or 10 (4+3+3) or 6/7 when ks = 0
        alpha: [nv, h, w, 1] from rast
        key_channel_se_pairs: pair of (key, channel_start, channel_end)
        out_dir: dir
        cam_name_list: if not None, save colors as cam_name
        save_res: if not None, force set save resolution
        bg_type: white/black/random 
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path_list = []
    for key, cs, ce in key_channel_se_pairs:
        color_tex = color[..., cs:ce]
        if 'normal' in key:
            color_tex = (color_tex + 1.)/2.     # [-1, 1] - > [0, 1]

        # if rgb, mix bg
        if color_tex.shape[-1] == 3:
            color_tex = color_tex[..., :3]

            if bg_type == 'black':
                background = torch.zeros_like(color_tex)
            elif bg_type == 'white':
                background = torch.ones_like(color_tex)
            elif bg_type == 'random':
                background = torch.rand_like(color_tex)
            else:
                assert False, "Unknown background type %s" % bg_type
            color_tex = mix_rgba(color_tex, alpha, background)

        for idx in range(color_tex.shape[0]):
            img = color_tex[idx]
            if cam_name_list and len(cam_name_list) == color_tex.shape[0]:
                out_path = os.path.join(out_dir, f'{key}_{cam_name_list[idx]}.png')
            else:
                out_path = os.path.join(out_dir, f'{key}_{idx:03d}.png')
            img_cpu = img.detach().cpu().numpy()

            if save_res and img.shape[-2] != save_res:
                img_cpu = cv2.resize(img_cpu, (save_res, save_res), interpolation=cv2.INTER_CUBIC)

            util.save_image(out_path, img_cpu)

            out_path_list.append(out_path)

    print(f'render {len(out_path_list)} views to {out_dir}')
    return


def render_gif(
    in_mesh: Mesh,
    out_gif,
    lrm_mode=True
):
    import imageio
    glctx = dr.RasterizeCudaContext()

    vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx   # [Nv, 3],  [Nf, 3]
    vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]    
    

    optim_cfg = {'render_res': 512, "max_mip_level":4}

    # generate rotate gif, just for vis
    # TODO from json
    frame_k = np.array([
        [703.354248046875, 0.0, 255.5],
        [0.0, 703.354248046875, 255.5],
        [0.0, 0.0, 1.0]])
    mvp = rotate_scene_mvp(frame_k, itr_all=40, cam_radius=3.8, lrm_mode=lrm_mode)['mvp']

    color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, in_mesh.material['kd'].data,
                                                int(optim_cfg['render_res']), optim_cfg['max_mip_level'])
    background = torch.ones_like(color_opt)
    
    color_tex = mix_rgba(color_opt, alpha_opt, background)

    img_cpu_list = []
    for idx in range(color_tex.shape[0]):
        img = color_tex[idx]

        img_cpu = img.detach().cpu().numpy()
        img_cpu = np.clip(np.rint(img_cpu * 255.0), 0, 255).astype(np.uint8)
        img_cpu_list.append(img_cpu)

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, img_cpu_list, duration=0.05)
    print(f"save gif to {out_gif}")

    return