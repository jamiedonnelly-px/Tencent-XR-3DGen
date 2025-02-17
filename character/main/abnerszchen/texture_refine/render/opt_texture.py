import os
import numpy  as np
import torch
import torch.nn as nn
import cv2
import nvdiffrast.torch as dr

from render.mesh import load_mesh, Mesh
from render.material import Material
from render.util import mse_to_psnr, load_image
from render.obj import write_obj
import render.texture as texture
from render.render_mesh import parse_pose_json, render_texture_views, save_textures, get_tensor_range, util_merge_tex
from render.util import load_image

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

def init_opt_tex(init_tex, key_channel_se_pairs, use_init=True, device='cuda', texture_res=None):
    """init trainable texture.

    Args:
        init_tex: [1, h, w, x]. TODO can be None
        key_channel_se_pairs: pair of (key, channel_start, channel_end)
        texture_res: TODO can be not None.
    Return:
        tex_merge: [1, h, w, x] nn.Parameter
    """
    # TODO(csz) ->cfg
    kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    ks_max              = [ 1.0,  1.0,  1.0]
    nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    nrm_max             = [ 1.0,  1.0,  1.0]    
    custom_mip = True
    
    kd_min, kd_max = torch.tensor(kd_min, dtype=torch.float32, device=device), torch.tensor(kd_max, dtype=torch.float32, device=device)
    ks_min, ks_max = torch.tensor(ks_min, dtype=torch.float32, device=device), torch.tensor(ks_max, dtype=torch.float32, device=device)
    nrm_min, nrm_max = torch.tensor(nrm_min, dtype=torch.float32, device=device), torch.tensor(nrm_max, dtype=torch.float32, device=device)
        
    if texture_res:
        raise NotImplementedError

    if use_init:
        tex_merge = torch.nn.Parameter(init_tex.clone().detach(), requires_grad=True)
        # tex_map_opt = texture.create_trainable(init, texture_res, not custom_mip, [ks_min, ks_max])
    else:
        tex_init_list = []
        for key, cs, ce in key_channel_se_pairs:
            if key == 'kd':
                tex_init = torch.zeros_like(init_tex[..., cs:ce])
                # tex_init = torch.rand_like(init_tex[..., cs:ce])
            elif key == 'ks':
                init = torch.rand_like(init_tex[..., cs:ce])
                init[..., 0] *= 0.01
                init[..., 1] = torch.lerp(ks_min[1], ks_max[1], init[..., 1])
                init[..., 1] = torch.lerp(ks_min[2], ks_max[2], init[..., 2])
                tex_init = init
            elif key == 'normal':
                tex_init = torch.ones_like(init_tex[..., cs:ce])
                tex_init[..., :-1] *= 0
            
            tex_init_list.append(tex_init)
        tex_merge = torch.cat(tex_init_list, dim=-1)    
        tex_merge = torch.nn.Parameter(tex_merge.clone().detach(), requires_grad=True)
    
    return tex_merge

def render_opt_obj_texture(in_obj, in_pose_json, out_dir):
    """render obj then optimize texture. for debug

    Args:
        in_obj: obj path.
        in_pose_json: like data/cams/cam_parameters_select.json
        out_dir: out dir

    Returns:
        _description_
    """
    # TODO(csz) ->cfg
    ref_res, res_train = 2048, 2048
    need_texs = ['kd', 'normal']
    max_mip_level = 9
    lr_base           = 1e-2
    lr_ramp           = 0.1
    max_iter          = 1000
    log_interval      = 100
            
    if not os.path.exists(in_obj):
        print(f'can not find obj {in_obj}')
        return None
    raw_mesh : Mesh = load_mesh(in_obj)
    
    frames = parse_pose_json(in_pose_json)
    frames = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in frames.items()}
    
    
    glctx = dr.RasterizeCudaContext()
    
    print('debug begin render..')
    ## render gt

    ### common geomtry 
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx
    mvp = frames['mvp']

    ### merge tex channels, use common rast, uv, interpolate.
    tex : Material = raw_mesh.material # class Material   
    tex_data_merge, key_channel_se_pairs = util_merge_tex(tex, need_texs)
    tex_data_opt = init_opt_tex(tex_data_merge, key_channel_se_pairs)

    # Adam optimizer for texture with a learning rate ramp.
    optimizer    = torch.optim.Adam([tex_data_opt], lr=lr_base)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    print('debug tex_data_merge ', tex_data_merge.shape, get_tensor_range(tex_data_merge[..., :3]))  
    print('debug tex_data_opt ', tex_data_opt.shape, get_tensor_range(tex_data_opt[..., :3]))
    print('debug key_channel_se_pairs ', key_channel_se_pairs)  
    
    color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_merge, ref_res, max_mip_level)
    while color.shape[1] > res_train:
        color = bilinear_downsample(color)
        alpha = bilinear_downsample(alpha)
        
    for it in range(max_iter + 1):
        color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt, res_train, max_mip_level)
        
        # loss_tex = torch.mean((tex_data_merge - tex_data_opt)**2)
        loss_tex = 0
        
        loss = loss_tex + torch.mean((color - color_opt)**2) # L2 pixel loss.
        optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        if log_interval and (it % log_interval == 0):
            psnr = mse_to_psnr(loss.item())
            s = "iter=%d,loss=%f,psnr=%f" % (it, loss.item(), psnr)
            print(s)
        
    # save imgs
    print('debug begin save..')
    save_textures(color, alpha, key_channel_se_pairs, out_dir)
    save_textures(color_opt, alpha_opt, key_channel_se_pairs, os.path.join(out_dir, 'opt'))
    save_textures(tex_data_merge, torch.ones_like(tex_data_merge[..., :1]), key_channel_se_pairs, os.path.join(out_dir, 'tex_raw'))
    save_textures(tex_data_opt, torch.ones_like(tex_data_opt[..., :1]), key_channel_se_pairs, os.path.join(out_dir, 'tex_opt'))
    
    print('debug tex_data_opt final ', tex_data_opt.shape, get_tensor_range(tex_data_opt[..., :3]))
    tex_psnr = mse_to_psnr(torch.mean((tex_data_merge[..., :4] - tex_data_opt[..., :4])**2).item())
    mask = tex_data_merge[..., :4] > 0
    tex_psnr_alpha = mse_to_psnr(torch.mean((tex_data_merge[..., :4][mask] - tex_data_opt[..., :4][mask])**2).item())
    print('tex_psnr/tex_psnr_alpha ', tex_psnr, tex_psnr_alpha)
    
    # TODO refine saving
    new_mesh = raw_mesh.clone()
    new_mesh.material = raw_mesh.material
    if tex_data_opt.shape[-1] > 4:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :-3])
        new_mesh.material['norml'] = texture.Texture2D(tex_data_opt[..., -3:])
    else:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :3])
        
    write_obj(os.path.join(out_dir, 'opt_mesh'), new_mesh)
    
    return




def temp_render_opt_obj_texture(in_obj, in_pose_json, cond_img_path, out_dir):
    """render obj then optimize texture. for debug

    Args:
        in_obj: obj path.
        in_pose_json: like data/cams/cam_parameters_select.json
        out_dir: out dir

    Returns:
        _description_
    """
    # TODO(csz) ->cfg
    ref_res, res_train = 2048, 512
    need_texs = ['kd']
    max_mip_level = 4
    lr_base           = 1e-2
    lr_ramp           = 0.1
    max_iter          = 400
    log_interval      = 100
            
    if not os.path.exists(in_obj):
        print(f'can not find obj {in_obj}')
        return None
    raw_mesh : Mesh = load_mesh(in_obj)
    
    frames = parse_pose_json(in_pose_json)
    frames = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in frames.items()}
    
    cond_img = load_image(cond_img_path)
    cond_img = torch.tensor(cond_img).to('cuda').unsqueeze(0)
    
    glctx = dr.RasterizeCudaContext()
    
    print('debug begin render..')
    ## render gt

    ### common geomtry 
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx
    mvp = frames['mvp']

    ### merge tex channels, use common rast, uv, interpolate.
    tex : Material = raw_mesh.material # class Material   
    tex_data_merge, key_channel_se_pairs = util_merge_tex(tex, need_texs)
    tex_data_opt = init_opt_tex(tex_data_merge, key_channel_se_pairs, use_init=True)

    # Adam optimizer for texture with a learning rate ramp.
    optimizer    = torch.optim.Adam([tex_data_opt], lr=lr_base)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    print('debug tex_data_merge ', tex_data_merge.shape, get_tensor_range(tex_data_merge[..., :3]))  
    print('debug tex_data_opt ', tex_data_opt.shape, get_tensor_range(tex_data_opt[..., :3]))
    print('debug key_channel_se_pairs ', key_channel_se_pairs)  
    print('debug cond_img ', cond_img.shape)  
    print('debug mvp ', mvp.shape)  
    
    color, alpha = cond_img[..., :3], cond_img[..., -1:]
    mask = (alpha[0].cpu().numpy() * 255).astype(np.uint8)

    ksize = 5
    kernel = np.ones((ksize, ksize), dtype=np.uint8)    
    alpha = torch.from_numpy(cv2.erode(mask, kernel, iterations=1)).float().to('cuda') / 255
    alpha = alpha.unsqueeze(0).unsqueeze(-1)
    print('alpha ', alpha.shape)
    # color, alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_merge, ref_res, max_mip_level)
    # while color.shape[1] > res_train:
    #     color = bilinear_downsample(color)
    #     alpha = bilinear_downsample(alpha)
    import time
    
    st = time.time()
    for it in range(max_iter + 1):
        color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt, res_train, max_mip_level)
        
        # loss_tex = torch.mean((tex_data_merge - tex_data_opt)**2)
        loss_tex = 0
        
        loss = loss_tex + torch.mean((color * alpha - color_opt * alpha)**2) # L2 pixel loss.
        # loss = loss_tex + torch.mean((color - color_opt)**2) # L2 pixel loss.
        optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        if log_interval and (it % log_interval == 0):
            psnr = mse_to_psnr(loss.item())
            s = "iter=%d,loss=%f,psnr=%f" % (it, loss.item(), psnr)
            print(s)
    use_t = time.time() - st
        
    # save imgs
    print('debug begin save..', use_t)
    save_textures(color, alpha, key_channel_se_pairs, out_dir)
    save_textures(color_opt, alpha_opt, key_channel_se_pairs, os.path.join(out_dir, 'opt'))
    save_textures(tex_data_merge, torch.ones_like(tex_data_merge[..., :1]), key_channel_se_pairs, os.path.join(out_dir, 'tex_raw'))
    save_textures(tex_data_opt, torch.ones_like(tex_data_opt[..., :1]), key_channel_se_pairs, os.path.join(out_dir, 'tex_opt'))
    
    print('debug tex_data_opt final ', tex_data_opt.shape, get_tensor_range(tex_data_opt[..., :3]))
    tex_psnr = mse_to_psnr(torch.mean((tex_data_merge[..., :4] - tex_data_opt[..., :4])**2).item())
    mask = tex_data_merge[..., :4] > 0
    tex_psnr_alpha = mse_to_psnr(torch.mean((tex_data_merge[..., :4][mask] - tex_data_opt[..., :4][mask])**2).item())
    print('tex_psnr/tex_psnr_alpha ', tex_psnr, tex_psnr_alpha)
    
    # TODO refine saving
    new_mesh = raw_mesh.clone()
    new_mesh.material = raw_mesh.material
    if tex_data_opt.shape[-1] > 4:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :-3])
        new_mesh.material['norml'] = texture.Texture2D(tex_data_opt[..., -3:])
    else:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :3])
        
    write_obj(os.path.join(out_dir, 'opt_mesh'), new_mesh)
    
    return