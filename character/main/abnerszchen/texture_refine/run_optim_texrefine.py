import os
import argparse
import time
import torch
from PIL import Image
import numpy as np
import nvdiffrast.torch as dr

from render.mesh import load_mesh, Mesh
from render.obj import write_obj
from render.material import Material
import render.texture as texture
from render.util import mse_to_psnr
from render.render_mesh import parse_pose_json, render_texture_views, save_textures, util_merge_tex, mix_rgba
from render.opt_texture import render_opt_obj_texture, init_opt_tex
from sd_tex_refine.pipeline_stable_diffusion_tex_refine import StableDiffusionTexRefinePipeline
from dataset.utils_dataset import load_rgba_as_rgb

def init_model_glctx(in_model_path, in_pose_json, device='cuda'):
    """init diffusion model and render pose/glctx. only need once.

    Args:
        in_model_path: diffusion model path
        in_pose_json: poses like data/cams/cam_parameters_select.json

    Returns:
        pipeline, generator, 
        mvp: [nv, 4, 4] = proj(intri) * mv, proj is relative about fovy and aspect, not about resolution
        glctx
    """
    pipeline = StableDiffusionTexRefinePipeline.from_pretrained(in_model_path, torch_dtype=torch.float16).to(device)
    generator = torch.Generator(device)
    
    frames = parse_pose_json(in_pose_json)
    mvp = frames['mvp'].to(device)
    cam_name_list = frames['cam_name_list']
    print('debug cam_name_list ', cam_name_list, mvp.shape)
    
    glctx = dr.RasterizeCudaContext()    
    return pipeline, generator, mvp, glctx



def optim_texrefine(pipeline, generator, mvp, glctx, optim_cfg, in_obj, in_condi):
    # 1. load obj and render raw images
    raw_mesh : Mesh = load_mesh(in_obj)
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    vtx_uv, uv_idx = raw_mesh.v_tex, raw_mesh.t_tex_idx

    ### merge tex channels, use common rast, uv, interpolate.
    tex : Material = raw_mesh.material # class Material   
    tex_data_merge, key_channel_se_pairs = util_merge_tex(tex, optim_cfg['need_texs'])
    
    # [b, render_res, render_res, x=3 or tex_data_merge.shape[-1] / 1]
    raw_color, raw_alpha = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_merge, 
                                        optim_cfg['render_res'], optim_cfg['max_mip_level'])
    
    # 2. infer texrefine model, get new images
    ## TODO input tensor
    raw_rgb = raw_color[..., :3]
    background = torch.ones_like(raw_rgb)   # need white backgroud
    raw_render_imgs = mix_rgba(raw_rgb, raw_alpha, background)[..., :3]  # [b, render_res, render_res, 3]
    
    array = raw_render_imgs.detach().cpu().numpy()
    images = [Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)) for arr in array]

    prompt_img_path = [in_condi] * len(images)
    # list of PIL, TODO just return tensor, output_type=np.array?
    infer_images = pipeline(
            prompt_img_path,
            image=images,
            num_inference_steps=optim_cfg['num_inference_steps'],
            image_guidance_scale=optim_cfg['image_guidance_scale'],
            guidance_scale=optim_cfg['guidance_scale'],
            generator=generator,
        ).images 
    if 0:
        for i, image in enumerate(images):
            image.save(f'debug_{i}.jpg')
        for i, image in enumerate(infer_images):
            image.save(f'debug_infer_{i}.jpg')
                
        condi_img = load_rgba_as_rgb(in_condi)
        condi_img.save('debug_condi.jpg')
            
    numpy_arrays = [np.array(img) for img in infer_images]
    # tensor [b, render_res, render_res, 3]
    infer_images = torch.from_numpy(np.stack(numpy_arrays)).to(pipeline.device)
    infer_images = infer_images.to(torch.float32) / 255.
    
    # 3. optim tex Adam optimizer for texture with a learning rate ramp.
    tex_data_opt = init_opt_tex(tex_data_merge, key_channel_se_pairs, use_init=True, device=pipeline.device)
    optimizer    = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x)/float(optim_cfg['max_iter'])))

    # TODO alpha
    for it in range(optim_cfg['max_iter'] + 1):
        color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt, 
                                                    optim_cfg['render_res'], optim_cfg['max_mip_level'])
        loss_tex = torch.mean((tex_data_merge - tex_data_opt)**2) * optim_cfg['lamda_tex']
        # loss_tex = 0
        # raw_color, raw_alpha
        # loss = loss_tex + torch.mean((infer_images - color_opt)**2) # L2 pixel loss.
        loss = loss_tex + torch.mean((infer_images * raw_alpha - color_opt * raw_alpha)**2) # L2 pixel loss.
        optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        if 'log_interval' in optim_cfg and (it % optim_cfg['log_interval'] == 0):
            psnr = mse_to_psnr(loss.item())
            s = "iter=%d,loss=%f,psnr=%f" % (it, loss.item(), psnr)
            print(s)
            
    new_mesh = raw_mesh.clone()
    new_mesh.material = raw_mesh.material
    if tex_data_opt.shape[-1] > 4:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :-3])
        new_mesh.material['norml'] = texture.Texture2D(tex_data_opt[..., -3:])
    else:
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :3])
            
    return new_mesh

def optim_texrefine_pipeline(in_model_path, in_obj, in_condi, in_pose_json, out_dir, device='cuda'):
    if not os.path.exists(in_model_path) or not os.path.exists(in_obj) or not os.path.exists(in_condi) or not os.path.exists(in_pose_json):
        raise ValueError(f'can not find valid {in_pose_json} / {in_obj} / {in_condi} / {in_pose_json}')
        
    # 1. Init model, generator, frames and glctx. / optim_cfg
    pipeline, generator, mvp, glctx = init_model_glctx(in_model_path, in_pose_json, device)
    optim_cfg = {'need_texs':['kd'], 'render_res':512, 'max_mip_level':6,
                 'num_inference_steps':20, 'image_guidance_scale':1.5, 'guidance_scale':2,
                 'lamda_tex':0.05, 'lr_base' : 1e-2, 'lr_ramp':0.1, 'max_iter':1000, 'log_interval':200}
   
    # 2. load obj and render raw images
    # infer texrefine model, get new images
    # optim tex
    new_mesh : Mesh = optim_texrefine(pipeline, generator, mvp, glctx, optim_cfg, in_obj, in_condi)
    
    
    # 3. output new obj
    os.makedirs(out_dir, exist_ok=True)
    raw_mesh : Mesh = load_mesh(in_obj)
    write_obj(os.path.join(out_dir, 'raw_mesh'), raw_mesh)
    write_obj(os.path.join(out_dir, 'opt_mesh'), new_mesh)
    os.system(f'cp {in_condi} {out_dir}')
    return


def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_model_path', type=str, help='texrefine model path')
    parser.add_argument('in_obj', type=str)
    parser.add_argument('in_condi', type=str, help='input condition image cam-0100.png')
    parser.add_argument('out_dir', type=str, help='out dir with new obj and mtl, texture map')
    parser.add_argument('--in_pose_json', type=str, default='data/cams/cam_parameters_select.json')
    args = parser.parse_args()

    # Run.
    in_pose_json = args.in_pose_json
    if not os.path.exists(args.in_pose_json):
        codedir = os.path.dirname(os.path.abspath(__file__))
        in_pose_json = os.path.join(codedir, args.in_pose_json)
    
    optim_texrefine_pipeline(args.in_model_path, args.in_obj, args.in_condi, in_pose_json, args.out_dir)

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
