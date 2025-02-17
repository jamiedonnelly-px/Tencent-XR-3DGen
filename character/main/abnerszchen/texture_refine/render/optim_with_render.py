import os
import json
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import os
import random
from PIL import Image

import torch
import numpy as np
import nvdiffrast.torch as dr
import cv2
import torch.nn.functional as F
from render import util
from render.mesh import Mesh, auto_normals, load_mesh
from render import geom_utils
from render.grid_put import mipmap_linear_grid_put_2d
import render.texture as texture
from render import render_mesh

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

def tensor_bhwc_to_pils(tensor):
    bhwc_np = (tensor.cpu().numpy() * 255).round().astype("uint8")
    bhwc_pils = [Image.fromarray(array) for array in bhwc_np]
    return bhwc_pils

def make_inpaint_condition(rgb, inpaint_mask):
    """_summary_

    Args:
        rgb: [b,h,w,3] in [0,1]
        inpaint_mask: [b,h,w,1] in [0,1]

    Returns:
        list of pil, list of pil, tensor [b,3,h,w]
    """
    assert rgb.shape[1:2] == inpaint_mask.shape[1:2], "image and image_mask must have the same image size"
    rgb_pils = tensor_bhwc_to_pils(rgb)
    inpaint_mask_pils = tensor_bhwc_to_pils(inpaint_mask)
    
    control_image = rgb.permute(0, 2, 3, 1) # to b,c,h,w
    in_inpaint_mask = inpaint_mask.permute(0, 2, 3, 1) # to b,c,h,w
    
    control_image[in_inpaint_mask.float() > 0.5] = -1.0  # set as masked pixel
    return rgb_pils, inpaint_mask_pils, control_image     
    
class InpaintMesh():
    def __init__(self, in_model_path="/aigc_cfs/model/control_v11p_sd15_inpaint", device='cuda'):
        self.in_model_path = in_model_path
        self.device = device

        try:
            if not os.path.exists(self.in_model_path):
                print(f'Error: can not find valid sd model {self.in_model_path}')

            self.pipeline, self.generator, self.glctx = self.init_model_glctx(in_model_path=in_model_path)
            print(f'load inpaint model {self.in_model_path} done')
        except:
            raise ValueError('init InpaintMesh failed')

    def init_model_glctx(self,
                         in_model_path="/aigc_cfs/model/control_v11p_sd15_inpaint",
                         sd_model_path="/aigc_cfs/model/stable-diffusion-v1-5",
                         seed=42):
        """init diffusion model and render pose/glctx. only need once.

        Args:
            in_model_path: diffusion model path

        Returns:
            pipeline, generator,  glctx
        """
        controlnet = ControlNetModel.from_pretrained(in_model_path, torch_dtype=torch.float16)
        pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(sd_model_path,
                                                                            controlnet=controlnet,
                                                                            torch_dtype=torch.float16)
        pipeline.safety_checker = None

        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_model_cpu_offload()

        generator = torch.Generator(self.device).manual_seed(seed)

        glctx = dr.RasterizeCudaContext()
        return pipeline, generator, glctx

    def load_pose_json(self, pose_json, lrm_mode=False, select_views=[], skip_views=[]):
        """load render pose from json

        Args:
            pose_json: data/cams/cam_parameters_select.json
            lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
            select_view: _description_. Defaults to [].

        Returns:
            mvp [nv, 4, 4] = proj * mv, proj(intri) is relative about fovy and aspect, not about resolution
            w2c [nv, 4, 4] 
        """
        frames = render_mesh.parse_pose_json(pose_json, lrm_mode=lrm_mode)
        mvp = frames['mvp'].to(self.device)
        w2c = frames['w2c'].to(self.device)
        views = [i for i in range(mvp.shape[0]) if (not select_views or i in select_views) and i not in skip_views]
        mvp, w2c = mvp[views], w2c[views]
     
        return mvp, w2c
    
    def inpaint_mesh_with_render(self, in_mesh: Mesh, in_mask, in_pose_json):
        """_summary_

        Args:
            in_mesh: _description_
            in_mask: [h, w] 1 means seen
            in_pose_json: _description_
        """
        optim_cfg = {"render_res":256, "tex_res": 512, "lr_base": 0.01, "lr_ramp": 0.1, "max_iter": 100, "batch_size":8}
        # 0. init tex
        if in_mesh.material and 'kd' in in_mesh.material:
            tex_data_opt = torch.nn.Parameter(in_mesh.material['kd'].data.clone(), requires_grad=True)
        else:
            tex_data_opt = init_opt_tex(optim_cfg['tex_res'], background='gray', device=self.device)
            
        optimizer = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x) / float(optim_cfg['max_iter'])))

        # 1. load pose and select views
        select_views = []    # TODO
        mvp_all, w2c_all = self.load_pose_json(in_pose_json, select_views=select_views, skip_views=range(300, 380))
        if in_mask.dim() == 2:
            in_mask = in_mask.unsqueeze(0).unsqueeze(-1)
            
        # 2. optim
        for iter in range(optim_cfg['max_iter']):
            views = random.sample(range(mvp_all.shape[0]), optim_cfg['batch_size'])
            mvp, w2c = mvp_all[views], w2c_all[views]
            
            # [b,h,w,c], [b,h,w,1] in [0,1]
            rgb, inpaint_mask = self.render_inpaint_inputs(in_mesh, tex_data_opt, mvp, w2c, optim_cfg['render_res'])
            rgb_pils, inpaint_mask_pils, control_image = make_inpaint_condition(rgb, inpaint_mask)
            
            # return b,c,h,w tensor
            new_images = self.pipeline(
                "complete, high quality",
                num_inference_steps=20,
                generator=self.generator,
                image=rgb_pils,
                mask_image=inpaint_mask_pils,
                control_image=control_image,
                controlnet_conditioning_scale=1.0,
                output_type='pt',
            ).images
            
            new_images = new_images.permute(0, 2, 3, 1)
        
            loss = torch.mean(((new_images * inpaint_mask - rgb * inpaint_mask)**2))  # L2 pixel loss.
            optimizer.zero_grad()
            # loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

        new_mesh = in_mesh.clone()
        new_mesh.material = in_mesh.material
        new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :-3])
        
        return new_mesh
    


    def render_inpaint_inputs(self, in_mesh: Mesh, tex_data, in_mask, mvp, w2c, render_res, max_mip_level=None):
        """_summary_

        Args:
            in_mesh: class Mesh
            tex_data: [1, ht, wt, x]
            in_mask: [1, h, w, 1]
            mvp: [nv, 4, 4]
            w2c: _description_
            render_res: _description_
            max_mip_level: _description_. Defaults to None.

        Returns:
            [nv,h,w,3], [nv,h,w,1]
        """
        vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx   # [Nv, 3],  [Nf, 3]
        vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]
        
        ## 1. rast
        pos_clip = render_mesh.transform_pos(vtx_pos, mvp)
        rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, pos_idx.int(), resolution=[render_res, render_res])

        ## 2. interpolate, fill uv of res^2. uv coords [b, res, res, 2] range:[0, 1] or texture will auto mod to [0, 1]
        # texc, _ = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int())
        texc, texd = dr.interpolate(vtx_uv[None, ...], rast_out, uv_idx.int(), rast_db=rast_out_db, diff_attrs='all')

        ## 3. sample texture and mask
        if max_mip_level and max_mip_level > 0:
            color = dr.texture(tex_data, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
            render_mask = dr.texture(tex_data, in_mask, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            color = dr.texture(tex_data, texc, filter_mode='linear')
            render_mask = dr.texture(tex_data, in_mask, filter_mode='linear')

        # color = dr.antialias(color, rast_out, pos_clip, pos_idx.int())
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        print('debug render_mask ', render_mask.shape)
        print('debug alpha ', alpha.shape)
        inpaint_mask = alpha * (render_mask < 0.5)  # alpha - render_mask
        rgb = color * alpha + (1 - alpha)   # white background
        return rgb, inpaint_mask
    
    def interface_inapint_mesh(in_obj_path, out_obj_path):

        if not os.path.exists(in_obj_path):
            print(f'ERROR can not find in_obj {in_obj_path}')
            return None
        
        raw_mesh: Mesh = load_mesh(in_obj_path)        

        return