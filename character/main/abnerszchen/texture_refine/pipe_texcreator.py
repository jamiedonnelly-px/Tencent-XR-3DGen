import os
import argparse
import time
import json
import torch
import random
import torch.nn.functional as F
from PIL import Image
import numpy as np
import nvdiffrast.torch as dr
import cv2
import trimesh
import miniball
import math
import imageio
import uuid

from render.mesh import load_mesh, Mesh
from render.obj import write_obj
from render.geom_utils import mesh_normalized, clean_decimate_mesh, align_y_mesh, align_z_mesh
from render.uv_conditions import mesh_xatlas
import render.texture as texture
from render.util import mse_to_psnr
from render.render_mesh import (
    recover_texture_grid,
    parse_pose_json,
    render_texture_views,
    render_depth_views,
    rotate_scene_mvp,
    mix_rgba,
    revert_texc_as_mask,
    refine_texture_knn,
    render_normal_cos,
    auto_normals,
)
from render.bake_utils import inpaint_refine_uv_tex
from sd_tex_creator.pipeline_stable_diffusion_tex_creator import (
    StableDiffusionTexCreatorPipeline,
)
from dataset.utils_dataset import load_json, load_rgba_as_rgb, depth_normalize_tensor, depth_normalize, concatenate_images_2d, concatenate_images_horizontally
from grpc_backend.client_sr import SrGenClient


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

def resize_render(in_tensor, scale):
    """resize bhwc tensor

    Args:
        in_tensor: b, h, w, c
        scale: _description_

    Returns:
        b, h, w, c
    """
    in_tensor = in_tensor.permute(0, 3, 1, 2)
    in_tensor = F.interpolate(in_tensor, scale_factor=scale, mode="bilinear", align_corners=False)
    # in_tensor = F.interpolate(in_tensor, scale_factor=scale, mode="bilinear", align_corners=False).squeeze(0)
    in_tensor = in_tensor.permute(0, 2, 3, 1).contiguous()
    return in_tensor


def mask_refine(depth, infer_images, raw_alpha, thres=180, k=20):
    """dilate and erode, Reduce white edge

    Args:
        depth: tensor [b, w, h, 1]
        infer_images: tensor [b, w, h, 3]
        raw_alpha: tensor [b, w, h, 1]
        thres: _description_. Defaults to 180.
        k: _description_. Defaults to 20.

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

def log_time_list(time_list):
    if not time_list:
        return
    print('time_list ', time_list)
    prev_timestamp = time_list[0][1]
    time_sum = 0
    for name, time_stp in time_list:
        stage_duration = time_stp - prev_timestamp
        time_sum += stage_duration
        print("{} stage duration: {:.2f} seconds".format(name, stage_duration))
        prev_timestamp = time_stp
    print('time_sum ', time_sum)

    return

class ObjTexCreatorPipeline:
    def __init__(self, in_model_path, optim_cfg_json, pose_json, lrm_mode, device='cuda'):
        """End-End object tex creator pipeline class

        Args:
            in_model_path: path of StableDiffusionTexCreatorPipeline
            optim_cfg_json: like grpc_backend/configs/optim_cfg_high.json
            pose_json: like data/cams/cam_parameters_srender8.json
            lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
            device: cuda
        """
        print("debug ObjTexCreatorPipeline input ", in_model_path, optim_cfg_json, pose_json, lrm_mode, device)
        self.in_model_path = in_model_path
        self.device = device

        try:
            assert os.path.exists(self.in_model_path), f'Error: can not find valid sd model {self.in_model_path}'
            assert os.path.exists(optim_cfg_json), f'Error: can not find valid optim_cfg_json {optim_cfg_json}'
            assert os.path.exists(pose_json), f'Error: can not find valid pose_json {pose_json}'

            self.optim_cfg = load_json(optim_cfg_json)

            self.pipeline, self.generator, self.glctx = self.init_model_glctx(self.in_model_path)
            print(f'load sd model {self.in_model_path} done')

            self.mvp, self.w2c = self.load_pose_json(pose_json, lrm_mode=lrm_mode)
            self.lrm_mode = lrm_mode
            print(f'load {self.mvp.shape[0]} pose from {pose_json} done. init done')

            self.sr_client = SrGenClient(server_addr="ip_addr:80")
            self.sr_temp_dir = "/aigc_cfs_3/sz/server/sr/"

        except:
            raise ValueError('init ObjTexCreatorPipeline failed')

    def init_model_glctx(self, in_model_path):
        """init diffusion model and render pose/glctx. only need once.

        Args:
            in_model_path: diffusion model path

        Returns:
            pipeline, generator,  glctx
        """
        pipeline = StableDiffusionTexCreatorPipeline.from_pretrained(
            in_model_path, torch_dtype=torch.float16).to(self.device)
        generator = torch.Generator(self.device)

        glctx = dr.RasterizeCudaContext()
        return pipeline, generator, glctx

    def load_pose_json(self, pose_json, lrm_mode, select_view=[]):
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
        mvp = frames['mvp'].to(self.device)
        w2c = frames['w2c'].to(self.device)
        if select_view and len(select_view) > 0:
            view_cnt = mvp.shape[0]
            filtered_list = [x for x in select_view if x < view_cnt]
            mvp, w2c = mvp[filtered_list], w2c[filtered_list]
        return mvp, w2c

    # TODO decimate_target... input infer cfg
    def optim_texcreator(
        self,
        in_obj,
        in_condi,
        out_debug_dir=None,
        decimate_target=30000,
        verbose=False,
        debug_paste_condi=False,
    ):
        """render depth, infer with SD, optim the texture map

        Args:
            in_obj: raw obj path with uv coord
            in_condi: condi img path
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Returns:
            new mesh : class Mesh
        """
        pipeline: StableDiffusionTexCreatorPipeline = self.pipeline
        generator: torch.Generator = self.generator
        glctx: dr.RasterizeCudaContext() = self.glctx
        optim_cfg: dict = self.optim_cfg
        mvp = self.mvp
        w2c = self.w2c

        # 1. load obj and render raw depth
        time_list = []
        time_list.append(('start', time.time()))
        if not os.path.exists(in_obj):
            print(f'ERROR can not find in_obj {in_obj}')
            return None
        if not os.path.exists(in_condi):
            print(f'ERROR can not find in_condi {in_condi}')
            return None
        print('debug optim_cfg ', optim_cfg)

        # debug_need_pro_mesh = False
        debug_need_pro_mesh = False
        debug_need_align_y_mesh = True
        debug_need_align_z_mesh = True
        use_optim = True    # optim tex or grid put
        # TODO only for debug
        main_view_id = 2    # 2 when 9 pose,  3 when 8 pose
        paste_condi_view = main_view_id
        main_views, ignore_views = [main_view_id], []
        # main_views, ignore_views = [main_view_id], [-1]
        main_weight, ignore_weight = 5, 0.5
        print('debug_paste_condi debug_need_pro_mesh ', debug_paste_condi, debug_need_pro_mesh)

        if debug_need_pro_mesh:
            raw_mesh: Mesh = load_mesh(in_obj)
            mesh_normalized(raw_mesh)
            clean_decimate_mesh(raw_mesh, decimate_target=decimate_target)
            in_mesh = mesh_xatlas(raw_mesh)
            in_mesh.material = raw_mesh.material
            print('process mesh done')
            time_list.append(('process_mesh_done', time.time()))
        else:
            in_mesh: Mesh = load_mesh(in_obj)

        if debug_need_align_z_mesh:
            in_mesh = align_z_mesh(in_mesh)
            print(f'Warning! need rm when all dataset fix u-up and re-train!!!!!!!!!')

        vtx_pos, pos_idx = in_mesh.v_pos, in_mesh.t_pos_idx   # [Nv, 3],  [Nf, 3]
        vtx_uv, uv_idx = in_mesh.v_tex, in_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]

        ## render depth as input, depth_bhwc [b, render_res, render_res, 1]
        resolution_depth = optim_cfg["render_res"] // 8
        depth_bhwc = render_depth_views(
            glctx, vtx_pos, pos_idx, mvp, w2c, optim_cfg["render_res"]
        )
        depth_np = depth_bhwc.detach().cpu().numpy()
        # condi_depth_tensor: [b, 1, render_res/8, render_res/8] as bchw
        condi_depth_tensor = F.interpolate(
            depth_bhwc.permute(0, 3, 1, 2),
            scale_factor=1 / 8,
            mode="bilinear",
            align_corners=False,
        )
        condi_depth_tensor = depth_normalize_tensor(condi_depth_tensor)

        # 2. infer texrefine model, get new images
        prompt_img_path = [in_condi] * condi_depth_tensor.shape[0]
        # list of PIL, TODO just return tensor, output_type=np.array?
        infer_images_pil = pipeline(
            prompt_img_path,
            depth=condi_depth_tensor,
            step_mask=False,
            num_inference_steps=optim_cfg['num_inference_steps'],
            image_guidance_scale=optim_cfg['image_guidance_scale'],
            guidance_scale=optim_cfg['guidance_scale'],
            generator=generator,
        ).images
        time_list.append(('infer_done', time.time()))

        ## SR TODO, need 120s
        if optim_cfg.get("use_sr", False):
            infer_images_pil = self.sr_infer_pils(infer_images_pil)
            time_list.append(('sr_done', time.time()))


        numpy_arrays = [np.array(img) for img in infer_images_pil]
        # infer_images tensor [b, render_res, render_res, 3]
        infer_images = torch.from_numpy(np.stack(numpy_arrays)).to(pipeline.device)
        infer_images = infer_images.to(torch.float32) / 255.
        time_list.append(('infer_tensor_done', time.time()))

        if use_optim:
            bake_res = optim_cfg.get("bake_res", 1024)
            depth_sp = bake_res // depth_bhwc.shape[1]
            infer_sp = bake_res // infer_images.shape[1]
            if depth_sp != 1:
                depth_bhwc = resize_render(depth_bhwc, depth_sp)
            if infer_sp != 1:
                infer_images = resize_render(infer_images, infer_sp)
            print('debug depth_sp infer_sp ', depth_sp, infer_sp)
            
            
        raw_alpha = depth_bhwc > 0   # [b, render_res, render_res, 1]
        uv_refine_type = optim_cfg.get("uv_refine_type", "pos_knn")
        if uv_refine_type == "raw" or uv_refine_type == "uv_knn":
            # TODO(csz)
            _, raw_alpha = mask_refine(depth_bhwc, infer_images, raw_alpha, thres=180, k=50)
        time_list.append(('dilate_done', time.time()))

        # 3. optim tex Adam optimizer for texture with a learning rate ramp.
        optim_render_res = int(optim_cfg.get("bake_res", 1024))
        if debug_paste_condi:
            # get tensor [h, w, 3]
            condi_pil = load_rgba_as_rgb(in_condi, res=optim_render_res)
            condi_image = (torch.from_numpy(np.array(condi_pil)) / 255.).to(infer_images.device)

        if use_optim:
            tex_data_opt = init_opt_tex(optim_cfg['tex_res'], background='gray', device=pipeline.device)
            optimizer = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x)/float(optim_cfg['max_iter'])))

            min_cos = optim_cfg.get("min_cos", 0.1)
            if min_cos > 0:
                in_mesh = auto_normals(in_mesh)
                viwecos_all = render_normal_cos(glctx, vtx_pos, pos_idx, mvp, in_mesh.v_nrm, w2c, optim_render_res)
                viwecos_mask_all = (viwecos_all > min_cos).float()
                print('debug min_cos ', min_cos)
            else:
                viwecos_mask_all = None


            for it in range(optim_cfg['max_iter'] + 1):
                # color_opt [nv, resolution, resolution, x]
                color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt,
                                                            optim_render_res, optim_cfg['max_mip_level'])
                if min_cos > 0:
                    alpha_opt = viwecos_mask_all * alpha_opt

                # [b, optim_render_res, optim_render_res, 3]
                weight = torch.ones_like(color_opt)
                if main_views and not debug_paste_condi:
                    weight[main_views, ...] *= main_weight
                if ignore_views:
                    weight[ignore_views, ...] *= ignore_weight

                # loss_tex = torch.mean((tex_data_merge - tex_data_opt)**2) * optim_cfg['lamda_tex']
                loss_tex = 0
                if debug_paste_condi:
                    condi_alpha = alpha_opt[paste_condi_view]
                    condi_render = color_opt[paste_condi_view]
                    loss_tex = torch.mean((condi_image * condi_alpha - condi_render * condi_alpha)**2)
                    loss_tex *= optim_cfg.get('lamda_condi', 5)

                loss = loss_tex + torch.mean(weight * ((infer_images * raw_alpha - color_opt * raw_alpha)**2))  # L2 pixel loss.
                optimizer.zero_grad()
                # loss.backward()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

                ### refine uv tex
                if it == max(10, optim_cfg['max_iter'] - 20) and uv_refine_type != "raw":
                    # refine texture uv: update tex_data_opt in-place
                    if uv_refine_type != "raw":
                        inpaint_refine_uv_tex(
                            glctx,
                            in_mesh,
                            tex_data_opt,
                            mvp,
                            uv_refine_type=uv_refine_type,
                            viwecos_mask_all=viwecos_mask_all,
                            mask_sp=optim_cfg.get("mask_sp", 4),
                            debug_dir=out_debug_dir,
                        )
                        print('debug update tex')

                if 'log_interval' in optim_cfg and (it % optim_cfg['log_interval'] == 0) and out_debug_dir is not None:
                    psnr = mse_to_psnr(loss.item())
                    s = "iter=%d,loss=%f,psnr=%f" % (it, loss.item(), psnr)
                    print(s)
                    print("debug optim res ", color_opt.shape[1])

            # 4. generate new texture map and new obj
            new_mesh = in_mesh.clone()
            new_mesh.material = in_mesh.material
            if tex_data_opt.shape[-1] > 4:
                new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :-3])
                new_mesh.material['norml'] = texture.Texture2D(tex_data_opt[..., -3:])
            else:
                new_mesh.material['kd'] = texture.Texture2D(tex_data_opt[..., :3])

        else:
            # tensor [h, w, 3] and np [uv_res, uv_res] [uv_res, uv_res]
            uv_tex_new, inpaint_region, texture_mask = recover_texture_grid(infer_images, in_mesh, glctx, mvp, w2c,
                                                                            optim_cfg.get('uv_res', 512), # TODO
                                                                            optim_cfg.get('min_view_cos', -100))    # TODO
            new_mesh = in_mesh.clone()
            new_mesh.material = in_mesh.material
            uv_tex_new = uv_tex_new.unsqueeze(0)
            new_mesh.material['kd'] = texture.Texture2D(uv_tex_new)

        time_list.append(('texture_done', time.time()))
        if debug_need_align_y_mesh:
            new_mesh = align_y_mesh(new_mesh)

        # 5. vis for debug
        try:
            if out_debug_dir is not None:
                print('debug condi_depth_tensor ', condi_depth_tensor.shape)
                print('debug infer_images ', infer_images.shape)
                print('debug raw_alpha ', raw_alpha.shape)
                print('debug out_debug_dir ', out_debug_dir)
                # save masked/infer/depth/condi/opti imgs
                masked_vis = []
                for i, infer_image in enumerate(infer_images):
                    masked_infer = (raw_alpha[i] * infer_image).detach().cpu().numpy()
                    masked_infer = np.clip(np.rint(masked_infer * 255.0), 0, 255).astype(np.uint8)
                    masked_vis.append(Image.fromarray(masked_infer).resize(
                        (optim_cfg['render_res'], optim_cfg['render_res'])))

                infer_vis = [image.resize((optim_cfg['render_res'], optim_cfg['render_res'])) for i, image in enumerate(infer_images_pil)]
                depth_vis = [Image.fromarray(depth_normalize(depth[..., 0])) for i, depth in enumerate(depth_np)]

                condi_img = load_rgba_as_rgb(in_condi)
                condi_vis = [condi_img for i in range(infer_images.shape[0])]

                # condi_img.save(os.path.join(out_debug_dir, 'debug_condi.jpg'))

                vis_pil = [depth_vis, condi_vis, infer_vis, masked_vis]
                try:
                    if color_opt is not None and torch.is_tensor(color_opt):
                        opt_vis = [
                            Image.fromarray(
                                np.clip(np.rint(color_opt[idx].detach().cpu().numpy() * 255.0), 0,
                                        255).astype(np.uint8)).resize((optim_cfg['render_res'], optim_cfg['render_res']))
                            for idx in range(color_opt.shape[0])
                        ]
                        vis_pil.append(opt_vis)
                except:
                    pass
                concatenate_images_2d(vis_pil, os.path.join(out_debug_dir, f'debug_vis.jpg'))

                # TODO
                debug_save_infer_vis = True
                if debug_save_infer_vis:
                    res_dir = os.path.join(out_debug_dir, "bake")
                    os.makedirs(res_dir, exist_ok=True)
                    for i, image in enumerate(infer_images_pil):
                        image.save(os.path.join(res_dir, f"cam-{i:04d}.png"))

                if not use_optim:
                    try:
                        if texture_mask is not None:
                            Image.fromarray(np.clip(texture_mask * 255., 0, 255).astype(np.uint8)).save(os.path.join(out_debug_dir, f'debug_mask.jpg'))
                        if inpaint_region is not None:
                            Image.fromarray(inpaint_region).save(os.path.join(out_debug_dir, f'debug_inpaint.jpg'))
                    except Exception as e:
                        print('Warn vis for texture_mask failed. skip', e)

                # generate rotate gif, just for vis
                # TODO from json
                frame_k = np.array([
                    [703.354248046875, 0.0, 255.5],
                    [0.0, 703.354248046875, 255.5],
                    [0.0, 0.0, 1.0]])
                mvp = rotate_scene_mvp(frame_k, itr_all=40, cam_radius=3.8, lrm_mode=self.lrm_mode)['mvp']

                color_opt, alpha_opt = render_texture_views(glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, new_mesh.material['kd'].data,
                                                            int(optim_cfg['render_res']), optim_cfg['max_mip_level'])
                print('debug color_opt ', color_opt.shape)
                background = torch.ones_like(color_opt)
                color_tex = mix_rgba(color_opt, alpha_opt, background)

                img_cpu_list = []
                for idx in range(color_tex.shape[0]):
                    img = color_tex[idx]

                    img_cpu = img.detach().cpu().numpy()
                    img_cpu = np.clip(np.rint(img_cpu * 255.0), 0, 255).astype(np.uint8)
                    img_cpu_list.append(img_cpu)

                imageio.mimsave(os.path.join(out_debug_dir, 'output.gif'), img_cpu_list, duration=0.05)
        except Exception as e:
            print('Warn vis for debug failed. skip', e)
            return new_mesh
        time_list.append(('create_done', time.time()))

        try:
            log_time_list(time_list)
        except Exception as e:
            print('log_time_list failed. skip', e)

        return new_mesh

    def sr_infer_pils(self, infer_images_pil, up_scale = 2):
        timestamp = int(time.time())
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
        out_sr_dir = os.path.join(self.sr_temp_dir, str(unique_id))
        os.makedirs(out_sr_dir, exist_ok=True)       
        sr_pils = []
        
        for idx, infer_image_pil in enumerate(infer_images_pil):
            # TODO direct with pil
            raw_path = os.path.join(out_sr_dir, f"raw_{idx}.png")
            infer_image_pil.save(raw_path)
            out_mesh_paths = self.sr_client.client_sr(raw_path, out_sr_dir, up_scale)
            if out_mesh_paths and len(out_mesh_paths) >= 1:
                sr_img = out_mesh_paths[0]
                sr_pils.append(load_rgba_as_rgb(sr_img))
            else:
                print(f"Error, sr failed {raw_path}")
                sr_pils.append(infer_image_pil)
        
        return sr_pils

    # interface for grpc
    def interface_set_render_pose(self, pose_json, lrm_mode, select_view=[]):
        """set render pose from pose_json and lrm_mode

        Args:
            pose_json: like data/cams/cam_parameters_srender8.json
            lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
            select_view: _description_. Defaults to [].

        Returns:
            pose_json
        """
        self.mvp, self.w2c = self.load_pose_json(pose_json, lrm_mode=lrm_mode, select_view=select_view)
        self.lrm_mode = lrm_mode
        return pose_json

    def interface_obj_generate_tex(self, in_obj, in_condi, out_obj, out_debug_dir=None, debug_paste_condi=False):
        """render depth from in_obj, then infer SD with in_condi, get tex rgb and optim uv texture

        Args:
            in_obj: raw obj path with uv coord
            in_condi: condi img path
            out_obj: output obj path
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Returns:
            out_obj or ''
        """
        if not self.pipeline:
            print(f'ERROR: The pipeline has not been initialized')
            return ''
        try:
            new_mesh = self.optim_texcreator(in_obj, in_condi, out_debug_dir=out_debug_dir, debug_paste_condi=debug_paste_condi)
            os.makedirs(os.path.dirname(out_obj), exist_ok=True)
            write_obj(os.path.dirname(out_obj), new_mesh)
            print('SUCCEED out_obj ', out_obj)
        except Exception as e:
            print(f'ERROR: optim_texcreator failed', e)
            return ''

        if not os.path.exists(out_obj):
            print(f'ERROR: The pipeline has not been initialized')
            return ''

        return out_obj

    # unit test
    def test_pipe_obj(self, pose_json, in_obj, in_condi, out_obj, lrm_mode, select_view=[], out_debug_dir=None, debug_paste_condi=False):
        print('input debug1 ', pose_json, lrm_mode, select_view)
        assert self.interface_set_render_pose(pose_json, lrm_mode, select_view=select_view) != ''
        print('render pose set: ', self.mvp.shape, self.w2c.shape)

        print('input debug2 ', in_obj, in_condi, out_obj, out_debug_dir)
        assert self.interface_obj_generate_tex(in_obj, in_condi, out_obj, out_debug_dir=out_debug_dir, debug_paste_condi=debug_paste_condi) != ''
        assert os.path.exists(os.path.join(os.path.dirname(out_obj), 'texture_kd.png'))
        print('new obj: ', out_obj)

        print('test_pipe_obj passed')
        return
