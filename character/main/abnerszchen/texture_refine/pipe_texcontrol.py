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
import shutil
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from easydict import EasyDict as edict

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj
from render.geom_utils import mesh_normalized
import render.texture as texture
from render.uv_conditions import render_geometry_uv, cvt_geom_to_pil
from dataset.utils_dataset import (
    load_json,
    save_lines,
    depth_normalize_tensor,
    load_rgba_as_rgb,
    concatenate_images_2d,
    concatenate_images_horizontally,
)


def init_opt_tex(tex_res, background="gray", device="cuda"):
    if background == "gray":
        tex = torch.ones((1, tex_res, tex_res, 3)).to(device) * 0.5
    elif background == "black":
        tex = torch.zeros((1, tex_res, tex_res, 3)).to(device)
    elif background == "white":
        tex = torch.ones((1, tex_res, tex_res, 3)).to(device)
    else:
        print("invalid init_opt_tex background ", background)
        return None
    tex_merge = torch.nn.Parameter(tex.clone(), requires_grad=True)
    return tex_merge


class ObjTexControlPipeline:
    def __init__(self, in_model_path, in_sd_path, ip_adapter_model_path, device="cuda"):
        """End-End object tex UV control pipeline class

        Args:
            in_model_path: path of ControlNetModel
            in_sd_path: path of StableDiffusionControlNetPipeline
            ip_adapter_model_path: path of ip_adapter
            device: cuda
        """
        self.in_model_path = in_model_path
        self.device = device
        self.pipe_type = "tex_control"

        try:
            if not os.path.exists(self.in_model_path):
                print(f"Error: can not find valid sd model {self.in_model_path}")
            if not os.path.exists(in_sd_path):
                print(f"Error: can not find valid in_sd_path {in_sd_path}")
            if not os.path.exists(ip_adapter_model_path):
                print(
                    f"Error: can not find valid ip_adapter_model_path {ip_adapter_model_path}"
                )

            self.pipeline, self.generator, self.glctx = self.init_model_glctx(
                in_model_path, in_sd_path, ip_adapter_model_path
            )

            print(f"load sd model {self.in_model_path} done")
            self.cfg = self.init_cfg()

        except:
            raise ValueError("init ObjTexControlPipeline failed")

    def init_model_glctx(self, in_model_path, in_sd_path, ip_adapter_model_path, seed=42):
        """init diffusion model and render pose/glctx. only need once.

        Args:
            in_model_path: path of ControlNetModel
            in_sd_path: path of StableDiffusionControlNetPipeline
            ip_adapter_model_path: path of ip_adapter
            seed

        Returns:
            pipeline, generator, glctx
        """
        controlnet = ControlNetModel.from_pretrained(
            in_model_path, torch_dtype=torch.float16
        )
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            in_sd_path, controlnet=controlnet, torch_dtype=torch.float16
        ).to(self.device)
        pipeline.load_ip_adapter(
            ip_adapter_model_path, subfolder="models", weight_name="ip-adapter_sd15.bin"
        )
        pipeline.safety_checker = None

        generator = torch.Generator(self.device).manual_seed(seed)

        glctx = dr.RasterizeCudaContext()
        return pipeline, generator, glctx
    
    def init_cfg(self):
        """init default run cfg

        Returns:
            _description_
        """
        cfg = {
            "uv_res": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "debug_save": True,
        }

        cfg = edict(cfg)
        return cfg
    
    def load_normal(self, normal_path):
        if not os.path.exists(normal_path):
            print(f"ERROR can not find normal_path {normal_path}")
            return None
        return Image.open(normal_path).convert("RGB")

    def infer_uv_normal(
        self,
        in_prompts,
        in_uv_normal_pils,
        in_condi_img_pils=None,
        num_inference_steps=20,
        guidance_scale=7.5,
    ):
        """run controlnet SD

        Args:
            in_prompts (list): list of text, can be list of []
            in_uv_normal_pils (list of pil) list of normal pil(0-255, h, w, 3). TODO(csz) need support list of str/pil/tensor/np
            in_condi_img_pils (list of pil) or None
            num_inference_steps (int, optional): _description_. Defaults to 20.
            guidance_scale (int, optional): _description_. Defaults to 7.5.

        Returns:
            out_uv_pils: list of uv pil
        """
        if len(in_uv_normal_pils) != len(in_prompts):
            print(
                "ERROR invalid input, not equal list ",
                len(in_prompts),
                len(in_uv_normal_pils),
            )
            return None

        pipeline: StableDiffusionControlNetPipeline = self.pipeline
        generator: torch.Generator = self.generator

        if in_condi_img_pils is not None:
            out_uv_pils = pipeline(
                in_prompts,
                image=in_uv_normal_pils,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                ip_adapter_image=in_condi_img_pils,
            ).images
        else:
            print("debug ip_adapter_image=None ", in_condi_img_pils)
            pipeline.set_ip_adapter_scale(0)
            # gray with be zero after pre-process
            gray_imgs = [Image.new("RGB", (512, 512), (128, 128, 128))] * len(
                in_uv_normal_pils
            )
            out_uv_pils = pipeline(
                in_prompts,
                image=in_uv_normal_pils,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                ip_adapter_image=gray_imgs,
            ).images
            pipeline.set_ip_adapter_scale(1)
            
        return out_uv_pils

    def obj_texcontrol(
        self,
        in_obj,
        in_prompts: Union[str, List[str]] = None,
        in_condi_img=None,
        uv_res=512,
        num_inference_steps=20,
        guidance_scale=3,
        out_debug_dir=None,
    ):
        """for one obj, render UV normal, generate uv texture maps

        Args:
            in_obj: raw obj path with uv coord
            in_prompts: string or list of prompt string for one obj, or None
            in_condi_img: condition image or None,  only support one condition img with multi text now TODO(csz)
            uv_res: render normal and out texture resolution
            num_inference_steps
            guidance_scale
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Returns:
            list of new mesh : list of class Mesh
        """

        # 1. load obj and render uv normal
        if not os.path.exists(in_obj):
            print(f"ERROR can not find in_obj {in_obj}")
            return None

        if in_prompts is None:
            in_prompts = [""]
        elif isinstance(in_prompts, str):
            in_prompts = [in_prompts]
        elif isinstance(in_prompts, list):
            if len(in_prompts) < 1 or not isinstance(in_prompts[0], str):
                print(f"ERROR empty in_prompts {in_prompts}")
                return None
        in_condi_img_pils = None
        if in_condi_img is not None and os.path.exists(in_condi_img):
            condi_img_pil = load_rgba_as_rgb(in_condi_img)
            in_condi_img_pils = [condi_img_pil] * len(in_prompts)

        raw_mesh: Mesh = load_mesh(in_obj)
        raw_in_mesh = raw_mesh.clone()

        try:
            mesh_normalized(raw_mesh)
        except:
            print("warning mesh_normalized failed. skip mesh_normalized")
        raw_mesh = auto_normals(raw_mesh)

        gb_xyz, gb_normal, gb_mask = render_geometry_uv(self.glctx, raw_mesh, uv_res)
        normal_pil = cvt_geom_to_pil(gb_normal, gb_mask)
        
        # 2. infer texrefine model, get new images
        in_uv_normal_pils = [normal_pil] * len(in_prompts)
        out_uv_pils = self.infer_uv_normal(
            in_prompts,
            in_uv_normal_pils,
            in_condi_img_pils=in_condi_img_pils,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if not out_uv_pils:
            print("ERROR infer_uv_normal failed")
            return None

        # 3. generate new texture map and new obj
        new_meshs, tex_list = [], []
        for i, out_uv_pil in enumerate(out_uv_pils):
            new_mesh = raw_in_mesh.clone()
            new_mesh.material = copy.deepcopy(
                raw_in_mesh.material
            )  # need deep copy, avoid in-place modifications
            # TODO(csz) refine. skip pil to tensor
            image = np.array(out_uv_pil)
            image = torch.from_numpy(image).unsqueeze(0) / 255.0  # [1,h,w,3]
            tex = image.to(self.device)
            new_mesh.material["kd"] = texture.Texture2D(tex)

            new_meshs.append(new_mesh)
            tex_list.append(tex)

            del new_mesh
            del tex

        # 5. vis for debug
        try:
            if out_debug_dir is not None:
                print("debug in_obj ", in_obj)
                print("debug in_prompts ", in_prompts)
                print("debug out_uv_pils cnt ", len(out_uv_pils))
                print("debug new_meshs ", len(new_meshs))

                # save uv-normal/out-tex imgs
                vis_pil = [in_uv_normal_pils, out_uv_pils]
                concatenate_images_2d(
                    vis_pil, os.path.join(out_debug_dir, f"debug_vis.jpg")
                )

                # # generate rotate gif, just for vis
                # # TODO from json
                # frame_k = np.array([
                #     [703.354248046875, 0.0, 255.5],
                #     [0.0, 703.354248046875, 255.5],
                #     [0.0, 0.0, 1.0]])
                # mvp = rotate_scene_mvp(frame_k, itr_all=40, cam_radius=3.8)['mvp']

                # color_opt, alpha_opt = render_texture_views(self.glctx, vtx_pos, pos_idx, vtx_uv, uv_idx, mvp, tex_data_opt, 512)
                # print('debug color_opt ', color_opt.shape)
                # background = torch.ones_like(color_opt)
                # color_tex = mix_rgba(color_opt, alpha_opt, background)

                # img_cpu_list = []
                # for idx in range(color_tex.shape[0]):
                #     img = color_tex[idx]

                #     img_cpu = img.detach().cpu().numpy()
                #     img_cpu = np.clip(np.rint(img_cpu * 255.0), 0, 255).astype(np.uint8)
                #     img_cpu_list.append(img_cpu)

                # imageio.mimsave(os.path.join(out_debug_dir, 'output.gif'), img_cpu_list, duration=0.05)

        except:
            print("Warn vis for debug failed. skip")
            return new_meshs
        return new_meshs

    # interface for grpc
    def interface_obj_texcontrol(
        self, in_obj, out_objs_dir, in_prompts=None, in_condi_img=None, run_cfg=None
    ):
        """grpc interface: get uv texture from obj, text and image. pipe_control

        Args:
            in_obj: raw obj path with uv coord
            out_objs_dir: output dir with multi output objs
            in_prompts: input text, list text or None
            in_condi_img: condi img path or None
            run_cfg: json or dict or easydict of run-cfg, refer: codedir/configs/tex_gen.json
                    use key uv_res, num_inference_steps, guidance_scale, debug_save

        Returns:
            out_obj_paths is list of obj path or '' means failed
        """
        if not self.pipeline:
            print(f"ERROR: The pipeline has not been initialized")
            return ""

        if run_cfg is None:
            run_cfg = self.cfg
        # TODO(csz) complete cfg
        cfg = edict(run_cfg)

        out_obj_paths = []
        try:
            new_meshs = self.obj_texcontrol(
                in_obj,
                in_prompts=in_prompts,
                in_condi_img=in_condi_img,
                uv_res=cfg.uv_res,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                out_debug_dir=out_objs_dir if cfg.debug_save else None,
            )
            if new_meshs is None:
                print(f"ERROR: obj_teximguv run ok but return nothing")
                return ""

            os.makedirs(os.path.dirname(out_objs_dir), exist_ok=True)
            for idx, new_mesh in enumerate(new_meshs):
                out_obj = os.path.join(out_objs_dir, f"obj_{idx:03d}/mesh.obj")
                write_obj(os.path.dirname(out_obj), new_mesh)
                if os.path.exists(out_obj):
                    out_obj_paths.append(out_obj)

            try:
                if cfg.debug_save:
                    in_prompts_ = [""] if in_prompts is None else in_prompts
                    if isinstance(in_prompts_, str):
                        in_prompts_ = [in_prompts_]
                    print('in_prompts_', in_prompts_)
                    save_lines(
                        in_prompts_, os.path.join(out_objs_dir, f"in_prompts.txt")
                    )
                    if in_condi_img is not None and os.path.exists(in_condi_img):
                        shutil.copyfile(
                            in_condi_img, os.path.join(out_objs_dir, f"in_condi.png")
                        )
            except Exception as e:
                print(f"Warn: skip debug_save", e)

        except Exception as e:
            print(f"ERROR: obj_teximguv failed", e)
            return ""

        return out_obj_paths

    # unit test
    def test_pipe_obj_texcontrol(
        self, in_obj, out_objs_dir, in_prompts=None, in_condi_img=None, debug_save=True, run_cfg=None
    ):  
        if not run_cfg:
            run_cfg = self.cfg
        run_cfg.debug_save = debug_save

        print("input test_pipe_obj_texcontrol: ", in_obj, out_objs_dir, in_prompts, in_condi_img, debug_save)
        print('test run_cfg ', run_cfg)
        out_obj_paths = self.interface_obj_texcontrol(
            in_obj,
            out_objs_dir,
            in_prompts=in_prompts,
            in_condi_img=in_condi_img,
            run_cfg=run_cfg,
        )
        print("out_obj_paths ", out_obj_paths)

        return
