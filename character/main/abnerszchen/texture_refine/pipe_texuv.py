import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import nvdiffrast.torch as dr
import shutil
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from easydict import EasyDict as edict
import PIL.Image
import traceback
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj
from render.geom_utils import mesh_normalized
import render.texture as texture
from render.uv_conditions import render_geometry_uv, cvt_geom_to_pil
from dataset.utils_dataset import (
    load_json,
    save_lines,
    save_json,
    load_rgba_as_rgb,
    concatenate_images_2d,
    concatenate_images_horizontally,
)

def query_gpu_mem():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory_size = gpu_properties.total_memory
        gpu_memory_size_g = gpu_memory_size / (1024 * 1024 * 1024)
        print(f"GPU memory size: {gpu_memory_size_g:.2f} G")
        return gpu_memory_size_g
    else:
        print("GPU is not available.")
        return -1

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


class ObjTexUVPipeline:
    def __init__(self, in_model_path, in_sd_path, pretrained_vae_model_name_or_path, ip_adapter_model_path, device="cuda"):
        """End-End object tex UV xl+control pipeline class

        Args:
            in_model_path: path of ControlNetModel
            in_sd_path: path of StableDiffusionXLControlNetPipeline
            pretrained_vae_model_name_or_path: vae for fp16
            ip_adapter_model_path: path of ip_adapter
            device: cuda
        """
        self.in_model_path = in_model_path
        self.device = device
        self.pipe_type = "tex_uv"

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
                in_model_path, in_sd_path, pretrained_vae_model_name_or_path, ip_adapter_model_path, device=device
            )

            print(f"load sd model {self.in_model_path} done")
            self.cfg = self.init_cfg()

        except:
            raise ValueError("init ObjTexUVPipeline failed")

    def init_model_glctx(
        self,
        in_model_path,
        in_sd_path,
        pretrained_vae_model_name_or_path,
        ip_adapter_model_path,
        device="cuda",
        use_ip_mode="vit-h",
        seed=42,
        run_t10=False,
        run_compile=False,
    ):
        """init diffusion model and glctx. only need once.

        Args:
            in_model_path: path of ControlNetModel
            in_sd_path: path of StableDiffusionXLControlNetPipeline
            pretrained_vae_model_name_or_path: vae for fp16
            ip_adapter_model_path: path of ip_adapter
            device: cuda or npu
            use_ip_mode: ip adapter xl mode: raw / vit-h / plus_vit-h. Defaults to "plus_vit-h".
            seed
            run_t10: _description_. Defaults to False.

            # models/image_encoder: OpenCLIP-ViT-H-14 with 632.08M parameter
            # sdxl_models/image_encoder: OpenCLIP-ViT-bigG-14 with 1844.9M parameter
            # ip-adapter_sdxl.bin: use global image embedding from OpenCLIP-ViT-bigG-14 as condition
            # ip-adapter_sdxl_vit-h.bin: same as ip-adapter_sdxl, but use OpenCLIP-ViT-H-14
            # ip-adapter-plus_sdxl_vit-h.bin: use patch image embeddings from OpenCLIP-ViT-H-14 as condition, closer to the reference image than ip-adapter_xl and ip-adapter_sdxl_vit-h

        Returns:
            pipeline, generator, glctx
        """
        weight_dtype = torch.float16
        controlnet = ControlNetModel.from_pretrained(in_model_path, torch_dtype=weight_dtype)
        if pretrained_vae_model_name_or_path is not None:
            vae = AutoencoderKL.from_pretrained(
                pretrained_vae_model_name_or_path, torch_dtype=weight_dtype
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                in_sd_path, subfolder="vae", torch_dtype=weight_dtype
            )

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            in_sd_path,
            vae=vae,
            controlnet=controlnet,
            variant="fp16",
            torch_dtype=weight_dtype,
            use_safetensors=True,
        ).to(device)

        if use_ip_mode == "raw":
            ip_image_encoder_folder = "sdxl_models/image_encoder"
            weight_name = "ip-adapter_sdxl.safetensors"
        elif use_ip_mode == "vit-h":
            ip_image_encoder_folder = "models/image_encoder"
            weight_name = "ip-adapter_sdxl_vit-h.safetensors"
        elif use_ip_mode == "plus_vit-h":
            ip_image_encoder_folder = "models/image_encoder"
            weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
        else:
            raise ValueError(f"invalid use_ip_mode {use_ip_mode}")

        pipeline.load_ip_adapter(
            ip_adapter_model_path,
            subfolder="sdxl_models",
            weight_name=weight_name,
            image_encoder_folder=ip_image_encoder_folder,
        )
        print(f"load ip adapter done from {ip_adapter_model_path}, use_ip_mode: {use_ip_mode}")

        pipeline.safety_checker = None

        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        # less memory but slower. used in t10 instead of v100
        run_t10 = query_gpu_mem() < 20
        if run_t10:
            print('run in t10')
            # # # remove this if torch.__version__ >= 2.0.0
            # pipeline.enable_xformers_memory_efficient_attention()
            # memory optimization.
            pipeline.enable_model_cpu_offload()

        # TODO(csz) need upgrade cuda
        if run_compile:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            pipeline.controlnet = torch.compile(pipeline.controlnet, mode="reduce-overhead", fullgraph=True)
            # TODO infer once

        generator = torch.Generator(self.device).manual_seed(seed)

        glctx = None
        try:
            glctx = dr.RasterizeCudaContext()
            print(f"dr.RasterizeCudaContext() ok")
        except Exception as e:
            print(f"[warn] dr.RasterizeCudaContext() failed, e = {e}")
            try:
                glctx = dr.RasterizeGLContext()
                print(f"dr.RasterizeGLContext() ok")
            except Exception as e:
                print(f"[warn] dr.RasterizeGLContext() failed, e = {e}, can not use rast mode")
                
        return pipeline, generator, glctx

    # TODO can also load SD

    def init_cfg(self):
        """init default run cfg

        Returns:
            edict
        """
        cfg = {
            "uv_res": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 9.0,
            "controlnet_conditioning_scale": 0.8,
            "ip_adapter_scale": 0.8,
            "debug_save": True,
        }

        cfg = edict(cfg)
        return cfg

    def load_condi(self, condi_path):
        if not os.path.exists(condi_path):
            print(f"ERROR can not find condi_path {condi_path}")
            return None
        return Image.open(condi_path).convert("RGB")

    def add_prompt_desc(self, in_prompts:list):
        append_desc = ", HDR, UHD, 4K"
        new_prompts = []
        for raw_prompt in in_prompts:
            if raw_prompt and append_desc not in raw_prompt:
                new_prompts.append(raw_prompt + append_desc)
            else:
                new_prompts.append(raw_prompt)

        return new_prompts

    def infer_uv_xl_geom(
        self,
        in_prompts,
        in_uv_geom_pils,
        in_condi_img_pils=None,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
    ):
        """run controlnet SDXL

        Args:
            in_prompts (list): list of text, can be list of []
            in_uv_geom_pils (list of pil) list of geom pil(0-255, h, w, 3/x). TODO(csz) need support list of str/pil/tensor/np
            in_condi_img_pils (list of pil) or None
            num_inference_steps (int, optional): _description_. Defaults to 20.
            guidance_scale (int, optional): The larger the value, the closer it is to the text prompt. Defaults to 9.0.
            controlnet_conditioning_scale (int, optional): The higher the value, the better the control but the worse the quality. in [0, 1]
            ip_adapter_scale (int, optional): The larger the value, the closer it is to the image. in [0, 1]

        Returns:
            out_uv_pils: list of uv pil
        """
        if len(in_uv_geom_pils) != len(in_prompts):
            print(
                "ERROR invalid input, not equal list ",
                len(in_prompts),
                len(in_uv_geom_pils),
            )
            return None

        pipeline: StableDiffusionXLControlNetPipeline = self.pipeline
        generator: torch.Generator = self.generator

        if in_condi_img_pils is not None:
            pipeline.set_ip_adapter_scale(ip_adapter_scale)
            ip_in = in_condi_img_pils
        else:
            # text only mode
            print("debug ip_adapter_image=None ", in_condi_img_pils)
            pipeline.set_ip_adapter_scale(0)
            # gray with be zero after pre-process, TODO use ip_adapter_image_embed
            gray_imgs = [Image.new("RGB", (1024, 1024), (128, 128, 128))] * len(in_uv_geom_pils)
            ip_in=gray_imgs

        out_uv_pils = pipeline(
            prompt=in_prompts,
            image=in_uv_geom_pils,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image=ip_in,
        ).images

        return out_uv_pils

    def parse_in_prompts_and_in_condi_img(self, in_prompts, in_condi_img):
        """prompts add prompt_desc, and load condition image if exists

        Args:
            in_prompts: "" or None or str
            in_condi_img: path

        Returns:
            in_prompts:list of str or [""], in_condi_img_pils, pil
        """
        if in_prompts is None:
            in_prompts = [""]
        elif isinstance(in_prompts, str):
            in_prompts = [in_prompts]
        elif isinstance(in_prompts, list):
            if len(in_prompts) < 1 or not isinstance(in_prompts[0], str):
                print(f"ERROR empty in_prompts {in_prompts}")
                return None
        in_prompts = self.add_prompt_desc(in_prompts)

        in_condi_img_pils = None
        if in_condi_img is not None and in_condi_img and os.path.exists(in_condi_img):
            condi_img_pil = load_rgba_as_rgb(in_condi_img)
            in_condi_img_pils = [condi_img_pil] * len(in_prompts)
        
        return in_prompts, in_condi_img_pils
                
    def obj_texuv(
        self,
        in_obj,
        in_prompts: Union[str, List[str]] = None,
        in_condi_img=None,
        in_geom_png=None,
        uv_res=1024,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        out_debug_dir=None,
    ):
        """for one obj, render UV geom, generate uv texture maps

        Args:
            in_obj: raw obj path with uv coord
            in_prompts: string or list of prompt string for one obj, or None
            in_condi_img: condition image or None,  only support one condition img with multi text now TODO(csz)
            in_geom_png: input pre-rendered geom png (masked uv pos)
            uv_res: render geom and out texture resolution
            num_inference_steps
            guidance_scale (int, optional): The larger the value, the closer it is to the text prompt. Defaults to 9.0.
            controlnet_conditioning_scale (int, optional): The higher the value, the better the control but the worse the quality. in [0, 1]
            ip_adapter_scale (int, optional): The larger the value, the closer it is to the image. in [0, 1]
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Returns:
            list of new mesh : list of class Mesh
        """

        # 1. load obj and render uv geom
        if not os.path.exists(in_obj):
            print(f"ERROR can not find in_obj {in_obj}")
            return None

        in_prompts, in_condi_img_pils = self.parse_in_prompts_and_in_condi_img(in_prompts, in_condi_img)

        raw_mesh: Mesh = load_mesh(in_obj)

        ## use pre-rendered pil to speed up
        if in_geom_png is not None and os.path.exists(in_geom_png):
            raw_in_mesh = raw_mesh
            in_geom_pil = PIL.Image.open(in_geom_png)
        else:
            raw_in_mesh = raw_mesh.clone()
            try_cnt = 5
            for i in range(try_cnt):
                try:
                    # may raise error. try try try..
                    mesh_normalized(raw_mesh)
                    break 
                except Exception as e:
                    print("warning mesh_normalized failed. skip mesh_normalized")
                    print(f"Exception occurred: {e}. Retrying {i + 1}/{try_cnt}")
                    if i == try_cnt - 1:
                        print("[ERROR] mesh_normalized failed!!!")
                        break

                    
            ####### Pay attention to the Settings here!(csz)
            raw_mesh = auto_normals(raw_mesh)
            gb_xyz, gb_normal, gb_mask = render_geometry_uv(self.glctx, raw_mesh, uv_res)
            in_geom_pil = cvt_geom_to_pil(gb_xyz, gb_mask)

        # 2. infer texrefine model, get new images
        in_uv_geom_pils = [in_geom_pil] * len(in_prompts)
        out_uv_pils = self.infer_uv_xl_geom(
            in_prompts,
            in_uv_geom_pils,
            in_condi_img_pils=in_condi_img_pils,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
        )

        if not out_uv_pils:
            print("ERROR infer_uv_xl_geom failed")
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

                # save uv-geom/out-tex imgs
                vis_pil = [in_uv_geom_pils, out_uv_pils]
                concatenate_images_2d(vis_pil, os.path.join(out_debug_dir, f"debug_vis.jpg"))
                save_lines(in_prompts, os.path.join(out_debug_dir, f"debug_in_prompts.txt"))
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
    
    def save_out_uv_pils(self, out_uv_pils, out_objs_dir, in_uv_geom_pils=None, debug_save=False):
        """save pils

        Args:
            out_uv_pils: _description_
            out_objs_dir: _description_

        Returns:
            suc_flag: T/F, out_pngs [str..] or ["]
        """
        if not out_uv_pils:
            print('invalid out_uv_pils')
            return False, [""]
        out_pngs = []
        os.makedirs(out_objs_dir, exist_ok=True)
        for idx, out_uv_pil in enumerate(out_uv_pils):
            out_png = os.path.join(out_objs_dir, f"out_tex_{idx:03d}.png")
            out_uv_pil.save(out_png)
            out_pngs.append(out_png)
            
        if debug_save and in_uv_geom_pils is not None and isinstance(in_uv_geom_pils, list):
            vis_pil = [in_uv_geom_pils, out_uv_pils]
            concatenate_images_2d(vis_pil, os.path.join(out_objs_dir, f"debug_vis.jpg"))
                        
        suc_flag = True if out_pngs else False
        
        return suc_flag, out_pngs
    
    # interface for grpc
    def interface_obj_texuv(
        self, in_obj, out_objs_dir, in_prompts=None, in_condi_img=None, in_geom_png=None, run_cfg=None
    ):
        """grpc interface: get uv texture from obj, text and image. pipe_type = "tex_uv"

        Args:
            in_obj: raw obj path with uv coord
            out_objs_dir: output dir with multi output objs
            in_prompts: input text, list text or None
            in_condi_img: condi img path or None
            in_geom_png: input pre-rendered geom png (masked uv pos)
            run_cfg: json or dict or easydict of run-cfg, refer: codedir/configs/tex_gen.json
                    use key uv_res, num_inference_steps, guidance_scale, debug_save

        Returns:
            out_obj_paths is list of obj path or '' means failed
        """
        if not self.pipeline:
            print(f"ERROR: The pipeline has not been initialized")
            return ""

        # copy the scheduler for each thread to make it thread-safe
        # TODO(csz) https://github.com/huggingface/diffusers/issues/3672#issuecomment-1587564138
        self.pipeline.scheduler = self.pipeline.scheduler.from_config(self.pipeline.scheduler.config)


        if run_cfg is None:
            run_cfg = self.cfg
        # TODO(csz) complete cfg
        cfg = edict(run_cfg)

        out_obj_paths = []
        try:
            new_meshs = self.obj_texuv(
                in_obj,
                in_prompts=in_prompts,
                in_condi_img=in_condi_img,
                in_geom_png=in_geom_png,
                uv_res=cfg.uv_res,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
                ip_adapter_scale=cfg.ip_adapter_scale,
                out_debug_dir=out_objs_dir if cfg.debug_save else None,
            )
            if new_meshs is None:
                print(f"ERROR: obj_texuv run ok but return nothing")
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
                    print("in_prompts_", in_prompts_)
                    save_lines(
                        in_prompts_, os.path.join(out_objs_dir, f"in_prompts.txt")
                    )
                    if in_condi_img is not None and os.path.exists(in_condi_img):
                        shutil.copyfile(
                            in_condi_img, os.path.join(out_objs_dir, f"in_condi.png")
                        )
                    save_json(cfg, os.path.join(out_objs_dir, f"cfg.json"))

            except Exception as e:
                print(f"Warn: skip debug_save", e)

        except Exception as e:
            print(f"ERROR: obj_teximguv failed", e)
            traceback.print_exc()
            return ""

        return out_obj_paths

    # unit test
    def test_pipe_obj_texuv(
        self, in_obj, out_objs_dir, in_prompts=None, in_condi_img=None, in_geom_png=None, debug_save=True, run_cfg=None
    ):
        if not run_cfg:
            run_cfg = self.cfg
        run_cfg.debug_save = debug_save

        print("input test_pipe_obj_texuv: ", in_obj, out_objs_dir, in_prompts, in_condi_img, in_geom_png, debug_save)
        print('test run_cfg ', run_cfg)
        out_obj_paths = self.interface_obj_texuv(
            in_obj,
            out_objs_dir,
            in_prompts=in_prompts,
            in_condi_img=in_condi_img,
            in_geom_png=in_geom_png,
            run_cfg=run_cfg,
        )
        print("out_obj_paths ", out_obj_paths)

        return out_obj_paths
