import os
from datetime import datetime
import torch
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import copy

import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

from src.ip_adapter import IPAdapterPlusXL

def load_rgba_as_rgb(img_path, res=None):
    """load with RGBA and convert to RGB with white backgroud, if is RGB just return

    Args:
        img_path: _description_

    Returns:
        PIL.Image [h, w, 3]
    """
    img = Image.open(img_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    if res is not None and isinstance(res, int):
        img = img.resize((res, res))
    return img

def init_pipe_sdxl(
                    # sd_path="/aigc_cfs_gdp/model/helloworldXL70",
                    sd_path="/aigc_cfs_2/model/helloworldXL70",
                    # sd_path="/aigc_cfs/model/stable-diffusion-xl-base-1.0/",
                   vae_path="/aigc_cfs_gdp/model/sdxl-vae-fp16-fix",
                   control_path="/aigc_cfs_gdp/model/xinsir/controlnet-depth-sdxl-1.0",
                   ip_adapter_model_path="/aigc_cfs_gdp/model/IP-Adapter-plus",
                   use_ip_mode="plus_vit-h",
                   ):
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler

    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    # control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0-small"
    # controlnet = ControlNetModel.from_pretrained(control_path, variant="fp16", torch_dtype=torch.float16)

    controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16, use_safetensors=True)

    # pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model,
    #     vae=vae,
    #     torch_dtype=torch.float16,
    #     scheduler=eulera_scheduler,
    # )
    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sd_path,
        vae=vae,
        controlnet=controlnet,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        scheduler=eulera_scheduler,
    ).to("cuda")

    if use_ip_mode == "raw":
        ip_image_encoder_folder = "sdxl_models/image_encoder"
        weight_name = "ip-adapter_sdxl.safetensors"
    elif use_ip_mode == "vit-h":
        ip_image_encoder_folder = "models/image_encoder"
        weight_name = "ip-adapter_sdxl_vit-h.safetensors"
    elif use_ip_mode == "plus_vit-h":
        ip_image_encoder_folder = os.path.join(ip_adapter_model_path, "models/image_encoder")
        ip_ckpt = os.path.join(ip_adapter_model_path, "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin")
        # weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
    else:
        raise ValueError(f"invalid use_ip_mode {use_ip_mode}")
    pipe.safety_checker = None

    ip_model = None
    
    # pipe_text = pipe
    ## use more gpu mem, but speed up text only mode, run in L20
    pipe_text = copy.deepcopy(pipe)
    
    if use_ip_mode == "plus_vit-h":
        ip_model = IPAdapterPlusXL(pipe, ip_image_encoder_folder, ip_ckpt, pipe.device, num_tokens=16)
    # else:
    #     ip_model = None
    #     pipe.load_ip_adapter(
    #         ip_adapter_model_path,
    #         subfolder="sdxl_models",
    #         weight_name=weight_name,
    #         image_encoder_folder=ip_image_encoder_folder,
    #     )

    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    return pipe_text, ip_model


def init_pipe(sd_path="/aigc_cfs/model/stable-diffusion-v1-5/",
              control_path="/aigc_cfs/model/control_v11f1p_sd15_depth/"):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler

    controlnet = ControlNetModel.from_pretrained(control_path, variant="fp16", torch_dtype=torch.float16)

    # control_path = "/aigc_cfs/model/Paint3d_UVPos_Control/"
    # controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_path, controlnet=controlnet,
                                                             torch_dtype=torch.float16).to("cuda")

    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sd_path, subfolder="scheduler")
    pipe.feature_extractor = None
    pipe.requires_safety_checker = False

    pipe.safety_checker = None
    return pipe


class DepthControlPipe():

    def __init__(self, use_sdxl=True, seed=1234):

        if use_sdxl:
            self.pipe, self.ip_model = init_pipe_sdxl()
        else:
            self.pipe = init_pipe()

        self.generator = torch.Generator("cuda").manual_seed(seed)
        print(f"init DepthControlPipe done with use_sdxl={use_sdxl}")

    def infer_depth(
        self,
        in_depth,
        prompt="",
        add_prompt=", with beautiful color and lots of details, blender 3D Object Rendering with Empty Background. High Quality, HDR, UHD, 4K",
        in_image_path_list=None,
        negative_prompt="",
        controlnet_conditioning_scale=0.8,
        guidance_scale=5.0,
        num_inference_steps=30,
        ip_adapter_scale=0.8,
        res=None,
    ):
        """control text to image

        Args:
            in_depth: [h, w, 3] np/tensor/pil
            prompt: _description_. Defaults to "".
            negative_prompt: _description_. Defaults to "".
            controlnet_conditioning_scale: _description_. Defaults to 0.9.
            guidance_scale: _description_. Defaults to 7.0.
            num_inference_steps: _description_. Defaults to 20.
            ip_adapter_scale: ip adapter for image crtl
            res: if not None, resize->res. Defaults to None.

        Returns:
            PIL, shape is res or in_depth.shape
        """
        image = self.wrap_depth(in_depth, res=res)
        if not prompt or len(prompt) <1:
            {"run_mode":"text"}
            prompt += add_prompt

        condi_img_pil = None
        if in_image_path_list is not None and isinstance(in_image_path_list, list) and len(in_image_path_list) > 0:
            in_condi_img = in_image_path_list[0]
            if not os.path.exists(in_condi_img):
                raise ValueError(f"[ERROR] input condi img but not exists: {in_condi_img}")
            condi_img_pil = load_rgba_as_rgb(in_condi_img)

        if condi_img_pil is not None:
            print('debug text + image mode')
            if not self.ip_model:
                print('warn old ipa')
                raise ValueError('warn old ipa')
                # self.pipe.set_ip_adapter_scale(ip_adapter_scale)
            ip_in = condi_img_pil
            result = self.ip_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                image=image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                pil_image=ip_in,
                scale=ip_adapter_scale,
                # generator=self.generator,
            )[0]            
        else:
            # text only mode
            print('debug text only mode')
            # if not self.ip_model:
            #     print('warn old ipa')
            #     self.pipe.set_ip_adapter_scale(0)
            # # gray with be zero after pre-process, TODO use ip_adapter_image_embed to speed up
            # gray_imgs = Image.new("RGB", (1024, 1024), (128, 128, 128))
            # ip_in=gray_imgs
            # ip_adapter_scale_use = 0
            
            result = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                image=image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
            ).images[0]
                        
        
        # if not self.ip_model:
        #     print('warn old ipa')
        #     result = self.pipe(
        #         prompt,
        #         negative_prompt=negative_prompt,
        #         controlnet_conditioning_scale=controlnet_conditioning_scale,
        #         image=image,
        #         guidance_scale=guidance_scale,
        #         num_inference_steps=num_inference_steps,
        #         ip_adapter_image=ip_in,
        #         generator=self.generator,
        #     ).images[0]
        # else:
        #     result = self.ip_model.generate(
        #         prompt=prompt,
        #         negative_prompt=negative_prompt,
        #         controlnet_conditioning_scale=controlnet_conditioning_scale,
        #         image=image,
        #         guidance_scale=guidance_scale,
        #         num_inference_steps=num_inference_steps,
        #         pil_image=ip_in,
        #         scale=ip_adapter_scale_use,
        #         # generator=self.generator,
        #     )[0]

        return result

    def wrap_depth(self, in_depth, res=None):
        """wrap input depth to PIL Image [res, res, 3] in [0, 255]

        Args:
            in_depth: [h, w, 3] np/tensor/pil
            res: if not None, resize->res

        Returns:
            PIL Image [res, res, 3] in [0, 255]
        """
        if isinstance(in_depth, Image.Image):
            return in_depth
        if isinstance(in_depth, torch.Tensor):
            in_array = in_depth.detach().cpu().numpy()
        elif isinstance(in_depth, np.ndarray):
            in_array = in_depth

        assert in_array.ndim == 3, f"invalid in_depth ndim= {in_depth.ndim}"
        assert in_array.shape[-1] == 3, f"invalid in_depth shape= {in_depth.shape}"

        if in_array.min() < 0 or in_array.max() > 255 or in_array.max() < 2:
            min_ = in_array.min()
            in_array = (in_array - min_) / (in_array.max() - min_)
            in_array *= 255.

        in_array = in_array.astype(np.uint8)

        out_pil = Image.fromarray(in_array)
        if res is not None and isinstance(res, int):
            out_pil = out_pil.resize((res, res))

        return out_pil


if __name__ == "__main__":

    use_sdxl = True
    use_mview = True
    use_text = True
    seed = 1234

    dc_pipe = DepthControlPipe(use_sdxl, seed)

    # render_png = "data/shirt/untitledortho_True.png"/
    # render_npy = "/aigc_cfs_2/sz/data/mvc/top_3/uv_condition/meshortho_True_512.npy"
    # prompt= "orange puffy jacket with a white t-shirt underneath, 3d assert, 4views"

    render_npy = "/aigc_cfs_gdp/jiawei/data/general_generate/5a7b172a-4aa7-47d8-a57b-20a7fd388535/texture_mesh/5a7b172a-4aa7-47d8-a57b-20a7fd388535ortho_True_512.npy"
    # render_npy = "/aigc_cfs_gdp/sz/d2rgb_tex/0909/text_in/ac1d3421bf2d4a5886c4d7f4ebf82224ortho_True_512.npy"
    if use_text:
        prompt = "Medieval style shield, exquisite, HD, 3d assert"
        if use_mview:
            prompt += ", 4views"
        in_image_path_list = []
    else:
        prompt = ""
        in_image_path_list = ["/aigc_cfs_gdp/sz/d2rgb_tex/0909/text_in/kafeiwhite.png"]

    # render_npy = "/aigc_cfs_2/sz/data/mvc/mario_tripo/untitledortho_True.npy"
    # render_png = "data/top_3/uv_condition/meshortho_True.png"
    # prompt = "a yellow t shirt with a brown bear on it"
    # prompt= "A white T-shirt with a bear on the front, 2 views"
    # prompt= "A beautiful treasure chest, 4 views"
    # prompt= "Mario with one hand raised, the image is high quality and clear, blender render, 3d assert, 4views"

    negative_prompt = ""

    controlnet_conditioning_scale = 0.9
    guidance_scale = 7.0
    num_inference_steps = 20
    # controlnet_conditioning_scale = 0.7 #TODO
    # controlnet_conditioning_end_scale = 0.9
    control_guidance_start = 0.0
    control_guidance_end = 1.0

    pils = []
    pils_masked = []
    try_cnt = 3
    res = 1024
    infer_out = os.path.join(os.path.dirname(render_npy), f"new_{res}_{controlnet_conditioning_scale}_text{use_text}_mv_{use_mview}_sdxl_{use_sdxl}.png")
    # infer_out = render_npy.replace(".npy", f"new_{res}_{controlnet_conditioning_scale}_text{use_text}_mv_{use_mview}_sdxl_{use_sdxl}.png")

    for j in range(try_cnt):
        depth_raw = np.load(render_npy)  # [4, h, w, 3]
        upper_row = np.concatenate((depth_raw[0], depth_raw[1]), axis=1)
        lower_row = np.concatenate((depth_raw[2], depth_raw[3]), axis=1)
        merged_array = np.concatenate((upper_row, lower_row), axis=0)

        print('merged_array ', merged_array.shape, isinstance(merged_array, np.ndarray), merged_array.max(),
              merged_array.dtype)
        # control = torch.from_numpy(depth_raw.copy()).float().cuda() / 255.0
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # render_pil = load_image(render_png)
        if use_mview:
            result = dc_pipe.infer_depth(merged_array,
                                         prompt=prompt,
                                         in_image_path_list=in_image_path_list,
                                         negative_prompt=negative_prompt,
                                         controlnet_conditioning_scale=controlnet_conditioning_scale,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps,
                                         res=None)
            pils.append(result)

        else:
            # one by one
            control_list = [Image.fromarray(d_np) for d_np in depth_raw]
            once_pils = []
            for control in control_list:
                result = dc_pipe.infer_depth(control,
                                             prompt=prompt,
                                             in_image_path_list=in_image_path_list,
                                             negative_prompt=negative_prompt,
                                             controlnet_conditioning_scale=controlnet_conditioning_scale,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             res=res)
                once_pils += [result]
            once_pil = make_image_grid(once_pils, 1, len(once_pils))
            pils.append(once_pil)

    make_image_grid(pils, len(pils), 1).save(infer_out)
