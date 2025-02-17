import os
from datetime import datetime
import torch
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np

def init_pipe_sdxl():
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    sd_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0/"
    vae_path = "/aigc_cfs/model/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(
        vae_path, torch_dtype=torch.float16
    )
    # control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0-small"
    # controlnet = ControlNetModel.from_pretrained(control_path, variant="fp16", torch_dtype=torch.float16)
    control_path = "/aigc_cfs/model/xinsir/controlnet-depth-sdxl-1.0"
    controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16, use_safetensors=True)

    # pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
    #     vae=vae,
    #     torch_dtype=torch.float16,
    #     scheduler=eulera_scheduler,
    # )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sd_path,
        vae=vae,
        controlnet=controlnet,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    return pipe

def init_pipe():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
    sd_path = "/aigc_cfs/model/stable-diffusion-v1-5/"
    control_path = "/aigc_cfs/model/control_v11f1p_sd15_depth/"
    controlnet = ControlNetModel.from_pretrained(control_path, variant="fp16", torch_dtype=torch.float16)

    # control_path = "/aigc_cfs/model/Paint3d_UVPos_Control/"
    # controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16)
   
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
		sd_path, controlnet=controlnet, torch_dtype=torch.float16
	).to("cuda")

    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sd_path, subfolder="scheduler")
    pipe.feature_extractor = None
    pipe.requires_safety_checker = False
        
    pipe.safety_checker = None
    return pipe

def split_image(image, n):
    width, height = image.size

    single_width = width // n

    images = []
    for i in range(n):
        left = i * single_width
        right = (i + 1) * single_width
        sub_image = image.crop((left, 0, right, height))
        images.append(sub_image)

    return images

use_sdxl = True
use_mview = True

if use_sdxl:
    pipe = init_pipe_sdxl()
else:
    pipe = init_pipe()

seed = 1234
generator=torch.Generator("cuda").manual_seed(seed)


# render_png = "data/shirt/untitledortho_True.png"/
render_npy = "/aigc_cfs_2/sz/data/mvc/top_3/uv_condition/meshortho_True_512.npy"
# render_npy = "/aigc_cfs_2/sz/data/mvc/mario_tripo/untitledortho_True.npy"
# render_png = "data/top_3/uv_condition/meshortho_True.png"
# prompt = "a yellow t shirt with a brown bear on it"
# prompt= "A white T-shirt with a bear on the front, 2 views"
# prompt= "A beautiful treasure chest, 4 views"
# prompt= "Mario with one hand raised, the image is high quality and clear, blender render, 3d assert, 4views"
prompt= "orange puffy jacket with a white t-shirt underneath, 3d assert, 4views"
negative_prompt= ""


controlnet_conditioning_scale = 0.9
guidance_scale=7.0
num_inference_steps=20
# controlnet_conditioning_scale = 0.7 #TODO
# controlnet_conditioning_end_scale = 0.9
control_guidance_start = 0.0
control_guidance_end = 1.0

pils = []
pils_masked = []
try_cnt = 3
res = 1024
infer_out = render_npy.replace(".npy", f"_{res}_{controlnet_conditioning_scale}_mv_{use_mview}_sdxl_{use_sdxl}.png")

for j in range(try_cnt):
    depth_raw = np.load(render_npy) # [4, h, w, 3]
    upper_row = np.concatenate((depth_raw[0], depth_raw[1]), axis=1)
    lower_row = np.concatenate((depth_raw[2], depth_raw[3]), axis=1)
    merged_array = np.concatenate((upper_row, lower_row), axis=0)    
    
    print('merged_array ', merged_array.shape)
        # control = torch.from_numpy(depth_raw.copy()).float().cuda() / 255.0
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()    
    # render_pil = load_image(render_png)
    if use_mview:
        images = pipe(prompt, # + f", {direction_names[dir_id[i]]} view",
                    negative_prompt = negative_prompt,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    # height=res,
                    # width=res*2,
                    image=Image.fromarray(merged_array),
                    guidance_scale = guidance_scale,
                    num_inference_steps = num_inference_steps,
                    generator=generator
                    ).images
        pils += images
    else:
        # one by one
        control_list = split_image(render_pil, 2)
        once_pils = []
        for control in control_list:
            images = pipe(prompt, # + f", {direction_names[dir_id[i]]} view",
                        negative_prompt = negative_prompt,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height=res,
                        width=res,
                        image=control,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        generator=generator
                        ).images
            once_pils += images
        once_pil = make_image_grid(once_pils, 1, len(once_pils))
        pils.append(once_pil)
    

make_image_grid(pils, len(pils), 1).save(infer_out)
