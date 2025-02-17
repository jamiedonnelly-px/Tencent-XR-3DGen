import argparse
import os
import time
import numpy as np
import torch
import PIL

from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

use_910b = False
use_fp16_vae = True
use_fp32 = False

if use_910b:
    base_model_path = "/data5/sz/model/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path="/data5/sz/model/sdxl-vae-fp16-fix"
    device = "npu"
else:
    base_model_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path="/aigc_cfs/model/sdxl-vae-fp16-fix"
    device = "cuda"
ip_adapter_model_path="/aigc_cfs/model/IP-Adapter"

weight_dtype = torch.float16
if not use_fp16_vae:
    pretrained_vae_model_name_or_path = None
if use_fp32:
    weight_dtype = torch.float32

test_ip = False
use_pos = True
dataset_type = "mcwy2"
# dataset_type = "ready"
if use_pos:
    if dataset_type== "ready":
        controlnet_path = "/aigc_cfs_3/sz/result/tex_control_2024/xl_ready/g4_pre_xyz_fixvae"
    elif dataset_type== "mcwy2":
        controlnet_path = "/aigc_cfs_3/sz/result/tex_control_2024/xl_mcwy2/g4_pre_right_pos_4class/checkpoint-3000/controlnet"
# controlnet_path = "/aigc_cfs_3/sz/result/tex_control_2024/circle/xl_1024_g4_vaefp16"


controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=weight_dtype)
if pretrained_vae_model_name_or_path is not None:
    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_model_name_or_path, torch_dtype=weight_dtype
    )
else:
    vae = AutoencoderKL.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=weight_dtype
    )

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    controlnet=controlnet,
    variant="fp16",
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)
pipe.safety_checker = None   


print('pipe1 ', pipe.device)
print('dtype ', pipe.unet.dtype, pipe.vae.dtype)

# ip-adapter_sdxl.bin: use global image embedding from OpenCLIP-ViT-bigG-14 as condition
# ip-adapter_sdxl_vit-h.bin: same as ip-adapter_sdxl, but use OpenCLIP-ViT-H-14
# ip-adapter-plus_sdxl_vit-h.bin: use patch image embeddings from OpenCLIP-ViT-H-14 as condition, closer to the reference image than ip-adapter_xl and ip-adapter_sdxl_vit-h

if test_ip:
    pipe.load_ip_adapter(
        ip_adapter_model_path, subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
    )
    
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print('debug pipe.scheduler ', pipe.scheduler)

# remove following line if xformers is not installed or when using Torch 2.0.


# # less memory but slower. use in t10 instead of v100
# if not use_910b:
#     pipe.enable_xformers_memory_efficient_attention() 
    
#     # memory optimization.
#     pipe.enable_model_cpu_offload()

res = 1024
# control_image = load_image("./test_input/bottom_mcwy2.png").resize((res, res))
# prompts = ["tiger style"]

if use_pos:
    if dataset_type== "ready":
        condi_img_path = "./test_input/pos_top_ready.png"
    elif dataset_type== "mcwy2":
        condi_img_path = "./test_input/pos_top_mcwy2.png"
else:
    condi_img_path = "./test_input/top_ready.png"
    
control_image = load_image(condi_img_path).resize((res, res))
prompts = ["red dragon style, high quality, DSLR."]
# prompts = ["india style, a woman, high quality, DSLR."]
# prompts = ["red chinese dragon, Hd, high quality", "firefighter style, Hd, high quality"]

# control_image = load_image("./conditioning_image_1.png").resize((res, res))
# prompts = ["cyan circle with brown floral background"]


# generate image
num_inference_steps = 20

for guidance_scale in [9.]:
    for controlnet_conditioning_scale in [0.8]:
        save_dir = os.path.join(controlnet_path, f"infer/ccs_{controlnet_conditioning_scale}")
        os.makedirs(
            save_dir, exist_ok=True
        )

        generator = torch.manual_seed(42)
        if test_ip:
            pipe.set_ip_adapter_scale(0)
            sd_images = pipe(
                prompt=prompts,
                image=control_image,
                num_inference_steps=30,
                guidance_scale=7.5,  # text only
                generator=generator,
                controlnet_conditioning_scale=0.0,  # for text only
                ip_adapter_image=control_image,  # useless
            ).images   
            print('sd_images ', sd_images)         
            pipe.set_ip_adapter_scale(1)
            breakpoint()
        
        torch.cuda.synchronize()
        start = time.time()
        print(f"begin infer")
        images = pipe(
            prompts,
            image=control_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            ip_adapter_image=sd_images[0] if test_ip else None,  # useless
        ).images
        print(f"end infer")
        torch.cuda.synchronize()
        use_t = time.time() - start
        print(f"use_t {use_t}")
                
        for prompt, image in zip(prompts, images):
            prompt_name = prompt.replace(" ", "-")
            image.save(
                os.path.join(
                    save_dir,
                    f"out_{res}_gs_{guidance_scale}_vae16_{use_fp16_vae}_num_{num_inference_steps}_{prompt_name}.png",
                )
            )
        print('save_dir ', save_dir)
        
# breakpoint()
