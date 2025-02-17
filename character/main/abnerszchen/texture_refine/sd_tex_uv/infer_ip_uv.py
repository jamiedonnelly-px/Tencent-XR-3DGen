from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import numpy as np
import PIL
import argparse
import os
from diffusers.utils import load_image


import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally, load_rgba_as_rgb

def infer_ip_adapter(pipeline, generator, normal_pils, ip_pils):
    guidance_scale = 5
    images = pipeline(
        prompt='', 
        # prompt='best quality, high quality', 
        image=normal_pils,
        ip_adapter_image=ip_pils,
        # negative_prompt="", 
        # negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
        num_inference_steps=20,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images
    return images

def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    # parser.add_argument('model_path',
    #                     type=str,
    #                     default="/aigc_cfs_3/sz/result/tex_control/ready_text_cap2_argum")
    parser.add_argument('in_normal_img', type=str)
    parser.add_argument('in_ip_img', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    in_normal_img, in_ip_img, out_dir = args.in_normal_img, args.in_ip_img, args.out_dir
        
        
    controlnet_model_path = "/aigc_cfs_3/sz/result/tex_control/ready_text_cap2_argum"
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "/aigc_cfs/model/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
    pipeline.safety_checker = None
    pipeline.to("cuda")
    pipeline.load_ip_adapter("/aigc_cfs/model/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    generator = torch.Generator(device="cuda").manual_seed(42)

    adapter_scale = 3
    pipeline.set_ip_adapter_scale(adapter_scale)
    # pipeline.unet.set_default_attn_processor()
    
    ip_pils = [load_rgba_as_rgb(in_ip_img, res=512)]
    normal_pils = [load_rgba_as_rgb(in_normal_img, res=512)] 

    images = infer_ip_adapter(pipeline, generator, normal_pils, ip_pils)
    os.makedirs(out_dir, exist_ok=True)
    concatenate_images_2d([normal_pils, ip_pils, images], os.path.join(out_dir, 'infer.jpg'))
    
    
if __name__ == "__main__":
    main()
