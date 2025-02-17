import os
import argparse
import torch
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
import time
import torch.nn.functional as F
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
import numpy as np
import cv2


def init_sdxl_controlnet(
    sdxl_path="/aigc_cfs/model/stable-diffusion-xl-base-1.0",
    #  control_path="/aigc_cfs/model/xinsir/controlnet-tile-sdxl-1.0",
    control_path="/aigc_cfs/model/TTPLanet_SDXL_Controlnet_Tile_Realistic",
    ip_adapter_model_path="/aigc_cfs_gdp/model/IP-Adapter",
    use_ip_mode="vit-h",
):
    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sdxl_path, subfolder="scheduler")

    controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16)

    # when test with other base model, you need to change the vae also.
    vae = AutoencoderKL.from_pretrained("/aigc_cfs/model/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sdxl_path,
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        safety_checker=None,
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
        ip_image_encoder_folder = "models/image_encoder"
        weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
    else:
        raise ValueError(f"invalid use_ip_mode {use_ip_mode}")
    pipe.load_ip_adapter(
        ip_adapter_model_path,
        subfolder="sdxl_models",
        weight_name=weight_name,
        image_encoder_folder=ip_image_encoder_folder,
    )

    return pipe


def infer_tile(pipe, in_img, condi_img, ip_adapter_scale=0.6, controlnet_conditioning_scale=0.7):
    prompt = "A clear face with detailed features."
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    controlnet_img = Image.open(in_img)
    controlnet_img = controlnet_img.resize((1024, 1024))
    new_width, new_height = 1024, 1024
    ip_in = Image.open(condi_img)

    pipe.set_ip_adapter_scale(ip_adapter_scale)
    # need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=controlnet_img,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        width=new_width,
        height=new_height,
        ip_adapter_image=ip_in,
        num_inference_steps=30,
    ).images

    return images[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='render human obj')
    parser.add_argument(
        '--in_img',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/render_face_only/render_split.png")
    parser.add_argument(
        '--condi_img',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/render_face_only/image.87.png")
    parser.add_argument('--out_dir',
                        type=str,
                        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/render_face_only/tile")

    args = parser.parse_args()

    in_img = args.in_img
    condi_img = args.condi_img
    out_dir = args.out_dir

    use_real = False

    for use_real in [True, False]:
        for ip_adapter_scale in [0.3, 0.5, 0.7, 0.9]:
            for controlnet_conditioning_scale in [0.3, 0.5, 0.7, 0.9]:
                if use_real:
                    control_path = "/aigc_cfs/model/TTPLanet_SDXL_Controlnet_Tile_Realistic"
                    model_name = "Realistic"
                else:
                    control_path = "/aigc_cfs/model/xinsir/controlnet-tile-sdxl-1.0"
                    model_name = "sdxl"

                pipe = init_sdxl_controlnet(control_path=control_path)
                os.makedirs(out_dir, exist_ok=True)

                out_img = infer_tile(pipe,
                                     in_img,
                                     condi_img,
                                     ip_adapter_scale=ip_adapter_scale,
                                     controlnet_conditioning_scale=controlnet_conditioning_scale)
                out_img.save(
                    os.path.join(out_dir, f"{model_name}_ip{ip_adapter_scale}_c{controlnet_conditioning_scale}.png"))
