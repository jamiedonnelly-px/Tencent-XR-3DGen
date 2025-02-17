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

from diffusers import ControlNetModel, StableDiffusionImg2ImgPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
import numpy as np
import cv2


def init_sd(
    sd_path="/aigc_cfs/model/stable-diffusion-v1-5/",
    ip_adapter_model_path="/aigc_cfs_gdp/model/IP-Adapter",
    use_ip_mode="vit-h",
):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_path,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.load_ip_adapter(ip_adapter_model_path, subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.6)

    return pipe


def infer_i2i(
    pipe,
    in_img,
    condi_img,
    prompt="A clear face with detailed features.",
    res=512,
    strength=0.7,
    num_inference_steps=50,
    guidance_scale=7.5,
    ip_adapter_scale=0.6,
):

    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    init_image = Image.open(in_img).resize((res, res))
    ip_in = Image.open(condi_img)

    pipe.set_ip_adapter_scale(ip_adapter_scale)
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ip_adapter_image=ip_in,
    ).images

    return images[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='render human obj')
    parser.add_argument(
        '--in_img',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/742ce15b-cfe6-4853-b7ca-f43094846ee9/render_face_only/render_split.png")
    parser.add_argument(
        '--condi_img',
        type=str,
        default="/aigc_cfs_gdp/sz/threeviews/742ce15b-cfe6-4853-b7ca-f43094846ee9/render_face_only/image_ipa.png")
    parser.add_argument('--out_dir',
                        type=str,
                        default="/aigc_cfs_gdp/sz/threeviews/742ce15b-cfe6-4853-b7ca-f43094846ee9/render_face_only/i2i_ip")

    args = parser.parse_args()

    in_img = args.in_img
    condi_img = args.condi_img
    out_dir = args.out_dir

    pipe = init_sd()

    prompt = ""
    for res in [768]:
        for ip_adapter_scale in [0.3, 0.5, 0.7, 0.9]:
            for strength in [0.3, 0.5, 0.7, 0.9]:
                for guidance_scale in [5, 7, 9]:

                    os.makedirs(out_dir, exist_ok=True)

                    out_img = infer_i2i(
                        pipe,
                        in_img,
                        condi_img,
                        res=res,
                        strength=strength,
                        num_inference_steps=50,
                        guidance_scale=guidance_scale,
                        ip_adapter_scale=ip_adapter_scale,
                        prompt=prompt,
                    )
                    out_img.save(os.path.join(out_dir, f"res{res}_ip{ip_adapter_scale}_str{strength}_gs{guidance_scale}.png"))
