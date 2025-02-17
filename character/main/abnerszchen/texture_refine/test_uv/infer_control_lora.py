import os
from datetime import datetime
import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import numpy as np

from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers import AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import cv2

sd_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0/"
vae_path = "/aigc_cfs/model/sdxl-vae-fp16-fix"
# control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0-small"
# control_lora_path = "/aigc_cfs/model/control-lora/control-LoRAs-rank256/"

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

image = load_image("/aigc_cfs_2/sz/proj/tex_cq/test_uv/debug/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

lora_id = "/aigc_cfs/model/control-lora"
lora_filename = "control-LoRAs-rank128/control-lora-canny-rank128.safetensors"

unet = UNet2DConditionModel.from_pretrained(
    sd_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

controlnet = ControlNetModel.from_unet(unet).to(device="cuda", dtype=torch.float16)
controlnet.load_lora_weights(lora_id, weight_name=lora_filename, controlnet_config=controlnet.config)

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    sd_path,
    unet=unet,
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
# pipe.load_lora_weights(lora_id, weight_name=lora_filename, controlnet_config=controlnet.config)

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_images_per_prompt=4
).images

final_image = [image] + images
grid = make_image_grid(final_image, 1, 5)
grid.save("/aigc_cfs_2/sz/proj/tex_cq/test_uv/debug/lora_out.png")

