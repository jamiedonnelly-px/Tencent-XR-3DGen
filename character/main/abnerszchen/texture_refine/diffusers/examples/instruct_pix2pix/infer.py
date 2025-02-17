# import requests
import torch
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "/apdcephfs_cq8/share_2909871/shenzhou/result/tex/init" # <- replace this 
# model_id = "instruct-pix2pix-model" # <- replace this 
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

# url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"

# def download_image(url):
#     image = PIL.Image.open(requests.get(url, stream=True).raw)
#     image = PIL.ImageOps.exif_transpose(image)
#     image = image.convert("RGB")
#     return image
# image = download_image(url)
image = Image.open("test_pix2pix_4.png").convert("RGB")


prompt = "wipe out the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 6

edited_image = pipe(prompt, 
    image=image, 
    num_inference_steps=num_inference_steps, 
    image_guidance_scale=image_guidance_scale, 
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]
edited_image.save("edited_image_g6.png")