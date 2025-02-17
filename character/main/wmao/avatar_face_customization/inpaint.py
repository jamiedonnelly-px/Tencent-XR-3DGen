import torch
from diffusers import AutoPipelineForInpainting, KandinskyV22InpaintPipeline,KandinskyV22PriorPipeline
from diffusers.utils import load_image, make_image_grid
from ipdb import set_trace as st
from PIL import Image, ImageOps, ImageFilter

pipeline = AutoPipelineForInpainting.from_pretrained(
    "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
#     "/aigc_cfs_2/weimao/pretrained_model_cache//kandinsky-2-2-prior", torch_dtype=torch.float16
#     )
# pipe_prior.to("cuda")
# pipeline = KandinskyV22InpaintPipeline.from_pretrained(
#     "/aigc_cfs_2/weimao/pretrained_model_cache/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16, variant="fp16"
# )
# pipeline.to("cuda")

# load base and mask image
init_image = load_image("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_combined_inpaint_0.9.png")
mask_image = load_image("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_inpaint_mask2.png")
# mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius = 20))
# mask_image.save("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_head_mask_blur.png")
# invert the mask
# mask_image = ImageOps.invert(mask_image)

generator = torch.Generator("cuda").manual_seed(92)
prompt = "bald cartoon head, pure color background, no hair, skin color, no shadow, no shading, no grey color, bright color"
negative_prompt = "hair, black, shading, shadow, highlight"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator,
                guidance_scale=10.0, strength=0.9).images[0]
# image_emb, zero_image_emb = pipe_prior(prompt=prompt, negative_prompt=negative_prompt, return_dict=False)
# image = pipeline(image=init_image, mask_image=mask_image, image_embeds=image_emb, negative_image_embeds=zero_image_emb,
#                 height=1024, width=1024, num_inference_steps=50).images[0]
# image = image.resize(init_image.size,Image.Resampling.BILINEAR)
image.save('/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_combined_inpaint_0.9.png')

# make_image_grid([init_image, mask_image, image], rows=1, cols=3)