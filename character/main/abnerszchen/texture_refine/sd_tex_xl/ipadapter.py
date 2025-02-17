import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils import load_image

ip_adapter_model_path="/aigc_cfs/model/IP-Adapter"
base_model_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0"


# models/image_encoder: OpenCLIP-ViT-H-14 with 632.08M parameter
# sdxl_models/image_encoder: OpenCLIP-ViT-bigG-14 with 1844.9M parameter
# ip-adapter_sdxl.bin: use global image embedding from OpenCLIP-ViT-bigG-14 as condition
# ip-adapter_sdxl_vit-h.bin: same as ip-adapter_sdxl, but use OpenCLIP-ViT-H-14
# ip-adapter-plus_sdxl_vit-h.bin: use patch image embeddings from OpenCLIP-ViT-H-14 as condition, closer to the reference image than ip-adapter_xl and ip-adapter_sdxl_vit-h

use_mode = "plus_vit-h"
if use_mode == "raw":
    image_encoder_folder = "sdxl_models/image_encoder"
    weight_name = "ip-adapter_sdxl.safetensors"
elif use_mode == "vit-h":
    image_encoder_folder = "models/image_encoder"
    weight_name = "ip-adapter_sdxl_vit-h.safetensors"
elif use_mode == "plus_vit-h":
    image_encoder_folder = "models/image_encoder"
    weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
else:
    raise ValueError(f"invalid use_mode {use_mode}")


weight_dtype = torch.float16
pipeline = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    variant="fp16",
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  ip_adapter_model_path,
  subfolder="sdxl_models",
  weight_name=weight_name,
  image_encoder_folder=image_encoder_folder,
)
generator = torch.Generator(device="cpu").manual_seed(0)
style_images = load_image(f"debug/img0.png")

for ip_s in [0.3, 0.6, 0.8, 1.0] :
    pipeline.set_ip_adapter_scale([ip_s])
    # pipeline.enable_model_cpu_offload()

    images = pipeline(
        prompt="wonderwoman",
        ip_adapter_image=style_images,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=50, num_images_per_prompt=1,
        generator=generator,
    ).images
    for i, image in enumerate(images):
        image.save(f"debug/test_{i}_{use_mode}_s_{ip_s}.png")