import os
from datetime import datetime
import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import numpy as np

def init_pipe_sdxl():
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    sd_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0/"
    vae_path = "/aigc_cfs/model/sdxl-vae-fp16-fix"
    # control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0-small"
    control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0"
    vae = AutoencoderKL.from_pretrained(
        vae_path, torch_dtype=torch.float16
    )
    controlnet = ControlNetModel.from_pretrained(control_path, variant="fp16", torch_dtype=torch.float16)

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

def save_conditional_images(conditional_images, out_dir):
    """_summary_

    Args:
        conditional_images: [b, c, h, w] in [0, 1]
        out_dir: _description_

    Returns:
        _description_
    """
    os.makedirs(out_dir, exist_ok=True)
    
    vis_np = conditional_images.permute(0, 2, 3, 1).cpu().numpy()
    vis_np = (vis_np * 255).clip(0, 255).astype(np.uint8)

    out_pngs = []
    pils = []
    for i, img_np in enumerate(vis_np):
        img = Image.fromarray(img_np, mode="RGB")
        pils.append(img)

        out_png = os.path.join(out_dir, f"image_{i}.png")
        img.save(out_png)
        out_pngs.append(out_png)
    
    make_image_grid(pils, 1, len(pils)).save(os.path.join(out_dir, f"image_merge.png"))
    return out_pngs


@torch.no_grad()
def decode_normalized_depth(depths, batched_norm=False):
    # [b,h,w], 1/d and normalized to [-, 1] where mask = 1, =0 where mask =0
    view_z, mask = depths.unbind(-1)
    view_z = view_z * mask + 100 * (1-mask)
    inv_z = 1 / view_z
    inv_z_min = inv_z * mask + 100 * (1-mask)
    if not batched_norm:
        max_ = torch.max(inv_z, 1, keepdim=True)
        max_ = torch.max(max_[0], 2, keepdim=True)[0]

        min_ = torch.min(inv_z_min, 1, keepdim=True)
        min_ = torch.min(min_[0], 2, keepdim=True)[0]
            
    else:
        max_ = torch.max(inv_z)
        min_ = torch.min(inv_z_min)
    inv_z = (inv_z - min_) / (max_ - min_)
    inv_z = inv_z.clamp(0,1)
    inv_z = inv_z[...,None].repeat(1,1,1,3)

    return inv_z

def convert_dpeth_to_midas_one(render_depth_one, s=245, t=125):
    """_summary_

    Args:
        render_depth_one: h, w, 2
        s: _description_. Defaults to 245.
        t: _description_. Defaults to 125.

    Returns:
        h,w,3
    """
    view_z, mask = render_depth_one.unbind(-1)  # h, w
    view_z = view_z * mask + 100. * (1-mask)
    disp = 1 / (view_z).float()   # disp where mask=1, 0.01 when mask = 0
    # disp = torch.rand_like(view_z)   ##### ? rand?
    
    valid_data = disp[mask > 0].view(-1)
    min_, max_ = torch.min(valid_data), torch.max(valid_data)
    print("inv", min_, max_)
    vmin, vmax = 0.2, 0.8
   
    disp_scaled = (disp - min_) / (max_ - min_)
    print("disp_scaled", disp_scaled.min(), disp_scaled.max(), disp_scaled[mask > 0].min())

    disp_normalized = disp_scaled * (vmax - vmin) + vmin
    valid_disp_normalized = disp_normalized[mask > 0].view(-1)
    
    print("disp_normalized", disp_normalized.min(), disp_normalized.max(), valid_disp_normalized.min(), valid_disp_normalized.max())
    
    # breakpoint()
    disp_normalized = disp_normalized.clamp(0,1)
    print("disp_normalized clamp", disp_normalized.min(), disp_normalized.max())
    # breakpoint()
    
    # disp_as_midas = (disp_as_midas - min_) / (max_ - min_)
    # disp_as_midas = disp_as_midas * mask
    # disp_as_midas = mask * 0.7
    # disp_as_midas = disp_as_midas.clamp(0,1)
    disp_normalized = disp_normalized[...,None].repeat(1,1,3)    
    return disp_normalized

def convert_pos_to_midas(uv_pos, uv_mask, s=245, t=125):
    disp_as_midas_list = []
    for i, render_depth_one in enumerate(depths):
        disp_as_midas = convert_dpeth_to_midas_one(render_depth_one)
        disp_as_midas_list.append(disp_as_midas)
    inv_z = torch.stack(disp_as_midas_list, axis=0)
    return inv_z

def easy_zoom(gb_geom : torch.Tensor, gb_mask : torch.Tensor):
    """save geom [-1, 1] to [0, 1]

    Args:
        gb_geom (torch.Tensor): tensor geometry [H, W, 3] in [-1, 1], position or normal
        gb_mask (torch.Tensor): tensor [H, W, 1]

    Returns:
        PIL.Image: pil in [H, W, 3] h,w,3
    """
    mask_expanded = gb_mask.expand(-1, -1, 3)
    selected_pixels = torch.masked_select(gb_geom, mask_expanded == 1)
    gmin, gmax = torch.min(selected_pixels).item(), torch.max(selected_pixels).item()
    assert -1.1 <= gmin <= gmax <= 1.1, f"geom need normalized to [-1, 1]. xyz in [-1, 1] cube, normal in [-1, 1], but get [{gmin}, {gmax}]"
    
    gb_geom_pro = (gb_geom + 1.) / 2.
    
    gb_geom_pro_mask = gb_geom_pro * gb_mask
    return gb_geom_pro_mask

def zoom_to_ab(data, mask, vmin, vmax):
    valid_data = data[mask > 0].view(-1)
    min_, max_ = torch.min(valid_data), torch.max(valid_data)
    
    disp_scaled = (data - min_) / (max_ - min_)

    disp_normalized = disp_scaled * (vmax - vmin) + vmin
    masked = disp_normalized * mask    
    return masked

def cvt_geom_to_disp(gb_geom : torch.Tensor, gb_mask : torch.Tensor):
    gb_dist = torch.sqrt(torch.sum(gb_geom**2, dim=-1))
    gb_dist = gb_dist.unsqueeze(-1)
    gb_disp = 1 / (gb_dist + 1e-6)
    gb_disp_masked = gb_disp * gb_mask
    gb_disp_normalized = zoom_to_ab(gb_disp_masked, gb_mask, 0.2, 0.8)
    return gb_disp_normalized

def cvt_geom_to_dist(gb_geom : torch.Tensor, gb_mask : torch.Tensor):
    gb_dist = torch.sqrt(torch.sum(gb_geom**2, dim=-1))
    gb_dist = gb_dist.unsqueeze(-1)
    gb_dist_masked = gb_dist * gb_mask
    print("gb_dist_masked", gb_dist_masked.min(), gb_dist_masked.max(), gb_dist_masked[gb_mask > 0].min())
    
    gb_dist_normalized = zoom_to_ab(gb_dist_masked, gb_mask, 0.2, 0.8)
    print("gb_dist_normalized", gb_dist_normalized.min(), gb_dist_normalized.max(), gb_dist_normalized[gb_mask > 0].min())
    return gb_dist_normalized

# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_depth_conditioning_images(uv_pos, uv_mask, output_size=1024, blur_filter=5):
    normals_transforms = Compose([
     Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True),
    #  GaussianBlur(blur_filter, blur_filter//3+1),
    ]
    )
    
    # keep raw
    conditional_images = uv_pos.unsqueeze(0).permute(0, 3, 1, 2)
    
    # # conditional_images = easy_zoom(uv_pos, uv_mask)
    # conditional_images = cvt_geom_to_disp(uv_pos, uv_mask)
    # # conditional_images = cvt_geom_to_dist(uv_pos, uv_mask)
    # conditional_images = conditional_images.unsqueeze(0)
    # conditional_images = conditional_images.permute(0, 3, 1, 2)
    # conditional_images = conditional_images[:, :1, ...].repeat(1, 3, 1, 1) # [b, 3, h, w]
    
    print("debug conditional_images ", conditional_images.shape, conditional_images.dtype, conditional_images.min(), conditional_images.max())
    # conditional_images = normals_transforms(conditional_images)

    return conditional_images

use_sdxl = True
# uv_condition_dir = "/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/MCWY_2_Top/BR_TOP_1_F_T/uv_condition"
uv_condition_dir = "/aigc_cfs_2/sz/proj/Paint3D/outputs/mcwy/BR_TOP/stage1/res-0/myuv"
out_dir = "/aigc_cfs_3/layer_tex/mcwy_2/v3/debug/pos_rawxyz"
uv_pos_npy = os.path.join(uv_condition_dir, "uv_pos.npy")
uv_pos_png = os.path.join(uv_condition_dir, "uv_pos.png")
uv_mask_png = os.path.join(uv_condition_dir, "uv_mask.png")
os.makedirs(out_dir, exist_ok=True)

# uv_pos = torch.tensor(np.load(uv_pos_npy)).to("cuda") # [h, w, 3] xyz raw in [-1, 1]
uv_pos = torch.tensor(np.array(load_image(uv_pos_png))).to("cuda") / 255. # [h, w, 3] xyz raw in [0, 1]
uv_mask = torch.tensor(np.array(load_image(uv_mask_png))).to("cuda")  
uv_mask = uv_mask[..., :1] / 255    # [h, w, 1] mask
print('uv_pos ', uv_pos.shape, uv_pos.dtype, uv_pos.min(), uv_pos.max())
print('uv_mask ', uv_mask.shape, uv_mask.dtype, uv_mask.min(), uv_mask.max())


conditional_images = get_depth_conditioning_images(uv_pos, uv_mask, output_size=1024)

save_conditional_images(conditional_images/2+0.5, os.path.join(out_dir, f"condi_vis_{use_sdxl}"))
condi_pngs = save_conditional_images(conditional_images, os.path.join(out_dir, f"pro_conditional_images_xl_{use_sdxl}"))
torch.save(conditional_images.cpu(), os.path.join(out_dir, "conditional_images_my.pt"))

print('condi_pngs ', condi_pngs)

# exit()

if use_sdxl:
    pipe = init_pipe_sdxl()
else:
    pipe = init_pipe()

seed = 1234
# prompt = "a yellow t shirt with a brown bear on it"
# prompt = "a yellow t shirt with a brown bear on it"
# negative_prompt= ""
prompt= "a yellow t shirt with a brown bear on it, high quality"
negative_prompt= "blur, low quality, noisy image, over-exposed"
direction_names = ["", "front", "side", "back", "top", "bottom", "face close-up"]
dir_id = [3, 3, 2, 1, 1, 1, 2, 3]

controlnet_conditioning_scale = 1.0
guidance_scale=7.0
num_inference_steps=20
control_guidance_start = 0.0
control_guidance_end = 0.99

os.makedirs(os.path.join(out_dir, f"out_rgb_xl_{use_sdxl}"), exist_ok=True)
pils = []
for i, condi_png in enumerate(condi_pngs):
    images = pipe(prompt, # + f", {direction_names[dir_id[i]]} view",
                negative_prompt = negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                height=1024,
                width=1024,
                image=load_image(condi_png),
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
                generator=torch.manual_seed(seed)
                ).images
    pils += images
    images[0].save(os.path.join(out_dir, f"out_rgb_xl_{use_sdxl}/image_{i}.png"))

make_image_grid(pils, 1, len(pils)).save(os.path.join(out_dir, f"out_rgb_xl_{use_sdxl}/image_merge.png"))
