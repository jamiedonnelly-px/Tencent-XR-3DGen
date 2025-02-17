import argparse
import time
import torch
import json
import os
import random
from PIL import Image
from diffusers.utils import load_image

def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data
def save_json(json_data, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return
def parse_objs_json(objs_json):
    """ parse standard json to dict and pair list
    return: objs_dict: dict
    key_pair_list: list of pair ('data', dtype, oname)
    """
    if not os.path.exists(objs_json):
        print('[Error] can not find objs_json '.format(objs_json))
        return dict(), []
    objs_dict = load_json(objs_json)
    if 'data' not in objs_dict:
        print('[Error] not standard json '.format(objs_json))
        return dict(), []
    key_pair_list = []
    for dataset, dataset_dict in objs_dict['data'].items():
        key_pair_list += [('data', dataset, obj_name) for obj_name in list(dataset_dict.keys())]

    return objs_dict, key_pair_list
def load_rgba_as_rgb(img_path, res=None):
    """load with RGBA and convert to RGB with white backgroud, if is RGB just return

    Args:
        img_path: _description_

    Returns:
        PIL.Image [h, w, 3]
    """
    img = Image.open(img_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    if res is not None and isinstance(res, int):
        img = img.resize((res, res))
    return img        


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
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
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

    cfs_1 = "/aigc_cfs"
    depth_estimator = DPTForDepthEstimation.from_pretrained(f"{cfs_1}/model/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained(f"{cfs_1}/model/dpt-hybrid-midas")
    return pipe, depth_estimator, feature_extractor

def get_depth_map(depth_estimator, feature_extractor, image, size=1024):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def infer_sdxl_once(model, depth_estimator, feature_extractor, in_render_path, in_prompt, out_path, size = 1024):
    assert os.path.exists(in_render_path), in_render_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    depth_image = get_depth_map(depth_estimator, feature_extractor, load_rgba_as_rgb(in_render_path, res=size), size=size)
    controlnet_conditioning_scale = 0.8
    
    images = model(
        prompt=in_prompt, 
        image=depth_image, #depth_map,
        negative_prompt="lowres, bad anatomy, worst quality, low quality", 
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=30,
    ).images

    images[0].save(out_path)
    return


def b_run_depth(in_json, out_dir):
    assert os.path.exists(in_json), in_json
    use_sdxl = True
    model, depth_estimator, feature_extractor = init_pipe_sdxl()
    os.makedirs(out_dir, exist_ok=True)

    objs_dict, key_pair_list = parse_objs_json(in_json)
    random.seed(1234)
    random.shuffle(key_pair_list)

    print(f"need run{len(key_pair_list)} use_sdxl={use_sdxl}")
    cnt = 0
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]

        out_path = os.path.join(out_dir, dname, f"{oname}.png")
        if use_sdxl:
            infer_sdxl_once(model, depth_estimator, feature_extractor, meta["condi_imgs_train"][0], meta["caption"][0], out_path, size = 1024)
            meta["infer_c_sdxl"] = out_path
            meta["infer_c_sdxl_in"] = [meta["condi_imgs_train"][0], meta["caption"][0]]
        cnt += 1

    out_json = os.path.join(out_dir, "out.json")
    save_json(objs_dict, out_json)
    print(f"batch infer {cnt}/{len(key_pair_list)} done, use_sdxl={use_sdxl}, save to {out_json}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json', type=str, default='/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right_test.json', help='select model. can be uv_mcwy, control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly')
    parser.add_argument('out_dir', type=str, default='uv_mcwy', help='select model. can be uv_mcwy, control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly')
    args = parser.parse_args()

    b_run_depth(args.in_json, args.out_dir)
