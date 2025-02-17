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

import os
from datetime import datetime
import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import numpy as np

def init_pipe_sdxl_t2i():
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    sd_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0/"
    vae_path = "/aigc_cfs/model/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(
        vae_path, torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        sd_path,
        vae=vae,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    return pipe


def infer_t2i_sdxl_once(model, in_prompt, out_path, size = 1024):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    images = model(
        prompt=in_prompt, 
        negative_prompt="lowres, bad anatomy, worst quality, low quality", 
        num_inference_steps=30,
    ).images

    images[0].save(out_path)
    return


def b_run_depth(in_json, out_dir):
    assert os.path.exists(in_json), in_json
    use_sdxl = True
    model = init_pipe_sdxl_t2i()
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
            infer_t2i_sdxl_once(model, meta["caption"][0], out_path, size = 1024)
            meta["infer_sdxl"] = out_path
            meta["infer_sdxl_in"] = [meta["condi_imgs_train"][0], meta["caption"][0]]
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
