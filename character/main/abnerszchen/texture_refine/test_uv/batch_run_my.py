import argparse
import time
import torch
import json
import os
import random
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import parse_objs_json, load_json, concatenate_images_2d, save_json, load_rgba_as_rgb
from render.render_obj import render_obj_with_in_kd, save_render
from render.mesh import load_mesh, Mesh, auto_normals
from test_uv.random_desc import dname_desc_map


def init_pipe_my_sdxl():
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    sd_path = "/aigc_cfs/model/stable-diffusion-xl-base-1.0/"
    vae_path = "/aigc_cfs/model/sdxl-vae-fp16-fix"
    # control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0-small"
    # control_path = "/aigc_cfs/model/controlnet-depth-sdxl-1.0"
    control_path = "/aigc_cfs_3/sz/result/tex_control_2024/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000/controlnet"
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16)

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


def infer_my_sdxl_once(
    model,
    in_obj,
    in_uv_pos,
    in_prompt,
    size=1024,
    vis_size=512,
    num_inference_steps=20,
    guidance_scale=9.0,
    controlnet_conditioning_scale=0.8,
):
    assert os.path.exists(in_uv_pos), in_uv_pos

    # infer xl
    in_uv_geom_pil = load_image(in_uv_pos)
    assert in_uv_geom_pil.size[0] == size, in_uv_geom_pil

    out_uv_pil = model(
        prompt=in_prompt,
        image=in_uv_geom_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cuda").manual_seed(1234),
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    out_t2i_pil = model(
        prompt=in_prompt,
        image=in_uv_geom_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(1234),
        controlnet_conditioning_scale=0.0,
    ).images[0]

    # render
    color, alpha = render_obj_with_in_kd(
        in_obj,
        out_uv_pil,
        "/aigc_cfs_2/sz/proj/tex_cq/data/cams/cam_parameters_srender1.json",
        lrm_mode=True,
        render_res=vis_size,
        use_normalized=True,
    )
    render_pil = save_render(color, alpha, bg_type="white", row=1)
    vis_pils = [out_t2i_pil.resize((vis_size, vis_size)), out_uv_pil.resize((vis_size, vis_size)), render_pil]
    cat_pil = concatenate_images_2d([vis_pils])
    return cat_pil


def b_run_my(in_json, out_dir, select_prompt_cnt=3, vis_size=256):
    assert os.path.exists(in_json), in_json
    use_sdxl = True
    model = init_pipe_my_sdxl()
    os.makedirs(out_dir, exist_ok=True)

    objs_dict, key_pair_list = parse_objs_json(in_json)
    random.seed(1234)
    random.shuffle(key_pair_list)

    select_keys = []  # ["MCWY_2_Bottom"]
    # append_desc = ""
    append_desc = ", HDR, UHD, 4K"
    key_pair_list = [key_pair for key_pair in key_pair_list
                     if key_pair[1] in select_keys] if select_keys else key_pair_list
    print(f"need run{len(key_pair_list)} use_sdxl={use_sdxl}")
    cnt = 0
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]

        descriptions = dname_desc_map[dname]
        sel_ids = random.sample(range(len(descriptions)), select_prompt_cnt)
        prompts = [meta["caption"][0]] + [descriptions[sel_id] for sel_id in sel_ids]
        prompts = [prompt + append_desc for prompt in prompts]
        out_path = os.path.join(out_dir, dname, f"{oname}.png")

        out_pils = [
            load_rgba_as_rgb(meta["uv_kd"], res=vis_size),
            load_rgba_as_rgb(meta["uv_pos"], res=vis_size),
            load_rgba_as_rgb(meta["condi_imgs_train"][0], res=vis_size)
        ]
        for in_prompt in prompts:
            # sdxl, infer, render
            once_pil = infer_my_sdxl_once(model,
                                          meta["Mesh_obj_pro"],
                                          meta["uv_pos"],
                                          in_prompt,
                                          size=1024,
                                          vis_size=vis_size)
            out_pils.append(once_pil)

        concatenate_images_2d([out_pils], out_path)
        meta["infer_uv_sdxl"] = out_path
        meta["infer_uv_sdxl_in"] = prompts
        print('prompts ', prompts)
        cnt += 1


    out_json = os.path.join(out_dir, "out.json")
    save_json(objs_dict, out_json)
    print(f"batch infer {cnt}/{len(key_pair_list)} done, use_sdxl={use_sdxl}, save to {out_json}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json',
                        type=str,
                        default='/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right_test.json',
                        help='')
    parser.add_argument('out_dir',
                        type=str,
                        help='')
    args = parser.parse_args()

    b_run_my(args.in_json, args.out_dir)
