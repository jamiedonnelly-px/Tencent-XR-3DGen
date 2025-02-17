import argparse
import os
import random
import numpy as np
import torch
import PIL

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from render.render_obj import render_obj_with_in_kd, save_render
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally, concatenate_images_vertically, load_rgba_as_rgb, load_depth, scale_depth_pil_255


def vis_render_kd(edited_images, uv_kd_paths, render_res=256):
    try:
        with torch.cuda.amp.autocast(enabled=False):  # render need float32
            in_pose_json = os.path.join(
                codedir, "data/cams/cam_parameters_srender4.json"
            )
            render_pils = []
            if os.path.exists(in_pose_json):
                for j, edited_image in enumerate(edited_images):
                    in_obj = os.path.join(os.path.dirname(uv_kd_paths[j]), "mesh.obj")
                    color, alpha = render_obj_with_in_kd(
                        in_obj,
                        edited_image,
                        in_pose_json,
                        lrm_mode=True,
                        render_res=render_res,
                        use_normalized=True,
                    )
                    render_pil = save_render(color, alpha, bg_type="white", row=2)
                    render_pils.append(render_pil)
    except Exception as e:
        print("vis render failed, skip", e)
        return []
    return render_pils


def infer_batch(pipeline,
                generator,
                prompt_texts,
                uv_condi_paths,
                uv_kd_paths,
                batch_size=2,
                resolution=512,
                infer_sd=False):
    """batch infer with input list of image paths

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        prompt_texts: list of text
        uv_condi_paths: list of condi image path
        uv_kd_paths: list of gt texture image path
        batch_size: pipeline input batch_size. Defaults to 2.
        resolution: pipeline input res. Defaults to 512.
        infer_sd: add result of sd without controlnet if True. Defaults to False
    Return:
        output_image_list: list of concated image (rows: in-uv, out-infer, gt, (sd text2img)). len is len(prompt_img_paths)//batch_size
    """
    in_cnt = len(prompt_texts)

    output_image_list = []
    for i in range(0, in_cnt, batch_size):
        prompt_text = prompt_texts[i:i + batch_size]
        normal_pils = [load_rgba_as_rgb(img, res=resolution) for img in uv_condi_paths[i:i + batch_size]]
        gt_pils = [load_rgba_as_rgb(img, res=resolution) for img in uv_kd_paths[i:i + batch_size]]

        # TODO(csz) cfg
        num_inference_steps = 20
        guidance_scale = 7.5
        controlnet_conditioning_scale = 0.8

        edited_images = pipeline(
            prompt_text,
            normal_pils,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        if infer_sd:
            sd_images = pipeline(
                prompt_text,
                normal_pils,
                num_inference_steps=30,
                guidance_scale=7.5,  # text only
                generator=generator,
                controlnet_conditioning_scale=0.0, #for text only
            ).images

        vis_pils_2d = [normal_pils, edited_images, gt_pils]
        if infer_sd:
            vis_pils_2d += [sd_images]

        vis_render = True
        if vis_render:
            render_pils = vis_render_kd(edited_images, uv_kd_paths[i:i + batch_size])
            if render_pils:
                vis_pils_2d.append(render_pils)
        
        output_image = concatenate_images_2d(vis_pils_2d)
        output_image_list.append(output_image)

    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024**3)
    print(f"Peak memory: {allocated_memory} G")

    return output_image_list


def make_infer_data(objs_dict, sample_key_pair_list, use_obj_all=False, sample_render_img=False):
    """_summary_

    Args:
        objs_dict: _description_
        sample_key_pair_list: _description_
        use_obj_all: if True, use all caption of one obj, else random select one
        sample_render_img: if True, sample sample_render_img

    Returns:
        prompt_texts, uv_condi_paths, uv_kd_paths, (gt_render_imgs can be []), list of str or path
    """
    # run inference
    prompt_texts, uv_condi_paths, uv_kd_paths, gt_render_imgs = [], [], [], []

    # get batch input
    for d, dname, oname in sample_key_pair_list:
        meta_dict = objs_dict[d][dname][oname]
        caption_list = meta_dict['caption']
        uv_condi = meta_dict['uv_pos']  # TODO
        uv_kd = meta_dict['uv_kd']
        pair_cnt = len(caption_list)
        if pair_cnt == 0:
            continue

        if not use_obj_all:
            select_i = random.randint(0, pair_cnt - 1)
            caption_list = [caption_list[select_i]]
            pair_cnt = 1
        print('caption_list', caption_list)

        if sample_render_img:
            render_imgs = None
            if 'render_imgs' in meta_dict:
                render_imgs = meta_dict['render_imgs']
            elif 'condi_imgs_in' in meta_dict:
                render_imgs = meta_dict['condi_imgs_in']
            if render_imgs:
                img_cnt = len(render_imgs)
                indices = random.sample(range(img_cnt), min(img_cnt, pair_cnt))
                gt_render_imgs += [render_imgs[i] for i in indices]

        prompt_texts += caption_list
        uv_condi_paths += [uv_condi] * pair_cnt
        uv_kd_paths += [uv_kd] * pair_cnt

    return prompt_texts, uv_condi_paths, uv_kd_paths, gt_render_imgs


def infer_control_batch_helper(pipeline, generator, objs_dict, sample_key_pair_list, infer_batch_size, resolution=512, infer_sd=False):
    """batch infer with input json and key pairs

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        objs_dict: first return of parse_objs_json(json), raw dict
        sample_key_pair_list: sample of second return of parse_objs_json(json), list of (d, dname, oname), len is obj num
        infer_batch_size: sd batch
        infer_sd: add result of sd without controlnet if True. Defaults to False
    Return:
        output_image: concated image.
        captions: use captions, list of str
    """
    # run inference
    prompt_texts, uv_condi_paths, uv_kd_paths, gt_render_imgs = make_infer_data(objs_dict, sample_key_pair_list, sample_render_img=True)

    output_image_list = infer_batch(pipeline,
                                    generator,
                                    prompt_texts,
                                    uv_condi_paths,
                                    uv_kd_paths,
                                    batch_size=infer_batch_size,
                                    resolution=resolution,
                                    infer_sd=infer_sd)
    output_image = concatenate_images_horizontally(output_image_list)

    if gt_render_imgs and len(gt_render_imgs) == len(uv_condi_paths):
        pils = [load_rgba_as_rgb(img).resize((resolution, resolution)) for img in gt_render_imgs]
        render_img = concatenate_images_horizontally(pils)
        output_image = concatenate_images_vertically([output_image, render_img])

    return output_image, prompt_texts
