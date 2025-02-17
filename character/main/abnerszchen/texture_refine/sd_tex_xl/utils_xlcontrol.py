import argparse
import os
import random
import numpy as np
import torch
import PIL

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

# from render.render_obj import render_obj_with_in_kd, save_render
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally, load_rgba_as_rgb


def infer_batch(pipeline,
                generator,
                prompt_texts,
                uv_condi_paths,
                uv_kd_paths,
                batch_size=2,
                resolution=1024,
                infer_sd=False):
    """batch infer with input list of image paths

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        prompt_img_paths: list of condi image path
        render_depth_paths: list of render image path
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
        print('debug once pipe ', len(prompt_text))

        # TODO(csz) cfg
        num_inference_steps = 20
        guidance_scale = 9.0
        controlnet_conditioning_scale = 0.8

        # TODO
        with torch.autocast(
            str(pipeline.device).replace(":0", ""), enabled=True
        ):                   
        # with torch.autocast(
        #     str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
        # ):          
            # print('debug prompt_text ', prompt_text)         
            # print('debug normal_pils ', normal_pils)         
            # print('debug pipeline ', pipeline.device)         
            edited_images = pipeline(
                prompt=prompt_text,
                image=normal_pils,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale, 
            ).images

            if infer_sd:
                sd_images = pipeline(
                    prompt=prompt_text,
                    image=normal_pils,
                    num_inference_steps=20,
                    guidance_scale=7.5,  # text only
                    generator=generator,
                    controlnet_conditioning_scale=0.0, #for text only
                ).images

        vis_pils_2d = [normal_pils, edited_images, gt_pils]
        if infer_sd:
            vis_pils_2d += [sd_images]
        
        output_image = concatenate_images_2d(vis_pils_2d)
        output_image_list.append(output_image)


    return output_image_list


def make_infer_data(select_dicts, sample_render_img=False):
    """_summary_

    Args:
        select_dicts
        sample_render_img: if True, sample sample_render_img

    Returns:
        prompt_texts, uv_condi_paths, uv_kd_paths  list of str or path
    """
    # run inference
    prompt_texts, uv_condi_paths, uv_kd_paths = [], [], []

    # get batch input
    for select_dict in select_dicts:
        prompt_texts.append(select_dict["text"])
        uv_condi_paths.append(select_dict["conditioning_image"])
        uv_kd_paths.append(select_dict["image"])
        
    return prompt_texts, uv_condi_paths, uv_kd_paths

def infer_xlcontrol_batch_helper(pipeline, generator, select_dicts, infer_batch_size, resolution=1024, infer_sd=False, test_mode=False):
    """batch infer with input select_dicts

    Args:
        pipeline: 
        generator: torch.Generator("cuda").manual_seed(42)
        select_dicts: list of dict in dataset
        infer_batch_size: sd batch
        resolution 1024
        infer_sd: add result of sd without controlnet if True. Defaults to False
        test_mode: use mannul prompt
    Return:
        output_image: concated image.
        captions: use captions, list of str
    """    
    prompt_texts, uv_condi_paths, uv_kd_paths = make_infer_data(select_dicts)
    if test_mode:
        prompt_texts = ["red chinese dragon"] * len(uv_condi_paths)
    
    output_image_list = infer_batch(pipeline,
                                    generator,
                                    prompt_texts,
                                    uv_condi_paths,
                                    uv_kd_paths,
                                    batch_size=infer_batch_size,
                                    resolution=resolution,
                                    infer_sd=infer_sd)
    output_image = concatenate_images_horizontally(output_image_list)

    return output_image, prompt_texts
    
    return

