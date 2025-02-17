import argparse
import os
import random
import torch


import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally, load_rgba_as_rgb

def infer_batch(pipeline, generator, prompt_img_paths, render_img_paths, gt_image_paths, batch_size=2, resolution = 512):
    """batch infer with input list of image paths

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        prompt_img_paths: list of condi image path
        render_img_paths: list of render image path
        gt_image_paths: list of gt path # TODO can be None.. future
        batch_size: pipeline input batch_size. Defaults to 2.
        resolution: pipeline input res. Defaults to 512.
    Return:
        output_image_list: list of concated image. len is len(prompt_img_paths)//batch_size
    """
    in_cnt = len(prompt_img_paths)
    
    output_image_list = []
    for i in range(0, in_cnt, batch_size):
        prompt_img_path = prompt_img_paths[i:i+batch_size]
        render_img = [load_rgba_as_rgb(img).resize((resolution, resolution)) for img in render_img_paths[i:i+batch_size]]
        condi_img = [load_rgba_as_rgb(img) for img in prompt_img_paths[i:i+batch_size]]
        gt_img = [load_rgba_as_rgb(img) for img in gt_image_paths[i:i+batch_size]]
        
        num_inference_steps = 20 
        image_guidance_scale = 1.5 
        guidance_scale = 2 

        edited_images = pipeline(
                prompt_img_path,    # input list of path
                image=render_img,   # input list of Image
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
        
        output_image = concatenate_images_2d([render_img, condi_img, edited_images, gt_img])
        output_image_list.append(output_image)
    
    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())  / (1024 ** 3)
    print(f"Peak memory: {allocated_memory} G")            
    
    return output_image_list

def infer_batch_helper(pipeline, generator, objs_dict, sample_key_pair_list, infer_cnt, infer_batch_size):
    """batch infer with input json and key pairs

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        objs_dict: first return of parse_objs_json(json), raw dict
        sample_key_pair_list: second return of parse_objs_json(json), list of (d, dname, oname)
        infer_cnt: total cnt
        infer_batch_size: sd batch
    Return:
        output_image: concated image.
    """
    # run inference
    prompt_img_paths, render_img_paths, gt_image_paths = [], [], []
    
    # get batch input paths
    for i in range(infer_cnt):
        d, dname, oname = sample_key_pair_list[i]
        meta_dict = objs_dict[d][dname][oname]
        select_view = random.randint(0, 6)
        
        prompt_img_path = meta_dict['Condition_img']
        print(meta_dict['tex_pairs'])
        gt_image_path, render_img_path = meta_dict['tex_pairs'][select_view][0], meta_dict['tex_pairs'][select_view][1]
        prompt_img_paths.append(prompt_img_path)
        render_img_paths.append(render_img_path)
        gt_image_paths.append(gt_image_path)
    
    output_image_list = infer_batch(pipeline, generator, prompt_img_paths, render_img_paths, gt_image_paths, batch_size=infer_batch_size)
    output_image = concatenate_images_horizontally(output_image_list)
    return output_image

