import argparse
import os
import random
import numpy as np
import torch
import PIL

import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, concatenate_images_horizontally, load_rgba_as_rgb, load_depth, scale_depth_pil_255

def check_is_prompt_image(prompt):
    if prompt is None or not prompt:
        return False
    if prompt is not None and isinstance(prompt, str):
        return os.path.exists(prompt)
    elif prompt is not None and isinstance(prompt, list):
        return os.path.exists(prompt[0])
    
    return False


def infer_batch(pipeline, generator, condis, render_depth_paths, gt_image_paths, batch_size=2, resolution = 512):
    """batch infer with input list of image paths

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        condis: list of condi image path or list of condition text
        render_depth_paths: list of render image path
        gt_image_paths: list of gt path # TODO can be None.. future
        batch_size: pipeline input batch_size. Defaults to 2.
        resolution: pipeline input res. Defaults to 512.
    Return:
        output_image_list: list of concated image. len is len(condis)//batch_size
    """
    in_cnt = len(condis)
    resolution_depth = resolution // 8
    
    output_image_list = []
    for i in range(0, in_cnt, batch_size):
        prompt = condis[i:i+batch_size]
        
        # in mm
        render_depth_pils = [scale_depth_pil_255(img) for img in render_depth_paths[i:i+batch_size]] # justrfor vis, not resize
        render_depth = [load_depth(img, resolution_depth).unsqueeze(0) for img in render_depth_paths[i:i+batch_size]]
        render_depth_tensor = torch.cat(render_depth, dim=0)    # [b, 1, 64, 64]
        
        condi_img = [load_rgba_as_rgb(img) for img in condis[i:i+batch_size] if check_is_prompt_image(img)]
        gt_img = [load_rgba_as_rgb(img) for img in gt_image_paths[i:i+batch_size]]
        
        num_inference_steps = 20 
        depth_guidance_scale = 1.5 
        guidance_scale = 2 

        edited_images = pipeline(
                prompt,    # input list of path or list of str text
                depth=render_depth_tensor,   # tensor [b, 1, 64, 64]
                num_inference_steps=num_inference_steps,
                depth_guidance_scale=depth_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
        
        render_masked_pils, gt_masked_pils = [], []
        for i, render_depth_pil in enumerate(render_depth_pils):
            alpha = (np.array(render_depth_pil)[..., None] > 0)
            render_masked_pils.append(PIL.Image.fromarray(np.array(edited_images[i]) * alpha))
            gt_masked_pils.append(PIL.Image.fromarray(np.array(gt_img[i]) * alpha))
        
        vis = [render_depth_pils, edited_images, render_masked_pils, gt_masked_pils, gt_img]
        if len(condi_img) == len(render_depth_pils):
            vis.append(condi_img)
        output_image = concatenate_images_2d(vis)
        output_image_list.append(output_image)
    
    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())  / (1024 ** 3)
    print(f"Peak memory: {allocated_memory} G")            
    
    return output_image_list

def infer_creator_batch_helper(pipeline, generator, objs_dict, sample_key_pair_list, infer_cnt, infer_batch_size, use_img_condi=True):
    """batch infer with input json and key pairs

    Args:
        pipeline: StableDiffusionTexRefinePipeline
        generator: torch.Generator("cuda").manual_seed(42)
        objs_dict: first return of parse_objs_json(json), raw dict
        sample_key_pair_list: second return of parse_objs_json(json), list of (d, dname, oname)
        infer_cnt: total cnt
        infer_batch_size: sd batch
        use_img_condi use condition image if True else use text
    Return:
        output_image: concated image.
    """
    # run inference
    condis, render_depth_paths, gt_image_paths = [], [], []
    
    # get batch input paths
    for i in range(infer_cnt):
        d, dname, oname = sample_key_pair_list[i]
        meta_dict = objs_dict[d][dname][oname]
        
        if use_img_condi:
            if 'Condition_img' in meta_dict:
                condi = meta_dict['Condition_img']
            elif 'condition_imgs' in meta_dict:
                condi = random.choice(meta_dict['condition_imgs'])
            raise ValueError(f"can not find condition image")
        else:
            assert 'caption' in meta_dict, meta_dict
            condi = random.choice(meta_dict['caption'])
        tex_pairs = meta_dict['tex_pairs']
        select_view = random.randint(0, len(tex_pairs) - 1)
        gt_image_path, render_depth_path = tex_pairs[select_view][0], tex_pairs[select_view][1]
        condis.append(condi)
        render_depth_paths.append(render_depth_path)
        gt_image_paths.append(gt_image_path)
    
    print('debug gt_image_paths', len(gt_image_paths), gt_image_paths)
    
    output_image_list = infer_batch(pipeline, generator, condis, render_depth_paths, gt_image_paths, batch_size=infer_batch_size)
    output_image = concatenate_images_horizontally(output_image_list)
    return output_image

