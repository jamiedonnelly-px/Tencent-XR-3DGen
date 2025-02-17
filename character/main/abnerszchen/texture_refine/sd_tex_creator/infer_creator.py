import argparse
import os
import random
import numpy as np
import torch
import PIL

from pipeline_stable_diffusion_tex_creator import StableDiffusionTexCreatorPipeline
from diffusers.utils.torch_utils import randn_tensor

import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, parse_objs_json, load_rgba_as_rgb, load_depth, scale_depth_pil_255
import torch.nn.functional as F

def debug_load_d_as_train(depth_path, resolution):
    render_depth = load_depth(depth_path, resolution=resolution)  # TODO need check resize depth
    render_depth = F.interpolate(render_depth.unsqueeze(0), size=(resolution // 8, resolution // 8), 
                                 mode='bilinear', align_corners=False).squeeze(0)    
    return render_depth

def check_is_prompt_image(prompt):
    if prompt is None or not prompt:
        return False
    if prompt is not None and isinstance(prompt, str):
        return os.path.exists(prompt)
    elif prompt is not None and isinstance(prompt, list):
        return os.path.exists(prompt[0])
    
    return False

def infer_temp_batch(pipeline, condis, render_depth_paths, gt_image_paths, out_dir, batch_size=2):
    # generator = torch.Generator("cuda")
    generator = torch.Generator("cuda").manual_seed(42)
    resolution = 512
    resolution_depth = resolution // 8
    os.makedirs(out_dir, exist_ok=True)
    
    output_image_list = []
    for i in range(0, len(condis), batch_size):
        prompt = condis[i:i+batch_size]
        print("prompt ", prompt)
        
        # in mm
        render_depth_pils = [scale_depth_pil_255(img) for img in render_depth_paths[i:i+batch_size]] # justrfor vis, not resize
        # render_depth = [load_depth(img, resolution_depth).unsqueeze(0) for img in render_depth_paths[i:i+batch_size]]
        render_depth = [debug_load_d_as_train(img, resolution).unsqueeze(0) for img in render_depth_paths[i:i+batch_size]]
        render_depth_tensor = torch.cat(render_depth, dim=0)    # [b, 1, 64, 64]
        
        condi_img = [load_rgba_as_rgb(img) for img in condis[i:i+batch_size] if check_is_prompt_image(img)]
        gt_img = [load_rgba_as_rgb(img) for img in gt_image_paths[i:i+batch_size]]
        
        num_inference_steps_l = [20] #[10, 20, 50]
        depth_guidance_scale_l = [1.5] # [0.5, 1, 1.5, 2, 3, 5, 7]
        guidance_scale_l = [2] # [0.5, 1, 1.5, 2, 3, 5, 7]
        latents = None
        if not check_is_prompt_image(prompt[0]):
            depth_guidance_scale_l = [1.5, 3, 5] # [0.5, 1, 1.5, 2, 3, 5, 7]
            guidance_scale_l = [2, 3, 5, 7]
            latents = randn_tensor([1, 4, resolution_depth, resolution_depth], generator=generator, device=pipeline.device, dtype=pipeline.dtype)
            latents = latents.repeat(len(prompt), 1, 1, 1)
            
        for num_inference_steps in num_inference_steps_l:
            for depth_guidance_scale in depth_guidance_scale_l:
                for guidance_scale in guidance_scale_l:
                    edited_images = pipeline(
                            prompt,
                            depth=render_depth_tensor,
                            num_inference_steps=num_inference_steps,
                            depth_guidance_scale=depth_guidance_scale,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            latents=latents,
                        ).images
                    
                    render_alpha_pils = []
                    for idx, render_depth_pil in enumerate(render_depth_pils):
                        alpha = (np.array(render_depth_pil)[..., None] > 0)
                        
                        render_alpha_pils.append(PIL.Image.fromarray(np.array(edited_images[idx]) * alpha))
                    vis = [render_depth_pils, edited_images, render_alpha_pils, gt_img]
                    if len(condi_img) == len(render_depth_pils):
                        vis.append(condi_img)
                    output_image = concatenate_images_2d(vis,
                                            os.path.join(out_dir, f'b_{i}_n_{num_inference_steps}_ds_{depth_guidance_scale}_gs_{guidance_scale}.jpg'))
                    output_image_list.append(output_image)
                    
    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())  / (1024 ** 3)
    print(f"Peak memory: {allocated_memory} G")
    
    return output_image_list

def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('model_path', type=str, default="/aigc_cfs/sz/result/tex_creator/condi_g1/first_2k_b16a1_nsddim")
    parser.add_argument('in_json', type=str, default='/aigc_cfs/sz/result/tex/first_2k/tex_creator_test.json')
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--infer_cnt', type=int, default=1, help='infer obj cnt')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--condi_mode', type=str, default="image")
    args = parser.parse_args()

    pipeline = StableDiffusionTexCreatorPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
    
    # Run.
    objs_dict, key_pair_list = parse_objs_json(args.in_json)
    random.shuffle(key_pair_list)
    sample_key_pair_list = key_pair_list[:args.infer_cnt]
    
    # sample_key_pair_list = [('data', 'guofenggame', 'C0019_clean_gaojishiwei')]
    # sample_key_pair_list = [('data', 'Designcenter_20231201', 'c020c525321687919d1eba0175d73340a9dbdcd2_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'Designcenter_1', '00d5b741c37918dae29ad83bb01876c53056992c_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1505326246213873410_manifold_full_output_512_MightyWSB')]
    
    out_dir = args.out_dir
    condi_mode = args.condi_mode
    
    print(f'infer {len(sample_key_pair_list)}/{len(key_pair_list)} ')
    
    
    # run inference
    # condi_imgs, render_depths, edited_images, gt_images = [], [], [], []
    condis, render_depth_paths, gt_image_paths = [], [], []
    
    # get batch input
    for i in range(args.infer_cnt):
        d, dname, oname = sample_key_pair_list[i]
        meta_dict = objs_dict[d][dname][oname]
        tex_pairs = meta_dict['tex_pairs']
        pair_cnt = len(tex_pairs)
        print(tex_pairs)

        if condi_mode == "image":
            if 'Condition_img' in meta_dict:
                condi = meta_dict['Condition_img']
            elif 'condition_imgs' in meta_dict:
                # prompt_img_path = meta_dict['condition_imgs'][-3]       
                condi = random.choice(meta_dict['condition_imgs'])             
        elif condi_mode == "text":
            assert 'caption' in meta_dict, meta_dict
            condi = random.choice(meta_dict['caption'])
                        
        condis += [condi] * pair_cnt
        gt_image_paths += [pair[0] for pair in tex_pairs]
        render_depth_paths += [pair[1] for pair in tex_pairs]
    
    infer_temp_batch(pipeline, condis, render_depth_paths, gt_image_paths, out_dir, batch_size=args.batch_size)

    
    print(f"Done., save to {out_dir}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
