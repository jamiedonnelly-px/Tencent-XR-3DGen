import argparse
import os
import random
import torch
import PIL

from pipeline_stable_diffusion_tex_refine import StableDiffusionTexRefinePipeline

import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, parse_objs_json, load_rgba_as_rgb


def infer_batch(pipeline, prompt_img_paths, render_img_paths, gt_image_paths, out_dir, batch_size=2):
    generator = torch.Generator("cuda")
    # generator = torch.Generator("cuda").manual_seed(42)
    resolution = 512
    in_cnt = len(prompt_img_paths)
    os.makedirs(out_dir, exist_ok=True)
    
    output_image_list = []
    for i in range(0, in_cnt, batch_size):
        prompt_img_path = prompt_img_paths[i:i+batch_size]
        render_img = [load_rgba_as_rgb(img).resize((resolution, resolution)) for img in render_img_paths[i:i+batch_size]]
        condi_img = [load_rgba_as_rgb(img) for img in prompt_img_paths[i:i+batch_size]]
        gt_img = [load_rgba_as_rgb(img) for img in gt_image_paths[i:i+batch_size]]
        
        num_inference_steps_l = [20] #[10, 20, 50]
        image_guidance_scale_l = [1.5] # [0.5, 1, 1.5, 2, 3, 5, 7]
        guidance_scale_l = [2] # [0.5, 1, 1.5, 2, 3, 5, 7]
        
        for num_inference_steps in num_inference_steps_l:
            for image_guidance_scale in image_guidance_scale_l:
                for guidance_scale in guidance_scale_l:
                    edited_images = pipeline(
                            prompt_img_path,
                            image=render_img,
                            num_inference_steps=num_inference_steps,
                            image_guidance_scale=image_guidance_scale,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images
                    
                    output_image = concatenate_images_2d([render_img, condi_img, edited_images, gt_img],
                                            os.path.join(out_dir, f'b_{i}_n_{num_inference_steps}_is_{image_guidance_scale}_gs_{guidance_scale}.jpg'))
                    output_image_list.append(output_image)
                    
    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())  / (1024 ** 3)
    print(f"Peak memory: {allocated_memory} G")            
    
    return output_image_list

def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('model_path', type=str, default="/aigc_cfs/sz/result/tex/condi_g1/first_2k")
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--infer_cnt', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()

    pipeline = StableDiffusionTexRefinePipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
    
    # Run.
    objs_dict, key_pair_list = parse_objs_json(args.in_json)
    # sample_key_pair_list = [('data', 'guofenggame', 'C0019_clean_gaojishiwei')]
    random.shuffle(key_pair_list)
    sample_key_pair_list = key_pair_list[:args.infer_cnt]
    
    out_dir = args.out_dir
    
    print(f'infer {len(sample_key_pair_list)}/{len(key_pair_list)} ')
    
    
    # run inference
    # condi_imgs, render_imgs, edited_images, gt_images = [], [], [], []
    prompt_img_paths, render_img_paths, gt_image_paths = [], [], []
    
    # get batch input
    for i in range(args.infer_cnt):
        d, dname, oname = sample_key_pair_list[i]
        meta_dict = objs_dict[d][dname][oname]
        tex_pairs = meta_dict['tex_pairs']
        pair_cnt = len(tex_pairs)
        print(tex_pairs)
        
        prompt_img_path = meta_dict['Condition_img']
        prompt_img_paths += [prompt_img_path] * pair_cnt
        gt_image_paths += [pair[0] for pair in tex_pairs]
        render_img_paths += [pair[1] for pair in tex_pairs]
    
    infer_batch(pipeline, prompt_img_paths, render_img_paths, gt_image_paths, out_dir, batch_size=args.batch_size)

    
    print(f"Done., save to {out_dir}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
