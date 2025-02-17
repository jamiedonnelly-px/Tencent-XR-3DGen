import argparse
import os
import random
import numpy as np
import torch
import PIL
import time

from diffusers import (ControlNetModel)

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import concatenate_images_2d, parse_objs_json, load_rgba_as_rgb
from sd_tex_img_uv.utils_img_uv import make_infer_data
from sd_tex_uv.pipeline_sd_control_tex_uv import SDControlNetUVPipeline


def infer_enum_batch(pipeline,
                     generator,
                     prompt_texts,
                     uv_pos_paths,
                     uv_normal_paths,
                     condi_img_paths,
                     uv_kd_paths,
                     gt_render_imgs,
                     out_dir,
                     batch_size=2,
                     test_ip=True,
                     only_text=False):
    # generator = torch.Generator("cuda")
    resolution = 512
    os.makedirs(out_dir, exist_ok=True)

    output_image_list = []
    for i in range(0, len(prompt_texts), batch_size):
        prompt_text = prompt_texts[i:i + batch_size]

        pos_pils = [load_rgba_as_rgb(img, res=resolution) for img in uv_pos_paths[i:i + batch_size]]
        normal_pils = [load_rgba_as_rgb(img, res=resolution) for img in uv_normal_paths[i:i + batch_size]]
        condi_img_pils = [load_rgba_as_rgb(img, res=resolution) for img in condi_img_paths[i:i + batch_size]]
        
        gt_pils = [load_rgba_as_rgb(img, res=resolution) for img in uv_kd_paths[i:i + batch_size]]
        render_pils = [load_rgba_as_rgb(img, res=resolution) for img in gt_render_imgs[i:i + batch_size]]

        num_inference_steps_l = [20]  #[10, 20, 50]
        guidance_scale_l = [4, 5, 7.5, 10]  # [0.5, 1, 1.5, 2, 3, 5, 7]

        for num_inference_steps in num_inference_steps_l:
            for guidance_scale in guidance_scale_l:
                
                if only_text:
                    set_scale_st = time.time()
                    pipeline.set_ip_adapter_scale(0)
                    set_scale_time = time.time() - set_scale_st
                    
                    print(f"set_scale_time: {set_scale_time}")
                    print('prompt_text raw ', prompt_text)
                    # prompt_text = ["red"] * len(prompt_text)
                    # print('prompt_text new ', prompt_text)
                    
                # always activate ip-adapter
                edited_images = pipeline(
                    prompt_text,
                    pos_pils,
                    normal_pils,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ip_adapter_image=condi_img_pils,
                ).images
                
                if only_text:
                    pipeline.set_ip_adapter_scale(1)
                
                sd_images = pipeline(
                    prompt_text,
                    pos_pils,
                    normal_pils,
                    num_inference_steps=30,
                    guidance_scale=7.5,  # text only
                    generator=generator,
                    controlnet_conditioning_scale=0.0,  # for text only
                    ip_adapter_image=condi_img_pils,
                ).images

                result_pils = [pos_pils, normal_pils, condi_img_pils, edited_images, gt_pils, sd_images]
                # result_pils = [pos_pils, normal_pils, condi_img_pils, edited_images, gt_pils, sd_images, render_pils]

                if test_ip:
                    img2_images = pipeline([""] * len(prompt_text),
                                           pos_pils,
                                           normal_pils,
                                           num_inference_steps=num_inference_steps,
                                           guidance_scale=guidance_scale,
                                           generator=generator,
                                           ip_adapter_image=sd_images,
                                           ).images
                    textimg2_images = pipeline(prompt_text,
                                               pos_pils,
                                               normal_pils,
                                               num_inference_steps=num_inference_steps,
                                               guidance_scale=guidance_scale,
                                               generator=generator,
                                               ip_adapter_image=sd_images,
                                               ).images
                    result_pils += [img2_images, textimg2_images]
                output_image = concatenate_images_2d(
                    result_pils, os.path.join(out_dir, f'b_{i}_n_{num_inference_steps}__gs_{guidance_scale}.jpg'))

                output_image_list.append(output_image)

    allocated_memory = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024**3)
    print(f"Peak memory: {allocated_memory} G")

    return output_image_list


def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('model_path',
                        type=str,
                        default="/aigc_cfs_3/sz/result/tex_img_uv/g1/lowpoly_debug")
    parser.add_argument('in_json', type=str, default='/aigc_cfs/sz/data/tex/lowpoly/add_imgs_test.json')
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--infer_cnt', type=int, default=2, help='infer obj cnt')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sd_model_path', type=str, default='/aigc_cfs/model/stable-diffusion-v1-5')
    parser.add_argument('--ip_adapter_model_path', type=str, default='/aigc_cfs/model/IP-Adapter')
    parser.add_argument('--test_ip', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    args = parser.parse_args()

    model_path = args.model_path
    in_json = args.in_json
    out_dir = args.out_dir
    sd_model_path = args.sd_model_path
    ip_adapter_model_path = args.ip_adapter_model_path
    seed = args.seed

    assert os.path.exists(model_path)
    assert os.path.exists(in_json)
    assert os.path.exists(sd_model_path)

    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = SDControlNetUVPipeline.from_pretrained(sd_model_path, controlnet=controlnet,
                                                      torch_dtype=torch.float16).to("cuda")
    if os.path.exists(ip_adapter_model_path):
        pipeline.load_ip_adapter(ip_adapter_model_path, subfolder="models", weight_name="ip-adapter_sd15.bin")
        print(f'load ip adapter done from {ip_adapter_model_path}')

    pipeline.safety_checker = None
    generator = torch.Generator("cuda").manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Run.
    objs_dict, key_pair_list = parse_objs_json(args.in_json)
    # random.shuffle(key_pair_list)
    sample_cnt = min(args.infer_cnt, len(key_pair_list))
    sample_key_pair_list = key_pair_list[:sample_cnt]  # sample obj

    # sample_key_pair_list = [('data', 'readplayerMe', 'MCWY_2_Top_Tops_M_A_TOP_955_M_A_TOP_955_fbx2020_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'Designcenter_20231201', 'c020c525321687919d1eba0175d73340a9dbdcd2_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'Designcenter_1', '00d5b741c37918dae29ad83bb01876c53056992c_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1505326246213873410_manifold_full_output_512_MightyWSB')]
    print(f'infer {len(sample_key_pair_list)}/{len(key_pair_list)} ')

    # output_image, prompt_texts = infer_control_batch_helper(pipeline, generator, objs_dict, sample_key_pair_list, args.batch_size)
    # output_image.save(os.path.join(out_dir, 'check_code.png'))

    # run inference
    prompt_texts, uv_pos_paths, uv_normal_paths, condi_img_paths, uv_kd_paths, gt_render_imgs = make_infer_data(objs_dict,
                                                                                               sample_key_pair_list,
                                                                                               use_obj_all=False,
                                                                                               sample_render_img=True)
    infer_enum_batch(pipeline,
                     generator,
                     prompt_texts,
                     uv_pos_paths,
                     uv_normal_paths,
                     condi_img_paths,
                     uv_kd_paths,
                     gt_render_imgs,
                     out_dir,
                     batch_size=args.batch_size,
                     test_ip=args.test_ip,
                     only_text=args.only_text)

    print(f"Done., save to {out_dir}")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
