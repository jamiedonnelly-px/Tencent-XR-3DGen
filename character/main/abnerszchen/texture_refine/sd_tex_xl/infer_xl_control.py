import argparse
import os
import random
import numpy as np
import torch
import PIL

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler, 
    DDPMScheduler,
)


import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import (
    parse_objs_json,
)


from sd_tex_control.utils_control import (
    make_infer_data,
    vis_render_kd,
    load_rgba_as_rgb,
    concatenate_images_2d,
)


def infer_enum_batch(
    pipeline,
    generator,
    prompt_texts,
    uv_condi_paths,
    uv_kd_paths,
    gt_render_imgs,
    out_dir,
    batch_size=2,
    test_ip=True,
    resolution=1024,
    ip_adapter_scale=0.8,
):

    # generator = torch.Generator("cuda")
    os.makedirs(out_dir, exist_ok=True)

    output_image_list = []
    for i in range(0, len(prompt_texts), batch_size):
        prompt_text = prompt_texts[i : i + batch_size]

        condi_pils = [
            load_rgba_as_rgb(img, res=resolution)
            for img in uv_condi_paths[i : i + batch_size]
        ]
        gt_pils = [
            load_rgba_as_rgb(img, res=resolution)
            for img in uv_kd_paths[i : i + batch_size]
        ]
        gt_render_pils = [
            load_rgba_as_rgb(img, res=resolution)
            for img in gt_render_imgs[i : i + batch_size]
        ]

        num_inference_steps_l = [20]  # [10, 20, 50]
        guidance_scale_l = [7.5, 9.0, 11.0]  # [0.5, 1, 1.5, 2, 3, 5, 7]
        controlnet_conditioning_scale_l = [0.6, 0.8, 1.0]
        # guidance_scale_l = [4, 5, 7.5, 10]  # [0.5, 1, 1.5, 2, 3, 5, 7]

        for num_inference_steps in num_inference_steps_l:
            for guidance_scale in guidance_scale_l:
                for controlnet_conditioning_scale in controlnet_conditioning_scale_l:

                    # disable ip-adapter for text only
                    if test_ip:
                        pipeline.set_ip_adapter_scale(0)
                        print(
                            "ip addition_embed_type ",
                            pipeline.unet.config.addition_embed_type,
                        )  # None
                        print(
                            "ip image_projection_layers ",
                            pipeline.unet.encoder_hid_proj.image_projection_layers,
                        )  # [ImageProjection]

                    # pipeline.unet.set_default_attn_processor()

                    sd_images = pipeline(
                        prompt=prompt_text,
                        image=condi_pils,
                        num_inference_steps=30,
                        guidance_scale=7.5,  # text only
                        generator=generator,
                        controlnet_conditioning_scale=0.0,  # for text only
                        ip_adapter_image=condi_pils if test_ip else None,  # useless because of set_ip_adapter_scale(0)
                    ).images

                    edited_images = pipeline(
                        prompt=prompt_text,
                        image=condi_pils,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        ip_adapter_image=condi_pils if test_ip else None,  # useless because of set_ip_adapter_scale(0)
                    ).images
                        
                    result_pils = [
                        condi_pils,
                        edited_images,
                        gt_pils,
                        sd_images,
                        gt_render_pils,
                    ]
                    
                    try:
                        render_pils = vis_render_kd(
                            edited_images,
                            uv_kd_paths[i : i + batch_size],
                            render_res=resolution // 2,
                        )
                        result_pils.append(render_pils)
                    except Exception as e:
                        print("render failed, skip render", e)

                    ## ip
                    if test_ip:
                        pipeline.set_ip_adapter_scale(ip_adapter_scale)
                        img2_images = pipeline(
                            prompt=[""] * len(prompt_text),
                            image=condi_pils,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            ip_adapter_image=sd_images,
                        ).images
                        textimg2_images = pipeline(
                            prompt=prompt_text,
                            image=condi_pils,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            ip_adapter_image=sd_images,
                        ).images
                        result_pils += [img2_images, textimg2_images]
                        
                    output_image = concatenate_images_2d(
                        result_pils,
                        os.path.join(
                            out_dir,
                            f"b_{i}_n_{num_inference_steps}_gs_{guidance_scale}_ccs_{controlnet_conditioning_scale}.jpg",
                        ),
                    )

                    output_image_list.append(output_image)

    allocated_memory = (
        torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
    ) / (1024**3)
    print(f"Peak memory: {allocated_memory} G")

    return output_image_list


def load_xl_pipeline(
    model_path,
    sd_model_path,
    pretrained_vae_model_name_or_path=None,
    ip_adapter_model_path=None,
    use_ip_mode = "plus_vit-h",
    run_t10=False,
):
    """load sdxl (with ip-adapter)

    Args:
        model_path: _description_
        sd_model_path: _description_
        pretrained_vae_model_name_or_path: _description_. Defaults to None.
        ip_adapter_model_path: _description_. Defaults to None.
        use_ip_mode: raw / vit-h / plus_vit-h. Defaults to "plus_vit-h".
        run_t10: _description_. Defaults to False.

    # models/image_encoder: OpenCLIP-ViT-H-14 with 632.08M parameter
    # sdxl_models/image_encoder: OpenCLIP-ViT-bigG-14 with 1844.9M parameter
    # ip-adapter_sdxl.bin: use global image embedding from OpenCLIP-ViT-bigG-14 as condition
    # ip-adapter_sdxl_vit-h.bin: same as ip-adapter_sdxl, but use OpenCLIP-ViT-H-14
    # ip-adapter-plus_sdxl_vit-h.bin: use patch image embeddings from OpenCLIP-ViT-H-14 as condition, closer to the reference image than ip-adapter_xl and ip-adapter_sdxl_vit-h

    Raises:
        ValueError: _description_

    Returns:
        _description_
    """
    weight_dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=weight_dtype)
    if pretrained_vae_model_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(
            pretrained_vae_model_name_or_path, torch_dtype=weight_dtype
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae", torch_dtype=weight_dtype
        )

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        sd_model_path,
        vae=vae,
        controlnet=controlnet,
        variant="fp16",
        torch_dtype=weight_dtype,
        use_safetensors=True,
    ).to("cuda")
            
    if ip_adapter_model_path is not None and os.path.exists(ip_adapter_model_path):
        if use_ip_mode == "raw":
            ip_image_encoder_folder = "sdxl_models/image_encoder"
            weight_name = "ip-adapter_sdxl.safetensors"
        elif use_ip_mode == "vit-h":
            ip_image_encoder_folder = "models/image_encoder"
            weight_name = "ip-adapter_sdxl_vit-h.safetensors"
        elif use_ip_mode == "plus_vit-h":
            ip_image_encoder_folder = "models/image_encoder"
            weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
        else:
            raise ValueError(f"invalid use_ip_mode {use_ip_mode}")        
        
        pipeline.load_ip_adapter(
            ip_adapter_model_path,
            subfolder="sdxl_models",
            weight_name=weight_name,
            image_encoder_folder=ip_image_encoder_folder,
        )
        print(f"load ip adapter done from {ip_adapter_model_path}, use_ip_mode: {use_ip_mode}")
          
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # less memory but slower. used in t10 instead of v100
    if run_t10:
        pipeline.enable_xformers_memory_efficient_attention()
        # memory optimization.
        pipeline.enable_model_cpu_offload()

    pipeline.safety_checker = None
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="batch infer xl control")
    parser.add_argument(
        "model_path",
        type=str,
        default="/aigc_cfs_3/sz/result/tex_control_2024/xl_ready/g4_pre_xyz_fixvae/",
    )
    parser.add_argument(
        "in_json",
        type=str,
        default="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_top_right_test.json",
    )
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--infer_cnt", type=int, default=2, help="infer obj cnt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="/aigc_cfs/model/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="/aigc_cfs/model/sdxl-vae-fp16-fix",
    )
    parser.add_argument(
        "--ip_adapter_model_path", type=str, default="/aigc_cfs/model/IP-Adapter"
    )
    parser.add_argument(
        "--test_ip", help="test ip-adapter if true", action="store_true"
    )
    args = parser.parse_args()

    model_path = args.model_path
    in_json = args.in_json
    out_dir = args.out_dir
    infer_cnt = args.infer_cnt
    batch_size = args.batch_size
    seed = args.seed
    sd_model_path = args.sd_model_path
    pretrained_vae_model_name_or_path = args.pretrained_vae_model_name_or_path
    ip_adapter_model_path = args.ip_adapter_model_path
    test_ip = args.test_ip
    if test_ip:
        batch_size = 1
        print("diffuser0.27 only support one ip image. set batch_size =1 TODO(csz)")
    else:
        ip_adapter_model_path = None

    assert os.path.exists(model_path)
    assert os.path.exists(in_json)
    assert os.path.exists(sd_model_path)

    pipeline = load_xl_pipeline(
        model_path,
        sd_model_path,
        pretrained_vae_model_name_or_path=pretrained_vae_model_name_or_path,
        ip_adapter_model_path=ip_adapter_model_path,
    )
    generator = torch.Generator("cuda").manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Run.
    objs_dict, key_pair_list = parse_objs_json(in_json)
    # random.shuffle(key_pair_list)
    sample_cnt = min(infer_cnt, len(key_pair_list))
    sample_key_pair_list = key_pair_list[:sample_cnt]  # sample obj

    print(f"infer {len(sample_key_pair_list)}/{len(key_pair_list)} ")

    # output_image, prompt_texts = infer_control_batch_helper(pipeline, generator, objs_dict, sample_key_pair_list, batch_size)
    # output_image.save(os.path.join(out_dir, 'check_code.png'))

    # run inference
    prompt_texts, uv_condi_paths, uv_kd_paths, gt_render_imgs = make_infer_data(
        objs_dict, sample_key_pair_list, use_obj_all=False, sample_render_img=True
    )
    infer_enum_batch(
        pipeline,
        generator,
        prompt_texts,
        uv_condi_paths,
        uv_kd_paths,
        gt_render_imgs,
        out_dir,
        batch_size=batch_size,
        test_ip=test_ip,
    )

    print(f"Done., save to {out_dir}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
