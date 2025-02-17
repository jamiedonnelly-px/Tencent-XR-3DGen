import argparse
import os
import random
import numpy as np
import torch
import PIL
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import (
    concatenate_images_2d,
    concatenate_images_horizontally,
    concatenate_images_vertically,
    parse_objs_json,
    load_rgba_as_rgb,
)
from sd_tex_control.utils_control import (
    make_infer_data,
    infer_batch,
    infer_control_batch_helper,
    vis_render_kd,
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
):
    resolution = 512

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

                    # disable ip-adapter
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
                        ip_adapter_image=condi_pils if test_ip else None,  # useless
                    ).images

                    edited_images = pipeline(
                        prompt=prompt_text,
                        image=condi_pils,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        ip_adapter_image=condi_pils if test_ip else None,  # useless
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
                            edited_images, uv_kd_paths[i : i + batch_size]
                        )
                        result_pils.append(render_pils)
                    except Exception as e:
                        print("render failed, skip render", e)

                    if test_ip:
                        print("debug sd_images ", len(sd_images))
                        print("debug prompt_text ", len(prompt_text))
                        # activate ip-adapter
                        pipeline.set_ip_adapter_scale(1)
                        img2_images = pipeline(
                            prompt=[""] * len(prompt_text),
                            image=condi_pils,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            ip_adapter_image=sd_images if test_ip else None,
                        ).images
                        textimg2_images = pipeline(
                            prompt=prompt_text,
                            image=condi_pils,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            ip_adapter_image=sd_images if test_ip else None,
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


def main():
    parser = argparse.ArgumentParser(description="batch infer control")
    parser.add_argument(
        "model_path",
        type=str,
        default="/aigc_cfs/sz/result/tex_creator/condi_g1/first_2k_b16a1_nsddim",
    )
    parser.add_argument(
        "in_json",
        type=str,
        default="/aigc_cfs/sz/result/tex/first_2k/tex_creator_test.json",
    )
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--infer_cnt", type=int, default=2, help="infer obj cnt")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sd_model_path", type=str, default="/aigc_cfs/model/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--ip_adapter_model_path", type=str, default="/aigc_cfs/model/IP-Adapter"
    )
    parser.add_argument("--test_ip", help="test ip-adapter if true", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    in_json = args.in_json
    out_dir = args.out_dir
    infer_cnt = args.infer_cnt
    batch_size = args.batch_size
    seed = args.seed
    sd_model_path = args.sd_model_path
    ip_adapter_model_path = args.ip_adapter_model_path
    test_ip = args.test_ip
    if test_ip:
        batch_size = 1
        print("diffuser0.27 only support one ip image. set batch_size =1 TODO(csz)")

    assert os.path.exists(model_path)
    assert os.path.exists(in_json)
    assert os.path.exists(sd_model_path)

    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_path, controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")
    if test_ip and os.path.exists(ip_adapter_model_path):
        pipeline.load_ip_adapter(
            ip_adapter_model_path, subfolder="models", weight_name="ip-adapter_sd15.bin"
        )
        print(f"load ip adapter done from {ip_adapter_model_path}")
        batch_size = 1
        print("diffuser0.27 only support one ip image. set batch_size =1 TODO(csz) checking..")
        
    pipeline.safety_checker = None
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

    # sample_key_pair_list = [('data', 'readplayerMe', 'MCWY_2_Top_Tops_M_A_TOP_955_M_A_TOP_955_fbx2020_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'Designcenter_20231201', 'c020c525321687919d1eba0175d73340a9dbdcd2_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'Designcenter_1', '00d5b741c37918dae29ad83bb01876c53056992c_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]
    # sample_key_pair_list = [('data', 'vroid', '0_1505326246213873410_manifold_full_output_512_MightyWSB')]
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
