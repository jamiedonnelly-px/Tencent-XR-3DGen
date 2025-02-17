import os
import argparse
import time
import random
import torch.nn.functional as F
from easydict import EasyDict as edict

from dataset.utils_dataset import parse_objs_json, load_json
from pipe_texuv import ObjTexUVPipeline


def query_data_one_obj(objs_dict, key_pair):
    d, dname, oname = key_pair
    meta_dict = objs_dict[d][dname][oname]
    return dname, oname, meta_dict


def texuv_pipeline(in_model_path,
                   in_sd_path,
                   pretrained_vae_model_name_or_path,
                   ip_adapter_model_path,
                   in_dataset_json,
                   out_dir,
                   infer_cnt=4,
                   cfg_json="configs/client_texgen.json",
                   model_key="uv_mcwy",
                   device='cuda',
                   debug_save=True):
    if not os.path.exists(in_model_path) or not os.path.exists(in_dataset_json):
        raise ValueError(f'can not find valid {in_model_path} / {in_dataset_json} ')

    objs_dict, key_pair_list = parse_objs_json(in_dataset_json)
    random.seed(1234)
    random.shuffle(key_pair_list)

    # key_pair_list = [('data', 'Designcenter_1', '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB')]
    # key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]

    obj_tex_pipe = None
    # Each obj
    infer_cnt = min(infer_cnt, len(key_pair_list))
    time_use_list = []
    for i in range(infer_cnt):
        dname, oname, meta_dict = query_data_one_obj(objs_dict, key_pair_list[i])
        uv_kd = meta_dict['uv_kd']
        in_prompts = meta_dict['caption']
        in_condi_img = meta_dict['condi_imgs_in'][-1]  # select 32 TODO
        in_obj = meta_dict["Mesh_obj_pro"]
        # in_obj = os.path.join(os.path.dirname(uv_kd), 'mesh.obj')

        out_objs_dir = os.path.join(out_dir, dname, oname)
        os.makedirs(out_objs_dir, exist_ok=True)
        out_obj = os.path.join(out_objs_dir, 'mesh.obj')

        # 1. Init model, generator, frames and glctx. / optim_cfg TODO(lrm_mode)
        if obj_tex_pipe is None:
            obj_tex_pipe = ObjTexUVPipeline(
                in_model_path,
                in_sd_path,
                pretrained_vae_model_name_or_path,
                ip_adapter_model_path,
                device=device,
            )

        optim_cfg_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_json)
        cfg = load_json(optim_cfg_json)[model_key]
        # cfg = {
        #     "uv_res": 1024,
        #     "num_inference_steps": 20,
        #     "guidance_scale": 9.0,
        #     "controlnet_conditioning_scale": 0.8,
        #     "ip_adapter_scale": 0.8,
        #     "debug_save": True,
        # }
        cfg = edict(cfg)

        # 2. load obj and render raw images, infer texrefine model, get new images, optim tex
        in_prompts = in_prompts[0]  # need batch size =1 TODO

        # test mix mode with small ip_adapter_scale
        # mix_ip_division_scale = 4.0 if "mix_ip_division_scale" not in cfg else cfg.mix_ip_division_scale
        # print('mix_ip_division_scale ', mix_ip_division_scale)
        # cfg.ip_adapter_scale = cfg.ip_adapter_scale / mix_ip_division_scale
        # obj_tex_pipe.test_pipe_obj_texuv(
        #     in_obj,
        #     out_objs_dir,
        #     in_prompts=in_prompts,
        #     in_condi_img=in_condi_img,
        #     debug_save=debug_save,
        #     run_cfg=cfg,
        # )
        # cfg.ip_adapter_scale = cfg.ip_adapter_scale * mix_ip_division_scale

        # 3. test text only and image only modes
        ts = time.time()
        obj_tex_pipe.test_pipe_obj_texuv(
            in_obj,
            os.path.join(out_dir, dname, "text_" + oname),
            in_prompts=in_prompts,
            in_condi_img=None,
            debug_save=debug_save,
            run_cfg=cfg,
        )
        time_use_list.append(time.time() - ts)

        # obj_tex_pipe.test_pipe_obj_texuv(
        #     in_obj,
        #     os.path.join(out_dir, dname, "img_" + oname),
        #     in_prompts=None,
        #     in_condi_img=in_condi_img,
        #     debug_save=debug_save,
        #     run_cfg=cfg,
        # )
    print('time_use_list ', time_use_list)

    return


def main():
    parser = argparse.ArgumentParser(
        description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_model_path', type=str, help='path of trained ControlNetModel')
    parser.add_argument('in_sd_path', type=str, help='path of StableDiffusionXLControlNetPipeline')
    parser.add_argument('pretrained_vae_model_name_or_path', type=str, help='path of VAE')
    parser.add_argument('ip_adapter_model_path', type=str, help='path of ip adapter')
    parser.add_argument('in_dataset_json', type=str)
    parser.add_argument('out_dir', type=str, help='out dir with new obj and mtl, texture map')
    parser.add_argument('--infer_cnt', type=int, default=4, help='infer obj cnt, sample from json')
    parser.add_argument('--cfg_json', type=str, default="configs/client_texgen.json", help='in configs/')
    parser.add_argument('--model_key', type=str, default="uv_mcwy", help='in cfg_json')
    args = parser.parse_args()

    texuv_pipeline(
        args.in_model_path,
        args.in_sd_path,
        args.pretrained_vae_model_name_or_path,
        args.ip_adapter_model_path,
        args.in_dataset_json,
        args.out_dir,
        infer_cnt=args.infer_cnt,
        cfg_json=args.cfg_json,
        model_key=args.model_key,
    )

    # Done.
    print("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
