import os
import argparse
import torch
import random
import torch.nn.functional as F
from easydict import EasyDict as edict

from dataset.utils_dataset import parse_objs_json, load_json, save_json
from pipe_texcontrol import ObjTexControlPipeline


def query_data_one_obj(objs_dict, key_pair):
    d, dname, oname = key_pair
    meta_dict = objs_dict[d][dname][oname]
    return dname, oname, meta_dict


def texcontrol_pipeline(in_model_path, in_sd_path, ip_adapter_model_path, in_dataset_json, out_dir, infer_cnt=4,
                              device='cuda', debug_save=True):
    if not os.path.exists(in_model_path) or not os.path.exists(in_dataset_json):
        raise ValueError(f'can not find valid {in_model_path} / {in_dataset_json} ')

    objs_dict, key_pair_list = parse_objs_json(in_dataset_json)
    random.shuffle(key_pair_list)

    # key_pair_list = [('data', 'Designcenter_1', '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB')]
    # key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]

    obj_tex_pipe = None
    # Each obj
    infer_cnt = min(infer_cnt, len(key_pair_list))
    for i in range(infer_cnt):
        dname, oname, meta_dict = query_data_one_obj(objs_dict, key_pair_list[i])
        uv_kd = meta_dict['uv_kd']
        in_prompts = meta_dict['caption']
        in_condi_img=meta_dict['condi_imgs_in'][-1] # select 32 TODO
        in_obj = meta_dict["Mesh_obj_pro"]
        # in_obj = os.path.join(os.path.dirname(uv_kd), 'mesh.obj')

        out_objs_dir = os.path.join(out_dir, dname, oname)
        os.makedirs(out_objs_dir, exist_ok=True)
        out_obj = os.path.join(out_objs_dir, 'mesh.obj')

        # 1. Init model, generator, frames and glctx. / optim_cfg TODO(lrm_mode)
        if obj_tex_pipe is None:
            obj_tex_pipe = ObjTexControlPipeline(
                in_model_path, in_sd_path, ip_adapter_model_path, device=device
            )

        cfg = {
            "uv_res": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "debug_save": True,
        }
        cfg = edict(cfg)
        
        # 2. load obj and render raw images, infer texrefine model, get new images, optim tex
        obj_tex_pipe.test_pipe_obj_texcontrol(
            in_obj,
            out_objs_dir,
            in_prompts=in_prompts,
            in_condi_img=in_condi_img,
            debug_save=debug_save,
            run_cfg=cfg,
        )

        # 3. test text only and image only modes
        obj_tex_pipe.test_pipe_obj_texcontrol(
            in_obj,
            os.path.join(out_dir, dname, "text_" + oname),
            in_prompts=in_prompts,
            in_condi_img=None,
            debug_save=debug_save,
            run_cfg=cfg,
        )
        obj_tex_pipe.test_pipe_obj_texcontrol(
            in_obj,
            os.path.join(out_dir, dname, "img_" + oname),
            in_prompts=None,
            in_condi_img=in_condi_img,
            debug_save=debug_save,
            run_cfg=cfg,
        )

    return


def main():
    parser = argparse.ArgumentParser(
        description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_model_path', type=str, help='path of trained ControlNetModel')
    parser.add_argument('in_sd_path', type=str, help='path of StableDiffusionControlNetPipeline')
    parser.add_argument('ip_adapter_model_path', type=str, help='path of ip adapter')
    parser.add_argument('in_dataset_json', type=str)
    parser.add_argument('out_dir', type=str, help='out dir with new obj and mtl, texture map')
    parser.add_argument('--infer_cnt', type=int, default=4, help='infer obj cnt, sample from json')
    args = parser.parse_args()

    texcontrol_pipeline(
        args.in_model_path,
        args.in_sd_path,
        args.ip_adapter_model_path,
        args.in_dataset_json,
        args.out_dir,
        infer_cnt=args.infer_cnt,
    )

    # Done.
    print("Done.")

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
