import os
import argparse
import torch
import random
import torch.nn.functional as F

from dataset.utils_dataset import parse_objs_json, load_json, save_json
from pipe_texcreator import ObjTexCreatorPipeline

def gaussian_filter(kernel_size, sigma):
    x = torch.linspace(-1, 1, kernel_size)
    y = torch.linspace(-1, 1, kernel_size)
    x, y = torch.meshgrid(x, y)
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-((d * d) / (2.0 * sigma * sigma)))
    return g / g.sum()


def random_opt_tex(tex_res, device='cuda'):
    tex = torch.randn((1, tex_res, tex_res, 3)).to(device)
    kernel_size = 7
    sigma = 1.5
    gaussian_kernel = gaussian_filter(kernel_size, sigma).to(device)

    smoothed_tensor = torch.zeros_like(tex)
    for i in range(tex.shape[-1]):
        smoothed_tensor[..., i] = F.conv2d(tex[..., i].unsqueeze(
            0), gaussian_kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2).squeeze()

    bounded_tensor = torch.sigmoid(smoothed_tensor)
    tex_merge = torch.nn.Parameter(bounded_tensor.clone(), requires_grad=True)
    return tex_merge


def query_data_one_obj(objs_dict, key_pair):
    d, dname, oname = key_pair
    meta_dict = objs_dict[d][dname][oname]
    return dname, oname, meta_dict



def optim_texcreator_pipeline(in_model_path, in_dataset_json, out_dir, infer_cnt=4, lrm_mode=False, pose_json='',
                              device='cuda', high_res=True, debug_save=True):
    if not os.path.exists(in_model_path) or not os.path.exists(in_dataset_json):
        raise ValueError(f'can not find valid {in_model_path} / {in_dataset_json} ')

    objs_dict, key_pair_list = parse_objs_json(in_dataset_json)
    # use_dnames = ['Designcenter_1', 'Designcenter_20231201']
    # key_pair_list = [pair for pair in key_pair_list if pair[1] in use_dnames]
    # random.shuffle(key_pair_list)
    # key_pair_list = [('data', 'Designcenter_1', '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB')]
    # key_pair_list = [('data', 'vroid', '0_1150871373227540420_manifold_full_output_512_MightyWSB')]

    obj_tex_pipe = None
    # Each obj
    infer_cnt = min(infer_cnt, len(key_pair_list))
    for i in range(infer_cnt):
        dname, oname, meta_dict = query_data_one_obj(objs_dict, key_pair_list[i])
        # if oname != '0b5110b0b4d94930b061e332a6af0c79_manifold_full_output_512_MightyWSB':
        #     continue
        select_view = []
        if pose_json != '' and os.path.exists(pose_json):
            pose_json_use = pose_json
            print('mannul load pose_json ', pose_json)
        else:
            pose_json_use = meta_dict['pose_json']
            if lrm_mode:
                select_view = [9, 11, 13, 15,  32, 34, 36, 38]
            else:
                select_view = [11, 48, 89, 100, 132, 167, 286, 318]
                # pose_dict = load_json(pose_json_use)
                # new_dict = {}
                # for v in select_view:
                #     name = f'cam-{v:04d}'
                #     new_dict[name] = pose_dict[name]
                # save_json(new_dict, 'data/cams/cam_parameters_human8.json')

        if 'Condition_img' in meta_dict:
            in_condi = meta_dict['Condition_img']
        elif 'condition_imgs' in meta_dict:
            if lrm_mode:
                in_condi = random.choice(meta_dict['condition_imgs'])
            else:  # TODO
                in_condi = meta_dict['condition_imgs'][-1]
        in_obj = meta_dict['diffusion_obj']

        # 1. Init model, generator, frames and glctx. / optim_cfg TODO(lrm_mode)
        if obj_tex_pipe is None:
            optim_cfg_json = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'configs', 'optim_cfg_high.json' if high_res else 'optim_cfg_low.json')
            obj_tex_pipe = ObjTexCreatorPipeline(in_model_path, optim_cfg_json, pose_json_use, lrm_mode, device=device)

        # 2. load obj and render raw images, infer texrefine model, get new images, optim tex
        batch_test_cnt = 10
        for idx in range(batch_test_cnt):
            obj_out_dir = os.path.join(out_dir, dname, oname, f"{idx:03d}")
            os.makedirs(obj_out_dir, exist_ok=True)
            out_obj = os.path.join(obj_out_dir, 'mesh.obj')        
            out_debug_dir = obj_out_dir if debug_save else None
            obj_tex_pipe.test_pipe_obj(pose_json_use, in_obj, in_condi, out_obj, lrm_mode,
                                    select_view=select_view, out_debug_dir=out_debug_dir)

    return


def main():
    parser = argparse.ArgumentParser(
        description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_model_path', type=str, help='texrefine model path')
    parser.add_argument('in_dataset_json', type=str)
    parser.add_argument('out_dir', type=str, help='out dir with new obj and mtl, texture map')
    parser.add_argument('--infer_cnt', type=int, default=4, help='infer obj cnt, sample from json')
    parser.add_argument("--lrm_mode", action="store_true", default=False,
                        help="use lrm mode. temp. need remove in future!TODO")
    parser.add_argument('--pose_json', type=str, default='',
                        help='if is empty, use json in in_dataset_json, else force use this json as render pose')
    args = parser.parse_args()

    optim_texcreator_pipeline(args.in_model_path, args.in_dataset_json, args.out_dir,
                              infer_cnt=args.infer_cnt, lrm_mode=args.lrm_mode, pose_json=args.pose_json)

    # Done.
    print("Done.")

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
