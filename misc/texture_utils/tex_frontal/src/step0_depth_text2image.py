import numpy as np
import torch
from PIL import Image
import argparse
import traceback
import torchvision.transforms as transforms

import logging
import os

import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.infer_control_depth_sdxl import DepthControlPipe
from src.render_control import RenderControl
from src.utils_render import save_normalized_geom, concatenate_images_horizontally
from sam_preprocess.main_rmgb2 import Rmgb2SegPipe


# render-infer
class DepthT2IPipeline:

    def __init__(self, geom_res=1024, device="cuda", use_sdxl=True, seed=1234):
        self.device = device

        self.render_control = RenderControl(geom_res, device=device)
        self.geom_res = geom_res

        self.seg_pipe = Rmgb2SegPipe()
        self.dc_pipe = DepthControlPipe(use_sdxl=use_sdxl, seed=seed)

    def mesh_depth_crtl_t2i(
        self,
        in_obj,
        prompt="",
        in_image_path_list=None,
        mesh_process="keep_raw",
        use_ortho=True,
        scale_factor=1.0,
        scale_xyz=0.9,
        control_scale=0.8,
        cfg_scale=5.0,
        num_inference_steps=30,
        ip_adapter_scale=1.5,
        add_prompt=", with beautiful color and lots of details, blender 3D Object Rendering with Empty Background. High Quality, HDR, UHD, 4K",
        negative_prompt="worst quality, low quality, low res, blurry, nsfw, nude, censored",
        out_dir=None,
    ):
        """obj render depth and generate image

        Args:
            in_obj: in mesh path
            in_image: in image path
            prompt: _description_. Defaults to "".
            use_ortho: _description_. Defaults to True.
            scale_factor: for mesh normalized. Defaults to 1.0.
            scale_xyz: for ortho. Defaults to 0.9.
            control_scale: control_scale. Defaults to 0.9.
            cfg_scale: cfg. Defaults to 5.
            out_dir: save mid results if is not None. Defaults to None.

        Returns:
            depths_control [4, geom_res, geom_res, 3] tensor in [0, 1]
            result_image PIL geom_res, geom_res, 3
        """
        prompt = prompt if prompt is not None else ""

        # [4, geom_res, geom_res, 3] tensor in [0, 1]
        if use_ortho:
            depths_control, out_obj = self.render_control.render_mesh_ortho_depth(in_obj,
                                                                         mesh_process=mesh_process,
                                                                         image_size=self.geom_res,
                                                                         scale_factor=scale_factor,
                                                                         scale_xyz=scale_xyz)
        else:
            depths_control, out_obj = self.render_control.render_mesh_depth(in_obj)

        in_depth = depths_control[0]
        # zoom depth
        zoom_range, zoom_offset = 0.2, 0.4
        in_depth[in_depth > 0] = in_depth[in_depth > 0] * zoom_range + zoom_offset

        normals_transforms = transforms.Compose([
            transforms.GaussianBlur(5, 5 // 3 + 1)
        ])
        
        in_depth_ = in_depth.permute(2, 0, 1)
        in_depth = normals_transforms(in_depth_).permute(1, 2, 0)
        print('in_depth ', in_depth.min(), in_depth.max())
        
        
        result_image = self.dc_pipe.infer_depth(
            in_depth,
            prompt=prompt,
            add_prompt=add_prompt,
            in_image_path_list=in_image_path_list,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=control_scale,
            guidance_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            ip_adapter_scale=ip_adapter_scale,
            res=None,
        )

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

            save_name = prompt.replace(" ", "_")
            result_image.save(os.path.join(out_dir, f"cfg{cfg_scale}_c{control_scale}_{save_name}.png"))
            # save_normalized_geom(depths_control, os.path.join(out_dir, f"in_depth.png"))

            render_mask = (in_depth[..., -1] > 0).detach().cpu().numpy()
            out_view = np.array(result_image)
            out_view[~render_mask] = (0.6 * 255.0, 0, 0)

            out_view_pil = Image.fromarray((out_view).round().astype("uint8")).convert("RGB")
            out_depth_pil = Image.fromarray((in_depth.detach().cpu().numpy() * 255.0).round().astype("uint8"))

            concatenate_images_horizontally([result_image, out_view_pil, out_depth_pil],
                                            os.path.join(out_dir, f"masked.png"))

        return depths_control, result_image, out_obj

    def main_infer_mesh2image(self,
                              in_obj,
                              prompt="",
                              in_image_path_list=None,
                              out_image_path=None,
                              mesh2image_extra_params=dict()):
        """render depth from in_obj and infer depth-crtl sdxl. generate image

        Args:
            in_obj: obj path
            prompt: _description_. Defaults to "".
            in_image_path_list: list of str or None
            out_image_path: need. Defaults to None.
            mesh2image_extra_params: {
                mesh_process: "keep_raw", #rot2
            }

        Returns:
            suc_flag: T/F
            result: dict of in/out and feedback
        """
        suc_flag = False
        result = {
            "in_obj": in_obj,
            "out_obj": in_obj,
            "prompt": prompt,
            "in_image_path_list": in_image_path_list,
            "mesh_process": None,
            "out_path": None,
            "feedback": "ok",
        }
        try:
            assert os.path.exists(in_obj), f"can not fin in_obj={in_obj}"
            mesh_process = mesh2image_extra_params.get("mesh_process", "keep_raw")
            result["mesh_process"] = mesh_process
            result["control_scale"] = mesh2image_extra_params.get("control_scale", 0.6)
            result["cfg_scale"] = mesh2image_extra_params.get("cfg_scale", 7.0)
            result["num_inference_steps"] = mesh2image_extra_params.get("num_inference_steps", 20)
            result["ip_adapter_scale"] = mesh2image_extra_params.get("ip_adapter_scale", 0.8)
            depths_control, result_image, out_obj = self.mesh_depth_crtl_t2i(in_obj, prompt=prompt,
                                                                    in_image_path_list=in_image_path_list,
                                                                    mesh_process=result["mesh_process"],
                                                                    control_scale=result["control_scale"],
                                                                    cfg_scale=result["cfg_scale"],
                                                                    num_inference_steps=result["num_inference_steps"],
                                                                    ip_adapter_scale=result["ip_adapter_scale"],
                                                                    )
            if out_image_path is not None:
                result["out_obj"] = out_obj
                os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
                out_pre, out_ext = os.path.splitext(out_image_path)
                if out_ext == ".png" or out_ext == ".jpg":
                    result_image.save(out_pre + "_infer.jpg")
                    self.seg_pipe.seg_img(result_image, cvt_gray_bg=True, out_image_path=out_image_path)
                    result["out_path"] = out_image_path

                    Image.fromarray((depths_control[0].detach().cpu().numpy() * 255.0).astype(
                        np.uint8)).save(out_pre + "_depth.png")

                    suc_flag = True
                else:
                    result["feedback"] = f"invalid out ext={out_ext}"
        except Exception as e:
            print(f"[ERROR] main_infer_mesh2image failed  e={e}")
            result["feedback"] = str(e)
            traceback.print_exc()

        return suc_flag, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render obj global xyz and normal')
    # parser.add_argument(
    #     '--in_obj',
    #     type=str,
    #     default="/aigc_cfs_gdp/jiawei/data/aitexture/2ec5531b-7b74-47d4-886c-2d014331da30/geom/raw_mesh.obj",
    #     help='path of obj')
    # # parser.add_argument('--prompt', type=str, default="red tops with floral patterns")
    # parser.add_argument('--prompt', type=str, default="Floral style short sleeves")
    # parser.add_argument('--in_image_path', type=str, default="")
    # # parser.add_argument('--in_image_path', type=str, default="/aigc_cfs_gdp/jiawei/data/aitexture/93e3e8a8-58bf-4a22-8092-69ec2728963d/huluobo.jpg")
    # parser.add_argument('--out_dir',
    #                     type=str,
    #                     default="/aigc_cfs_gdp/jiawei/data/aitexture/2ec5531b-7b74-47d4-886c-2d014331da30/ipa_plus/text_smooth",
    #                     help='out root path')
    # parser.add_argument(
    #     '--in_obj',
    #     type=str,
    #     default="/aigc_cfs_gdp/jiawei/data/aitexture/56add50a-7278-4e76-9783-fe0de56cf74a/geom/raw_mesh.obj",
    #     help='path of obj')
    # # parser.add_argument('--prompt', type=str, default="")
    # parser.add_argument('--prompt', type=str, default="Bloodstained")
    # parser.add_argument('--in_image_path', type=str, default="")
    # # parser.add_argument('--in_image_path', type=str, default="/aigc_cfs_gdp/jiawei/data/aitexture/56add50a-7278-4e76-9783-fe0de56cf74a/liuyifei.jpg")
    # parser.add_argument('--out_dir',
    #                     type=str,
    #                     default="/aigc_cfs_gdp/jiawei/data/aitexture/56add50a-7278-4e76-9783-fe0de56cf74a/ipa_plus/text",
    #                     help='out root path')
    parser.add_argument(
        '--in_obj',
        type=str,
        default="/aigc_cfs_gdp/jiawei/data/aitexture/5d76c2cd-2a92-448e-990d-28819b6cc43e/geom/raw_mesh.obj",
        help='path of obj')
    parser.add_argument('--prompt', type=str, default="Cute little dog")
    parser.add_argument('--in_image_path', type=str, default="")
    # parser.add_argument('--in_image_path', type=str, default="/aigc_cfs_gdp/sz/result/pipe_test/f467fb42-4c8a-4d86-ada5-a24e0e3cb923/hello_cfg/mesh2image.png")
    parser.add_argument('--out_dir',
                        type=str,
                        default="/aigc_cfs_gdp/jiawei/data/aitexture/5d76c2cd-2a92-448e-990d-28819b6cc43e/text",
                        help='out root path')
    
    parser.add_argument('--use_ortho', type=int, default=1, help='')
    parser.add_argument('--control_scale', type=float, default=0.7, help='')
    parser.add_argument('--cfg_scale', type=float, default=7.0, help='')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='')
    parser.add_argument('--ip_adapter_scale', type=float, default=0.8, help='')
    args = parser.parse_args()

    in_obj = args.in_obj
    prompt = args.prompt
    in_image_path = args.in_image_path
    in_image_path_list = [in_image_path] if in_image_path else None
    out_dir = args.out_dir
    use_ortho = bool(args.use_ortho)

    control_scale = args.control_scale
    cfg_scale = args.cfg_scale
    num_inference_steps = args.num_inference_steps
    ip_adapter_scale = args.ip_adapter_scale

    dt2i_pipe = DepthT2IPipeline(seed=8986)

    for cfg_scale in [cfg_scale]:
        for control_scale in [control_scale]:
            for ip_adapter_scale in [ip_adapter_scale]:
    # for cfg_scale in [5, 7]:
    #     for control_scale in [0.5, 0.6]:
    #         for ip_adapter_scale in [0.4, 0.6, 0.8]:
                mesh2image_extra_params = {
                    "mesh_process": "keep_raw",
                    "control_scale": control_scale,
                    "cfg_scale": cfg_scale,
                    "num_inference_steps": num_inference_steps,
                    "ip_adapter_scale": ip_adapter_scale,
                }
                out_dir_sub = os.path.join(out_dir, f"cfg_{cfg_scale}_c{control_scale}_ipa{ip_adapter_scale}")
                out_image_path = os.path.join(out_dir_sub, "mesh2image.png")
                suc_flag, result = dt2i_pipe.main_infer_mesh2image(in_obj,
                                                prompt=prompt,
                                                in_image_path_list=in_image_path_list,
                                                out_image_path=out_image_path,
                                                mesh2image_extra_params=mesh2image_extra_params)
    assert suc_flag
    print('result', result)

    # for cfg_scale in [cfg_scale]:
    #     for control_scale in [control_scale]:
    # # for cfg_scale in [3, 5, 7, 9, 11]:
    # #     for control_scale in [0.5, 0.8, 0.9, 1.0]:
    #         out_dir_sub = os.path.join(out_dir, f"cfg_{cfg_scale}_c{control_scale}")
    #         depths_control, result_image = dt2i_pipe.mesh_depth_crtl_t2i(in_obj,
    #                                                                      prompt=prompt,
    #                                                                      use_ortho=use_ortho,
    #                                                                      control_scale=control_scale,
    #                                                                      cfg_scale=cfg_scale,
    #                                                                      out_dir=out_dir_sub)
