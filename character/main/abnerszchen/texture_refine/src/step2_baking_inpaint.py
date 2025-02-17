import numpy as np
import torch
from PIL import Image
import argparse

import logging
import os
import time
import traceback
import sys
import cv2

current_script_path = os.path.abspath(__file__)
src_root = os.path.dirname(current_script_path)
project_root = os.path.dirname(src_root)
sys.path.append(src_root)
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.nv_diff_bake import NvdiffRender
from src.utils_render import split_image, make_image_grid, numpy_to_pil, util_print_times

from src.utils_uv import main_inpaint_mesh
from src.rmbg_interface import RmbgMain

# from tdmq.tdmq_interface_cvt import BlenderCvtInterface, init_job_id


class BakePipeline():

    def __init__(self, texture_size=1024, render_size=2048):
        """bake 4 views and inpaint knn

        Args:
            texture_size: uv resolution. Defaults to 1024.
            render_size: render resolution. Defaults to 1024.
        """
        self.device = "cuda"
        self.texture_size = texture_size
        self.render_size = render_size

        self.seg_pipe = RmbgMain()
        # self.blender_cvt_interface = BlenderCvtInterface(os.path.join(project_root, "tdmq/configs/tdmq_blender.json"))

    def close(self):
        # self.blender_cvt_interface.close()
        return

    def bake_inpaint_from_4views(self,
                                 in_mesh_path,
                                 image_nps,
                                 out_dir=None,
                                 mesh_process="keep_raw",
                                 scale_factor=1.0,
                                 scale_xyz=0.9,
                                 out_glb=False, # TODO
                                 save_debug=False):
        """direct bake 4 views with cos-weight.

        Args:
            in_mesh_path: obj path
            image_nps: [4, 1024, 1024, 3] np in [0,1]
            out_dir: save obj and mid results. 
            mesh_process: rot2, / keep_raw / ..
            scale_factor: _description_. Defaults to 1.0.
            scale_xyz: _description_. Defaults to 0.9.

        Returns:
            out_mesh_path out mesh path, glb path if out_glb else obj path
        """
        try:
            assert os.path.exists(in_mesh_path), f"can not find in_mesh_path {in_mesh_path}"

            t_list = []
            t_list.append((time.time(), "start"))
            hard_masks = self.seg_erode_hard_masks(image_nps)
            t_list.append((time.time(), "seg_erode_hard_masks"))
            
            diff_render = NvdiffRender(render_res=self.render_size, tex_res=self.texture_size)
            mesh_path, see_mask, tex_hwc_np = diff_render.bake_mesh_4views(
                in_mesh_path,
                image_nps,
                hard_masks=hard_masks,
                mesh_process=mesh_process,
                scale_factor=scale_factor,
                scale_xyz=scale_xyz,
                out_dir=out_dir,
                save_debug=save_debug,
            )
            t_list.append((time.time(), "bake_mesh_4views"))
            logging.info(f"baking done")

            out_obj_path, uv_tex_new = main_inpaint_mesh(diff_render.raw_mesh, # set as Mesh in
                                                         see_mask,
                                                         tex_hwc_np,
                                                         out_dir,
                                                         resolution=self.texture_size,
                                                         mesh_process=mesh_process,
                                                         save_debug=save_debug)
            del diff_render
            t_list.append((time.time(), "main_inpaint_mesh"))

            logging.info(f"inpaint done, save to {out_obj_path}")

            out_mesh_path = out_obj_path
            if out_glb:
                out_glb = out_obj_path.replace(".obj", ".glb")

                job_id = init_job_id()
                result_dict = self.blender_cvt_interface.blocking_call_obj_to_glb(
                    job_id,
                    out_obj_path,
                    out_glb,
                )
                suc = result_dict[job_id]["blender_cvt"].get("success", False)
                if suc:
                    out_mesh_path = out_glb
                else:
                    logging.error(f"ERROR need glb but run blender_cvt failed job_id={job_id}")
            
            util_print_times(t_list, "bake_inpaint_from_4views")
        except Exception as e:
            print(f"[ERROR] when bake_inpaint_from_4views {e}")
            traceback.print_exc()
            return None
        return out_mesh_path

    def seg_erode_hard_masks(self, image_nps, erode_kernel=9, erode_iters=3,):
        seg_nps = self.seg_pipe.infer_image_nps(image_nps)
        hard_masks = []

        for i, seg_np in enumerate(seg_nps):
            render_mask = seg_np[..., -1:]
            if not erode_kernel or erode_kernel < 0:
                eroded_mask = (render_mask > 0)
            else:
                mask = (render_mask > 0).astype(np.uint8)
                kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=erode_iters)
                # [h,w,1]
                eroded_mask = torch.tensor(eroded_mask, device=self.device).unsqueeze(-1)
            hard_masks.append(eroded_mask)
        
        hard_masks = torch.stack(hard_masks, dim=0)
        return hard_masks
    
    def main_infer_from_geom2rgb(self,
                                   in_mesh_path,
                                   in_img_npy,
                                   out_dir=None,
                                   mesh_process="keep_raw", # "keep_raw"
                                   scale_factor=1.0,
                                   scale_xyz=0.9,
                                   save_debug=False,
                                   ):
        try:
            raw_image_nps = np.load(in_img_npy)  # [4, 3, h, w]
            print('debug load raw_image_nps ', raw_image_nps.shape)
            # TODO refine
            if raw_image_nps.shape[-1] == self.render_size:
                image_nps = raw_image_nps.transpose(0, 2, 3, 1) / 255.0
            else:
                image_nps = []
                for image_np in raw_image_nps:
                    pil = (Image.fromarray(image_np.transpose(1, 2, 0))).resize((self.render_size, self.render_size),
                                                                                Image.BILINEAR)
                    image_nps.append(np.array(pil) / 255.0)
                image_nps = np.stack(image_nps, axis=0)

            out_mesh_path = self.bake_inpaint_from_4views(in_mesh_path,
                                                        image_nps,
                                                        out_dir=out_dir,
                                                        mesh_process=mesh_process,
                                                        scale_factor=scale_factor,
                                                        scale_xyz=scale_xyz,
                                                        save_debug=save_debug)
        except Exception as e:
            print(f"[ERROR] e={e}")
            traceback.print_exc()
            return None
        return out_mesh_path

    def main_infer_from_image_grid(self,
                                   in_mesh_path,
                                   in_img,
                                   out_dir=None,
                                   mesh_process="keep_raw", # "rot2"
                                   scale_factor=1.0,
                                   scale_xyz=0.9,
                                   save_debug=False):
        image_pils = split_image(Image.open(in_img).resize((self.render_size * 2, self.render_size * 2)), 2,
                                 2)  # TODO SR
        if save_debug:
            os.makedirs(out_dir, exist_ok=True)
            image_pils[0].save(os.path.join(out_dir, f"in.png"))
        image_nps = [np.array(img) / 255.0 for img in image_pils]
        image_nps = np.stack(image_nps, axis=0)

        out_mesh_path = self.bake_inpaint_from_4views(in_mesh_path,
                                                      image_nps,
                                                      out_dir=out_dir,
                                                      mesh_process=mesh_process,
                                                      scale_factor=scale_factor,
                                                      scale_xyz=scale_xyz,
                                                      save_debug=save_debug)
        return out_mesh_path


if __name__ == "__main__":

    # in_img = f"/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus/18000_inpaint_1e_5_guidance_scale_3.0_conditioning_scale_1.5/03633dd2-26cf-49ac-9caa-440ef14455c9.png/mask.png"
    # in_mesh_path = f"/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus/tmp/03633dd2-26cf-49ac-9caa-440ef14455c9.obj"
    render_size = 2048
    scale_factor = 1.0
    scale_xyz = 0.9
    bake_pipe = BakePipeline(render_size=render_size)

    # in_img = f"/aigc_cfs_gdp/sz/d2rgb_tex/0909/text_in/debug_eulerad30/cfg_5_c0.8/cfg5_c0.8_a_high-end_espresso_machine_with_a_built-in_burr_grinder.png"
    # in_mesh_path = f"/aigc_cfs_gdp/sz/d2rgb_tex/0909/text_in/ac1d3421bf2d4a5886c4d7f4ebf82224.obj"
    # out_dir = f"/aigc_cfs_gdp/sz/d2rgb_tex/0909/text_in/init/step2_exp12_{render_size}"

    # in_img = f"/aigc_cfs_gdp/sz/d2rgb_tex/0918/kafeiji/imc_infer.png"
    # in_mesh_path = f"/aigc_cfs_gdp/sz/d2rgb_tex/0918/kafeiji/textured.obj"
    # out_dir = f"/aigc_cfs_gdp/sz/d2rgb_tex/0918/kafeiji/step2_exp12_{render_size}"

    # # in_img = f"/aigc_cfs/xibinsong/code/zero123plus_control/results/gray_18000_75_inpaint_1e_5_guidance_scale_3.5_conditioning_scale_0.75/c4e3a8fef092469294f0a1e94f1f811f/res.png"
    # # in_mesh_path = f"/aigc_cfs/weixuan/output/craftsman/2024-09-07-15:04:08/c4e3a8fef092469294f0a1e94f1f811f.obj"
    # # out_dir = f"/aigc_cfs_gdp/sz/d2rgb_tex/0911/init_test/step2_{render_size}_ero5"
    # out_mesh_path = bake_pipe.main_infer_from_image_grid(
    #     in_mesh_path,
    #     in_img,
    #     out_dir=out_dir,
    #     mesh_process="keep_raw",
    #     scale_factor=scale_factor,
    #     scale_xyz=scale_xyz,
    #     save_debug=False,
    # )

    in_dir = "/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf"
    # in_img_npy = f"{in_dir}/imgsr/z123.npy"
    in_img_npy = f"{in_dir}/d2rgb/out/imgsr/color.npy"
    in_mesh_path = f"{in_dir}/mesh_in/mesh.obj"
    # in_mesh_path = f"{in_dir}/clay/obj_mesh_mesh.obj"
    # in_mesh_path = f"{in_dir}/verts2tex/baking/tex_mesh.obj"
    out_dir = f"/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/use_nv"
    out_mesh_path = bake_pipe.main_infer_from_geom2rgb(
        in_mesh_path,
        in_img_npy,
        out_dir=out_dir,
        mesh_process="keep_raw",
        scale_factor=scale_factor,
        scale_xyz=scale_xyz,
        save_debug=True,
    )

    print(out_mesh_path)

    bake_pipe.close()
