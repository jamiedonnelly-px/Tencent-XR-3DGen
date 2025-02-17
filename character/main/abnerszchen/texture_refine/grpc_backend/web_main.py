from sympy import var
import os
import sys
from collections import deque
import time
import json
import random
import string
import logging
import subprocess
# os.environ["GRADIO_EXAMPLES_CACHE"] = "/path/to/your/cache/folder"  # TODO(csz)
import gradio as gr

import sys
web_ui_dir = os.path.dirname(os.path.abspath(__file__))
codedir = os.path.dirname(web_ui_dir)
sys.path.append(web_ui_dir)
sys.path.append(codedir)


from web_setup import ui_setup
from client_texgen import TexGenClient
from softlink_gradio import softlink_tmp_gradio

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)    
    return data

def log_time_list(time_list):
    try:
        if not time_list:
            return
        print('time_list ', time_list)
        prev_timestamp = time_list[0][1]
        time_sum = 0
        for name, time_stp in time_list:
            stage_duration = time_stp - prev_timestamp
            time_sum += stage_duration
            print("{} stage duration: {:.2f} seconds".format(name, stage_duration))
            prev_timestamp = time_stp
        print(f"time_sum: {time_sum:.2f} seconds")
    except Exception as e:
        print('skip log_time_list ', e)
    return


class WebTexImgUV:
    def __init__(self, cfg_json="client_texgen.json", model_name="uv_mcwy", device='cuda'):
        self.grpc_client = TexGenClient(client_cfg_json=cfg_json, model_name=model_name, device=device)
        self.device = device

        self.action = None
        self.ui = None
        self.log_queue = deque(maxlen=100)
        start_time_str = time.strftime("%Y_%m_%d_%H_%M")
        self.log_dir = os.path.join(self.grpc_client.run_cfg.log_root_dir, start_time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        softlink_tmp_gradio("/tmp/gradio", "/aigc_cfs_3/sz/server/tmp_gradio")  # TODO as cfg

        self.make_ui()
        logging.info(f"make_ui done")

    def make_ui(self):
        self.ui = ui_setup(self, self.grpc_client.model_name_list)
        self.ui.queue(max_size=10).launch(server_name='0.0.0.0', server_port=80)

    def reset_configs(self):
        """reset configs as default run_cfg

        Returns:
            _description_
        """
        self.grpc_client = TexGenClient(client_cfg_json=self.grpc_client.cfg_json, device=self.device)

        logging.info(f'reset_configs done , model_name:{self.grpc_client.model_name}, ip set: {self.grpc_client.run_cfg.server_addr}')
        return gr.Dropdown(value=self.grpc_client.model_name, choices=self.grpc_client.model_name_list)

    def select_models(self, model_name):
        self.grpc_client = TexGenClient(client_cfg_json=self.grpc_client.cfg_json, model_name=model_name, device=self.device)
        logging.info(f"select_models {model_name}, model_name:{self.grpc_client.model_name},, ip set: {self.grpc_client.run_cfg.server_addr}")
        return gr.Dropdown(value=self.grpc_client.model_name, choices=self.grpc_client.model_name_list)

    def wrap_input(self, blk_model_in, blk_image_in, blk_prompt, blk_negative_prompt, use_img=True, use_text=True):
        """convert glb to obj. convert '' as None

        Args:
            blk_model_in: _description_
            blk_image_in: _description_
            blk_prompt: _description_
            blk_negative_prompt: TODO
            use_img: _description_. Defaults to True.
            use_text: _description_. Defaults to True.

        Returns:
            _description_
        """
        in_mesh_path, out_objs_dir, in_prompts, in_condi_img = None, '', '', ''
        if os.path.exists(blk_model_in):
            in_mesh_path = blk_model_in
        else:
            return in_mesh_path, out_objs_dir, in_prompts, in_condi_img

        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        start_time_str = time.strftime("%d_%H_%M_%s")
        out_objs_dir = os.path.join(self.log_dir, start_time_str, random_string)

        if use_img and os.path.exists(blk_image_in):
            in_condi_img = blk_image_in
        if use_text and blk_prompt is not None:
            if isinstance(blk_prompt, list):
                in_prompts = ":::".join(blk_prompt)
            elif isinstance(blk_prompt, str):
                in_prompts = blk_prompt
            else:
                logging.error('[ERROR] invalid blk_prompt', blk_prompt)
                in_prompts = ''
        return in_mesh_path, out_objs_dir, in_prompts, in_condi_img

    def pipe_text(
        self,
        blk_model_in,
        blk_image_in,
        blk_prompt,
        blk_negative_prompt,
        slider_num_inference_steps,
        slider_guidance_scale,
        slider_controlnet_conditioning_scale,
        slider_ip_adapter_scale,
    ):
        out = self.pipe_helper(
            "pipe_text",
            blk_model_in,
            blk_image_in,
            blk_prompt,
            blk_negative_prompt,
            slider_num_inference_steps,
            slider_guidance_scale,
            slider_controlnet_conditioning_scale,
            slider_ip_adapter_scale,
        )
        return out

    def pipe_image(
        self,
        blk_model_in,
        blk_image_in,
        blk_prompt,
        blk_negative_prompt,
        slider_num_inference_steps,
        slider_guidance_scale,
        slider_controlnet_conditioning_scale,
        slider_ip_adapter_scale,
    ):
        out = self.pipe_helper(
            "pipe_image",
            blk_model_in,
            blk_image_in,
            blk_prompt,
            blk_negative_prompt,
            slider_num_inference_steps,
            slider_guidance_scale,
            slider_controlnet_conditioning_scale,
            slider_ip_adapter_scale,
        )
        return out

    def pipe_mix(
        self,
        blk_model_in,
        blk_image_in,
        blk_prompt,
        blk_negative_prompt,
        slider_num_inference_steps,
        slider_guidance_scale,
        slider_controlnet_conditioning_scale,
        slider_ip_adapter_scale,
    ):
        out = self.pipe_helper(
            "pipe_mix",
            blk_model_in,
            blk_image_in,
            blk_prompt,
            blk_negative_prompt,
            slider_num_inference_steps,
            slider_guidance_scale,
            slider_controlnet_conditioning_scale,
            slider_ip_adapter_scale,
        )
        return out


    def pipe_helper(
        self,
        action,
        blk_model_in,
        blk_image_in,
        blk_prompt,
        blk_negative_prompt,
        slider_num_inference_steps,
        slider_guidance_scale,
        slider_controlnet_conditioning_scale,
        slider_ip_adapter_scale,        
        progress=gr.Progress(),
    ):
        use_img, use_text = True, True
        if action == "pipe_mix":
            use_img = use_text = True
        elif action == "pipe_text":
            use_img = False
        elif action == "pipe_image":
            use_text = False
        else:
            raise NotImplementedError
        self.action = action
        start_time_str = time.strftime("%Y_%m_%d_%H_%M")
        logging.info(f"request {action} time : {start_time_str}")

        time_list = []
        time_list.append(("start", time.time()))
        print(
            f"input of {action}:",
            blk_model_in,
            blk_image_in,
            blk_prompt,
            blk_negative_prompt,
            slider_num_inference_steps,
            slider_guidance_scale,
            slider_controlnet_conditioning_scale,
            slider_ip_adapter_scale,
        )
        in_mesh_path, out_objs_dir, in_prompts, in_condi_img = self.wrap_input(
            blk_model_in,
            blk_image_in,
            blk_prompt,
            blk_negative_prompt,
            use_img=use_img,
            use_text=use_text,
        )
        print(
            f"wrap_input of {action}:",
            in_mesh_path,
            out_objs_dir,
            in_prompts,
            in_condi_img,
        )
        time_list.append(("wrap_input", time.time()))

        out_glb_path, out_tex_path = "", ""
        try:
            pipe_type = self.grpc_client.run_cfg.pipe_type
            logging.info(f"pipe_type {pipe_type}")
            in_obj_type = ""    # TODO
            out_mesh_paths = self.grpc_client.client_obj_tex_gen(
                pipe_type,
                in_mesh_path,
                out_objs_dir,
                in_prompts=in_prompts,
                in_condi_img=in_condi_img,
                in_obj_type=in_obj_type,
                out_mesh_format="glb",
                uv_res=1024,    # TODO
                num_inference_steps=int(slider_num_inference_steps),
                guidance_scale=float(slider_guidance_scale),
                controlnet_conditioning_scale=float(slider_controlnet_conditioning_scale),
                ip_adapter_scale=float(slider_ip_adapter_scale),
                debug_save=True,
            )
            if not out_mesh_paths or len(out_mesh_paths) < 1:
                raise ValueError(
                    f"invalid out_mesh_paths of {action}:client_obj_tex_gen failed with pipe_type {pipe_type}"
                )
            time_list.append(("client_obj_tex_gen", time.time()))

            print("debug out_mesh_paths ", out_mesh_paths)
            out_glb_path = out_mesh_paths[0]    # TODO(csz) only use first obj for gradio now
            out_tex_path = os.path.join(os.path.dirname(out_glb_path), 'texture_kd.png')
        except Exception as e:
            print(f"[ERROR]{action}:client_obj_tex_gen failed with pipe_type {pipe_type} ", e)

        # self.file_path_list = [x[0] for x in result_path_list]
        logging.info(f"done {action} time: {time.strftime('%Y-%m-%d-%H:%M:%S')}")
        log_time_list(time_list)

        out = [out_glb_path, out_tex_path]
        return out      

    # def generate_3dcharactor_textandimage(self, prompt, negative_prompt, ref_image, mix_weight, max_face_num, progress=gr.Progress()):
    #     print("request time : {}".format(time.strftime('%Y-%m-%d-%H:%M:%S')))
    #     if prompt is None or ref_image is None:
    #         return
    #     self.action = "generate_3dcharactor_image"
    #     self.image = preprocess_image(ref_image)
    #     result_path_list = self.pipeline.generate_with_textandimage(prompt, self.image, mix_weight, max_face_num, progress) # [(tar_path, gif_path), ...]
    #     self.file_path_list = [x[0] for x in result_path_list]
    #     print("success time: {}".format(time.strftime('%Y-%m-%d-%H:%M:%S')))
    #     return [gr.File.update(value=x[0], visible=True) for x in result_path_list] + \
    #         [x[1] for x in result_path_list] + \
    #         [gr.Button.update(variant="primary", visible=True) for x in range(4)]


if __name__ == "__main__":
    WebTexImgUV()
